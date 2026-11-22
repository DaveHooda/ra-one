"""Geometry and ground routing for the arena described by the server."""
from array import array
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from heapq import heappop, heappush
from math import ceil, floor, hypot, radians, tan
import json
from pathlib import Path
import sys
from time import perf_counter


def clamp(value, low, high):
    return min(high, max(low, value))


def edges(ring):
    points = list(zip(ring[::2], ring[1::2]))
    return [(x, y, nx - x, ny - y)
            for (x, y), (nx, ny) in zip(points, points[1:] + points[:1])]


def inside(x, y, ring):
    hit = False
    for sx, sy, dx, dy in ring:
        if (sy > y) != (sy + dy > y) and x < sx + (y - sy) * dx / dy:
            hit = not hit
    return hit


@dataclass
class Prism:
    base: float
    top: float
    rings: list
    bounds: tuple

    def contains(self, x, y):
        return inside(x, y, self.rings[0]) and not any(
            inside(x, y, hole) for hole in self.rings[1:]
        )

    def overlaps(self, x, y, radius):
        xlo, ylo, xhi, yhi = self.bounds
        if x < xlo - radius or x > xhi + radius or y < ylo - radius or y > yhi + radius:
            return False
        if self.contains(x, y):
            return True
        limit = radius * radius
        for ring in self.rings:
            for sx, sy, dx, dy in ring:
                length = dx * dx + dy * dy
                if length:
                    t = clamp(((x - sx) * dx + (y - sy) * dy) / length, 0, 1)
                    if (x - sx - t * dx) ** 2 + (y - sy - t * dy) ** 2 < limit:
                        return True
        return False

    def intersects(self, a, b):
        # Clip to the prism's vertical interval, then test that XY segment
        # against the outer ring and every hole boundary.
        ax, ay, az = a
        bx, by, bz = b
        dx, dy, dz = bx - ax, by - ay, bz - az
        if not dz:
            if not self.base <= az <= self.top:
                return False
            lo, hi = 0.0, 1.0
        else:
            t0, t1 = sorted(((self.base - az) / dz, (self.top - az) / dz))
            lo, hi = max(0.0, t0), min(1.0, t1)
            if lo > hi:
                return False
        xlo, ylo, xhi, yhi = self.bounds
        px, py = ax + lo * dx, ay + lo * dy
        qx, qy = ax + hi * dx, ay + hi * dy
        if max(px, qx) < xlo or min(px, qx) > xhi or max(py, qy) < ylo or min(py, qy) > yhi:
            return False
        if self.contains(px, py) or self.contains(qx, qy):
            return True
        for ring in self.rings:
            for sx, sy, ex, ey in ring:
                determinant = dx * ey - dy * ex
                if abs(determinant) < 1e-9:
                    continue
                rx, ry = sx - ax, sy - ay
                t = (rx * ey - ry * ex) / determinant
                if lo <= t <= hi and 0 <= (rx * dy - ry * dx) / determinant <= 1:
                    return True
        return False


class Arena:
    bucket = 20
    cell = 6

    def __init__(self, map_info, stats):
        directory = Path(map_info["dir"])
        world_file = directory / next(p for p in map_info["files"] if p.endswith("world.json"))
        world = json.loads(world_file.read_text())
        terrain = world["terrain"]
        self.gx, self.gy = terrain["minX"], terrain["minY"]
        self.nx, self.ny = terrain["nx"], terrain["ny"]
        self.step, self.scale = terrain["step"], terrain["scale"]
        self.heights = array("h", (world_file.parent / terrain["file"]).read_bytes())
        if sys.byteorder == "big":
            self.heights.byteswap()
        bounds = world["bounds"]
        self.bounds = (bounds["minX"], bounds["minY"], bounds["maxX"], bounds["maxY"])
        self.step_up = stats["step_up"]
        self.grade = tan(radians(stats["max_slope_deg"]))
        self.prisms = []
        self.spatial = defaultdict(list)
        for row in world["prisms"]:
            ring = row["ring"]
            xs, ys = ring[::2], ring[1::2]
            box = min(xs), min(ys), max(xs), max(ys)
            index = len(self.prisms)
            self.prisms.append(Prism(row["base"], row["top"],
                                     [edges(ring), *(edges(hole) for hole in row["holes"])], box))
            for key in self.buckets(*box):
                self.spatial[key].append(index)

    def buckets(self, xlo, ylo, xhi, yhi):
        size = self.bucket
        for ix in range(floor(xlo / size), floor(xhi / size) + 1):
            for iy in range(floor(ylo / size), floor(yhi / size) + 1):
                yield ix, iy

    def ground(self, x, y):
        x = clamp((x - self.gx) / self.step, 0, self.nx - 1)
        y = clamp((y - self.gy) / self.step, 0, self.ny - 1)
        ix, iy = min(int(x), self.nx - 2), min(int(y), self.ny - 2)
        dx, dy = x - ix, y - iy
        a = iy * self.nx + ix
        b = a + self.nx
        h = self.heights
        return self.scale * ((h[a] * (1 - dx) + h[a + 1] * dx) * (1 - dy)
                             + (h[b] * (1 - dx) + h[b + 1] * dx) * dy)

    def clear(self, a, b):
        x, y, z = a
        dx, dy, dz = (b[0] - x, b[1] - y, b[2] - z)
        count = max(1, ceil(hypot(dx, dy) / 3))
        # Step along the ray to avoid scanning all buckets in a diagonal box.
        nearby = set()
        ix, iy = floor(x / self.bucket), floor(y / self.bucket)
        nearby.update(self.spatial.get((ix, iy), ()))
        for i in range(count + 1):
            t = i / count
            sx, sy = x + t * dx, y + t * dy
            if z + t * dz < self.ground(sx, sy) + 0.05:
                return False
            nx, ny = floor(sx / self.bucket), floor(sy / self.bucket)
            if (nx, ny) != (ix, iy):
                nearby.update(self.spatial.get((nx, ny), ()))
                if nx != ix and ny != iy:
                    nearby.update(self.spatial.get((nx, iy), ()))
                    nearby.update(self.spatial.get((ix, ny), ()))
                ix, iy = nx, ny
        return not any(self.prisms[i].intersects(a, b) for i in nearby)

    def walkable(self, x, y, z, radius, height):
        xlo, ylo, xhi, yhi = self.bounds
        if not (xlo + radius <= x <= xhi - radius and ylo + radius <= y <= yhi - radius):
            return False
        for key in self.buckets(x - radius, y - radius, x + radius, y + radius):
            for index in self.spatial.get(key, ()):
                p = self.prisms[index]
                if p.top > z + self.step_up and p.base < z + height:
                    if p.overlaps(x, y, radius):
                        return False
        return True

    def walk_segment(self, start, end, radius, height):
        x, y, z = start
        dx, dy = end[0] - x, end[1] - y
        distance = hypot(dx, dy)
        count = max(1, ceil(distance / 2))
        on_ground = z <= self.ground(x, y) + self.step_up
        for i in range(1, count + 1):
            px, py = x + dx * i / count, y + dy * i / count
            ground = self.ground(px, py)
            if on_ground and ground - z > max(self.step_up, self.grade * distance / count):
                return False
            z = ground if on_ground else max(ground, z)
            if not self.walkable(px, py, z, radius, height):
                return False
        return True

    def grid_key(self, point):
        return round(point[0] / self.cell), round(point[1] / self.cell)

    @lru_cache(maxsize=160_000)
    def grid_cell(self, key, radius, height):
        x, y = key[0] * self.cell, key[1] * self.cell
        z = self.ground(x, y)
        return (x, y, z) if self.walkable(x, y, z, radius, height) else None

    def route(self, start, goal, radius, height):
        radius, height = ceil(radius * 2) / 2, ceil(height)
        begin = self.grid_key(start)
        end = self.grid_key(goal)
        # If a blocked objective is on a footprint, route to its nearest road.
        def road_cells(anchor):
            for ring in range(4):
                cells = []
                for dx in range(-ring, ring + 1):
                    for dy in range(-ring, ring + 1):
                        if max(abs(dx), abs(dy)) != ring:
                            continue
                        key = anchor[0] + dx, anchor[1] + dy
                        pos = self.grid_cell(key, radius, height)
                        if pos:
                            cells.append((key, pos))
                yield from cells

        end = next((key for key, _ in road_cells(end)), None)
        begin = next((key for key, pos in road_cells(begin)
                      if self.walk_segment(start, pos, radius - 0.5, height)), None)
        if begin is None or end is None:
            return []
        def remaining(key):
            return hypot(key[0] - end[0], key[1] - end[1])

        frontier = [(remaining(begin), 0.0, begin)]
        costs = {begin: 0.0}
        back = {}
        best = begin
        deadline = perf_counter() + 0.008
        steps = ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1))
        while frontier and perf_counter() < deadline:
            _, cost, key = heappop(frontier)
            if cost != costs[key]:
                continue
            if remaining(key) < remaining(best):
                best = key
            if key == end:
                break
            point = self.grid_cell(key, radius, height)
            for dx, dy in steps:
                adjacent = key[0] + dx, key[1] + dy
                distance = 1.4142 if dx and dy else 1
                total = cost + distance
                if total >= costs.get(adjacent, float("inf")):
                    continue
                pos = self.grid_cell(adjacent, radius, height)
                if pos is None or not self.walk_segment(point, pos, radius, height):
                    continue
                costs[adjacent] = total
                back[adjacent] = key
                heappush(frontier, (total + 1.45 * remaining(adjacent), total, adjacent))
        path = [best]
        while path[-1] != begin:
            path.append(back[path[-1]])
        return [self.grid_cell(key, radius, height) for key in reversed(path)]
