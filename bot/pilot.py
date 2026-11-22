"""Choose safe motion while preserving the ability to engage or loot."""
from math import atan2, cos, dist, hypot, pi, sin

from arena import clamp


def xy(point):
    return point[:2]


def bearing(a, b):
    return atan2(b[1] - a[1], b[0] - a[0])


class Pilot:
    def __init__(self, arena, config):
        self.arena = arena
        self.config = config
        self.path = []
        self.route_goal = None
        self.next_plan = 0
        self.last_change = 0
        self.last_move = (0, 0)
        self.samples = ()

    def zone_goal(self, obs):
        zone, me, now = obs["zone"], obs["self"], obs["t"]
        position = xy(me["pos"])
        current_margin = zone["radius"] - dist(position, zone["center"])
        distance = dist(position, zone["next_center"])
        target_radius = max(0, zone["next_radius"] - 26)
        overshoot = distance - target_radius
        travel = max(0, overshoot) / max(4, me["stats"]["walk_speed"] * 0.64)
        due = (overshoot > 0 and now + travel + 9 >= zone["shrink_end"])
        closing = now >= zone["shrink_start"] - 5
        if not me["in_zone"] or current_margin < 16 or due or (closing and overshoot > 70):
            if distance < 0.001:
                return tuple(zone["next_center"]), current_margin < 9
            # Rotate toward the nearest safe point in the next circle.
            fraction = clamp((overshoot + 18) / distance, 0, 1)
            return tuple(p + (c - p) * fraction for p, c in zip(position, zone["next_center"])), (
                current_margin < 10 or not me["in_zone"] or due
            )
        return None, False

    def waypoint(self, obs, goal):
        me, now = obs["self"], obs["t"]
        point = me["pos"]
        reach = me["stats"]
        radius, height = reach["radius"], reach["height"]
        distance = dist(xy(point), xy(goal))
        if distance < 2:
            return goal
        facing = bearing(point, goal)
        direct = (point[0] + cos(facing) * min(distance, 26),
                  point[1] + sin(facing) * min(distance, 26))
        if self.arena.walk_segment(point, direct, radius, height):
            self.path = []
            return goal
        changed = self.route_goal is None or dist(xy(goal), xy(self.route_goal)) > 13
        if now >= self.next_plan and (changed or not self.path or now >= self.next_plan + 0.7):
            self.path = self.arena.route(point, goal, radius, height)
            self.route_goal, self.next_plan = goal, now + 0.35
        while self.path and dist(xy(point), xy(self.path[0])) < 3:
            self.path.pop(0)
        for candidate in reversed(self.path[:6]):
            if self.arena.walk_segment(point, candidate, radius, height):
                return candidate
        return goal

    def cover(self, obs, enemies):
        me, now = obs["self"], obs["t"]
        if now < self.last_change and self.samples:
            return self.samples
        self.last_change = now + 0.65
        x, y, z = me["pos"]
        stats, zone = me["stats"], obs["zone"]
        candidates = []
        threats = enemies[:3]
        for ring in (17, 31):
            for k in range(12):
                a = k * pi / 6
                point = x + ring * cos(a), y + ring * sin(a)
                if dist(point, zone["center"]) > zone["radius"] - 4:
                    continue
                if not self.arena.walk_segment((x, y, z), point, stats["radius"], stats["height"]):
                    continue
                eye = (*point, self.arena.ground(*point) + stats["eye_height"] * 0.8)
                hidden = 0
                separation = 0
                for target in threats:
                    enemy_pos = target["pos"]
                    gap = dist(point, xy(enemy_pos))
                    separation += min(120, gap)
                    if gap >= 9 and not self.arena.clear((*enemy_pos[:2], enemy_pos[2] + 7), eye):
                        hidden += 1
                score = 55 * hidden + separation / len(threats) - 0.4 * ring
                candidates.append((score, point))
        self.samples = max(candidates, default=(0, ()))[1]
        return self.samples

    def incoming(self, obs, enemies):
        me = obs["self"]
        x, y, z = me["pos"]
        nearby = []
        for p in obs["projectiles"]:
            vx, vy, _ = p["vel"]
            dx, dy = x - p["pos"][0], y - p["pos"][1]
            speed2 = vx * vx + vy * vy
            if speed2 < 1:
                continue
            t = (dx * vx + dy * vy) / speed2
            if -0.05 < t < 0.65 and (dx - vx * t) ** 2 + (dy - vy * t) ** 2 < 20 ** 2:
                nearby.append((t, p))
        lasers = []
        for enemy in enemies:
            if enemy.get("charging") and dist(xy(enemy["pos"]), (x, y)) < 105:
                lasers.append(enemy)
        return [p for _, p in sorted(nearby, key=lambda item: item[0])[:6]], lasers[:3]

    def danger(self, me, velocity, bullets, lasers):
        x, y, z = me["pos"]
        mid = z + me["stats"]["height"] / 2
        radius = me["stats"]["radius"]
        penalty = 0.0
        for p in bullets:
            px, py, pz = p["pos"]
            vx, vy, vz = p["vel"]
            rvx, rvy = vx - velocity[0], vy - velocity[1]
            dx, dy = x - px, y - py
            t = (dx * rvx + dy * rvy) / max(1, rvx * rvx + rvy * rvy)
            if not 0 < t < 0.64:
                continue
            flight_z = pz + t * vz
            width = radius + 1.2
            weight = 23
            if p["kind"] == "mortar":
                settings = self.config["weapons"]["mortar"]
                flight_z -= 0.5 * settings["gravity"] * t * t
                width += settings["splash_radius"]
                weight = 47
            if abs(flight_z - mid) < me["stats"]["height"] * 0.58 + width:
                miss = hypot(dx - rvx * t, dy - rvy * t)
                penalty += weight * clamp(1.4 - miss / width, 0, 1)
        for enemy in lasers:
            yaw = enemy["aim"][0]
            dx, dy = x - enemy["pos"][0], y - enemy["pos"][1]
            cross = -dx * sin(yaw) + dy * cos(yaw)
            future = cross + (-velocity[0] * sin(yaw) + velocity[1] * cos(yaw)) * 0.3
            if abs(cross) < radius * 2.5:
                penalty += 16 * clamp(1.3 - abs(future) / (radius + 1), 0, 1)
        return penalty

    def move(self, obs, goal, urgent, enemies, target, recover, quiet):
        me, zone, now = obs["self"], obs["zone"], obs["t"]
        x, y, z = me["pos"]
        stats = me["stats"]
        waypoint = self.waypoint(obs, goal) if goal is not None else None
        bullets, lasers = self.incoming(obs, enemies)
        fighting = target is not None and target.get("kind") == "mech"
        preferred = 60
        equipped = {w["kind"] for w in me["weapons"] if w}
        lightning = next((w for w in me["weapons"] if w and w["kind"] == "lightning"), None)
        if lightning and me["energy"] > 20 and not recover:
            preferred = min(preferred, lightning["stats"]["range"] * 0.74)
        if "laser" in equipped:
            preferred = 74
        spacing = dist((x, y), xy(target["pos"])) if fighting else 0
        if fighting and "mortar" in target.get("weapons", ()):
            preferred = max(preferred, 55)
        active = fighting or bullets or lasers or len(enemies) > 1 or (quiet and recover and enemies)
        quiet = quiet and not active and not urgent
        boost = me["grounded"] and me["energy"] > (6 if urgent else 28) and (
            urgent or (active and (recover or lasers or any(p["kind"] == "mortar" for p in bullets)))
            or (not active and not quiet and me["energy"] > 75 and waypoint is not None and
                dist((x, y), xy(waypoint)) > 32 and "lightning" not in equipped)
        )
        speed = stats["boost_speed"] if boost else stats["walk_speed"]
        if any(w and w["kind"] == "laser" and w["charge"] for w in me["weapons"]):
            speed *= self.config["mech"]["charge_move_mult"]
        reference = bearing((x, y), waypoint) if waypoint is not None else (
            bearing(target["pos"], (x, y)) if fighting else 0
        )
        variations = (0, 1, -1, 2, -2, 3, -3, 4, -4, 5, -5, 6, 99)
        if not active and waypoint is None:
            variations = (99,)
        elif not active:
            variations = (0, 1, -1, 2, -2, 3, -3, 4, -4, 6, 99)
        best = (-float("inf"), (0.0, 0.0))
        for offset in variations:
            if offset == 99:
                move = (0.0, 0.0)
            else:
                yaw = reference + offset * pi / 6
                throttle = 1
                if not active and waypoint is not None:
                    throttle = clamp(dist((x, y), xy(waypoint)) / 5, 0, 1)
                if quiet:
                    throttle = min(throttle, 0.94 * self.config["sound"]["step_speed"] / speed)
                move = cos(yaw) * throttle, sin(yaw) * throttle
            vx, vy = move[0] * speed, move[1] * speed
            velocity = (vx * 0.8 + me["vel"][0] * 0.2, vy * 0.8 + me["vel"][1] * 0.2)
            look = (x + velocity[0] * 0.58, y + velocity[1] * 0.58)
            if offset != 99 and not self.arena.walk_segment(
                (x, y, z), look, max(0.7, stats["radius"] - 0.30), stats["height"]
            ):
                continue
            score = 0.0
            if waypoint is not None:
                score += (dist((x, y), xy(waypoint)) - dist(look, xy(waypoint))) * (
                    3.5 if urgent else 1.5 if recover else 0.9 if active else 1.8
                )
            future_margin = zone["radius"] - dist(look, zone["center"])
            score -= (4 if urgent else 2) * max(0, 7 - future_margin)
            if fighting and not urgent:
                gap = dist(look, xy(target["pos"]))
                if recover:
                    score += (gap - spacing) * 0.70
                else:
                    score += (abs(spacing - preferred) - abs(gap - preferred)) * 0.90
                    side = sin(reference - bearing((x, y), target["pos"]) + offset * pi / 6)
                    if offset != 99:
                        score += (1 if int(now / 1.8) % 2 else -1) * side * 2.1
            for enemy in enemies[:4]:
                gap = dist(look, xy(enemy["pos"]))
                if enemy is not target or recover:
                    score -= 1.5 * max(0, 39 - gap)
            score -= self.danger(me, velocity, bullets, lasers)
            score += 0.8 * (move[0] * self.last_move[0] + move[1] * self.last_move[1])
            if score > best[0]:
                best = score, move
        move = best[1]
        jump = False
        stuck = waypoint is not None and dist((x, y), xy(waypoint)) > 5 and hypot(*move) < 0.25
        if stuck and (urgent or not active) and me["energy"] > 29:
            # A narrow passage or a low wall may require climbing.
            yaw = bearing((x, y), waypoint)
            move = cos(yaw), sin(yaw)
            jump = me["grounded"] and hypot(*me["vel"][:2]) < 2.2
            boost = not me["grounded"]
        elif not me["grounded"] and me["energy"] > 12 and urgent and me["vel"][2] < -2:
            boost = True
        self.last_move = move
        return move, boost and bool(hypot(*move)), jump
