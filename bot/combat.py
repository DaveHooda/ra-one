"""Weapon selection and aim for Mech Royale's simulated weapons."""
from dataclasses import dataclass
from math import atan2, cos, dist, hypot, pi, radians, sin, sqrt

from arena import clamp


def angle(value):
    return (value + pi) % (2 * pi) - pi


def angles(a, b):
    dx, dy, dz = (b[i] - a[i] for i in range(3))
    return atan2(dy, dx), atan2(dz, hypot(dx, dy))


@dataclass
class Shot:
    aim: tuple
    horizontal: float
    vertical: float
    value: float
    slot: int


class Combat:
    def __init__(self, arena, config):
        self.arena = arena
        self.config = config
        self.tick = 1 / config["match"]["tick_hz"]
        self.interval = self.tick * config["match"]["decide_every"]

    def dimensions(self, target):
        if target.get("kind") == "mech":
            mech = self.config["mech"]
            scale = 1 + self.config["cubes"]["size_per_cube"] * target.get("cubes", 0)
            return mech["radius"] * scale, mech["height"] * scale
        if target.get("kind") == "npc":
            npc = self.config["npc"]
            return npc.get("radius", 1.8), npc.get("height", 3.5)
        return self.config["crates"]["radius"], self.config["crates"]["height"]

    def target_at(self, target, height, seconds, fraction):
        x, y, z = target["pos"]
        vx, vy, vz = target.get("vel", (0, 0, 0))
        px, py = x + vx * seconds, y + vy * seconds
        if abs(vz) > 0.5:
            gravity = self.config["mech"]["gravity"]
            fall = max(0, seconds - 0.12)
            z = max(self.arena.ground(px, py), z + vz * seconds - 0.5 * gravity * fall ** 2)
        return px, py, z + fraction * height

    def straight(self, origin, target, height, speed, launch, fraction):
        seconds = launch
        for _ in range(3):
            point = self.target_at(target, height, seconds, fraction)
            if not speed:
                break
            seconds = launch + max(0, dist(origin, point) - 3.0) / speed
        return point

    def mortar(self, origin, target, height, settings):
        speed, gravity = settings["speed"], settings["gravity"]
        # Find the first ballistic intercept. Later roots describe high arcs.
        def intercept(seconds):
            point = self.target_at(target, height, self.tick + seconds, 0.42)
            raised = (*point[:2], point[2] + 0.5 * gravity * seconds ** 2)
            return dist(origin, raised) - (speed * seconds + 3), raised

        limit = min(4.2, settings["ttl"])
        lo, hi = 0.03, 0.11
        while hi < limit and intercept(hi)[0] > 0:
            lo, hi = hi, hi + 0.12
        if hi >= limit:
            return None
        for _ in range(10):
            middle = (lo + hi) / 2
            if intercept(middle)[0] > 0:
                lo = middle
            else:
                hi = middle
        _, point = intercept(hi)
        yaw, pitch = angles(origin, point)
        if hypot(point[0] - origin[0], point[1] - origin[1]) < max(11, settings["splash_radius"] + 4):
            return None
        dx, dy, dz = cos(yaw) * cos(pitch), sin(yaw) * cos(pitch), sin(pitch)
        count = max(2, int(hi / 0.13) + 1)
        previous = origin
        for i in range(count + 1):
            t = hi * i / count
            travel = 3 + speed * t
            current = (origin[0] + dx * travel, origin[1] + dy * travel,
                       origin[2] + dz * travel - 0.5 * gravity * t * t)
            if not self.arena.clear(previous, current):
                return None
            previous = current
        return yaw, pitch

    def line_of_sight(self, me, target):
        stats = me["stats"]
        origin = (*me["pos"][:2], me["pos"][2] + stats["muzzle_height"])
        _, height = self.dimensions(target)
        for fraction in (0.65, 0.88, 0.42):
            point = self.target_at(target, height, 0, fraction)
            pitch = angles(origin, point)[1]
            if stats["min_pitch"] <= pitch <= stats["max_pitch"] and self.arena.clear(origin, point):
                return True
        return False

    def solve(self, me, target, slot):
        weapon = me["weapons"][slot]
        kind, settings = weapon["kind"], weapon["stats"]
        x, y, z = me["pos"]
        radius, height = self.dimensions(target)
        stats = me["stats"]
        origin = x, y, z + stats["muzzle_height"]
        distance = dist(origin, (*target["pos"][:2], target["pos"][2] + 0.60 * height))
        ready = weapon["cooldown"] <= self.interval + 1e-5
        if kind == "mortar":
            if not ready or distance > settings["speed"] ** 2 / settings["gravity"] * 1.07:
                return None
            result = self.mortar((x, y, z + stats["height"] + 0.5), target, height, settings)
            if result is None:
                return None
            yaw, pitch = result
            yaw_tolerance = clamp(1.2 * radius / max(distance, 1), 0.022, 0.11)
            pitch_tolerance = clamp(3 / max(distance, 1), 0.016, 0.09)
            value = 116
        else:
            if distance > settings["range"] + radius * 0.4:
                return None
            if kind == "lightning" and me["energy"] < settings["energy"]:
                return None
            for fraction in (0.60, 0.82, 0.40):
                point = self.straight(origin, target, height, settings.get("speed", 0),
                                      self.tick, fraction)
                yaw, pitch = angles(origin, point)
                if stats["min_pitch"] <= pitch <= stats["max_pitch"] and self.arena.clear(origin, point):
                    break
            else:
                return None
            horizontal = hypot(point[0] - x, point[1] - y)
            yaw_tolerance = clamp(0.65 * radius / max(horizontal, 1), 0.005, 0.26)
            vertical_margin = height * min(fraction, 1 - fraction)
            pitch_tolerance = clamp(0.72 * vertical_margin / max(distance, 1), 0.005, 0.20)
            value = 54 if kind == "gun" else 80
            if kind == "gun" and (weapon["reload"] > self.interval or not weapon["ammo"]):
                value *= 0.35
            if kind == "lightning":
                yaw_tolerance = radians(settings["cone_deg"]) * 0.46
                pitch_tolerance = pi
                value = 90 if ready else 20
            if kind == "laser":
                charge = weapon["charge"]
                desired = max(settings["min_charge"], settings["charge"] - self.tick / 2)
                cubes = self.config["cubes"]
                power = min(me["cubes"], cubes["stat_cap"])
                damage = settings["damage"] * (1 + cubes["damage_per_cube"] * power)
                if "mortar" in target.get("weapons", ()):
                    damage *= 1 + settings["cookoff"]
                hp = target.get("hp", float("inf"))
                fraction = hp / (damage * 0.96)
                if fraction < 1:
                    charge_fraction = (fraction - settings["min_frac"]) / (1 - settings["min_frac"])
                    desired = clamp(settings["charge"] * charge_fraction, settings["min_charge"], desired)
                value = (180 if charge >= desired else 85) if charge or ready else 18
        if not stats["min_pitch"] <= pitch <= stats["max_pitch"]:
            return None
        return Shot((yaw, pitch), yaw_tolerance, pitch_tolerance, value, slot)

    def act(self, me, target, close_pressure):
        plans = [self.solve(me, target, i)
                 for i, weapon in enumerate(me["weapons"]) if weapon] if target else []
        plans = [plan for plan in plans if plan]
        if not plans:
            return None, self.hold_charge(me)
        yaw, pitch = me["aim"]
        stats = me["stats"]
        yaw_step = stats["turn_rate"] * self.tick
        pitch_step = stats["pitch_rate"] * self.tick

        def aim_value(candidate):
            value = candidate.value
            for plan in plans:
                if plan is candidate:
                    continue
                if (abs(angle(candidate.aim[0] - plan.aim[0])) <= plan.horizontal and
                        abs(candidate.aim[1] - plan.aim[1]) <= plan.vertical):
                    value += plan.value * 0.8
            return value

        goal = max(plans, key=aim_value).aim
        after_yaw = yaw + clamp(angle(goal[0] - yaw), -yaw_step, yaw_step)
        after_pitch = pitch + clamp(goal[1] - pitch, -pitch_step, pitch_step)
        fire = self.hold_charge(me)
        for plan in plans:
            weapon = me["weapons"][plan.slot]
            kind, settings = weapon["kind"], weapon["stats"]
            yaw_error = abs(angle(after_yaw - plan.aim[0]))
            pitch_error = abs(after_pitch - plan.aim[1])
            aligned = yaw_error <= plan.horizontal and pitch_error <= plan.vertical
            ready = weapon["cooldown"] <= self.interval + 1e-5
            if kind == "laser":
                charge = weapon["charge"]
                if charge:
                    fire[plan.slot] = not (aligned and plan.value == 180)
                elif ready:
                    fire[plan.slot] = aligned or (
                        yaw_error < 0.27 and pitch_error < 0.20 and not close_pressure
                    )
            else:
                fire[plan.slot] = ready and aligned
        return goal, fire

    @staticmethod
    def hold_charge(me):
        # Keep a nearly charged laser through brief occlusion. Before forced
        # discharge, tap off so a stale lock cannot keep monopolizing the aim.
        return [bool(w and w["kind"] == "laser" and 0 < w["charge"] < w["stats"]["charge"])
                for w in me["weapons"]]
