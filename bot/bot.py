"""Ra-One's Mech Royale entry."""
from math import dist, pi, sin
from time import perf_counter

from arena import Arena
from combat import Combat, angle
import mechfight as mf
from pilot import Pilot, bearing, xy


UPGRADE_PRIORITY = {
    "nanites": 102, "gun_mag": 94, "laser_damage": 90, "gun_damage": 88,
    "armor": 83, "plating": 81, "mortar_reload": 75, "gun_rate": 72,
    "gun_velocity": 78, "servos": 74, "laser_charge": 76, "lightning_range": 71,
    "mortar_velocity": 60, "gyros": 56, "lightning_efficiency": 53,
    "mortar_splash": 50, "reactor": 45, "jets": 42, "sensors": 40,
    "lightning_chain": 33, "capacitor": 31, "laser_pierce": 28,
}
WEAPON_PRIORITY = {"gun": 0, "mortar": 1, "lightning": 2, "laser": 3}


class Bot:
    name = "Ra-One"

    def __init__(self, hello):
        self.config = hello["config"]
        self.arena = Arena(hello["map"], self.config["mech"])
        self.combat = Combat(self.arena, self.config)
        self.pilot = Pilot(self.arena, self.config)
        self.loot = {}
        self.last_target = None
        self.last_attack = None
        self.last_damage = -100
        self.recovering = False
        self.recent_attackers = {}
        self.free_for_all = False
        self.spawn = None

    def observe_combat(self, obs, enemies):
        me, now = obs["self"], obs["t"]
        damage = [event for event in obs["events"] if event["type"] == "damaged"]
        if damage:
            self.last_damage = now
        attackers = [event for event in damage if event["bearing"] is not None]
        self.recent_attackers = {key: when for key, when in self.recent_attackers.items()
                                 if now - when < 4}
        for event in attackers:
            if event["by"] is not None and event["by"] < self.config["match"]["mechs"]:
                self.recent_attackers[event["by"]] = now
        for event in obs["events"]:
            if event["type"] == "kill":
                self.recent_attackers.pop(event["target"], None)
        if attackers:
            attack = max(attackers, key=lambda event: event["amount"])
            self.last_attack = attack.get("by"), attack["bearing"], now
        if not self.free_for_all:
            # Lightning arc endpoints identify a third party more reliably than sounds.
            radius = me["stats"]["radius"]
            for beam in obs["beams"]:
                if beam["kind"] != "lightning" or len(beam["path"]) < 2:
                    continue
                end = beam["path"][1]
                if dist(xy(end), xy(me["pos"])) < radius + 5:
                    continue
                if any(enemy["id"] != beam["by"] and
                       dist(xy(end), xy(enemy["pos"])) <= radius and
                       abs(end[2] - enemy["pos"][2] - me["stats"]["height"] / 2) <= 4
                       for enemy in enemies):
                    self.free_for_all = True
                    break

    def upgrade(self, me, choices):
        equipped = {weapon["kind"] for weapon in me["weapons"] if weapon}

        def value(index):
            name = choices[index]["id"]
            weapon = name.split("_", 1)[0]
            score = UPGRADE_PRIORITY.get(name, 0)
            if weapon in WEAPON_PRIORITY and weapon not in equipped:
                return score * 0.2
            if name in ("gun_mag", "gun_damage") and len(equipped) > 1:
                score -= 8
            if name in ("servos", "reactor", "jets") and self.free_for_all:
                score += 18
            return score

        return max(range(len(choices)), key=value)

    def take_options(self, me, pickup):
        kind = pickup["kind"]
        weapons = me["weapons"]
        if kind == "gun" or any(w and w["kind"] == kind for w in weapons):
            return False, None
        if any(w is None for w in weapons):
            return True, None
        slot = next((i for i, w in enumerate(weapons) if w["kind"] != "gun"), None)
        if slot is None:
            return False, None
        old = weapons[slot]
        return (not old.get("charge") and WEAPON_PRIORITY[kind] > WEAPON_PRIORITY[old["kind"]]), slot

    def update_loot(self, obs):
        now = obs["t"]
        me = obs["self"]
        visible = {item["id"] for key in ("pickups", "cubes") for item in obs[key]}
        for item_id, (item, expiration) in list(self.loot.items()):
            if expiration < now or (item_id not in visible and dist(item["pos"], me["pos"]) < 7):
                del self.loot[item_id]
        for key in ("pickups", "cubes"):
            for item in obs[key]:
                self.loot[item["id"]] = (item, now + (12 if key == "cubes" else 8))

    def loot_goal(self, obs, enemy_count):
        me, zone = obs["self"], obs["zone"]
        position = me["pos"]
        options = []
        needs_secondary = any(weapon is None for weapon in me["weapons"])
        safe_radius = zone["radius"] - 5
        for item, _ in self.loot.values():
            point = item["pos"]
            distance = dist(xy(position), xy(point))
            weapon = "kind" in item
            seek_range = 175 if weapon and needs_secondary else 88
            if distance > seek_range or dist(xy(point), zone["center"]) > safe_radius:
                continue
            # Roof loot takes too long to reach from the street.
            if abs(position[2] - point[2]) > 5 and point[2] - self.arena.ground(*xy(point)) > 5:
                continue
            if weapon:
                if not self.take_options(me, item)[0]:
                    continue
                score = 71 + WEAPON_PRIORITY[item["kind"]] * 8
                if needs_secondary:
                    score += 60 if not enemy_count else 28
                else:
                    score -= 28
            else:
                score = 126 if me["cubes"] < self.config["cubes"]["stat_cap"] else 66
                nearby = sum(1 for other in obs["cubes"] if dist(xy(point), xy(other["pos"])) < 7)
                score += min(18, nearby * 4)
            score -= distance * (1.55 if enemy_count else 0.90)
            if score > 19:
                options.append((score, point))
        best_score, best_point = 19, None
        for score, point in sorted(options, reverse=True):
            if score < best_score:
                break
            if not self.arena.walk_segment(position, point, me["stats"]["radius"], me["stats"]["height"]):
                score -= 18
            if score > best_score:
                best_score, best_point = score, point
        return best_point

    def select_target(self, obs, enemies, recover):
        me = obs["self"]
        now = obs["t"]
        position = me["pos"]
        candidates = []
        for target in enemies:
            distance = dist(position, target["pos"])
            if distance > 223:
                continue
            score = 126 - distance * 0.41
            hp = target.get("hp", 1400)
            score += 46 * (1 - hp / max(1, target.get("max_hp", 1400)))
            score += 26 if hp < 175 else 0
            if target["id"] in self.recent_attackers:
                score += 18
            if target["id"] == self.last_target:
                score += 12
            if self.last_attack and self.last_attack[0] == target["id"] and now - self.last_attack[2] < 2:
                score += 20
            if target.get("charging"):
                score += 6
            if recover and target.get("hp", 500) > 175:
                continue
            candidates.append((score, target))
        if not recover:
            for target in (*obs["crates"], *(m for m in obs["mechs"] if m["kind"] == "npc")):
                distance = dist(position, target["pos"])
                if distance > 107 or dist(xy(target["pos"]), obs["zone"]["center"]) > obs["zone"]["radius"] - 8:
                    continue
                target_z = target["pos"][2]
                if target_z - position[2] > 6 and target_z - self.arena.ground(*xy(target["pos"])) > 6:
                    continue
                score = (99 if target.get("kind") == "npc" else 95) - distance * 0.80
                if self.recent_attackers:
                    score -= 40
                score += 15 * (1 - target.get("hp", 300) / max(1, target.get("max_hp", 350)))
                if target["id"] == self.last_target:
                    score += 11
                candidates.append((score, target))
        ranked = sorted(candidates, key=lambda pair: -pair[0])
        for _, target in ranked[:8]:
            if target.get("kind") != "mech" and not self.arena.walk_segment(
                    position, target["pos"], me["stats"]["radius"], me["stats"]["height"]):
                continue
            if self.combat.line_of_sight(me, target):
                return target
        # A checked ballistic trajectory can reach a tracked enemy behind cover.
        if not recover and any(w and w["kind"] == "mortar" and w["cooldown"] <= 0.1 for w in me["weapons"]):
            for _, target in ranked[:3]:
                if target.get("kind") == "mech" and 22 < dist(position, target["pos"]) < 125:
                    return target
        return None

    def act(self, obs):
        # Leave time for serialization and transport after tactical searches.
        deadline = perf_counter() + 0.030
        me, now, zone = obs["self"], obs["t"], obs["zone"]
        position = me["pos"]
        if self.spawn is None:
            self.spawn = tuple(xy(position))
        self.update_loot(obs)
        enemies = sorted((e for e in obs["mechs"] if e["kind"] == "mech"),
                         key=lambda e: dist(xy(position), xy(e["pos"])))
        self.observe_combat(obs, enemies)
        nearby = [e for e in enemies if dist(xy(position), xy(e["pos"])) < 155]
        close = bool(nearby and dist(xy(position), xy(nearby[0]["pos"])) < 65)
        threats = sorted(nearby, key=lambda e: e["id"] not in self.recent_attackers)
        ratio = me["hp"] / me["max_hp"]
        outnumbered = len(self.recent_attackers) >= 2
        if ratio < 0.46 or (ratio < 0.78 and outnumbered):
            self.recovering = True
        elif ratio > 0.80:
            self.recovering = False
        cautious = self.free_for_all and obs["alive"] >= 6
        recover = self.recovering or (cautious and close)
        target = self.select_target(obs, enemies, recover)
        pressure = bool(nearby and (outnumbered or
                        dist(xy(position), xy(nearby[0]["pos"])) < 38))
        goal, urgent = self.pilot.zone_goal(obs)
        rotating = goal is not None
        action = {"tick": obs["tick"]}

        if obs["offer"] and obs["offer"]["choices"]:
            action["pick"] = self.upgrade(me, obs["offer"]["choices"])

        reach = self.config["loot"]["pickup_radius"]
        pickups = sorted(obs["pickups"], key=lambda item: -WEAPON_PRIORITY[item["kind"]])
        for item in pickups:
            allowed, slot = self.take_options(me, item)
            if allowed and dist(xy(position), xy(item["pos"])) <= reach and abs(position[2] - item["pos"][2]) <= 4:
                action["take"] = item["id"]
                if slot is not None:
                    action["drop"] = slot
                break

        if not urgent:
            drop = self.supply_goal(obs, nearby)
            if drop is not None and not (cautious and close):
                goal = drop
            elif recover and threats:
                goal = self.pilot.cover(obs, threats, deadline - 0.010) or goal
            elif target is None or target.get("kind") != "mech" or not pressure:
                loot = self.loot_goal(obs, len(self.recent_attackers))
                if loot is not None:
                    goal = loot
                elif target and target.get("kind") != "mech" and dist(position, target["pos"]) > 24:
                    goal = target["pos"]
            if goal is None and target is None and not nearby:
                center = zone["next_center"]
                distance = dist(xy(position), center)
                # Head toward playable ground while scanning for early resources.
                hold_at = max(18, zone["next_radius"] * 0.82)
                if distance > hold_at and not cautious and not (self.recovering and now - self.last_damage < 8):
                    goal = center
            if goal is None and target is None and nearby and not recover and not cautious:
                nearest = nearby[0]
                if dist(xy(position), xy(nearest["pos"])) > 60:
                    goal = nearest["pos"]

        quiet = cautious and not rotating
        aim, fire = self.combat.act(me, target, pressure, now)
        move, boost, jump = self.pilot.move(obs, goal, urgent, threats, target, recover, quiet, deadline)
        if aim is None and cautious and recover:
            fire = [False, False]
        if aim is None:
            aim = self.watch(obs, goal, nearby)
        action.update(move=move, boost=boost, jump=jump, aim=aim, fire=fire)
        if any(w and w["kind"] == "gun" and not w["reload"] and w["ammo"] < w["stats"]["mag"] * 0.72
               for w in me["weapons"]):
            if target is None:
                action["reload"] = True
        self.last_target = target["id"] if target else None
        return action

    def supply_goal(self, obs, nearby):
        me, zone = obs["self"], obs["zone"]
        position = xy(me["pos"])
        choices = []
        for drop in obs["drops"]:
            point = xy(drop["pos"])
            distance = dist(position, point)
            if dist(point, zone["next_center"]) > zone["next_radius"] - 4:
                continue
            if distance > 110 or drop["lands_in"] > max(4, distance / 8):
                continue
            enemy_gap = min((dist(point, xy(e["pos"])) for e in nearby), default=200)
            if enemy_gap < drop["radius"] + 2 and drop["holder"] != me["id"]:
                continue
            if drop["holder"] == me["id"] or enemy_gap > min(53, distance + 9):
                choices.append((distance, point))
        return min(choices, default=(0, None))[1]

    def watch(self, obs, goal, nearby):
        me, now = obs["self"], obs["t"]
        position = me["pos"]
        if self.last_attack and now - self.last_attack[2] < 2.1:
            return self.last_attack[1], 0
        if nearby:
            return bearing(position, nearby[0]["pos"]), 0
        sounds = [sound for sound in obs["sounds"] if sound["dist"] != "far"
                  and sound["kind"] in ("gun", "mortar", "laser", "laser_charge", "lightning", "boost")]
        if sounds:
            return min(sounds, key=lambda s: s["age"] + (s["dist"] != "near") * 0.3)["bearing"], 0
        base = bearing(position, goal) if goal is not None else bearing(self.spawn, obs["zone"]["center"])
        return angle(base + 1.2 * sin(now * pi / 2.8)), -0.02


if __name__ == "__main__":
    mf.run(Bot)
