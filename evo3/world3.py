"""Version 3 world: three task families per life, difficulty set by a config.

    plants  ordered search with feedback; hidden laws: no action twice, "dry" never first
    tracks  an animal moves a fixed number of sites round a ring; say where it is now
    trade   pick the offer worth the most (arithmetic on quantities and values)

Energy is +4 for a success and -1 otherwise, summed over all rounds of all families.
Every agent in a generation meets the same world, so energy differences come from
behaviour only. `oracle=True` plays each family correctly and logs the exchange in the
same format the model would produce; that is used only for the CEILING reference.
"""
import asyncio
import itertools
import json
import random
import re

from evo2.world import nonsense
from evo3 import wide

WIN, LOSE = 4, -1
ALL_ACTIONS = ["soak", "pound", "roast", "dry"]
ITEMS = ["nut", "fish", "hide", "root", "shell"]
DEFAULT = {"plants": {"steps": 2, "actions": 4, "n": 2, "tries": 5},
           "tracks": {"ring": 6, "n": 2, "rounds": 4},
           "trade": {"offers": 3, "bundle": 1, "rounds": 8}}
# The wider world: the same three families plus every family in wide.TRAIN, `rounds` questions each per life.
WIDE = dict(DEFAULT, wide={"families": list(wide.TRAIN), "rounds": 3})


def legal_recipes(P) -> list:
    acts = ALL_ACTIONS[:P["actions"]]
    return [list(r) for r in itertools.product(acts, repeat=P["steps"])
            if len(set(r)) == len(r) and r[0] != "dry"]


def new_world(rng: random.Random, cfg=None) -> dict:
    cfg = cfg or DEFAULT
    P, T, D = cfg["plants"], cfg["tracks"], cfg["trade"]
    names = nonsense(rng, P["n"] + T["n"])
    plants, animals = names[:P["n"]], names[P["n"]:]
    acts, legal = ALL_ACTIONS[:P["actions"]], legal_recipes(P)
    values = dict(zip(rng.sample(ITEMS, 3), rng.sample(range(2, 10), 3)))
    rounds = []
    while len(rounds) < D["rounds"]:
        offers = []
        for _ in range(D["offers"]):
            its = rng.sample(sorted(values), D["bundle"])
            offers.append([(rng.randint(1, 6), it) for it in its])
        totals = [sum(q * values[it] for q, it in o) for o in offers]
        if totals.count(max(totals)) == 1:
            rounds.append(offers)
    out = {
        "cfg": cfg,
        "plants": {p: rng.choice(legal) for p in plants},
        "plant_rounds": [plants[i % P["n"]] for i in range(P["n"] * P["tries"])],
        "action_order": [rng.sample(acts, len(acts)) for _ in range(P["n"] * P["tries"])],
        "animals": {a: {"start": rng.randrange(T["ring"]), "step": rng.randint(1, T["ring"] // 2)} for a in animals},
        "animal_rounds": [animals[i % T["n"]] for i in range(T["n"] * T["rounds"])],
        "values": values, "trade_rounds": rounds,
    }
    if cfg.get("wide"):   # drawn last, so the three base families are the same with or without the wider part
        out["wide"] = {f: wide.items_for(f, rng, cfg["wide"]["rounds"]) for f in cfg["wide"]["families"]}
    return out


async def decide(llm, system, user, mock, oracle_fn, log):
    if oracle_fn is not None:
        d = oracle_fn()
        if log is not None:
            log.append({"system": system, "user": user, "out": json.dumps(d)})
        return d
    return await llm.ask_json(system, user, mock_fn=mock, max_tokens=24)


def tally(wins, rounds, invalid) -> dict:
    return {"energy": WIN * wins + LOSE * (rounds - wins), "wins": wins, "rounds": rounds, "invalid": invalid}


# ---- plants -------------------------------------------------------------------------
async def plants_life(llm, w, rng, oracle=False, log=None) -> dict:
    P = w["cfg"]["plants"]
    k, acts, legal = P["steps"], ALL_ACTIONS[:P["actions"]], legal_recipes(P)
    system = ("You are a forager. Each round one plant is in season. To eat it, prepare it with the right "
              f"{k} step{'s' if k > 1 else ''}{' in the right order' if k > 1 else ''}, each one of the listed actions. "
              "After each try you learn how many steps were right and in place. "
              f"Right preparation: +{WIN} energy. Anything else: {LOSE}. A plant keeps its preparation all season.\n"
              'Reply with only JSON: {"steps": [' + ", ".join(['"<action>"'] * k) + "]}")
    recs, wins, invalid, illegal = [], 0, 0, 0
    R = len(w["plant_rounds"])
    for r, plant in enumerate(w["plant_rounds"], 1):
        user = (f"Round {r} of {R}. In season: {plant}.\nActions: {', '.join(w['action_order'][r - 1])}\n"
                + ("Your records:\n" + "\n".join(f"- {x['plant']}: {', '.join(x['steps'])} -> {x['right']} of {k} right"
                                                 for x in recs) if recs else "Your records: none"))

        def best():
            mine = [x for x in recs if x["plant"] == plant]
            ok = [c for c in legal if all(sum(a == b for a, b in zip(c, x["steps"])) == x["right"] for x in mine)]
            return {"steps": rng.choice(ok)}

        d = await decide(llm, system, user, lambda: {"steps": [rng.choice(acts) for _ in range(k)]},
                         best if oracle else None, log)
        s = d.get("steps", d.get("_raw")) if isinstance(d, dict) else None
        if isinstance(s, str):   # lenient: "soak, dry" counts the same as ["soak", "dry"]
            s = [x for x in s.replace(" then ", ",").split(",") if x.strip()]
        s = [str(x).strip().lower() for x in s] if isinstance(s, list) else None
        s = s[:k] if s else s   # lenient: extra steps are ignored
        if not s or len(s) != k or any(x not in acts for x in s):
            invalid += 1
            continue
        right = sum(a == b for a, b in zip(w["plants"][plant], s))
        wins += right == k
        illegal += s not in legal
        recs.append({"plant": plant, "steps": s, "right": right})
    return dict(tally(wins, R, invalid), illegal=illegal)


# ---- tracks -------------------------------------------------------------------------
async def tracks_life(llm, w, rng, oracle=False, log=None) -> dict:
    n = w["cfg"]["tracks"]["ring"]
    system = (f"You are a tracker. Sites 1 to {n} lie in a circle; after site {n} comes site 1. "
              "Each animal moves the same number of sites forward every round. "
              "Each round, name the site where the animal is now. Then you learn where it was. "
              f"Right site: +{WIN} energy. Wrong: {LOSE}.\n"
              'Reply with only JSON: {"site": <number>}')
    pos = lambda a, t: (w["animals"][a]["start"] + w["animals"][a]["step"] * t) % n + 1
    seen = {a: [pos(a, 0), pos(a, 1)] for a in w["animals"]}
    wins = invalid = 0
    R = len(w["animal_rounds"])
    for r, a in enumerate(w["animal_rounds"], 1):
        user = (f"Round {r} of {R}. Animal: {a}.\n"
                f"Sightings of {a} so far, oldest first: {', '.join(map(str, seen[a]))}\nWhere is it now?")
        best = lambda: {"site": (seen[a][-1] - 1 + (seen[a][-1] - seen[a][-2])) % n + 1}
        d = await decide(llm, system, user, lambda: {"site": rng.randint(1, n)}, best if oracle else None, log)
        now = pos(a, len(seen[a]))
        m = re.search(r"\d+", str(d.get("site", d.get("_raw", "")))) if isinstance(d, dict) else None
        site = int(m.group(0)) if m else None
        if site is None or not 1 <= site <= n:
            invalid += 1
        wins += site == now
        seen[a].append(now)
    return tally(wins, R, invalid)


# ---- trade --------------------------------------------------------------------------
async def trade_life(llm, w, rng, oracle=False, log=None) -> dict:
    values = w["values"]
    system = ("You are a trader. Each round you may accept one offer. The best offer is the one worth the most "
              "in total, using the listed value of each item. "
              f"Best offer: +{WIN} energy. Any other: {LOSE}.\n"
              'Reply with only JSON: {"offer": "<letter>"}')
    wins = invalid = 0
    R = len(w["trade_rounds"])
    for r, offers in enumerate(w["trade_rounds"], 1):
        letters = "ABCDEF"[:len(offers)]
        totals = [sum(q * values[it] for q, it in o) for o in offers]
        user = (f"Round {r} of {R}.\nValues: " + ", ".join(f"{it} {v}" for it, v in values.items()) + ".\nOffers:\n"
                + "\n".join(f"{L}: " + " and ".join(f"{q} {it}" for q, it in o) for L, o in zip(letters, offers))
                + "\nWhich offer?")
        best = lambda: {"offer": letters[totals.index(max(totals))]}
        d = await decide(llm, system, user, lambda: {"offer": rng.choice(letters)}, best if oracle else None, log)
        m = re.search(rf"\b([{letters}])\b", str(d.get("offer", d.get("_raw", ""))).upper()) if isinstance(d, dict) else None
        pick = m.group(1) if m else ""
        if not pick:
            invalid += 1
        wins += pick == letters[totals.index(max(totals))]
    return tally(wins, R, invalid)


FAMILIES = {"plants": plants_life, "tracks": tracks_life, "trade": trade_life}


def families(cfg=None) -> list:
    """Names of every family in a life under this config."""
    return list(FAMILIES) + list(((cfg or DEFAULT).get("wide") or {}).get("families", []))


async def life(llm, w, seed: int, oracle=False, log=None) -> dict:
    """One whole life: every family of the world. Returns total energy and per-family results."""
    names = families(w["cfg"])
    jobs = [fn(llm, w, random.Random(seed * 7 + i), oracle, log) for i, fn in enumerate(FAMILIES.values())]
    jobs += [wide.family_life(f, llm, w["wide"][f], random.Random(seed * 7 + 3 + i), oracle, log)
             for i, f in enumerate(names[len(FAMILIES):])]
    parts = await asyncio.gather(*jobs)
    out = dict(zip(names, parts))
    out["energy"] = sum(p["energy"] for p in parts)
    out["invalid"] = sum(p["invalid"] for p in parts)
    out["decisions"] = sum(p["rounds"] for p in parts)
    return out


def max_energy(cfg=None) -> int:
    cfg = cfg or DEFAULT
    extra = len(cfg["wide"]["families"]) * cfg["wide"]["rounds"] if cfg.get("wide") else 0
    return WIN * (cfg["plants"]["n"] * cfg["plants"]["tries"] + cfg["tracks"]["n"] * cfg["tracks"]["rounds"]
                  + cfg["trade"]["rounds"] + extra)
