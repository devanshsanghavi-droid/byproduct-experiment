"""Breadth experiment: does transfer to unseen task families rise with the number of families trained on?

    POOL   one-question families that may be trained on (13 from wide.py + 52 from fam_a..fam_d)
    HELD   families in no training set (6 from wide.py + 12 from fam_held)
    public a slice of BIG-Bench Hard, written by other authors (files are fetched inside the Modal image)

Design and pass rules: SPEC3.md, "Breadth curve". Tests run at temperature 0 on fixed items.
"""
import json
import os
import random
import re
import zlib

from evo3 import fam_a, fam_b, fam_c, fam_d, fam_held, wide
from evo3.remote import BBH_TASKS

POOL = dict(wide.TRAIN)
for _m in (fam_a, fam_b, fam_c, fam_d):
    POOL.update(_m.FAMILIES)
HELD = dict(wide.HELD, **fam_held.FAMILIES)
wide.ALL.update(POOL)      # wide.system_for and wide.items_for look families up here
wide.ALL.update(HELD)
assert not set(POOL) & set(HELD)

LEVELS, DRAWS = [2, 4, 8, 16, 32, 64], 3
SIZES = {"pool": 8, "held": 32, "bbh": 40, "near": 32}
NEAR_N = 16                # rerun: pool families kept out of every training set
BBH_DIR = os.environ.get("BBH_DIR", "/data/bbh")
BBH_SYSTEM = ('Answer the question. If options are given, answer with the letter of the correct option. '
              'Reply with only JSON: {"answer": "<answer>"}')


def subset(k: int, draw: int) -> list:
    """The K families of one run. Every run draws its own."""
    return sorted(random.Random(1000 * k + draw).sample(sorted(POOL), k))


def reserved() -> list:
    """Rerun: the pool families no rerun trains on. Drawn once with a fixed seed, before any rerun result."""
    return sorted(random.Random(20261114).sample(sorted(POOL), NEAR_N))


def subset2(k: int, draw: int) -> list:
    """Rerun: the K families of one run, drawn from the pool families that are not reserved."""
    rest = sorted(set(POOL) - set(reserved()))
    return sorted(random.Random(50000 + 1000 * k + draw).sample(rest, k))


def reserved2() -> list:
    """Strengthening experiment A: a second reserved set, drawn from the pool families outside the first one."""
    rest = sorted(set(POOL) - set(reserved()))
    return sorted(random.Random(20261115).sample(rest, NEAR_N))


def subset3(k: int, draw: int) -> list:
    """Experiment A: the K families of one run, drawn from the pool families outside the second reserved set."""
    rest = sorted(set(POOL) - set(reserved2()))
    return sorted(random.Random(60000 + 1000 * k + draw).sample(rest, k))


def training_set(fams, total, seed) -> list:
    """`total` examples of correct play, split equally among the families."""
    rng, out = random.Random(seed), []
    for f in fams:
        for it in wide.items_for(f, rng, total // len(fams)):
            out.append({"system": wide.system_for(f, it["kind"]), "user": it["user"],
                        "out": json.dumps({"answer": it["answer"]})})
    return out


def test_items(fams, base, n) -> dict:
    return {f: wide.items_for(f, random.Random(base + zlib.crc32(f.encode()) % 100000), n) for f in sorted(fams)}


async def ask(llm, fam, it, rng):
    d = await llm.ask_json(wide.system_for(fam, it["kind"]), it["user"],
                           mock_fn=lambda: {"answer": wide.random_answer(it, rng)}, max_tokens=32)
    right, valid = wide.check(it, d)
    return float(right), float(valid)


# ---- public test -----------------------------------------------------------------------------
def bbh_items(n) -> list:
    out = []
    for t in BBH_TASKS:
        ex = json.load(open(f"{BBH_DIR}/{t}.json"))["examples"]
        for i in sorted(random.Random(4100).sample(range(len(ex)), n)):
            out.append((t, ex[i]["input"], ex[i]["target"]))
    return out


def bbh_right(target, got) -> bool:
    t, g = str(target).strip(), str(got).strip()
    m = re.fullmatch(r"\(([A-R])\)", t)
    if m:
        found = set(re.findall(r"\(([A-R])\)", g)) or set(re.findall(r"^\W*([A-R])(?:\W.*)?$", g, flags=re.S))
        return found == {m.group(1)}
    pairs = {"yes": "no", "no": "yes", "true": "false", "false": "true", "valid": "invalid", "invalid": "valid"}
    if t.lower() in pairs:
        return bool(re.search(rf"\b{t.lower()}\b", g.lower())) and not re.search(rf"\b{pairs[t.lower()]}\b", g.lower())
    if re.fullmatch(r"-?\d+", t):
        m = re.search(r"-?\d+", g.replace(",", ""))
        return bool(m) and int(m.group(0)) == int(t)
    return g.lower().strip(" .\"'") == t.lower()


def bbh_chance(target, text) -> float:
    """Chance rate of one item: 1 / number of options, 0.5 for two-way answers, 0 for open numbers."""
    if re.fullmatch(r"\([A-R]\)", target.strip()):
        return 1 / max(2, len(set(re.findall(r"^\(([A-R])\)", text, flags=re.M))))
    return 0.5 if not re.fullmatch(r"-?\d+", target.strip()) else 0.0


async def bbh_one(llm, text, target) -> float:
    d = await llm.ask_json(BBH_SYSTEM, text, mock_fn=lambda: {"answer": ""}, max_tokens=24)
    return float(bbh_right(target, d.get("answer", d.get("_raw", ""))))


# ---- battery -----------------------------------------------------------------------------------
async def battery(llm, trained, sizes=None, far=True, public=True, near=None) -> dict:
    """All tests for one adapter. `trained` is the list of pool families this run was trained on.
    `near` (rerun) is the list of reserved pool families, tested with their own larger item set."""
    from evo3 import tests3
    sz = dict(SIZES, **(sizes or {}))
    pool, held = test_items(POOL, 96000, sz["pool"]), test_items(HELD, 97000, sz["held"])
    keys = [("pool", f, it) for f, its in pool.items() for it in its] + [("held", f, it) for f, its in held.items() for it in its]
    if near:
        keys += [("near", f, it) for f, its in test_items(near, 98000, sz["near"]).items() for it in its]
    rng = random.Random(5)
    bbh = bbh_items(sz["bbh"]) if public else []
    old, llm.temp = llm.temp, 0.0
    res = await llm.run([ask(llm, f, it, rng) for _, f, it in keys] + [bbh_one(llm, x, t) for _, x, t in bbh])
    llm.temp = old
    scores, by = {}, {}
    for (part, f, _), (right, valid) in zip(keys, res):
        by.setdefault((part, f), []).append((right, valid))
    for (part, f), v in by.items():
        scores[f"{part}_{f}"] = sum(r for r, _ in v) / len(v)
    for part, fams in (("pool", POOL), ("held", HELD)) + ((("near", near),) if near else ()):
        scores[part] = sum(scores[f"{part}_{f}"] for f in fams) / len(fams)
        scores[part + "_invalid"] = 1 - sum(x for f in fams for _, x in by[(part, f)]) / sum(len(by[(part, f)]) for f in fams)
    seen, unseen = [f for f in POOL if f in trained], [f for f in POOL if f not in trained]
    if seen:
        scores["seen"] = sum(scores["pool_" + f] for f in seen) / len(seen)
    if unseen:
        scores["unseen"] = sum(scores["pool_" + f] for f in unseen) / len(unseen)
    items = {"held": [r for (part, _, _), (r, _) in zip(keys, res) if part == "held"]}
    if near:
        items["near"] = [r for (part, _, _), (r, _) in zip(keys, res) if part == "near"]
        items["near_valid"] = [v for (part, _, _), (_, v) in zip(keys, res) if part == "near"]
    if bbh:
        got = res[len(keys):]
        items["bbh"] = [float(x) for x in got]
        scores["bbh"] = sum(got) / len(got)
        for t in BBH_TASKS:
            v = [x for (name, _, _), x in zip(bbh, got) if name == t]
            scores["bbh_" + t] = sum(v) / len(v)
        scores["bbh_chance"] = sum(bbh_chance(t, x) for _, x, t in bbh) / len(bbh)
    if far:
        b = await tests3.battery(llm, None, {k: v for k, v in (sizes or {}).items() if k != "near"} or None,
                                 only=("code", "rule", "order", "retention"))
        scores.update(b["scores"])
        items.update(b["items"])
    return {"scores": {k: round(v, 4) for k, v in scores.items()}, "items": items}
