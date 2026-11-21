"""Version 3 tests. Fixed items, temperature 0, never used for selection.

    near       fresh worlds, same three families as the training world
    code       code-breaking (evo2.far), sized for the small model
    rule       rule induction (evo2.far)
    order      ordering from clues (evo2.far), sized for the small model
    retention  unrelated skills (sums, capitals, comparisons); detects general forgetting

`battery(llm)` returns {"scores": {test: mean}, "items": {test: [per-item score]}}.
Per-item lists let later analysis pair populations item by item.
"""
import asyncio
import random
import re
import sys
import types

sys.modules.setdefault("anthropic", types.ModuleType("anthropic"))  # evo.llm imports it; not used here
from evo2 import far  # noqa: E402
from evo3 import wide, world3 as W  # noqa: E402

SIZES = {"near": 24, "mid": 24, "code": 40, "rule": 40, "order": 60, "retention": 90}
FAR = {"CODE_LEN": 3, "CODE_SYMBOLS": "ABC", "CODE_TURNS": 8, "RUNNERS": 4}
HELD_OUT = ["code", "rule", "order"]

CAPITALS = [("France", "Paris"), ("Japan", "Tokyo"), ("Italy", "Rome"), ("Egypt", "Cairo"), ("Spain", "Madrid"),
            ("Germany", "Berlin"), ("Russia", "Moscow"), ("China", "Beijing"), ("Greece", "Athens"),
            ("Canada", "Ottawa"), ("Peru", "Lima"), ("Kenya", "Nairobi"), ("Norway", "Oslo"), ("Ireland", "Dublin"),
            ("Portugal", "Lisbon"), ("Austria", "Vienna"), ("Poland", "Warsaw"), ("Sweden", "Stockholm"),
            ("Cuba", "Havana"), ("Thailand", "Bangkok"), ("Argentina", "Buenos Aires"), ("Hungary", "Budapest"),
            ("Finland", "Helsinki"), ("Denmark", "Copenhagen"), ("Chile", "Santiago"), ("Iran", "Tehran"),
            ("Iraq", "Baghdad"), ("Belgium", "Brussels"), ("Ukraine", "Kyiv"), ("Mexico", "Mexico City")]
NAMES = ["Tom", "Sam", "Lee", "Ada", "Max", "Zoe", "Kim", "Roy"]


def retention_items() -> list:
    rng, out = random.Random(4242), []
    for _ in range(30):
        a, b, c = rng.randint(11, 79), rng.randint(11, 79), rng.randint(2, 9)
        out.append((f"What is {a} + {b} - {c}?", str(a + b - c)))
    for country, city in CAPITALS:
        out.append((f"What is the capital of {country}?", city))
    for _ in range(30):
        x, y, z = rng.sample(NAMES, 3)
        word, hi, lo = rng.choice([("taller", "tallest", "shortest"), ("older", "oldest", "youngest"),
                                   ("faster", "fastest", "slowest")])
        ask_hi = rng.random() < 0.5
        facts = [f"{x} is {word} than {y}.", f"{y} is {word} than {z}."]
        rng.shuffle(facts)
        out.append((" ".join(facts) + f" Who is the {hi if ask_hi else lo}?", x if ask_hi else z))
    return out


async def retention_one(llm, q, a) -> float:
    """Right if the answer appears as a whole word and no other name from the question does.
    A bare answer without JSON counts, so a change of reply format is not scored as forgetting."""
    d = await llm.ask_json('Answer the question. Reply with only JSON: {"answer": "<answer>"}', q,
                           mock_fn=lambda: {"answer": ""}, max_tokens=30)
    got = str(d.get("answer", d.get("_raw", ""))).lower().replace("kiev", "kyiv")
    if not re.search(rf"(?<![\w]){re.escape(a.lower())}(?![\w])", got):
        return 0.0
    others = [n.lower() for n in NAMES if n != a and re.search(rf"\b{n}\b", q)]
    return float(not any(re.search(rf"\b{n}\b", got) for n in others))


async def near_one(llm, cfg, i) -> dict:
    return await W.life(llm, W.new_world(random.Random(90000 + i), cfg), seed=90000 + i)


async def battery(llm, cfg=None, sizes=None, only=None) -> dict:
    """Run every test on the model as it is now. Sets temperature 0 and restores it."""
    sizes = dict(SIZES, **(sizes or {}))
    for k, v in FAR.items():
        setattr(far, k, v)
    jobs = {
        "near": [near_one(llm, cfg, i) for i in range(sizes["near"])],
        # families that are in no life and no training set; only run for the wider world
        "mid": [wide.heldout_life(llm, 95000 + i) for i in range(sizes["mid"] if (cfg or {}).get("wide") else 0)],
        "code": [far.code_breaking(llm, random.Random(71000 + i)) for i in range(sizes["code"])],
        "rule": [far.rule_induction(llm, random.Random(72000 + i)) for i in range(sizes["rule"])],
        "order": [far.ordering(llm, random.Random(73000 + i)) for i in range(sizes["order"])],
        "retention": [retention_one(llm, q, a) for q, a in retention_items()[:sizes["retention"]]],
    }
    if only:
        for k in list(jobs):
            if k not in only:
                for c in jobs.pop(k):
                    c.close()
    old, llm.temp = llm.temp, 0.0
    flat = await llm.run([c for cs in jobs.values() for c in cs])
    llm.temp = old
    items, scores, i = {}, {}, 0
    for name, cs in jobs.items():
        res = flat[i:i + len(cs)]
        i += len(cs)
        if name == "near":
            items["near"] = [r["energy"] for r in res]
            for fam in W.families(cfg):
                scores["near_" + fam] = sum(r[fam]["wins"] for r in res) / sum(r[fam]["rounds"] for r in res)
            scores["near_invalid"] = sum(r["invalid"] for r in res) / sum(r["decisions"] for r in res)
            if (cfg or {}).get("wide"):
                new = (cfg or {})["wide"]["families"]
                scores["near_wide"] = (sum(r[f]["wins"] for r in res for f in new)
                                       / sum(r[f]["rounds"] for r in res for f in new))
        elif name == "mid":
            if not res:
                continue
            items["mid"] = [r["rate"] for r in res]
            scores["mid_invalid"] = (sum(r[f]["invalid"] for r in res for f in wide.HELD)
                                     / sum(r[f]["rounds"] for r in res for f in wide.HELD))
            for fam in wide.HELD:
                scores["mid_" + fam] = sum(r[fam]["wins"] for r in res) / sum(r[fam]["rounds"] for r in res)
        else:
            items[name] = [float(x) for x in res]
        scores[name] = sum(items[name]) / len(items[name])
    held = [scores[k] for k in HELD_OUT if k in scores]
    if held:
        scores["general"] = sum(held) / len(held)
    return {"scores": {k: round(v, 4) for k, v in scores.items()}, "items": items}


async def transcripts(llm, n=2) -> dict:
    """Raw exchanges from the first `n` items of each held-out test, to show what the model actually replies."""
    for k, v in FAR.items():
        setattr(far, k, v)
    logs = {"code": [], "rule": [], "order": []}
    old, llm.temp = llm.temp, 0.0
    await llm.run([far.code_breaking(llm.proxy(logs["code"]), random.Random(71000 + i)) for i in range(n)]
                  + [far.rule_induction(llm.proxy(logs["rule"]), random.Random(72000 + i)) for i in range(n)]
                  + [far.ordering(llm.proxy(logs["order"]), random.Random(73000 + i)) for i in range(n)])
    llm.temp = old
    return {k: [{"user": e["user"][-260:], "out": e["out"][:200]} for e in v[:14]] for k, v in logs.items()}
