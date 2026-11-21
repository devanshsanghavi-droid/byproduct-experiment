"""Go / no-go check before building version 2: can prompt text change the frozen model's scores?

    python3 -m evo2.headroom --reps 24 --budget 2 --out runs2/headroom.json [--mock]

Four hand-written instinct sets are scored on the same task instances. If the "method"
set does not beat "bare" on the transfer tests, evolved text has no room to show a gain
there and the full experiment would be uninformative on that test.
"""
import argparse
import asyncio
import json
import random
import statistics as st

from evo.llm import LLM, BudgetExceeded
from evo2 import far, world

SETS = {
    "bare": [],
    "method": [
        "Before each try, work out what every past result rules out.",
        "Only try options that fit all past results.",
        "Change one thing at a time so each result teaches you something.",
        "Once something works, repeat it exactly.",
    ],
    "method+laws": [
        "Before each try, work out what every past result rules out.",
        "Only try options that fit all past results.",
        "Once something works, repeat it exactly.",
        "No preparation uses the same action twice.",
        "Dry is never the first step.",
    ],
    "code-expert": [
        "First try one letter in every position, then the next letter, to count how often each letter appears.",
        "Then move one letter at a time to find its position.",
        "Only try options that fit all past results.",
    ],
    "harmful": [
        "Always begin with soak, whatever happened before.",
        "After any failure, change every step at once.",
        "Past results are unreliable. Trust your first idea.",
    ],
}


async def gather_scores(coros):
    out = []
    for r in await asyncio.gather(*coros, return_exceptions=True):
        if isinstance(r, BudgetExceeded):
            continue
        if isinstance(r, Exception):
            raise r
        out.append(r)
    return out


async def run(a):
    llm = LLM(model=a.model, budget_usd=a.budget, max_concurrency=a.concurrency, mock=a.mock,
              thinking={"type": "disabled"} if a.no_thinking else None)
    out = {"model": a.model, "mock": a.mock, "thinking": "off" if a.no_thinking else "on", "reps": a.reps, "sets": SETS, "scores": {}}
    for name, inst in SETS.items():
        async def near(i, inst=inst):
            w = world.new_world(random.Random(f"hw-{i}"))
            r = await world.solo_lifetime(llm, w, random.Random(f"hr-{i}"), instinct=inst)
            return r
        near_res, mid, f1, f2 = await asyncio.gather(
            gather_scores([near(i) for i in range(a.reps)]),
            gather_scores([far.code_breaking(llm, random.Random(f"code-{i}"), inst) for i in range(a.reps)]),
            gather_scores([far.rule_induction(llm, random.Random(f"rule-{i}"), inst) for i in range(a.reps)]),
            gather_scores([far.ordering(llm, random.Random(f"ord-{i}"), inst) for i in range(a.reps * 2)]),
        )
        def ms(xs):
            return {"mean": round(st.mean(xs), 3) if xs else None,
                    "se": round(st.stdev(xs) / len(xs) ** 0.5, 3) if len(xs) > 1 else None, "n": len(xs)}
        out["scores"][name] = {
            "near_energy": ms([r["energy"] for r in near_res]),
            "near_rate": ms([r["rate"] for r in near_res]),
            "near_illegal_tries": ms([r["illegal_tries"] for r in near_res]),
            "near_invalid": sum(r["invalid"] for r in near_res),
            "mid_code": ms(mid), "far_rule": ms(f1), "far_order": ms(f2),
        }
        print(name, json.dumps(out["scores"][name]), flush=True)
    out["usage"] = llm.usage()
    json.dump(out, open(a.out, "w"), indent=1)
    print("usage", out["usage"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="claude-haiku-5-5")
    ap.add_argument("--reps", type=int, default=24)
    ap.add_argument("--budget", type=float, default=2.0)
    ap.add_argument("--concurrency", type=int, default=48)
    ap.add_argument("--out", default="runs2/headroom.json")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--no-thinking", action="store_true")
    asyncio.run(run(ap.parse_args()))
