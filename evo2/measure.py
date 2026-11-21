"""Transfer measurement for version 2. Real model calls with each agent's evolved text.

    python3 -m evo2.measure --runs runs2 --out runs2/measure.json --budget 7 [--mock]

Populations measured (16 agents each, agent slot i always gets task instance i of its seed,
so every population of a seed is scored on identical tasks):
    bare            no evolved text at all
    gen0            the shared generation 0 instincts of that seed
    final           each arm's newborn population after the last selection (instincts + notes)
    final_no_notes  the same newborns with their notes removed (arms with notes only)

Scores per agent: near_energy, near_rate, near_illegal (fresh world, same hidden laws),
mid (code-breaking), far (rule induction). Resumable: finished populations are skipped.
"""
import argparse
import asyncio
import glob
import json
import os
import random

from evo.llm import LLM, BudgetExceeded
from evo2 import far, world as W


async def score_agent(llm, seed, i, instinct, note):
    near = await W.solo_lifetime(llm, W.new_world(random.Random(f"m-world-{seed}-{i}")),
                                 random.Random(f"m-near-{seed}-{i}"), instinct=instinct, note=note)
    mid = await far.code_breaking(llm, random.Random(f"m-code-{seed}-{i}"), instinct, note)
    fr = await far.rule_induction(llm, random.Random(f"m-rule-{seed}-{i}"), instinct, note)
    return {"near_energy": near["energy"], "near_rate": near["rate"], "near_illegal": near["illegal_tries"],
            "near_solved": near["solved"], "mid": mid, "far": fr}


async def main_async(a):
    llm = LLM(model=a.model, budget_usd=a.budget, max_concurrency=a.concurrency, mock=a.mock,
              thinking=None if a.thinking == "on" else {"type": "disabled"})
    out = {"version": 2, "model": a.model, "mock": a.mock, "thinking": a.thinking, "populations": {}}
    if os.path.exists(a.out):
        old = json.load(open(a.out))
        if old.get("mock") == a.mock:
            out["populations"] = old.get("populations", {})
    pops = {}
    for path in sorted(glob.glob(os.path.join(a.runs, "*_s*.json"))):
        d = json.load(open(path))
        if not d.get("complete"):
            continue
        arm, seed = d["arm"], d["seed"]
        n = len(d["next_population"])
        pops.setdefault(f"bare|{seed}", {"arm": None, "seed": seed, "when": "bare", "agents": [([], "")] * n})
        pops.setdefault(f"gen0|{seed}", {"arm": None, "seed": seed, "when": "gen0",
                                         "agents": [(x["instinct"], "") for x in d["generations"][0]["agents"]]})
        pops[f"final|{arm}|{seed}"] = {"arm": arm, "seed": seed, "when": "final",
                                       "agents": [(x["instinct"], x["note_in"]) for x in d["next_population"]]}
        if any(x["note_in"] for x in d["next_population"]):
            pops[f"final_no_notes|{arm}|{seed}"] = {"arm": arm, "seed": seed, "when": "final_no_notes",
                                                    "agents": [(x["instinct"], "") for x in d["next_population"]]}
    lock = asyncio.Lock()

    async def one(key, p):
        if key in out["populations"]:
            return
        try:
            scores = await asyncio.gather(*[score_agent(llm, p["seed"], i, inst, note)
                                            for i, (inst, note) in enumerate(p["agents"])])
        except BudgetExceeded:
            print(f"budget reached before {key} finished", flush=True)
            return
        async with lock:
            out["populations"][key] = {"arm": p["arm"], "seed": p["seed"], "when": p["when"], "agents": scores}
            out["usage"] = llm.usage()
            json.dump(out, open(a.out + ".tmp", "w"))
            os.replace(a.out + ".tmp", a.out)
            m = lambda k: sum(s[k] for s in scores) / len(scores)
            print(f"{key}: near={m('near_energy'):.1f} illegal={m('near_illegal'):.2f} mid={m('mid'):.2f} far={m('far'):.2f} "
                  f"| {len(out['populations'])}/{len(pops)} ${llm.cost_usd:.2f}", flush=True)

    await asyncio.gather(*[one(k, p) for k, p in pops.items()])
    out["usage"] = llm.usage()
    out["complete"] = len(out["populations"]) == len(pops)
    json.dump(out, open(a.out, "w"))
    print(f"Finished: {len(out['populations'])}/{len(pops)} populations | usage {json.dumps(llm.usage())}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs2")
    ap.add_argument("--out", default="runs2/measure.json")
    ap.add_argument("--model", default="claude-haiku-5-5")
    ap.add_argument("--budget", type=float, default=7.0)
    ap.add_argument("--concurrency", type=int, default=64)
    ap.add_argument("--thinking", choices=["on", "off"], default="off")
    ap.add_argument("--mock", action="store_true")
    asyncio.run(main_async(ap.parse_args()))
