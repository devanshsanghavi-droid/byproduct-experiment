"""Version 3 simulation: the population's weights evolve.

Each generation: N lives sampled from the current weights at a fixed temperature, all in the
same world. Parents are the top `parents` by energy (DRIFT: a random `parents`). The adapter
is then trained on the parents' whole-life transcripts from the last `window` generations.
Energy is the only signal. Test scores are never used.
"""
import random
import time

from evo3 import tests3
from evo3 import world3 as W

ARMS = ("DRIFT", "SOLO")
DEFAULTS = {"G": 6, "N": 24, "parents": 6, "iters": 60, "lr": 1e-4, "batch_size": 8, "temp": 0.7, "window": 2}


def fresh(arm, seed, cfg, P) -> dict:
    return {"out": {"arm": arm, "seed": seed, "config": dict(P, world=cfg), "generations": [], "tests": {},
                    "complete": False},
            "buffer": [], "rng": random.Random(1000 * seed + 17).getstate()}


async def run_sim(llm, state, on_gen=None, tests=True, test_sizes=None, stop=None) -> dict:
    """Advance `state` (from fresh() or a checkpoint) to the end. Calls on_gen(state) after each generation."""
    out = state["out"]
    arm, seed, P = out["arm"], out["seed"], out["config"]
    cfg, rng = P["world"], random.Random()
    rng.setstate(state["rng"])
    for g in range(len(out["generations"]), P["G"]):
        if stop and stop():
            out["stopped"] = "budget"
            return out
        t = time.time()
        w = W.new_world(random.Random(seed * 100003 + g), cfg)   # same world for every arm of a seed
        logs = [[] for _ in range(P["N"])]
        llm.temp = P["temp"]
        res = await llm.run([W.life(llm.proxy(logs[i]), w, seed=seed * 1000 + g * 100 + i) for i in range(P["N"])])
        energy = [r["energy"] for r in res]
        if arm == "DRIFT":
            chosen = rng.sample(range(P["N"]), P["parents"])
        else:
            chosen = sorted(range(P["N"]), key=lambda i: (-energy[i], rng.random()))[:P["parents"]]
        state["buffer"] = (state["buffer"] + [[e for i in chosen for e in logs[i]]])[-P["window"]:]
        ex = [e for b in state["buffer"] for e in b]
        losses = llm.train(ex, P["iters"], P["lr"], P["batch_size"], seed=seed * 100 + g)
        gen = {"g": g, "energy": round(sum(energy) / len(energy), 3),
               "parent_energy": round(sum(energy[i] for i in chosen) / len(chosen), 3),
               "max_energy": W.max_energy(cfg),
               "invalid": round(sum(r["invalid"] for r in res) / sum(r["decisions"] for r in res), 4),
               "loss": losses, "examples": len(ex), "seconds": round(time.time() - t, 1),
               "sample": logs[chosen[0]][:3]}
        for fam in W.families(cfg):
            gen[fam] = round(sum(r[fam]["wins"] for r in res) / sum(r[fam]["rounds"] for r in res), 4)
        out["generations"].append(gen)
        state["rng"] = rng.getstate()
        if on_gen:
            on_gen(state)
    if tests and "final" not in out["tests"]:
        out["tests"]["final"] = await tests3.battery(llm, cfg, test_sizes)
    out["complete"] = True
    out["usage"] = llm.usage()
    if on_gen:
        on_gen(state)
    return out
