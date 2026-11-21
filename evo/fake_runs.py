"""Throwaway fake-data generator for developing analyze.py and the report page.

Writes run files and heldout.json in the SPEC.md schemas, all marked "mock": true.
No model is called. Fitness is a made-up function of the genes, so the numbers
mean nothing; only the file shapes are real.

    python3 -m evo.fake_runs --out runs_mock_c
"""
import argparse
import json
import os
import random

from evo import genome as G

ARMS = ["DRIFT", "SOLO", "COOP", "ALPHA", "DOMEST", "CULTURE", "MEDLEY"]
PAIR = {"COOP", "ALPHA", "DOMEST", "MEDLEY"}
GRAB = {"ALPHA", "DOMEST", "MEDLEY"}
VOTE = {"DOMEST", "MEDLEY"}
NOTES = {"CULTURE", "MEDLEY"}


def fake_gross(arm, g, rng, R):
    """Invented payoff. Capacity genes pay only where the arm uses them."""
    rate = 0.08 + 0.09 * g["MEM"]
    if arm in PAIR:
        rate += 0.07 * g["TALK"]
    if arm in NOTES:
        rate += 0.10 * g["LORE"]
    rate = min(0.92, rate)
    wins = sum(1 for _ in range(R) if rng.random() < rate)
    gross = 4 * wins - (R - wins)
    grabs = 0
    if arm in GRAB:
        grabs = sum(1 for _ in range(wins) if rng.random() < G.grab_prob(g))
        gross += 2 * grabs
    return gross, wins, grabs


def one_run(arm, seed, N, R, Gn, cost):
    init = random.Random(seed)                      # same generation 0 in every arm
    pop = [{"genome": G.random_genome(init), "parent": None} for _ in range(N)]
    rng = random.Random(f"{arm}-{seed}")
    gens = []
    samples = {"messages": [], "gossip": [], "notes": []}
    for gen in range(Gn):
        agents = []
        for i, p in enumerate(pop):
            g = p["genome"]
            gross, wins, grabs = fake_gross(arm, g, rng, R)
            up = G.upkeep(g, cost)
            votes = 0
            if arm in VOTE and grabs:
                votes = sum(1 for _ in range(N - 1) if rng.random() < 0.08 * grabs)
            agents.append({
                "id": f"a{i}", "parent": p["parent"], "genome": g,
                "gross": gross, "upkeep": up, "net": gross - up,
                "successes": wins, "trials": R, "grabs": grabs,
                "grabbed": rng.randint(0, 2) if arm in GRAB else 0,
                "votes_against": votes, "exiled": False,
                "note": ("zuvo soak then dry. kemi roast then pound."
                         if arm in NOTES and g["LORE"] > 0 else ""),
            })
        if arm in VOTE:
            named = sorted([a for a in agents if a["votes_against"] >= N / 4],
                           key=lambda a: -a["votes_against"])[: N // 4]
            for a in named:
                a["exiled"] = True
        mean = {t: sum(a["genome"][t] for a in agents) / N for t in G.TRAITS}
        mean["gross"] = sum(a["gross"] for a in agents) / N
        mean["net"] = sum(a["net"] for a in agents) / N
        mean["success_rate"] = sum(a["successes"] for a in agents) / (N * R)
        gens.append({"gen": gen, "agents": agents, "mean": mean,
                     "exiled": sum(a["exiled"] for a in agents), "parse_fail": 0})
        if arm in PAIR and len(samples["messages"]) < 20:
            samples["messages"].append({"gen": gen, "agent": "a1",
                                        "text": "zuvo step two is roast"})
        if arm in VOTE and len(samples["gossip"]) < 20:
            samples["gossip"].append({"gen": gen, "agent": "a4", "text": "a7 grabbed my share"})
        if arm in NOTES and len(samples["notes"]) < 20:
            samples["notes"].append({"gen": gen, "agent": "a2",
                                     "text": "zuvo: soak then dry. Test one step at a time."})
        eligible = [a for a in agents if not a["exiled"]]
        if arm == "DRIFT":
            parents = rng.sample(eligible, N // 2)
        else:
            parents = sorted(eligible, key=lambda a: -a["net"])[: N // 2]
        while len(parents) < N // 2:
            parents.append(rng.choice(eligible))
        pop = [{"genome": G.mutate(p["genome"], rng), "parent": p["id"]}
               for p in parents for _ in range(2)]
    calls = N * R * Gn * (2 if arm in PAIR else 1)
    return {
        "arm": arm, "seed": seed, "model": "mock", "mock": True,
        "config": {"N": N, "R": R, "G": Gn, "cost": cost}, "complete": True,
        "generations": gens, "final_population": pop,
        "usage": {"calls": calls, "cost_usd": round(calls * 0.00022, 4)},
        "samples": samples,
    }


def fake_heldout(reps=24):
    rng = random.Random(7)

    def cell(m, n):
        m = max(0.0, min(1.0, m + rng.uniform(-0.03, 0.03)))
        return {"mean": round(m, 3), "sd": round(rng.uniform(0.1, 0.3), 3), "n": n}

    return {
        "model": "mock", "mock": True, "reps": reps,
        "H1": {str(m): cell(0.08 + 0.2 * m, reps) for m in range(5)},
        "H3": {str(m): cell(0.5 + 0.07 * m, reps) for m in range(5)},
        "H2": {f"{m},{t}": cell(0.25 + 0.04 * m + 0.14 * t, reps // 2)
               for m in range(5) for t in range(5)},
        "usage": {"calls": 4300, "cost_usd": 0.95},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs_mock_c")
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--N", type=int, default=12)
    ap.add_argument("--R", type=int, default=12)
    ap.add_argument("--G", type=int, default=10)
    ap.add_argument("--cost", type=float, default=1.5)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for arm in ARMS:
        for seed in [int(s) for s in a.seeds.split(",")]:
            run = one_run(arm, seed, a.N, a.R, a.G, a.cost)
            with open(os.path.join(a.out, f"{arm}_s{seed}.json"), "w") as f:
                json.dump(run, f)
    with open(os.path.join(a.out, "heldout.json"), "w") as f:
        json.dump(fake_heldout(), f)
    print(f"wrote fake runs to {a.out}/")


if __name__ == "__main__":
    main()
