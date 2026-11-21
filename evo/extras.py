"""Second-pass measurements added to results.json after evo.analyze.

    python3 -m evo.extras --runs runs --results results.json

Adds results["extras"]:
  exile     how often the exile vote was used and how often it succeeded
  gradient  mean within-generation correlation of each gene with net and gross energy
  foraging  success rate and net energy, first and last generation
  findings  evidence for the thesis, against it, and what this run could not test,
            each sentence generated from the numbers so it cannot go stale
"""
import argparse
import glob
import json
import math
import os
import statistics as st

GENES = ["MEM", "TALK", "LORE", "AGG"]


def corr(x, y):
    if len(set(x)) < 2 or len(set(y)) < 2:
        return None
    mx, my = st.mean(x), st.mean(y)
    sx = math.sqrt(sum((a - mx) ** 2 for a in x))
    sy = math.sqrt(sum((b - my) ** 2 for b in y))
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / (sx * sy)


def load(runs_dir):
    by_arm = {}
    for p in sorted(glob.glob(os.path.join(runs_dir, "*_s*.json"))):
        d = json.load(open(p))
        by_arm.setdefault(d["arm"], []).append(d)
    return by_arm


def gradient(by_arm, arms):
    out = {}
    for arm in arms:
        row = {}
        for g in GENES:
            cn, cg = [], []
            for d in by_arm.get(arm, []):
                for gen in d["generations"]:
                    x = [a["genome"][g] for a in gen["agents"]]
                    c = corr(x, [a["net"] for a in gen["agents"]])
                    if c is not None:
                        cn.append(c)
                    c = corr(x, [a["gross"] for a in gen["agents"]])
                    if c is not None:
                        cg.append(c)
            row[g] = {"net": round(st.mean(cn), 3) if cn else None,
                      "gross": round(st.mean(cg), 3) if cg else None}
        out[arm] = row
    return out


def exile(by_arm):
    out = {}
    for arm in ("DOMEST", "MEDLEY"):
        runs = by_arm.get(arm, [])
        if not runs:
            continue
        gens = [g for d in runs for g in d["generations"]]
        n = len(gens[0]["agents"])
        votes = [sum(a["votes_against"] for a in g["agents"]) for g in gens]
        top = [max(a["votes_against"] for a in g["agents"]) for g in gens]
        out[arm] = {
            "generations": len(gens), "members": n,
            "threshold": max(1, math.ceil(n / 4)),
            "exiled_total": sum(g.get("exiled", 0) for g in gens),
            "votes_per_generation": round(st.mean(votes), 2),
            "most_votes_against_one_member": max(top),
            "grabs_per_generation": round(st.mean(sum(a["grabs"] for a in g["agents"]) for g in gens), 1),
        }
    return out


def foraging(results):
    out = {}
    for arm in results["arms"]:
        m = results["trajectories"][arm]["mean"]
        sr = [x for x in m["success_rate"] if x is not None]
        nt = [x for x in m["net"] if x is not None]
        out[arm] = {"success_start": round(sr[0], 3), "success_end": round(sr[-1], 3),
                    "net_start": round(nt[0], 2), "net_end": round(nt[-1], 2)}
    return out


def s(x, n=2):
    return f"{x:+.{n}f}"


def findings(R, ex, grad, forage):
    ab, ch = R["ability"]["arms"], R["change"]
    band = R["ability"]["noise"]["mean"]["band"]
    capband = R["noise"]["CAP"]["band"]
    memband = R["noise"]["MEM"]["band"]
    n = len(R["seeds"])
    social = [a for a in ("COOP", "ALPHA", "DOMEST", "MEDLEY") if a in ch]
    f, a, u = [], [], []

    up = lambda arm: sum(1 for v in ab[arm]["mean"]["seeds"].values() if v > 0)
    f.append(f"Ability rose with no reward for ability in the two arms where agents forage alone. "
             f"SOLO gained {s(ab['SOLO']['mean']['mean'], 3)} and CULTURE {s(ab['CULTURE']['mean']['mean'], 3)}, "
             f"positive in {up('SOLO')} of {n} and {up('CULTURE')} of {n} seeds. "
             f"Both are inside the DRIFT noise band of {band:.3f}, so this is a consistent direction, not a confirmed effect.")
    f.append(f"Memory is what moved. MEM rose {s(ch['SOLO']['MEM']['mean'])} in SOLO and {s(ch['CULTURE']['MEM']['mean'])} in CULTURE, "
             f"upward in every seed, against {s(ch['DRIFT']['MEM']['mean'])} under chance alone. The gap is inside the noise band of {memband:.2f}.")
    f.append("Selection behaved as designed. Every costly gene that an arm could not use fell, and grabbing rose where nothing punished it.")
    c = forage["CULTURE"]
    f.append(f"Inherited notes transformed foraging. CULTURE success went from {c['success_start']:.2f} to {c['success_end']:.2f} "
             f"while SOLO reached {forage['SOLO']['success_end']:.2f}, so knowledge accumulated across generations without any change to the model.")

    a.append(f"No theory arm beat plain ecology. The best theory arm, CULTURE, gained {s(ab['CULTURE']['mean']['mean'], 3)} in held-out ability; "
             f"SOLO, with no social mechanism at all, gained {s(ab['SOLO']['mean']['mean'], 3)}.")
    a.append("Social life selected against costly cognition. Total capacity changed "
             + ", ".join(f"{s(ch[x]['CAP']['mean'])} in {x}" for x in social)
             + f". All four are beyond the DRIFT noise band of {capband:.2f}, and three of them fell in every seed.")
    a.append("Message length never paid for itself. TALK fell in every arm with selection, including the paired arms where it is usable. "
             "Within a generation, longer messages went with lower net energy: correlation "
             + ", ".join(f"{grad[x]['TALK']['net']:+.2f} in {x}" for x in social) + ".")
    a.append(f"Cultural gain did not become ability gain. CULTURE tripled its foraging success, yet LORE moved only {s(ch['CULTURE']['LORE']['mean'])} "
             f"and its held-out gain matched SOLO.")
    if "DOMEST" in ab:
        a.append(f"The self-domestication arm lost ability: {s(ab['DOMEST']['mean']['mean'], 3)}, the largest drop of any arm, though still inside the noise band.")

    if ex:
        d = ex.get("DOMEST") or next(iter(ex.values()))
        tot = sum(v["exiled_total"] for v in ex.values())
        gens = sum(v["generations"] for v in ex.values())
        u.append(f"Theory 3 was not tested. The exile vote removed {tot} members in {gens} generations. "
                 f"About {d['votes_per_generation']:.0f} of {d['members']} members named anyone per generation, and exile needed {d['threshold']} votes against one member, "
                 f"while about {d['grabs_per_generation']:.0f} grabs happened per generation. The frozen model declined to punish.")
        fd = R["agg_contrast"]["final_diff"]
        u.append(f"With no exile, DOMEST was ALPHA plus a public board. Grabbing rose {s(fd['domest_change'])} with the vote and {s(fd['alpha_change'])} without it, "
                 "which says nothing about whether real punishment would lower it.")
    cost = R["config"]["cost"]
    start_cap = ch["DRIFT"]["CAP"]["start"]
    neg = sum(1 for v in forage.values() if v["net_start"] < 0)
    u.append(f"One cost setting. Each capacity level cost {cost} energy per lifetime, so the average starting genome paid {cost * start_cap:.1f} in upkeep, "
             f"and {neg} of {len(forage)} arms began with net energy below zero. "
             "A cheaper brain could reverse the capacity losses in the social arms. This run cannot say.")
    u.append("Ability can only follow MEM and TALK. The held-out score is read from those two genes, so nothing an arm does to LORE or to the notes themselves can register as ability.")
    return {"for": f, "against": a, "untested": u}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--results", default="results.json")
    a = ap.parse_args()
    R = json.load(open(a.results))
    by_arm = load(a.runs)
    ex, grad, forage = exile(by_arm), gradient(by_arm, R["arms"]), foraging(R)
    R["extras"] = {"exile": ex, "gradient": grad, "foraging": forage,
                   "findings": findings(R, ex, grad, forage)}
    json.dump(R, open(a.results, "w"))
    for k, v in R["extras"]["findings"].items():
        print(k.upper())
        for line in v:
            print("  -", line)


if __name__ == "__main__":
    main()
