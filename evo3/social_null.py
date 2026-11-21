"""Check 0 of SPEC4.md: the null model. Mechanics alone, no model calls, no learning.

    PYTHONPATH=. python3 -m evo3.social_null [--seeds 200] [--keep 0.5] [--vote 0.5] [--set gain=0.3 ...]
        -> runs3/null4.json and a printed table

Runs every arm with a frozen score table: once with the untrained model's stored per-family
scores, once with a perfect table. Reports the breadth each arm produces from its rules alone
and each arm's gap to its control, paired by seed.
"""
import argparse
import asyncio
import json
import math
import statistics as st

from evo3 import social as S


def untrained_table() -> dict:
    s = json.load(open("runs3/modal/nr-k00-base.json"))["scores"]
    return {f: s["pool_" + f] for f in S.TRAINABLE}


def mean_se(v):
    return round(st.mean(v), 4), round(st.stdev(v) / math.sqrt(len(v)), 4) if len(v) > 1 else 0.0


def run(table, seeds, P, keep, vote) -> dict:
    rows = {}
    for arm in S.ARMS:
        rows[arm] = []
        for seed in range(1, seeds + 1):
            llm = S.TableLLM(table, seed * 7 + 3, keep, vote)
            out = asyncio.run(S.run_sim(llm, S.fresh(arm, seed, P), tests=False))
            g = out["generations"]
            rows[arm].append({"breadth": out["breadth"], "late": out["late_breadth"], "rep_size": g[-1]["rep_size"],
                              "parent_fams": g[-1]["parent_fams"], "energy": g[-1]["energy"],
                              "added": sum(x["added"] for x in g)})
    res = {}
    for arm, v in rows.items():
        r = {k: mean_se([x[k] for x in v]) for k in v[0]}
        r["log2_breadth"] = mean_se([math.log2(x["breadth"]) for x in v])
        c = S.CONTROL.get(arm)
        if c:
            r["gap"] = mean_se([a["breadth"] - b["breadth"] for a, b in zip(v, rows[c])])
            r["gap_log2"] = mean_se([math.log2(a["breadth"] / b["breadth"]) for a, b in zip(v, rows[c])])
            r["control"] = c
        res[arm] = r
    lo, hi = min(r["log2_breadth"][0] for r in res.values()), max(r["log2_breadth"][0] for r in res.values())
    return {"arms": res, "largest_gap_doublings": round(hi - lo, 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=200)
    ap.add_argument("--keep", type=float, default=0.5)
    ap.add_argument("--vote", type=float, default=0.5)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--out", default="runs3/null4.json")
    a = ap.parse_args()
    P = dict(S.DEFAULTS)
    for kv in a.set:
        k, v = kv.split("=")
        P[k] = type(S.DEFAULTS[k])(v)
    tables = {"untrained": untrained_table(), "perfect": {f: 1.0 for f in S.TRAINABLE}}
    out = {"seeds": a.seeds, "keep_rate": a.keep, "vote_rate": a.vote, "config": P,
           "tables": {name: run(t, a.seeds, P, a.keep, a.vote) for name, t in tables.items()}}
    json.dump(out, open(a.out, "w"), indent=1)
    for name, t in out["tables"].items():
        print(f"\n== score table: {name}   largest gap between arms: {t['largest_gap_doublings']} doublings")
        print(f"{'arm':<8} {'breadth':>14} {'late':>14} {'rep size':>12} {'parent fams':>12} {'vs':>7} {'gap (families)':>18} {'gap (doublings)':>18}")
        for arm, r in t["arms"].items():
            gap = f"{r['gap'][0]:+.2f} ({r['gap'][1]:.2f})" if "gap" in r else ""
            gl = f"{r['gap_log2'][0]:+.3f} ({r['gap_log2'][1]:.3f})" if "gap" in r else ""
            print(f"{arm:<8} {r['breadth'][0]:>7.2f} ({r['breadth'][1]:.2f}) {r['late'][0]:>7.2f} ({r['late'][1]:.2f}) "
                  f"{r['rep_size'][0]:>12.2f} {r['parent_fams'][0]:>12.2f} {r.get('control', ''):>7} {gap:>18} {gl:>18}")


if __name__ == "__main__":
    main()
