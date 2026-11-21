"""Numbers for the near-transfer rerun, read from runs3/modal/nr-k*.json. Rules are in SPEC3.md, "Near-transfer rerun".

    PYTHONPATH=. python3 -m evo3.near_stats      -> runs3/near_stats.json and a printed table
    PYTHONPATH=. python3 -m evo3.near_stats n2   -> runs3/n2_stats.json (second reserved set)
    PYTHONPATH=. python3 -m evo3.near_stats m3   -> runs3/m3_stats.json (3B model, with 1.5B twins)
"""
import asyncio
import glob
import json
import math
import statistics as st
import sys

from evo3 import breadth as B
from evo3.breadth_stats import mean_se, ols


def chance(near):
    from evo3.local import LocalLLM
    mock = LocalLLM(mock=True)

    async def go():
        tot = {}
        for _ in range(20):
            sc = (await B.battery(mock, [], far=False, public=False, near=near))["scores"]
            for k in ("near", "held", "pool"):
                tot[k] = tot.get(k, 0) + sc[k] / 20
        return tot
    return asyncio.run(go())


def summary(prefix="nr"):
    near = B.reserved2() if prefix == "n2" else B.reserved()
    rest = sorted(set(B.POOL) - set(near))
    runs = [json.load(open(f)) for f in sorted(glob.glob(f"runs3/modal/{prefix}-k*.json"))]
    base = next((r for r in runs if r["k"] == 0), None)
    trained = [r for r in runs if r["k"] > 0]
    ks = sorted({r["k"] for r in trained})
    for r in runs:
        s = r["scores"]
        s["diff"] = s["near"] - s["held"]
        s["near_valid"] = 1 - s["near_invalid"]
        un = [f for f in rest if f not in r["families"]]
        s["rest_unseen"] = st.mean(s["pool_" + f] for f in un)
        if base:      # the earlier comparison, on the 49 non-reserved families
            s["rest_gain"] = st.mean(s["pool_" + f] - base["scores"]["pool_" + f] for f in un)
    out = {"prefix": prefix, "model": runs[0]["model"]["model"] if runs else None,
           "n_runs": len(trained), "reserved": near, "levels": {}, "fit": {}, "base": None,
           "usd": round(sum(r.get("usd") or 0 for r in runs), 2)}
    measures = ["near", "held", "diff", "seen", "near_valid", "rest_gain"]
    x = [math.log2(r["k"]) for r in trained]
    for m in measures:
        if all(m in r["scores"] for r in trained) and len(trained) > 2:
            out["fit"][m] = ols(x, [r["scores"][m] for r in trained])
        for k in ks:
            v = [r["scores"][m] for r in trained if r["k"] == k and m in r["scores"]]
            if v:
                out["levels"].setdefault(str(k), {})[m] = mean_se(v)
    if base:
        out["base"] = {m: base["scores"].get(m) for m in measures}
        n = len(base["items"]["near"])
        out["base"]["near_item_se"] = math.sqrt(base["scores"]["near"] * (1 - base["scores"]["near"]) / n)
        for k in ks:      # each level against the untrained baseline; SE from the runs at that level
            lv = out["levels"][str(k)]["near"]
            lv["minus_base"] = lv["mean"] - base["scores"]["near"]
        allv = [r["scores"]["near"] - base["scores"]["near"] for r in trained]
        out["trained_minus_base"] = {"near": mean_se(allv),
                                     "held": mean_se([r["scores"]["held"] - base["scores"]["held"] for r in trained])}
    lo, hi = out["levels"].get(str(ks[0]), {}).get("near"), out["levels"].get(str(ks[-1]), {}).get("near")
    if lo and hi:
        out["top_minus_bottom"] = {"gap": hi["mean"] - lo["mean"], "se": math.sqrt(lo["se"] ** 2 + hi["se"] ** 2),
                                   "from": ks[0], "to": ks[-1]}
    f1, f2 = out["fit"].get("near"), out["fit"].get("diff")
    out["rule1"] = bool(f1 and f1["slope"] > 0 and f1["t"] > 2)
    out["rule2"] = bool(out["rule1"] and f2 and f2["slope"] > 0 and f2["t"] > 2)
    if prefix == "nr":
        out["outcome"] = "Confirmed, and the near-far split is confirmed" if out["rule2"] else ("Confirmed" if out["rule1"] else "Not confirmed")
    elif len(trained) < 10:
        out["outcome"] = "Not completed"
    else:
        out["outcome"] = "Replicated, and the split is replicated" if out["rule2"] else ("Replicated" if out["rule1"] else "Not replicated")
    if prefix == "m3":      # each 3B run against its 1.5B twin (same families, same training seed)
        twin = {r["name"][2:]: r for r in (json.load(open(f)) for f in glob.glob("runs3/modal/nr-k*.json"))}
        pairs = [(r, twin[r["name"][2:]]) for r in trained if r["name"][2:] in twin]
        assert all(a["families"] == b["families"] and a["seed"] == b["seed"] for a, b in pairs)
        if len(pairs) > 2:
            xs = [math.log2(a["k"]) for a, _ in pairs]
            out["twin"] = {"n": len(pairs),
                           "near": mean_se([a["scores"]["near"] - b["scores"]["near"] for a, b in pairs]),
                           "held": mean_se([a["scores"]["held"] - b["scores"]["held"] for a, b in pairs]),
                           "near_fit": ols(xs, [a["scores"]["near"] - b["scores"]["near"] for a, b in pairs])}
            if base and "-k00-base" in twin:
                out["twin"]["base_near"] = base["scores"]["near"] - twin["-k00-base"]["scores"]["near"]
                out["twin"]["base_held"] = base["scores"]["held"] - twin["-k00-base"]["scores"]["held"]
    out["by_family"] = {f: dict({str(k): st.mean(r["scores"]["near_" + f] for r in trained if r["k"] == k) for k in ks},
                                **({"0": base["scores"]["near_" + f]} if base else {})) for f in near}
    out["chance"] = {k: round(v, 4) for k, v in chance(near).items()}
    out["runs"] = [{"name": r["name"], "k": r["k"], "families": r["families"], "usd": r.get("usd"),
                    **{m: r["scores"].get(m) for m in measures}} for r in runs]
    return out


if __name__ == "__main__":
    pre = sys.argv[1] if len(sys.argv) > 1 else "nr"
    s = summary(pre)
    json.dump(s, open("runs3/near_stats.json" if pre == "nr" else f"runs3/{pre}_stats.json", "w"), indent=1)
    ms = ["near", "held", "diff", "seen", "near_valid", "rest_gain"]
    print(f"{'run':<13}" + "".join(f"{m:>11}" for m in ms))
    for r in s["runs"]:
        print(f"{r['name']:<13}" + "".join(f"{r[m]:>11.3f}" if r.get(m) is not None else f"{'':>11}" for m in ms))
    print("chance:", s["chance"])
    for k, lv in s["levels"].items():
        n = lv["near"]
        print(f"K={k:<3} near {n['mean']:.4f} SE {n['se']:.4f}  minus untrained {n.get('minus_base', float('nan')):+.4f}"
              f" | held {lv['held']['mean']:.4f} SE {lv['held']['se']:.4f}")
    for m, f in s["fit"].items():
        print(f"fit {m:<11} slope per doubling {f['slope']:+.4f}  SE {f['se']:.4f}  t {f['t']:+.2f}  n {f['n']}")
    if "top_minus_bottom" in s:
        g = s["top_minus_bottom"]
        print(f"K={g['to']} minus K={g['from']} near {g['gap']:+.4f} SE {g['se']:.4f}")
    for m, v in s.get("trained_minus_base", {}).items():
        print(f"all trained runs minus untrained, {m}: {v['mean']:+.4f} SE {v['se']:.4f}")
    if "twin" in s:
        t = s["twin"]
        print(f"3B minus 1.5B twin: reserved {t['near']['mean']:+.4f} SE {t['near']['se']:.4f}; held {t['held']['mean']:+.4f} SE {t['held']['se']:.4f};"
              f" slope of the reserved difference {t['near_fit']['slope']:+.4f} SE {t['near_fit']['se']:.4f}")
    print("outcome:", s["outcome"], "| runs", s["n_runs"], "| cost $", s["usd"])
