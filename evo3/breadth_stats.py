"""Numbers for the breadth experiment, read from runs3/modal/br-*.json. Rules are in SPEC3.md.

    PYTHONPATH=. python3 -m evo3.breadth_stats      -> runs3/breadth_stats.json and a printed table
"""
import asyncio
import glob
import json
import math
import statistics as st

from evo3 import breadth as B

MEASURES = ["held", "unseen", "seen", "bbh", "general", "retention"]


def ols(xs, ys):
    """Least-squares slope of y on x with its standard error."""
    n, mx, my = len(xs), sum(xs) / len(xs), sum(ys) / len(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    a = my - b * mx
    res = sum((y - a - b * x) ** 2 for x, y in zip(xs, ys))
    se = math.sqrt(res / (n - 2) / sxx) if n > 2 else float("nan")
    return {"slope": b, "se": se, "t": b / se if se else float("nan"), "intercept": a, "n": n}


def mean_se(v):
    return {"mean": st.mean(v), "se": st.stdev(v) / math.sqrt(len(v)) if len(v) > 1 else None, "n": len(v), "runs": v}


def chance():
    from evo3.local import LocalLLM
    mock = LocalLLM(mock=True)

    async def go():
        tot = {}
        for _ in range(20):
            sc = (await B.battery(mock, [], far=False, public=False))["scores"]
            for k, v in sc.items():
                tot[k] = tot.get(k, 0) + v / 20
        return tot
    return asyncio.run(go())


def load():
    runs = []
    for f in sorted(glob.glob("runs3/modal/br-k*.json")):
        d = json.load(open(f))
        runs.append({"name": d["name"], "k": d["k"], "families": d["families"], "scores": d["scores"], "items": d["items"],
                     "usd": d.get("usd"), "loss": d.get("loss", [])})
    return runs


def summary():
    runs = load()
    base = next((r for r in runs if r["k"] == 0), None)
    trained = [r for r in runs if r["k"] > 0]
    out = {"n_runs": len(trained), "pool": len(B.POOL), "held_families": len(B.HELD), "levels": {}, "fit": {}, "top_minus_bottom": {},
           "base": base["scores"] if base else None, "usd": round(sum(r["usd"] or 0 for r in runs), 2)}
    ks = sorted({r["k"] for r in trained})
    for m in MEASURES:
        pts = [(math.log2(r["k"]), r["scores"][m]) for r in trained if m in r["scores"]]
        if len(pts) > 2:
            out["fit"][m] = ols([p[0] for p in pts], [p[1] for p in pts])
        for k in ks:
            v = [r["scores"][m] for r in trained if r["k"] == k and m in r["scores"]]
            if v:
                out["levels"].setdefault(str(k), {})[m] = mean_se(v)
        lo, hi = out["levels"].get(str(ks[0]), {}).get(m), out["levels"].get(str(ks[-1]), {}).get(m)
        if lo and hi and lo["se"] is not None and hi["se"] is not None:
            out["top_minus_bottom"][m] = {"gap": hi["mean"] - lo["mean"], "se": math.sqrt(lo["se"] ** 2 + hi["se"] ** 2),
                                          "from": ks[0], "to": ks[-1]}
    f = out["fit"].get("held")
    out["outcome"] = None if not f else ("Curve" if f["slope"] > 0 and f["t"] > 2 else "Flat")
    out["far"] = {m: bool(out["fit"].get(m) and out["fit"][m]["slope"] > 0 and out["fit"][m]["t"] > 2) for m in ("bbh", "general")}
    out["held_by_family"] = {h: {str(k): st.mean(r["scores"]["held_" + h] for r in trained if r["k"] == k) for k in ks}
                             for h in sorted(B.HELD)}
    if base:
        for h in out["held_by_family"]:
            out["held_by_family"][h]["0"] = base["scores"]["held_" + h]
    out["chance"] = {k: round(v, 4) for k, v in chance().items() if k in ("held", "pool") or k.startswith("held_")}
    out["bbh_chance"] = base["scores"].get("bbh_chance") if base else None
    out["bbh_by_task"] = {t: dict({str(k): st.mean(r["scores"]["bbh_" + t] for r in trained if r["k"] == k) for k in ks},
                                  **({"0": base["scores"]["bbh_" + t]} if base else {})) for t in B.BBH_TASKS}
    # ---- added after seeing the data (labelled as such on the page) ----
    post = {"trained_minus_base": {}, "unseen_gain_levels": {}, "unseen_n": {str(k): len(B.POOL) - k for k in ks}}
    if base:
        gains = []
        for r in trained:      # gain on the pool families this run did not train on, against the untrained model on the same families
            un = [f for f in B.POOL if f not in r["families"]]
            gains.append((r["k"], st.mean(r["scores"]["pool_" + f] - base["scores"]["pool_" + f] for f in un), len(un)))
        fair = [(math.log2(k), g) for k, g, n in gains if n >= 30]   # K=64 leaves 1 family per run, so it is left out of the fit
        post["unseen_gain_fit"] = ols([x for x, _ in fair], [y for _, y in fair])
        post["unseen_gain_fit"]["max_k"] = max(k for k, _, n in gains if n >= 30)
        for k in ks:
            post["unseen_gain_levels"][str(k)] = mean_se([g for kk, g, _ in gains if kk == k])
        for m in MEASURES:
            v = [r["scores"][m] - base["scores"][m] for r in trained if m in r["scores"] and m in base["scores"]]
            if v:
                post["trained_minus_base"][m] = mean_se(v)
    out["post"] = post
    out["runs"] = [{"name": r["name"], "k": r["k"], **{m: r["scores"].get(m) for m in MEASURES}, "usd": r["usd"]} for r in runs]
    return out


if __name__ == "__main__":
    s = summary()
    json.dump(s, open("runs3/breadth_stats.json", "w"), indent=1)
    print(f"{'run':<13}" + "".join(f"{m:>10}" for m in MEASURES))
    for r in s["runs"]:
        print(f"{r['name']:<13}" + "".join(f"{r[m]:>10.3f}" if r[m] is not None else f"{'':>10}" for m in MEASURES))
    print("chance: held", s["chance"].get("held"), "pool", s["chance"].get("pool"), "bbh", s["bbh_chance"])
    for m, f in s["fit"].items():
        print(f"fit {m:<10} slope per doubling {f['slope']:+.4f}  SE {f['se']:.4f}  t {f['t']:+.2f}")
    for m, g in s["top_minus_bottom"].items():
        print(f"K={g['to']} minus K={g['from']} {m:<10} {g['gap']:+.4f}  SE {g['se']:.4f}")
    f = s["post"]["unseen_gain_fit"]
    print(f"post hoc: unseen-pool gain vs untrained, K <= {f['max_k']}: slope {f['slope']:+.4f} SE {f['se']:.4f} t {f['t']:+.2f}")
    for k, v in s["post"]["unseen_gain_levels"].items():
        print(f"  K={k:<3} unseen families {s['post']['unseen_n'][k]:<3} gain {v['mean']:+.4f} SE {v['se']:.4f}")
    for m, v in s["post"]["trained_minus_base"].items():
        print(f"  all trained runs minus untrained, {m:<10} {v['mean']:+.4f} SE {v['se']:.4f} (n {v['n']})")
    print("outcome:", s["outcome"], "| far:", s["far"], "| cost $", s["usd"])
