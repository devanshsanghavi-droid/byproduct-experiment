"""Statistics and labels for version 4, by the rules in SPEC4.md (amended after check 0).

    PYTHONPATH=. python3 -m evo3.social_stats          -> runs3/s4_stats.json and a printed table
    PYTHONPATH=. python3 -m evo3.social_stats smoke    -> checks 1 and 2 from the two smoke runs

Every comparison is paired by seed. "Above" means the mean paired difference is more than 2
standard errors above zero.
"""
import glob
import json
import math
import statistics as st
import sys

from evo3 import social as S
from evo3.remote import RATE_GPU

UNTRAINED = {"inworld": 0.5102, "near": 0.543, "held": 0.406}


def base_scores() -> dict:
    return json.load(open("runs3/modal/nr-k00-base.json"))["scores"]


def mean_se(v):
    v = list(v)
    if not v:
        return None
    return [round(st.mean(v), 4), round(st.stdev(v) / math.sqrt(len(v)), 4) if len(v) > 1 else None, len(v)]


def verdict(m) -> str:
    """above / below / neither for a [mean, se, n] triple."""
    if not m or m[1] is None:
        return "neither"
    if m[1] == 0:
        return "above" if m[0] > 0 else ("below" if m[0] < 0 else "neither")
    return "above" if m[0] > 2 * m[1] else ("below" if m[0] < -2 * m[1] else "neither")


def load() -> dict:
    runs = {}
    for p in sorted(glob.glob("runs3/modal/s4-*-s*.json")):
        d = json.load(open(p))
        if "smoke" in d["name"] or not d.get("complete"):
            continue
        runs[(d["arm"], d["seed"])] = d
    return runs


def row(d) -> dict:
    g, G = d["generations"], d["config"]["G"]
    t = d["tests"][str(G)]
    keep = [x["keep_rate"] for x in g]
    late = [k for x, k in zip(g, keep) if x["g"] >= 5 and k is not None]
    return {"breadth": d["breadth"], "log2_breadth": math.log2(d["breadth"]) if d["breadth"] > 0 else None,
            "late_breadth": d["late_breadth"], "inworld": t["inworld"], "near": t["near"], "held": t["held"],
            "near_invalid": t.get("near_invalid"), "near_valid_only": t["near"] / (1 - t.get("near_invalid", 0)), "acc_first": g[0]["acc"], "acc_last": g[-1]["acc"],
            "rep_size": g[-1]["rep_size"], "parent_fams": g[-1]["parent_fams"], "invalid_last": g[-1]["invalid"],
            "keep_first": keep[0], "keep_late": st.mean(late) if late else None, "keep": keep,
            "barred_total": sum(x["barred"] for x in g), "copied_last": g[-1]["copied_parents"],
            "examples": sum(x["added"] for x in g), "usd": d.get("usd"), "seconds": d.get("seconds")}


def paired(R, arm, ctl, key, seeds):
    return mean_se(R[(arm, s)][key] - R[(ctl, s)][key] for s in seeds
                   if (arm, s) in R and (ctl, s) in R and R[(arm, s)][key] is not None and R[(ctl, s)][key] is not None)


def summary() -> dict:
    runs = load()
    R = {k: row(d) for k, d in runs.items()}
    seeds = sorted({s for _, s in R})
    null = json.load(open("runs3/null4.json")) if glob.glob("runs3/null4.json") else None
    out = {"n_runs": len(R), "seeds": seeds, "untrained": UNTRAINED, "arms": {}, "theories": {}, "null": bool(null)}
    for arm in S.ARMS:
        mine = [R[(arm, s)] for s in seeds if (arm, s) in R]
        if not mine:
            continue
        a = {k: mean_se(x[k] for x in mine if x[k] is not None) for k in
             ("breadth", "log2_breadth", "late_breadth", "inworld", "near", "held", "acc_first", "acc_last", "rep_size",
              "parent_fams", "invalid_last", "keep_first", "keep_late", "examples", "usd")}
        a["n"], a["control"] = len(mine), S.CONTROL.get(arm)
        a["keep_by_gen"] = [mean_se(x["keep"][i] for x in mine if x["keep"][i] is not None) for i in range(len(mine[0]["keep"]))]
        c = S.CONTROL.get(arm)
        if c:
            a["gap"] = {k: paired(R, arm, c, k, seeds) for k in ("breadth", "log2_breadth", "inworld", "near", "held")}
            a["gap_label"] = {k: verdict(v) for k, v in a["gap"].items()}
            if null:
                ng = [null["tables"][t]["arms"][arm]["gap_log2"][0] for t in ("untrained", "perfect")]
                a["null_gap_log2"] = {"untrained": ng[0], "perfect": ng[1]}
                real = a["gap"]["log2_breadth"]
                if real and real[1] is not None:
                    a["mechanics"] = ("more than mechanics" if real[0] - max(ng) > 2 * real[1]
                                      else "less than mechanics" if min(ng) - real[0] > 2 * real[1] else "within mechanics")
        if null:
            a["null_breadth"] = {t: null["tables"][t]["arms"][arm]["breadth"] for t in ("untrained", "perfect")}
        out["arms"][arm] = a

    def fired(arm):
        mine = [R[(arm, s)] for s in seeds if (arm, s) in R]
        if arm == "COOP":
            return len(mine), len(mine)
        if arm in ("ALPHA",):
            return sum(1 for x in mine if x["keep_first"] is not None and 0.05 <= x["keep_first"] <= 0.95), len(mine)
        if arm == "DOMEST":
            return sum(1 for x in mine if x["barred_total"] >= 1), len(mine)
        if arm == "CULTURE":
            return sum(1 for x in mine if x["copied_last"] >= 1), len(mine)
        return None

    for arm, title in S.THEORIES.items():
        a = out["arms"].get(arm)
        if not a:
            continue
        f = fired(arm)
        tested = f[0] >= 3
        gl = a["gap_label"]
        breadth = ("Not tested" if not tested else
                   "Narrows" if gl["breadth"] == "below" else
                   "No breadth change" if gl["breadth"] != "above" else
                   "Broadens, more than mechanics" if a.get("mechanics") == "more than mechanics" else "Broadens, mechanical")
        out["theories"][arm] = {"title": title, "control": a["control"], "fired": f, "tested": tested, "breadth_label": breadth,
                                "mechanics": a.get("mechanics"), "practised_skill": gl["inworld"],
                                "similar_descriptive": a["gap"]["near"], "new_skill_descriptive": a["gap"]["held"]}

    # rule 1 (learning check), rule 6 (selection), rule C (theory 3 behaviour)
    solo = [R[("SOLO", s)] for s in seeds if ("SOLO", s) in R]
    if solo:
        m = mean_se(x["inworld"] - UNTRAINED["inworld"] for x in solo)
        out["rule1"] = {"solo_minus_untrained_inworld": m, "passes": verdict(m) == "above",
                        "solo_minus_drift_inworld": paired(R, "SOLO", "DRIFT", "inworld", seeds)}
        out["rule1"]["credit"] = ("selection" if verdict(out["rule1"]["solo_minus_drift_inworld"]) == "above" else "the correctness filter")
        out["rule6"] = {k: paired(R, "SOLO", "DRIFT", k, seeds) for k in ("breadth", "log2_breadth", "near", "held", "near_valid_only")}
    alpha = [R[("ALPHA", s)] for s in seeds if ("ALPHA", s) in R]
    if alpha:
        spread = mean_se(x["keep_late"] - x["keep_first"] for x in alpha if x["keep_late"] is not None and x["keep_first"] is not None)
        restrain = paired(R, "DOMEST", "ALPHA", "keep_late", seeds)
        out["ruleC"] = {"alpha_late_minus_first": spread, "spreads": verdict(spread) == "above",
                        "domest_minus_alpha_late": restrain, "restrains": verdict(restrain) == "below"}
    # rule E (exploratory, added after the smoke test): each arm against the stored untrained scores
    out["ruleE"] = {}
    bs = base_scores()
    base_valid = bs["near"] / (1 - bs["near_invalid"])
    out["untrained_valid"] = {"near_invalid": round(bs["near_invalid"], 4), "near_valid_only": round(base_valid, 4),
                              "trained_near_invalid": round(st.mean(x["near_invalid"] for x in R.values()), 4)}
    for arm in out["arms"]:
        mine = [R[(arm, s)] for s in seeds if (arm, s) in R]
        e = {k: mean_se(x[k] - UNTRAINED[k] for x in mine) for k in ("near", "held", "inworld")}
        # post hoc check: the same comparison among valid replies only, because training also fixes reply format
        e["near_valid_only"] = mean_se(x["near_valid_only"] - base_valid for x in mine)
        out["ruleE"][arm] = dict(e, near_label=verdict(e["near"]), held_label=verdict(e["held"]),
                                 near_valid_only_label=verdict(e["near_valid_only"]))
    out["runs"] = [{"arm": a, "seed": s, "breadth": r["breadth"], "inworld": r["inworld"], "near": r["near"], "held": r["held"],
                    "acc": [x["acc"] for x in runs[(a, s)]["generations"]],
                    "rep": [x["rep_size"] for x in runs[(a, s)]["generations"]],
                    "pfams": [x["parent_fams"] for x in runs[(a, s)]["generations"]],
                    "inworld4": runs[(a, s)]["tests"].get("4", {}).get("inworld"), "near4": runs[(a, s)]["tests"].get("4", {}).get("near")}
                   for (a, s), r in sorted(R.items())]
    out["thesis_verdict"] = "Not testable at this scale (declared after check 0, before any model run)"
    out["cost"] = round(sum(x["usd"] or 0 for x in R.values()), 2)
    return out


def smoke() -> dict:
    base, out = base_scores(), {}
    for arm in ("SOLO", "MEDLEY"):
        p = f"runs3/modal/s4-smoke-{arm}.json"
        if not glob.glob(p):
            continue
        d = json.load(open(p))
        g, G = d["generations"], d["config"]["G"]
        t = d["tests"][str(G)]
        fams = sorted(d["draws"])
        before = st.mean(base["pool_" + f] for f in fams)
        after = st.mean(t["pool_" + f] for f in fams)
        gen_s = st.mean(x["seconds"] for x in g)
        other = d["seconds"] - sum(x["seconds"] for x in g)          # model load plus one test battery
        full_s = d["seconds"] + 4 * gen_s + max(0.0, other - 60)      # 4 more generations and one more battery
        out[arm] = {"practised_families": len(fams), "practised_before": round(before, 4), "practised_after": round(after, 4),
                    "practised_gain": round(after - before, 4), "inworld": t["inworld"], "near": t["near"], "held": t["held"],
                    "acc": [x["acc"] for x in g], "invalid": [x["invalid"] for x in g], "energy": [x["energy"] for x in g],
                    "added": [x["added"] for x in g], "rep_size": [x["rep_size"] for x in g], "loss": [x["loss"][-1] if x["loss"] else None for x in g],
                    "keep": [x["keep_rate"] for x in g], "keep_asked": [x.get("keep_asked") for x in g], "keep_invalid": [x["keep_invalid"] for x in g],
                    "votes": [x["votes"] for x in g], "barred": [x["barred"] for x in g],
                    "seconds": d["seconds"], "gen_seconds": round(gen_s, 1), "usd": d["usd"],
                    "full_run_usd_estimate": round(full_s * RATE_GPU, 3), "breadth": d["breadth"], "gpu": d["model"].get("gpu")}
    if "SOLO" in out:
        out["check1"] = {"learning": out["SOLO"]["practised_gain"] >= 0.05, "cost": out["SOLO"]["full_run_usd_estimate"] <= 0.26}
    if "MEDLEY" in out:
        k = out["MEDLEY"]["keep"][0]
        out["check2"] = {"keep_rate_gen1": k, "keep_ok": k is not None and 0.05 <= k <= 0.95,
                         "barred_any": sum(out["MEDLEY"]["barred"]) >= 1}
    return out


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "smoke":
        r = smoke()
        json.dump(r, open("runs3/s4_smoke.json", "w"), indent=1)
        print(json.dumps(r, indent=1))
    else:
        r = summary()
        json.dump(r, open("runs3/s4_stats.json", "w"), indent=1)
        print(f"{r['n_runs']} runs, seeds {r['seeds']}, cost ${r['cost']}")
        for arm, a in r["arms"].items():
            gap = a.get("gap", {})
            print(f"{arm:<8} n={a['n']} breadth {a['breadth'][0]:>6.2f} ({a['breadth'][1]}) inworld {a['inworld'][0]:.3f} near {a['near'][0]:.3f} held {a['held'][0]:.3f} "
                  f"keep {a['keep_first'][0] if a['keep_first'] else ''}->{a['keep_late'][0] if a['keep_late'] else ''} "
                  f"| vs {a.get('control')}: breadth {gap.get('breadth')} doublings {gap.get('log2_breadth')} {a.get('mechanics', '')} inworld {gap.get('inworld')}")
        print(json.dumps({k: r.get(k) for k in ("theories", "rule1", "rule6", "ruleC", "thesis_verdict")}, indent=1))
