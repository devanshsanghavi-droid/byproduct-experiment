"""Turn run files and the held-out ability surface into one results.json.

    python3 -m evo.analyze --runs runs --out results.json

Tolerates missing arms, missing seeds, partial runs and a missing heldout.json.
Every claim in the report is computed here; the page only displays it.
"""
import argparse
import datetime
import glob
import json
import math
import os

THESIS = ("All three surviving theories say intelligence was a byproduct. "
          "A faithful simulation therefore scores agents only on staying alive and being "
          "copied, then measures general ability separately on held-out tests.")

GENES = ["MEM", "TALK", "LORE", "AGG"]
CAPACITY = ["MEM", "TALK", "LORE"]
ARM_ORDER = ["DRIFT", "SOLO", "COOP", "ALPHA", "DOMEST", "CULTURE", "MEDLEY"]
THEORY_ARMS = ["COOP", "DOMEST", "CULTURE", "MEDLEY"]
TESTS = ["H1", "H2", "H3"]

ARM_META = {
    "DRIFT":   {"role": "noise floor", "theory": "Theory 1, chance alone", "foraging": "solo",
                "grab": False, "vote": False, "notes": False, "parents": "random half"},
    "SOLO":    {"role": "baseline", "theory": "Ecology only", "foraging": "solo",
                "grab": False, "vote": False, "notes": False, "parents": "top half by net energy"},
    "COOP":    {"role": "theory", "theory": "Theory 2, cooperation", "foraging": "pairs",
                "grab": False, "vote": False, "notes": False, "parents": "top half by net energy"},
    "ALPHA":   {"role": "control", "theory": "Control for theory 3", "foraging": "pairs",
                "grab": True, "vote": False, "notes": False, "parents": "top half by net energy"},
    "DOMEST":  {"role": "theory", "theory": "Theory 3, self-domestication", "foraging": "pairs",
                "grab": True, "vote": True, "notes": False, "parents": "top half by net energy"},
    "CULTURE": {"role": "theory", "theory": "Theory 4, cultural brain", "foraging": "solo",
                "grab": False, "vote": False, "notes": True, "parents": "top half by net energy"},
    "MEDLEY":  {"role": "theory", "theory": "Theories 2 + 3 + 4", "foraging": "pairs",
                "grab": True, "vote": True, "notes": True, "parents": "top half by net energy"},
}


# ---------- small helpers ----------

def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def sd(xs):
    xs = [x for x in xs if x is not None]
    if len(xs) < 2:
        return None
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def rnd(x, n=4):
    return None if x is None else round(x, n)


def fmt(x, n=2):
    return "n/a" if x is None else f"{x:+.{n}f}"


def load_json(path, warnings):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        warnings.append(f"Could not read {os.path.basename(path)}: {e.__class__.__name__}")
        return None


# ---------- per-run reduction ----------

def genome_means(members):
    """Mean of each gene plus CAP over a list of {genome: {...}} dicts."""
    gs = [m.get("genome") for m in members if isinstance(m, dict) and isinstance(m.get("genome"), dict)]
    if not gs:
        return None
    out = {g: mean([x.get(g) for x in gs]) for g in GENES}
    if all(out[g] is not None for g in CAPACITY):
        out["CAP"] = sum(out[g] for g in CAPACITY)
    else:
        out["CAP"] = None
    return out


def gen_stats(gen):
    """One generation -> gene means, energy, success rate, exiles."""
    agents = gen.get("agents") or []
    m = gen.get("mean") or {}
    genes = genome_means(agents)
    if genes is None:
        if not any(g in m for g in GENES):
            return None
        genes = {g: m.get(g) for g in GENES}
        genes["CAP"] = (sum(m[g] for g in CAPACITY) if all(g in m for g in CAPACITY) else None)
    out = dict(genes)
    out["gross"] = mean([a.get("gross") for a in agents]) if agents else m.get("gross")
    out["net"] = mean([a.get("net") for a in agents]) if agents else m.get("net")
    trials = sum(a.get("trials") or 0 for a in agents)
    if trials:
        out["success_rate"] = sum(a.get("successes") or 0 for a in agents) / trials
    else:
        out["success_rate"] = m.get("success_rate")
    if agents and any("exiled" in a for a in agents):
        out["exiled"] = sum(1 for a in agents if a.get("exiled"))
    else:
        out["exiled"] = gen.get("exiled", 0)
    return out


def ability_of(members, surface):
    """Mean held-out ability of a population, looked up gene by gene from the surface."""
    if not surface:
        return None
    per = {t: [] for t in TESTS}
    for m in members:
        g = m.get("genome") if isinstance(m, dict) else None
        if not isinstance(g, dict):
            continue
        cells = {
            "H1": (surface.get("H1") or {}).get(str(g.get("MEM"))),
            "H3": (surface.get("H3") or {}).get(str(g.get("MEM"))),
            "H2": (surface.get("H2") or {}).get(f"{g.get('MEM')},{g.get('TALK')}"),
        }
        for t, c in cells.items():
            if isinstance(c, dict) and c.get("mean") is not None:
                per[t].append(c["mean"])
    out = {t: mean(v) for t, v in per.items()}
    out["mean"] = mean([out[t] for t in TESTS])
    return out if out["mean"] is not None else None


def reduce_run(run, surface):
    gens = [s for s in (gen_stats(g) for g in run.get("generations") or []) if s]
    if not gens:
        return None
    first_agents = (run["generations"][0].get("agents") or [])
    final_pop = run.get("final_population") or []
    final_genes = genome_means(final_pop)
    end_members = final_pop if final_genes else (run["generations"][-1].get("agents") or [])
    end_genes = final_genes or {k: gens[-1].get(k) for k in GENES + ["CAP"]}
    start_genes = {k: gens[0].get(k) for k in GENES + ["CAP"]}
    return {
        "arm": run.get("arm"), "seed": run.get("seed"),
        "complete": bool(run.get("complete")) and bool(final_genes),
        "gens_done": len(gens),
        "gens_planned": (run.get("config") or {}).get("G"),
        "gens": gens,
        "final_genes": final_genes,
        "start": start_genes, "end": end_genes,
        "end_is_final_population": bool(final_genes),
        "ability_before": ability_of(first_agents, surface),
        "ability_after": ability_of(end_members, surface),
        "usage": run.get("usage") or {},
        "parse_fail": sum((g.get("parse_fail") or 0) for g in run.get("generations") or []),
        "mock": bool(run.get("mock")),
        "model": run.get("model"),
        "config": run.get("config") or {},
        "samples": run.get("samples") or {},
    }


# ---------- aggregation ----------

TRAJ_METRICS = GENES + ["CAP", "gross", "net", "success_rate", "exiled"]


def trajectories(runs):
    """Per arm: gene and energy series per seed and averaged. Gene series get one extra
    point for the unplayed final population when it exists."""
    out = {}
    for arm in sorted({r["arm"] for r in runs}, key=order_key):
        rs = [r for r in runs if r["arm"] == arm]
        seeds = {}
        for r in rs:
            series = {m: [g.get(m) for g in r["gens"]] for m in TRAJ_METRICS}
            if r["final_genes"]:
                for m in TRAJ_METRICS:
                    series[m].append(r["final_genes"].get(m) if m in GENES + ["CAP"] else None)
            seeds[str(r["seed"])] = {m: [rnd(v) for v in vs] for m, vs in series.items()}
        length = max(len(s["MEM"]) for s in seeds.values())
        avg = {}
        for m in TRAJ_METRICS:
            avg[m] = [rnd(mean([s[m][i] for s in seeds.values() if i < len(s[m])]))
                      for i in range(length)]
        out[arm] = {"gens": list(range(length)), "mean": avg, "seeds": seeds,
                    "played": max(r["gens_done"] for r in rs)}
    return out


def order_key(arm):
    return ARM_ORDER.index(arm) if arm in ARM_ORDER else len(ARM_ORDER)


def summarize(per_seed, drift, solo_mean):
    """per_seed: {seed: delta}. Compare the mean with the DRIFT noise floor and SOLO."""
    vals = [v for v in per_seed.values() if v is not None]
    m = mean(vals)
    out = {"seeds": {k: rnd(v) for k, v in per_seed.items()}, "mean": rnd(m), "n": len(vals),
           "exceeds_noise": None, "vs_drift": None, "vs_solo": None, "exceeds_solo": None,
           "consistent_sign": None}
    if m is None:
        return out
    if vals:
        out["consistent_sign"] = all(v > 0 for v in vals) or all(v < 0 for v in vals)
    if drift and drift.get("mean") is not None:
        out["vs_drift"] = rnd(m - drift["mean"])
        if drift.get("band") is not None:
            out["exceeds_noise"] = abs(m - drift["mean"]) > drift["band"]
    if solo_mean is not None:
        out["vs_solo"] = rnd(m - solo_mean)
        if drift and drift.get("band") is not None:
            out["exceeds_solo"] = abs(m - solo_mean) > drift["band"]
    return out


def noise_from(per_seed):
    """DRIFT's across-seed spread. Band = twice the sample SD of DRIFT's per-seed change."""
    vals = [v for v in per_seed.values() if v is not None]
    s = sd(vals)
    return {"mean": rnd(mean(vals)), "sd": rnd(s), "band": rnd(2 * s) if s is not None else None,
            "n": len(vals),
            "method": "band = 2 x sample SD of the DRIFT arm's per-seed change"}


def build_change(runs, key_fn, keys):
    """Generic per-arm change table. key_fn(run, key) -> (start, end) or None."""
    arms = sorted({r["arm"] for r in runs}, key=order_key)
    deltas = {arm: {k: {} for k in keys} for arm in arms}
    starts = {arm: {k: [] for k in keys} for arm in arms}
    ends = {arm: {k: [] for k in keys} for arm in arms}
    for r in runs:
        for k in keys:
            pair = key_fn(r, k)
            if pair is None or pair[0] is None or pair[1] is None:
                continue
            deltas[r["arm"]][k][str(r["seed"])] = pair[1] - pair[0]
            starts[r["arm"]][k].append(pair[0])
            ends[r["arm"]][k].append(pair[1])
    noise = {k: noise_from(deltas.get("DRIFT", {}).get(k, {})) for k in keys}
    solo = {k: mean(list(deltas.get("SOLO", {}).get(k, {}).values())) for k in keys}
    table = {}
    for arm in arms:
        table[arm] = {}
        for k in keys:
            s = summarize(deltas[arm][k], noise[k], solo[k] if arm != "SOLO" else None)
            s["start"] = rnd(mean(starts[arm][k]))
            s["end"] = rnd(mean(ends[arm][k]))
            table[arm][k] = s
    return table, noise


def gene_pair(r, k):
    return (r["start"].get(k), r["end"].get(k))


def ability_pair(r, k):
    if not r["ability_before"] or not r["ability_after"]:
        return None
    return (r["ability_before"].get(k), r["ability_after"].get(k))


def sanity_checks(change, arms):
    """Negative controls: a costly gene the arm cannot use should fall."""
    checks = []

    def add(name, arm, gene, expect):
        c = (change.get(arm) or {}).get(gene)
        if not c or c.get("mean") is None:
            return
        d = c["mean"]
        ok = (d < 0) if expect == "fall" else (d > 0)
        checks.append({"name": name, "arm": arm, "gene": gene, "expected": expect,
                       "observed": rnd(d), "seeds": c["seeds"],
                       "beyond_noise": c.get("exceeds_noise"), "pass": bool(ok)})

    for arm in ["SOLO", "COOP", "ALPHA", "DOMEST"]:
        if arm in arms:
            add(f"LORE is costly and unusable in {arm} (no notes)", arm, "LORE", "fall")
    for arm in ["SOLO", "CULTURE"]:
        if arm in arms:
            add(f"TALK is costly and unusable in {arm} (no partner)", arm, "TALK", "fall")
    if "ALPHA" in arms:
        add("Grabbing pays in ALPHA (no vote)", "ALPHA", "AGG", "rise")
    return checks


def agg_contrast(change, traj):
    out = {"available": "ALPHA" in traj and "DOMEST" in traj}
    for arm in ["ALPHA", "DOMEST"]:
        if arm in traj:
            out[arm] = {"gens": traj[arm]["gens"], "mean": traj[arm]["mean"]["AGG"],
                        "seeds": {s: v["AGG"] for s, v in traj[arm]["seeds"].items()}}
    if out["available"]:
        a, d = change["ALPHA"]["AGG"], change["DOMEST"]["AGG"]
        paired = {s: rnd(d["seeds"][s] - a["seeds"][s]) for s in d["seeds"] if s in a["seeds"]}
        diff = mean(list(paired.values()))
        out["final_diff"] = {
            "mean": rnd(diff), "seeds": paired,
            "alpha_change": a["mean"], "domest_change": d["mean"],
            "all_seeds_lower": bool(paired) and all(v < 0 for v in paired.values()),
            "note": "DOMEST change in AGG minus ALPHA change in AGG, same seed. "
                    "Negative means the vote pushed grabbing down.",
        }
    return out


def collect_samples(runs, per_kind=8):
    """Spread the saved examples across arms. Item shape is normalized defensively."""
    out = {"messages": [], "gossip": [], "notes": []}
    text_keys = ["text", "message", "msg", "content", "note", "gossip", "line"]
    for kind in out:
        buckets = []
        for r in sorted(runs, key=lambda r: (order_key(r["arm"]), str(r["seed"]))):
            items = (r["samples"] or {}).get(kind) or []
            norm = []
            for it in items:
                if isinstance(it, str):
                    it = {"text": it}
                if not isinstance(it, dict):
                    continue
                text = next((it[k] for k in text_keys if isinstance(it.get(k), str) and it[k].strip()), None)
                if not text:
                    continue
                norm.append({
                    "arm": r["arm"], "seed": r["seed"],
                    "gen": next((it[k] for k in ["gen", "generation"] if k in it), None),
                    "agent": next((it[k] for k in ["agent", "id", "agent_id", "from", "author"] if k in it), None),
                    "text": text.strip()[:600],
                })
            if norm:
                buckets.append(norm)
        i = 0
        while len(out[kind]) < per_kind and any(buckets):
            b = buckets[i % len(buckets)]
            if b:
                # take from the end first: later generations show evolved behavior
                out[kind].append(b.pop())
            i += 1
            if i > 10000:
                break
    return out


# ---------- verdict ----------

def verdict(primary_key, primary_label, table, noise, arms, runs, mock):
    """Supported / not supported / unclear, from the numbers only."""
    theory = [a for a in THEORY_ARMS if a in arms and table[a][primary_key]["mean"] is not None]
    checks, reasons = [], []
    drift = noise.get(primary_key) or {}
    solo = (table.get("SOLO") or {}).get(primary_key) or {}

    if not theory:
        return {"status": "no_data", "measure": primary_label,
                "headline": "No theory arm has data yet, so nothing can be said about the thesis.",
                "reasons": [], "checks": []}

    for a in theory:
        c = table[a][primary_key]
        seeds = list(c["seeds"].values())
        above_noise = bool(c["exceeds_noise"]) and (c["vs_drift"] or 0) > 0
        above_solo = bool(c["exceeds_solo"]) and (c["vs_solo"] or 0) > 0
        all_up = bool(seeds) and all(v > 0 for v in seeds)
        checks.append({"arm": a, "change": c["mean"], "seeds": c["seeds"],
                       "above_noise": above_noise, "above_solo": above_solo,
                       "all_seeds_up": all_up, "pass": above_noise and above_solo and all_up})
        reasons.append(
            f"{a}: {primary_label} changed {fmt(c['mean'], 3)} "
            f"(per seed {', '.join(fmt(v, 3) for v in seeds)}). "
            f"Above DRIFT noise: {'yes' if above_noise else 'no'}. "
            f"Above SOLO baseline: {'yes' if above_solo else 'no'}. "
            f"All seeds up: {'yes' if all_up else 'no'}.")

    if drift.get("mean") is not None:
        reasons.append(f"DRIFT (noise floor): mean change {fmt(drift['mean'], 3)}, "
                       f"band +/- {('%.3f' % drift['band']) if drift.get('band') is not None else 'unknown'} "
                       f"from {drift.get('n', 0)} seeds.")
    if solo.get("mean") is not None:
        solo_up = bool(solo.get("exceeds_noise")) and (solo.get("vs_drift") or 0) > 0
        reasons.append(f"SOLO (ecology only): mean change {fmt(solo['mean'], 3)}. "
                       f"Above DRIFT noise: {'yes' if solo_up else 'no'}.")

    partial = [r for r in runs if not r["complete"]]
    if partial:
        reasons.append(f"{len(partial)} of {len(runs)} runs are incomplete; their latest generation was used.")

    passed = [c["arm"] for c in checks if c["pass"]]
    above = [c["arm"] for c in checks if c["above_noise"]]
    prefix = "Mock data, not a finding. " if mock else ""

    if drift.get("band") is None:
        status = "unclear"
        head = (f"Unclear: the DRIFT arm has fewer than two seeds, so there is no noise floor "
                f"to judge the change in {primary_label} against.")
    elif len(passed) >= 2:
        status = "supported"
        head = (f"Supported in this run: {primary_label} rose beyond the DRIFT noise floor and beyond "
                f"the SOLO baseline in {join_names(passed)}, in every seed, although agents were "
                f"scored only on net energy.")
    elif not above:
        status = "not_supported"
        head = (f"Not supported in this run: no theory arm gained {primary_label} beyond the DRIFT "
                f"noise floor.")
    else:
        status = "unclear"
        why = []
        only_noise = [a for a in above if a not in passed]
        if passed:
            why.append(f"only {join_names(passed)} cleared every check")
        if only_noise:
            why.append(f"{join_names(only_noise)} rose above noise but not clearly above SOLO in every seed")
        head = f"Unclear: {'; '.join(why)}."
    return {"status": status, "measure": primary_label, "headline": prefix + head,
            "reasons": reasons, "checks": checks,
            "rule": ("Supported needs at least two theory arms whose mean change is above the DRIFT "
                     "noise band, above the SOLO baseline by more than that band, and positive in "
                     "every seed. Not supported means no theory arm is above the noise band. "
                     "Anything else is unclear.")}


def join_names(xs):
    xs = list(xs)
    if len(xs) <= 1:
        return "".join(xs)
    return ", ".join(xs[:-1]) + " and " + xs[-1]


# ---------- main ----------

def analyze(runs_dir):
    warnings = []
    heldout_path = os.path.join(runs_dir, "heldout.json")
    surface = load_json(heldout_path, warnings) if os.path.exists(heldout_path) else None
    if surface is None:
        warnings.append("No held-out ability surface found; ability columns are empty and the "
                        "verdict falls back to total capacity.")
    elif not any(isinstance(surface.get(t), dict) and surface.get(t) for t in TESTS):
        warnings.append("heldout.json has no test results; treating it as missing.")
        surface = None

    runs = []
    for path in sorted(glob.glob(os.path.join(runs_dir, "*.json"))):
        if os.path.basename(path) == "heldout.json":
            continue
        raw = load_json(path, warnings)
        if not isinstance(raw, dict) or "arm" not in raw or "generations" not in raw:
            continue
        red = reduce_run(raw, surface)
        if red is None:
            warnings.append(f"{os.path.basename(path)} has no finished generation; skipped.")
            continue
        runs.append(red)

    arms = sorted({r["arm"] for r in runs}, key=order_key)
    seeds = sorted({r["seed"] for r in runs}, key=lambda s: str(s))
    for a in ARM_ORDER:
        if a not in arms:
            warnings.append(f"Arm {a} has no runs.")
    mock = any(r["mock"] for r in runs) or bool(surface and surface.get("mock"))

    traj = trajectories(runs) if runs else {}
    change, noise = build_change(runs, gene_pair, GENES + ["CAP"]) if runs else ({}, {})
    ability_keys = TESTS + ["mean"]
    if surface and runs:
        ab_change, ab_noise = build_change(runs, ability_pair, ability_keys)
    else:
        ab_change, ab_noise = {}, {}

    ability = {"available": bool(surface and ab_change), "tests": ability_keys,
               "arms": ab_change, "noise": ab_noise, "per_run": []}
    for r in runs:
        ability["per_run"].append({
            "arm": r["arm"], "seed": r["seed"],
            "before": {k: rnd(v) for k, v in (r["ability_before"] or {}).items()},
            "after": {k: rnd(v) for k, v in (r["ability_after"] or {}).items()}})

    if ability["available"]:
        v = verdict("mean", "held-out ability", ab_change, ab_noise, arms, runs, mock)
    elif runs:
        v = verdict("CAP", "total capacity (MEM + TALK + LORE)", change, noise, arms, runs, mock)
        v["reasons"].insert(0, "Held-out ability is not available, so this verdict uses total "
                               "capacity as a stand-in.")
    else:
        v = {"status": "no_data", "measure": None, "reasons": [], "checks": [],
             "headline": "No run files were found."}

    run_calls = sum((r["usage"].get("calls") or 0) for r in runs)
    run_cost = sum((r["usage"].get("cost_usd") or 0) for r in runs)
    h_usage = (surface or {}).get("usage") or {}
    configs = []
    for r in runs:
        if r["config"] and r["config"] not in configs:
            configs.append(r["config"])
    if len(configs) > 1:
        warnings.append("Runs do not all share one configuration.")

    return {
        "thesis": THESIS,
        "generated": datetime.date.today().isoformat(),
        "mock": mock,
        "models": sorted({r["model"] for r in runs if r["model"]}),
        "heldout_model": (surface or {}).get("model"),
        "config": configs[0] if configs else {},
        "configs": configs,
        "arms": arms,
        "arm_meta": {a: ARM_META.get(a, {"role": "other", "theory": a}) for a in arms},
        "seeds": seeds,
        "runs": [{"arm": r["arm"], "seed": r["seed"], "complete": r["complete"],
                  "gens_done": r["gens_done"], "gens_planned": r["gens_planned"],
                  "calls": r["usage"].get("calls"), "cost_usd": r["usage"].get("cost_usd"),
                  "parse_fail": r["parse_fail"], "mock": r["mock"],
                  "end_is_final_population": r["end_is_final_population"]}
                 for r in sorted(runs, key=lambda r: (order_key(r["arm"]), str(r["seed"])))],
        "trajectories": traj,
        "change": change,
        "noise": noise,
        "ability": ability,
        "surface": ({t: surface.get(t) or {} for t in TESTS} | {"reps": surface.get("reps")}) if surface else None,
        "agg_contrast": agg_contrast(change, traj) if runs else {"available": False},
        "sanity": sanity_checks(change, arms) if runs else [],
        "usage": {"calls": run_calls, "cost_usd": round(run_cost, 4),
                  "heldout_calls": h_usage.get("calls"), "heldout_cost_usd": h_usage.get("cost_usd"),
                  "total_calls": run_calls + (h_usage.get("calls") or 0),
                  "total_cost_usd": round(run_cost + (h_usage.get("cost_usd") or 0), 4)},
        "samples": collect_samples(runs),
        "verdict": v,
        "warnings": warnings,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default="results.json")
    a = ap.parse_args()
    res = analyze(a.runs)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
    print(f"{a.out}: {len(res['runs'])} runs, arms {', '.join(res['arms']) or 'none'}, "
          f"verdict {res['verdict']['status']}, mock {res['mock']}")
    for w in res["warnings"]:
        print("  warning:", w)


if __name__ == "__main__":
    main()
