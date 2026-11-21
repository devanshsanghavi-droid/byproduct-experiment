"""Turn version 2 run files and the transfer measurement into one results2.json.

    python3 -m evo2.analyze --runs runs2 --out results2.json [--headroom-dir runs2]

Reads   <runs>/{ARM}_s{seed}.json   simulation runs (evo2.sim)
        <runs>/measure.json         transfer measurement (evo2.measure), optional
        <runs>/usage.json           simulation usage, optional
        headroom.json, headroom_nothink.json   pre-check, from <runs> or --headroom-dir

Tolerates a missing measure.json, missing arms, missing seeds and unfinished runs; each
gap is recorded in `warnings`. Makes no model calls. Every number on the page is computed
here; the page only displays it. `findings` is left null for a later pass to fill.
"""
import argparse
import datetime
import glob
import json
import math
import os
import sys
import types
from collections import Counter

try:
    from evo2.sim import ARMS as SIM_ARMS, EXILE_VOTES, L1, L2, MAX_RULES, NOTE_WORDS, P_MUTATE, RULE_WORDS, SEED_POOL
except ImportError:
    # evo2.sim imports the model client at module level. The analysis never calls it,
    # so a stand-in module is enough when the client library is not installed.
    sys.modules.setdefault("anthropic", types.ModuleType("anthropic"))
    from evo2.sim import ARMS as SIM_ARMS, EXILE_VOTES, L1, L2, MAX_RULES, NOTE_WORDS, P_MUTATE, RULE_WORDS, SEED_POOL

THESIS = ("All three surviving theories say intelligence was a byproduct. "
          "A faithful simulation therefore scores agents only on staying alive and being "
          "copied, then measures general ability separately on held-out tests.")

ARM_ORDER = ["DRIFT", "SOLO", "COOP", "ALPHA", "DOMEST", "CULTURE", "MEDLEY"]
THEORY_ARMS = ["COOP", "DOMEST", "CULTURE", "MEDLEY"]
GRAB_ARMS = ["ALPHA", "DOMEST", "MEDLEY"]
VOTE_ARMS = ["DOMEST", "MEDLEY"]

ROLE = {"DRIFT": "noise floor", "SOLO": "baseline", "COOP": "theory", "ALPHA": "control",
        "DOMEST": "theory", "CULTURE": "theory", "MEDLEY": "theory"}
THEORY = {"DRIFT": "Chance only", "SOLO": "Ecology only", "COOP": "Theory 2, cooperation",
          "ALPHA": "Control for theory 3", "DOMEST": "Theory 3, self-domestication",
          "CULTURE": "Theory 4, cultural brain", "MEDLEY": "Theories 2 + 3 + 4"}

MEASURES = ["near_energy", "near_illegal", "mid", "far", "general"]
MEASURE_LABEL = {
    "near_energy": "near-transfer energy",
    "near_illegal": "law-breaking tries",
    "mid": "code-breaking",
    "far": "rule induction",
    "general": "general ability",
}
# +1: a higher score is better. -1: a lower score is better.
MEASURE_DIRECTION = {"near_energy": 1, "near_illegal": -1, "mid": 1, "far": 1, "general": 1}
MEASURE_DECIMALS = {"near_energy": 1, "near_illegal": 2, "mid": 3, "far": 3, "general": 3}

HEADROOM_FILES = [("headroom.json", "on"), ("headroom_nothink.json", "off")]
HEADROOM_LABEL = {"bare": "none (bare model)", "method": "general method rules",
                  "method+laws": "method + the two world laws",
                  "code-expert": "code-breaking algorithm", "harmful": "harmful rules"}
HEADROOM_COLS = ["near_energy", "mid_code", "far_rule"]

RULE_TEXT = ("Supported needs at least two theory arms whose mean change is above the DRIFT arm's "
             "mean change by more than the noise band, above the SOLO arm's mean change by more "
             "than that band, and positive in every seed. Not supported means no theory arm is "
             "above the DRIFT band. Anything else is unclear. The noise band is twice the sample "
             "standard deviation of the DRIFT arm's per-seed change.")


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


def fmt(x, n=3):
    return "n/a" if x is None else f"{x:+.{n}f}"


def order_key(arm):
    return ARM_ORDER.index(arm) if arm in ARM_ORDER else len(ARM_ORDER)


def seed_key(s):
    try:
        return (0, int(s))
    except (TypeError, ValueError):
        return (1, str(s))


def join_names(xs):
    xs = list(xs)
    if len(xs) <= 1:
        return "".join(xs)
    return ", ".join(xs[:-1]) + " and " + xs[-1]


def load_json(path, warnings):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        warnings.append(f"Could not read {os.path.basename(path)}: {e.__class__.__name__}.")
        return None


def arm_meta(arm):
    t = SIM_ARMS.get(arm) or {}
    parents = "random half" if t.get("select") == "random" else "top half by energy"
    if t.get("vote"):
        parents += ", exiled excluded"
    return {"role": ROLE.get(arm, "other"), "theory": THEORY.get(arm, arm),
            "foraging": "pairs" if t.get("pairs") else "solo",
            "grab": bool(t.get("grab")), "vote": bool(t.get("vote")), "notes": bool(t.get("notes")),
            "parents": parents}


def final_members(run):
    """The newborn population after the last finished selection. Falls back to the last played one."""
    pop = run.get("next_population") or []
    if pop:
        return pop
    gens = run.get("generations") or []
    return (gens[-1].get("agents") or []) if gens else []


# ---------- trajectories ----------

def trajectories(runs):
    out = {}
    for arm in sorted({r["arm"] for r in runs}, key=order_key):
        rs = sorted([r for r in runs if r["arm"] == arm], key=lambda r: seed_key(r["seed"]))
        keys = []
        for r in rs:
            for g in r["generations"]:
                for k in (g.get("mean") or {}):
                    if k not in keys:
                        keys.append(k)
        keys.append("exiled")
        seeds = {}
        for r in rs:
            seeds[str(r["seed"])] = {
                k: [rnd(g.get("exiled", 0) if k == "exiled" else (g.get("mean") or {}).get(k))
                    for g in r["generations"]] for k in keys}
        length = max((len(r["generations"]) for r in rs), default=0)
        avg = {k: [rnd(mean([s[k][i] for s in seeds.values() if i < len(s[k])])) for i in range(length)]
               for k in keys}
        out[arm] = {"gens": list(range(length)), "keys": keys, "mean": avg, "seeds": seeds}
    return out


# ---------- transfer ----------

def pop_value(pop, measure):
    """Mean of one measure over the agents of one measured population."""
    agents = (pop or {}).get("agents") or []
    if not agents:
        return None
    if measure == "general":
        m, f = pop_value(pop, "mid"), pop_value(pop, "far")
        return None if m is None or f is None else (m + f) / 2
    return mean([a.get(measure) for a in agents if isinstance(a, dict)])


def seed_map(d):
    return {k: rnd(d[k]) for k in sorted(d, key=seed_key)}


def summarize(per_seed, drift, solo_mean):
    """Compare one arm's per-seed change with the DRIFT noise floor and the SOLO baseline."""
    vals = [v for v in per_seed.values() if v is not None]
    m = mean(vals)
    out = {"mean": rnd(m), "seeds": seed_map(per_seed), "n": len(vals),
           "vs_drift": None, "exceeds_noise": None, "vs_solo": None, "exceeds_solo": None,
           "all_seeds_up": bool(vals) and all(v > 0 for v in vals),
           "consistent_sign": (all(v > 0 for v in vals) or all(v < 0 for v in vals)) if vals else None}
    if m is None:
        return out
    if drift.get("mean") is not None:
        out["vs_drift"] = rnd(m - drift["mean"])
        if drift.get("band") is not None:
            out["exceeds_noise"] = abs(m - drift["mean"]) > drift["band"]
    if solo_mean is not None:
        out["vs_solo"] = rnd(m - solo_mean)
        if drift.get("band") is not None:
            out["exceeds_solo"] = abs(m - solo_mean) > drift["band"]
    return out


def build_transfer(pops, arms):
    """transfer[measure][arm], noise[measure] and reference levels, from measure.json populations."""
    by = {"bare": {}, "gen0": {}, "final": {}, "final_no_notes": {}}
    for key, p in (pops or {}).items():
        when = (p or {}).get("when") or key.split("|")[0]
        if when not in by:
            continue
        seed = str(p.get("seed") if p.get("seed") is not None else key.split("|")[-1])
        if when in ("bare", "gen0"):
            by[when][seed] = p
        else:
            arm = p.get("arm") or key.split("|")[1]
            by[when].setdefault(arm, {})[seed] = p

    transfer, noise, meta_bare, meta_gen0 = {}, {}, {}, {}
    for ms in MEASURES:
        bare = {s: pop_value(p, ms) for s, p in by["bare"].items()}
        gen0 = {s: pop_value(p, ms) for s, p in by["gen0"].items()}
        meta_bare[ms] = {"mean": rnd(mean(list(bare.values()))), "seeds": seed_map(bare)}
        meta_gen0[ms] = {"mean": rnd(mean(list(gen0.values()))), "seeds": seed_map(gen0)}

        raw = {}
        for arm in arms:
            end = {s: pop_value(p, ms) for s, p in (by["final"].get(arm) or {}).items()}
            end = {s: v for s, v in end.items() if v is not None}
            nn = {s: pop_value(p, ms) for s, p in (by["final_no_notes"].get(arm) or {}).items()}
            nn = {s: v for s, v in nn.items() if v is not None}
            raw[arm] = {
                "end": end, "nn": nn,
                "start": {s: gen0[s] for s in end if gen0.get(s) is not None},
                "bare": {s: bare[s] for s in end if bare.get(s) is not None},
                "change": {s: end[s] - gen0[s] for s in end if gen0.get(s) is not None},
                "vs_bare": {s: end[s] - bare[s] for s in end if bare.get(s) is not None},
                "nn_change": {s: nn[s] - gen0[s] for s in nn if gen0.get(s) is not None},
            }

        dvals = list((raw.get("DRIFT") or {}).get("change", {}).values())
        s_ = sd(dvals)
        noise[ms] = {"mean": rnd(mean(dvals)), "sd": rnd(s_), "band": rnd(2 * s_) if s_ is not None else None,
                     "n": len(dvals), "method": "band = 2 x sample SD of the DRIFT arm's per-seed change"}
        solo_mean = mean(list((raw.get("SOLO") or {}).get("change", {}).values()))

        transfer[ms] = {}
        for arm in arms:
            r = raw[arm]
            entry = {
                "start": rnd(mean(list(r["start"].values()))), "end": rnd(mean(list(r["end"].values()))),
                "bare": rnd(mean(list(r["bare"].values()))),
                "start_seeds": seed_map(r["start"]), "end_seeds": seed_map(r["end"]),
                "bare_seeds": seed_map(r["bare"]),
                "change": summarize(r["change"], noise[ms], solo_mean if arm != "SOLO" else None),
                "vs_bare": {"mean": rnd(mean(list(r["vs_bare"].values()))), "seeds": seed_map(r["vs_bare"])},
                "no_notes": None,
            }
            if r["nn"]:
                entry["no_notes"] = {
                    "end": rnd(mean(list(r["nn"].values()))), "end_seeds": seed_map(r["nn"]),
                    "change": {"mean": rnd(mean(list(r["nn_change"].values()))), "seeds": seed_map(r["nn_change"])},
                    # what the notes add on top of the instincts, same seed, same tasks
                    "notes_effect": rnd(mean([r["end"][s] - r["nn"][s] for s in r["nn"] if s in r["end"]])),
                }
            transfer[ms][arm] = entry

    measured = {arm: sorted((by["final"].get(arm) or {}), key=seed_key) for arm in arms}
    meta = {
        "available": any(measured.values()) and bool(by["gen0"]),
        "measures": MEASURES, "labels": MEASURE_LABEL, "direction": MEASURE_DIRECTION,
        "decimals": MEASURE_DECIMALS,
        "bare": meta_bare, "gen0": meta_gen0, "measured_seeds": measured,
        "agents_per_population": max((len((p or {}).get("agents") or []) for p in (pops or {}).values()), default=0),
        "note": ("Each population value is the mean over its agents. Agent slot i of a seed gets the "
                 "same task instances in every population. general = mean of mid and far."),
    }
    return transfer, noise, meta


# ---------- verdicts ----------

def verdict(measure, transfer, noise, arms, runs, mock, tail):
    label = MEASURE_LABEL[measure]
    dec = MEASURE_DECIMALS[measure]
    table = transfer.get(measure) or {}
    drift = noise.get(measure) or {}
    theory = [a for a in THEORY_ARMS if a in arms and ((table.get(a) or {}).get("change") or {}).get("mean") is not None]
    base = {"measure": measure, "label": label, "rule": RULE_TEXT, "reasons": [], "checks": [],
            "noise": {"mean": drift.get("mean"), "band": drift.get("band"), "n": drift.get("n", 0)}}
    if not theory:
        base.update(status="no_data",
                    headline=f"No theory arm has a measured final population yet, so nothing can be said about {label}.")
        return base

    checks, reasons = [], []
    for a in theory:
        c = table[a]["change"]
        seeds = list(c["seeds"].values())
        above_drift = bool(c["exceeds_noise"]) and (c["vs_drift"] or 0) > 0
        above_solo = bool(c["exceeds_solo"]) and (c["vs_solo"] or 0) > 0
        all_up = bool(seeds) and all(v > 0 for v in seeds)
        checks.append({"arm": a, "change": c["mean"], "seeds": c["seeds"],
                       "vs_drift": c["vs_drift"], "vs_solo": c["vs_solo"],
                       "above_drift": above_drift, "above_solo": above_solo, "all_seeds_up": all_up,
                       "pass": above_drift and above_solo and all_up})
        reasons.append(
            f"{a}: {label} changed {fmt(c['mean'], dec)} (per seed {', '.join(fmt(v, dec) for v in seeds)}). "
            f"Above DRIFT by more than the band: {'yes' if above_drift else 'no'}. "
            f"Above SOLO by more than the band: {'yes' if above_solo else 'no'}. "
            f"Positive in every seed: {'yes' if all_up else 'no'}.")

    if drift.get("mean") is not None:
        band = f"{drift['band']:.{dec}f}" if drift.get("band") is not None else "unknown"
        reasons.append(f"DRIFT (chance only): mean change {fmt(drift['mean'], dec)} over {drift.get('n', 0)} seeds. "
                       f"Noise band: {band} on either side of that.")
    solo = (table.get("SOLO") or {}).get("change") or {}
    if solo.get("mean") is not None:
        solo_up = bool(solo.get("exceeds_noise")) and (solo.get("vs_drift") or 0) > 0
        reasons.append(f"SOLO (ecology only): mean change {fmt(solo['mean'], dec)}. "
                       f"Above DRIFT by more than the band: {'yes' if solo_up else 'no'}.")
    partial = [r for r in runs if not r.get("complete")]
    if partial:
        reasons.append(f"{len(partial)} of {len(runs)} runs are unfinished and have no measured final population.")

    passed = [c["arm"] for c in checks if c["pass"]]
    above = [c["arm"] for c in checks if c["above_drift"]]
    if drift.get("band") is None:
        status = "unclear"
        head = (f"Unclear: the DRIFT arm has fewer than two measured seeds, so there is no noise band "
                f"to judge the change in {label} against.")
    elif len(passed) >= 2:
        status = "supported"
        head = (f"Supported in this run: {label} rose beyond the DRIFT noise band and beyond the SOLO "
                f"baseline in {join_names(passed)}, in every seed. {tail}")
    elif not above:
        status = "not_supported"
        head = f"Not supported in this run: no theory arm gained {label} beyond the DRIFT noise band."
    else:
        status = "unclear"
        why = []
        if passed:
            why.append(f"only {join_names(passed)} cleared every check")
        rest = [a for a in above if a not in passed]
        if rest:
            why.append(f"{join_names(rest)} rose beyond the DRIFT band and failed the SOLO check or the every-seed check")
        head = f"Unclear: {'; '.join(why)}."
    base.update(status=status, headline=("Mock data, not a finding. " if mock else "") + head,
                reasons=reasons, checks=checks)
    return base


def ceiling(transfer, noise, arms):
    """Does any final population beat the bare model by more than the noise band?"""
    out = {}
    for ms in ["mid", "far", "general"]:
        band = (noise.get(ms) or {}).get("band")
        per, best = {}, None
        for arm in arms:
            vb = ((transfer.get(ms) or {}).get(arm) or {}).get("vs_bare") or {}
            if vb.get("mean") is None:
                continue
            exceeds = (vb["mean"] > band) if band is not None else None
            per[arm] = {"diff": vb["mean"], "seeds": vb.get("seeds") or {}, "exceeds": exceeds}
            if best is None or vb["mean"] > per[best]["diff"]:
                best = arm
        flags = [v["exceeds"] for v in per.values()]
        out[ms] = {"label": MEASURE_LABEL[ms], "band": band, "arms": per, "best_arm": best,
                   "any_exceeds": (any(f is True for f in flags) if flags and band is not None else None)}
    return out


# ---------- social ----------

def social(runs, traj):
    grab = {}
    for arm in GRAB_ARMS:
        t = traj.get(arm)
        if not t or "grab_rate" not in t["mean"]:
            continue
        series = t["mean"]["grab_rate"]
        known = [v for v in series if v is not None]
        grab[arm] = {"gens": t["gens"], "mean": series,
                     "seeds": {s: v.get("grab_rate") for s, v in t["seeds"].items()},
                     "first": known[0] if known else None, "last": known[-1] if known else None,
                     "overall": rnd(mean(known))}
    exile = {}
    for arm in VOTE_ARMS:
        rs = [r for r in runs if r["arm"] == arm]
        gens = [g for r in rs for g in r["generations"]]
        if not gens:
            continue
        votes = [(g.get("mean") or {}).get("votes_cast") for g in gens]
        exile[arm] = {
            "exiled_total": sum(g.get("exiled") or 0 for g in gens),
            "generations_with_exile": sum(1 for g in gens if (g.get("exiled") or 0) > 0),
            "generations": len(gens),
            "votes_per_generation": rnd(mean(votes), 2),
            "members": len(gens[0].get("agents") or []),
            "threshold": EXILE_VOTES,
            "most_votes_against_one_member": max((a.get("votes_against") or 0 for g in gens for a in (g.get("agents") or [])), default=0),
            "grabs_per_generation": rnd(mean([sum(a.get("grabs") or 0 for a in (g.get("agents") or [])) for g in gens]), 2),
            "exiled_by_seed": {str(r["seed"]): sum(g.get("exiled") or 0 for g in r["generations"])
                               for r in sorted(rs, key=lambda r: seed_key(r["seed"]))},
        }
    return {"grab_rate": grab, "exile": exile}


# ---------- instincts ----------

def share(members, pred):
    return sum(1 for m in members if pred(m)) / len(members) if members else None


def rules_block(runs, arms):
    seed_set = set(SEED_POOL)
    by_arm = {a: sorted([r for r in runs if r["arm"] == a], key=lambda r: seed_key(r["seed"])) for a in arms}

    # (a) the 16 seed rules, generation 0 against the final population
    gen0_by_seed = {}
    for r in runs:
        gen0_by_seed.setdefault(str(r["seed"]), r["generations"][0].get("agents") or [])
    seed_rules = []
    for rule in SEED_POOL:
        holds = lambda m, rule=rule: rule in (m.get("instinct") or [])
        row = {"rule": rule, "L1": bool(L1.search(rule)), "L2": bool(L2.search(rule)),
               "gen0": rnd(mean([share(ag, holds) for ag in gen0_by_seed.values()])), "arms": {}}
        for a in arms:
            row["arms"][a] = {
                "gen0": rnd(mean([share(r["generations"][0].get("agents") or [], holds) for r in by_arm[a]])),
                "final": rnd(mean([share(final_members(r), holds) for r in by_arm[a]])),
            }
        seed_rules.append(row)

    # (b) most common rules in the final populations, exact string match
    top, summary = {}, {}
    for a in arms:
        rs = by_arm[a]
        counts = Counter()
        for r in rs:
            fm = final_members(r)
            if not fm:
                continue
            for rule, c in Counter(x for m in fm for x in set(m.get("instinct") or [])).items():
                counts[rule] += c / len(fm)
        n = max(1, len(rs))
        ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:6]
        top[a] = [{"rule": rule, "share": rnd(c / n), "seeded": rule in seed_set,
                   "L1": bool(L1.search(rule)), "L2": bool(L2.search(rule))} for rule, c in ranked]
        slots = [x for r in rs for m in final_members(r) for x in (m.get("instinct") or [])]
        summary[a] = {
            "rules_per_agent_gen0": rnd(mean([mean([len(m.get("instinct") or []) for m in (r["generations"][0].get("agents") or [])]) for r in rs]), 2),
            "rules_per_agent_final": rnd(mean([mean([len(m.get("instinct") or []) for m in final_members(r)]) for r in rs]), 2),
            "unseeded_share_final": rnd(sum(1 for x in slots if x not in seed_set) / len(slots)) if slots else None,
            "distinct_rules_final": len(set(slots)),
        }

    # (c) who mentions the hidden laws, by keyword
    def mention_shares(pops):
        out = {}
        for where, text in (("instinct", lambda m: " ".join(m.get("instinct") or [])),
                            ("note", lambda m: m.get("note_in") or ""),
                            ("any", lambda m: " ".join(m.get("instinct") or []) + " " + (m.get("note_in") or ""))):
            out[where] = {"L1": rnd(mean([share(p, lambda m: bool(L1.search(text(m)))) for p in pops])),
                          "L2": rnd(mean([share(p, lambda m: bool(L2.search(text(m)))) for p in pops]))}
        return out

    mentions = {"gen0": mention_shares(list(gen0_by_seed.values())), "final": {}}
    for a in arms:
        mentions["final"][a] = mention_shares([final_members(r) for r in by_arm[a]])
    return {"seed_rules": seed_rules, "top_final": top, "summary": summary, "law_mentions": mentions,
            "note": ("Shares are fractions of agents, averaged over seeds. Rules are matched as exact strings. "
                     "Law mentions are found by keyword and are approximate; some seed rules contain the keywords.")}


# ---------- samples ----------

def collect_samples(runs, per_kind=8):
    ordered = sorted(runs, key=lambda r: (order_key(r["arm"]), seed_key(r["seed"])))
    out = {}

    def spread(buckets):
        """Round-robin over arms, one new text at a time, skipping repeats."""
        picked, seen, i, idle = [], set(), 0, 0
        arms = [a for a in buckets if buckets[a]]
        while len(picked) < per_kind and arms and idle < len(arms):
            b = buckets[arms[i % len(arms)]]
            i += 1
            took = False
            while b:
                it = b.pop(0)
                key = it["text"].strip().lower()
                if key not in seen:
                    seen.add(key)
                    picked.append(it)
                    took = True
                    break
            idle = 0 if took else idle + 1
        return picked

    # notes carried by the final newborns
    buckets = {}
    for r in ordered:
        last = len(r["generations"]) - 1
        counts = Counter()
        for m in final_members(r):
            for line in (m.get("note_in") or "").split("\n"):
                line = line.strip()
                line = line[1:].strip() if line.startswith("-") else line
                if line:
                    counts[line] += 1
        for text, c in counts.most_common():
            buckets.setdefault(r["arm"], []).append(
                {"arm": r["arm"], "seed": r["seed"], "gen": last, "agent": None, "text": text[:600], "holders": c})
    out["notes"] = spread(buckets)

    for kind in ("new_rules", "messages", "votes"):
        buckets = {}
        for r in ordered:
            items = [it for it in ((r.get("samples") or {}).get(kind) or [])
                     if isinstance(it, dict) and isinstance(it.get("text"), str) and it["text"].strip()]
            # later generations first: they show evolved behaviour
            for it in sorted(items, key=lambda it: -(it.get("gen") or 0)):
                row = {"arm": r["arm"], "seed": r["seed"], "gen": it.get("gen"), "agent": it.get("agent"),
                       "text": it["text"].strip()[:600]}
                for k in ("topic", "to", "round"):
                    if it.get(k) is not None:
                        row[k] = it[k]
                buckets.setdefault(r["arm"], []).append(row)
        out[kind] = spread(buckets)
    return out


# ---------- headroom and usage ----------

def headroom_block(dirs, warnings):
    tables, seen = [], set()
    for fname, default_thinking in HEADROOM_FILES:
        for d in dirs:
            path = os.path.join(d, fname)
            if fname in seen or not os.path.exists(path):
                continue
            h = load_json(path, warnings)
            if not isinstance(h, dict) or not isinstance(h.get("scores"), dict):
                continue
            seen.add(fname)
            rows = []
            for name, sc in h["scores"].items():
                row = {"set": name, "label": HEADROOM_LABEL.get(name, name), "rules": (h.get("sets") or {}).get(name) or []}
                for col in HEADROOM_COLS + ["far_order", "near_illegal_tries"]:
                    c = sc.get(col) if isinstance(sc.get(col), dict) else {}
                    row[col] = {"mean": c.get("mean"), "se": c.get("se"), "n": c.get("n")}
                rows.append(row)
            tables.append({"file": fname, "thinking": h.get("thinking") or default_thinking, "model": h.get("model"),
                           "reps": h.get("reps"), "mock": bool(h.get("mock")), "rows": rows,
                           "usage": h.get("usage") or {}})
    return {"available": bool(tables), "columns": HEADROOM_COLS, "tables": tables} if tables else None


def usage_block(runs, runs_dir, measure, headroom, warnings):
    path = os.path.join(runs_dir, "usage.json")
    sim = load_json(path, warnings) if os.path.exists(path) else None
    run_calls = sum(r.get("calls") or 0 for r in runs)
    if isinstance(sim, dict):
        simulation = {"calls": sim.get("calls"), "cost_usd": sim.get("cost_usd"),
                      "input_tokens": sim.get("input_tokens"), "output_tokens": sim.get("output_tokens"),
                      "failures": sim.get("failures"), "source": "usage.json", "calls_in_run_files": run_calls}
    else:
        simulation = {"calls": run_calls if runs else None, "cost_usd": None, "source": "run files",
                      "calls_in_run_files": run_calls}
        if runs:
            warnings.append("No usage.json found; simulation calls are summed from the run files and the simulation spend is unknown.")
    mu = (measure or {}).get("usage") or {}
    measurement = {"calls": mu.get("calls"), "cost_usd": mu.get("cost_usd"),
                   "input_tokens": mu.get("input_tokens"), "output_tokens": mu.get("output_tokens")} if measure else None
    hruns = [{"file": t["file"], "thinking": t["thinking"], "calls": t["usage"].get("calls"),
              "cost_usd": t["usage"].get("cost_usd")} for t in (headroom or {}).get("tables", [])]
    head = {"calls": sum(h["calls"] or 0 for h in hruns), "cost_usd": round(sum(h["cost_usd"] or 0 for h in hruns), 4),
            "runs": hruns} if hruns else None
    parts = [simulation, measurement, head]
    return {"simulation": simulation, "measurement": measurement, "headroom": head,
            "total_calls": sum((p or {}).get("calls") or 0 for p in parts),
            "total_cost_usd": round(sum((p or {}).get("cost_usd") or 0 for p in parts), 4)}


# ---------- main ----------

def analyze(runs_dir, headroom_dir="runs2"):
    warnings = []
    runs = []
    for path in sorted(glob.glob(os.path.join(runs_dir, "*_s*.json"))):
        raw = load_json(path, warnings)
        if not isinstance(raw, dict) or "arm" not in raw or not isinstance(raw.get("generations"), list):
            continue
        if not raw["generations"]:
            warnings.append(f"{os.path.basename(path)} has no finished generation; skipped.")
            continue
        runs.append(raw)
    runs.sort(key=lambda r: (order_key(r["arm"]), seed_key(r["seed"])))
    if not runs:
        warnings.append(f"No run files found in {runs_dir}.")

    arms = sorted({r["arm"] for r in runs}, key=order_key)
    seeds = sorted({r["seed"] for r in runs}, key=seed_key)
    for a in ARM_ORDER:
        if a not in arms:
            warnings.append(f"Arm {a} has no runs.")
    for a in arms:
        have = {r["seed"] for r in runs if r["arm"] == a}
        miss = [s for s in seeds if s not in have]
        if miss:
            warnings.append(f"Arm {a} is missing seed {', '.join(str(s) for s in miss)}.")
    partial = [r for r in runs if not r.get("complete")]
    if partial:
        warnings.append(f"{len(partial)} of {len(runs)} runs are unfinished: " + ", ".join(
            f"{r['arm']} seed {r['seed']} ({len(r['generations'])} of {(r.get('config') or {}).get('G')})" for r in partial) + ".")

    mpath = os.path.join(runs_dir, "measure.json")
    measure = load_json(mpath, warnings) if os.path.exists(mpath) else None
    if not isinstance(measure, dict) or not isinstance(measure.get("populations"), dict):
        if runs:
            warnings.append("No transfer measurement (measure.json) found; transfer columns and both verdicts are empty.")
        measure = None
    pops = (measure or {}).get("populations") or {}
    if measure:
        if not measure.get("complete"):
            warnings.append("measure.json is marked incomplete; some populations are not measured yet.")
        for r in runs:
            if r.get("complete") and f"final|{r['arm']}|{r['seed']}" not in pops:
                warnings.append(f"Final population of {r['arm']} seed {r['seed']} is not measured.")

    configs = []
    for r in runs:
        if r.get("config") and r["config"] not in configs:
            configs.append(r["config"])
    if len(configs) > 1:
        warnings.append("Runs do not all share one configuration.")
    models = sorted({r["model"] for r in runs if r.get("model")} | ({measure["model"]} if measure and measure.get("model") else set()))
    if len(models) > 1:
        warnings.append("More than one model appears in the inputs.")
    think = sorted({str(r["thinking"]) for r in runs if r.get("thinking")})
    if measure and measure.get("thinking") and think and str(measure["thinking"]) not in think:
        warnings.append(f"Hidden reasoning was {'/'.join(think)} in the simulation and {measure['thinking']} in the measurement.")
    if len(think) > 1:
        warnings.append("Runs do not all share one hidden-reasoning setting.")
    if not think and measure and measure.get("thinking"):
        think = [str(measure["thinking"])]

    dirs = [runs_dir] + ([headroom_dir] if headroom_dir and os.path.abspath(headroom_dir) != os.path.abspath(runs_dir) else [])
    headroom = headroom_block(dirs, warnings)
    if headroom is None:
        warnings.append("No headroom pre-check files found.")

    mock = (any(r.get("mock") for r in runs) or bool(measure and measure.get("mock"))
            or any(t["mock"] for t in (headroom or {}).get("tables", [])))
    if measure and runs and bool(measure.get("mock")) != any(r.get("mock") for r in runs):
        warnings.append("The simulation and the measurement disagree on whether they are mock data.")

    traj = trajectories(runs) if runs else {}
    transfer, noise, tmeta = build_transfer(pops, arms)
    v_general = verdict("general", transfer, noise, arms, runs, mock,
                        "Agents were selected only on energy and no test score was ever used for selection.")
    v_knowledge = verdict("near_energy", transfer, noise, arms, runs, mock,
                          "The test world was new and kept the same two hidden laws.")

    return {
        "thesis": THESIS,
        "version": 2,
        "generated": datetime.date.today().isoformat(),
        "mock": bool(mock),
        "model": ", ".join(models) if models else None,
        "models": models,
        "thinking": think[0] if len(think) == 1 else ("mixed" if think else None),
        "config": configs[0] if configs else {},
        "configs": configs,
        "constants": {"max_rules": MAX_RULES, "rule_words": RULE_WORDS, "note_words": NOTE_WORDS,
                      "exile_votes": EXILE_VOTES, "p_mutate": P_MUTATE, "seed_rules": len(SEED_POOL),
                      "seed_rules_per_agent": 2},
        "arms": arms,
        "arm_meta": {a: arm_meta(a) for a in arms},
        "theory_arms": [a for a in THEORY_ARMS if a in arms],
        "seeds": seeds,
        "runs": [{"arm": r["arm"], "seed": r["seed"], "gens_done": len(r["generations"]),
                  "gens_planned": (r.get("config") or {}).get("G"), "complete": bool(r.get("complete")),
                  "calls": r.get("calls"), "parse_fail": r.get("parse_fail"), "mock": bool(r.get("mock")),
                  "measured": f"final|{r['arm']}|{r['seed']}" in pops} for r in runs],
        "trajectories": traj,
        "transfer": transfer,
        "transfer_meta": tmeta,
        "noise": noise,
        "verdict_general": v_general,
        "verdict_knowledge": v_knowledge,
        "ceiling": ceiling(transfer, noise, arms),
        "social": social(runs, traj),
        "rules": rules_block(runs, arms) if runs else None,
        "samples": collect_samples(runs),
        "headroom": headroom,
        "usage": usage_block(runs, runs_dir, measure, headroom, warnings),
        "findings": None,
        "warnings": warnings,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs2")
    ap.add_argument("--out", default="results2.json")
    ap.add_argument("--headroom-dir", default="runs2")
    a = ap.parse_args()
    res = analyze(a.runs, a.headroom_dir)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
    u = res["usage"]
    print(f"{a.out}: {len(res['runs'])} runs, arms {', '.join(res['arms']) or 'none'}, "
          f"general {res['verdict_general']['status']}, knowledge {res['verdict_knowledge']['status']}, "
          f"mock {res['mock']}, {u['total_calls']} calls, ${u['total_cost_usd']:.2f}, "
          f"{len(res['warnings'])} warnings")
    for w in res["warnings"]:
        print("  warning:", w)


if __name__ == "__main__":
    main()
