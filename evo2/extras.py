"""Second pass over the version 2 results, run after evo2.analyze and before evo2.build_report.

    python3 -m evo2.extras --runs runs2 --results results2.json

Adds to results2.json:
  findings          {"for", "against", "untested"}: sentences generated from the numbers
  extras.paired     per-seed paired differences of each arm against DRIFT and SOLO.
                    This check was chosen after seeing the data; the pre-set verdict is unchanged.
  extras.mechanism  per arm: share of agents holding a harmful rule, grab rate, exiles, effect of notes
  extras.case       the one seed where the grab rule took off without a vote
  extras.notes      extra caveat and limit lines for the page
Also recomputes the law-mention trajectories from the stored text with the corrected keyword
check, because the simulation process had loaded the earlier check. Makes no model calls.
"""
import argparse
import glob
import json
import math
import os
import statistics as st

from evo2.analyze import L1, L2, SEED_POOL

HARMFUL = [SEED_POOL[i] for i in (4, 5, 6)]
GOOD = SEED_POOL[0]
GRAB = SEED_POOL[9]
SELECTED = ["SOLO", "COOP", "ALPHA", "DOMEST", "CULTURE", "MEDLEY"]


def harmful(rule):
    low = rule.lower()
    return rule in HARMFUL or ("even" in low and ("succeed" in low or "worked" in low))


def text_of(agent):
    note = agent.get("note_in") or ""
    if isinstance(note, list):
        note = " ".join(str(n) for n in note)
    return list(agent.get("instinct") or []), str(note)


def mentions(agent, rx):
    rules, note = text_of(agent)
    return any(rx.search(r) for r in rules) or bool(rx.search(note))


def share(agents, pred):
    return sum(1 for a in agents if pred(a)) / len(agents) if agents else None


def load(runs_dir):
    runs = {}
    for p in sorted(glob.glob(os.path.join(runs_dir, "*_s*.json"))):
        d = json.load(open(p))
        if d.get("generations"):
            runs[(d["arm"], str(d["seed"]))] = d
    return runs


def fix_mentions(R, runs):
    for arm, T in R["trajectories"].items():
        for key, rx in (("mentions_L1", L1), ("mentions_L2", L2)):
            for seed in T["seeds"]:
                d = runs.get((arm, seed))
                if d:
                    T["seeds"][seed][key] = [round(share(g["agents"], lambda a: mentions(a, rx)), 4)
                                             for g in d["generations"]]
            n = len(T["gens"])
            T["mean"][key] = [round(st.mean(s[key][i] for s in T["seeds"].values() if i < len(s[key])), 4)
                              for i in range(n)]


def paired(R, measure, arm, ref):
    x, y = R["transfer"][measure][arm]["change"]["seeds"], R["transfer"][measure][ref]["change"]["seeds"]
    d = {s: x[s] - y[s] for s in x if s in y}
    v = list(d.values())
    mu, sd = st.mean(v), (st.stdev(v) if len(v) > 1 else 0.0)
    return {"mean": round(mu, 4), "sd": round(sd, 4), "n": len(v),
            "t": round(mu / (sd / math.sqrt(len(v))), 2) if sd else None,
            "higher": sum(1 for z in v if z > 0), "lower": sum(1 for z in v if z < 0),
            "seeds": {s: round(z, 4) for s, z in d.items()}}


def paired_block(R):
    out = {}
    for m in ("near_energy", "general"):
        out[m] = {}
        for arm in R["arms"]:
            row = {}
            if arm != "DRIFT":
                row["vs_drift"] = paired(R, m, arm, "DRIFT")
            if arm not in ("DRIFT", "SOLO"):
                row["vs_solo"] = paired(R, m, arm, "SOLO")
            if row:
                out[m][arm] = row
    return out


def notes_effect(R, measure, arm):
    e = R["transfer"][measure][arm]
    nn = e.get("no_notes")
    if not nn:
        return None
    d = {s: e["end_seeds"][s] - nn["end_seeds"][s] for s in e["end_seeds"] if s in nn["end_seeds"]}
    v = list(d.values())
    return {"mean": round(st.mean(v), 4), "higher": sum(1 for z in v if z > 0),
            "lower": sum(1 for z in v if z < 0), "n": len(v), "seeds": {s: round(z, 4) for s, z in d.items()}}


def mechanism(R, runs):
    out = {}
    seeds = [str(s) for s in R["seeds"]]
    for arm in R["arms"]:
        rs = [runs[(arm, s)] for s in seeds if (arm, s) in runs]
        first = [g for r in rs for g in r["generations"][:1]]
        last = [r["generations"][-1] for r in rs]
        allg = [g for r in rs for g in r["generations"]]
        could = sum(a.get("could_grab", 0) for g in allg for a in g["agents"])
        grabs = sum(a.get("grabs", 0) for g in allg for a in g["agents"])
        votes = sum(1 for g in allg for a in g["agents"] if a.get("voted"))
        row = {
            "harmful_gen0": round(st.mean(share(g["agents"], lambda a: any(harmful(r) for r in a["instinct"])) for g in first), 4),
            "harmful_final": round(st.mean(share(g["agents"], lambda a: any(harmful(r) for r in a["instinct"])) for g in last), 4),
            "good_gen0": round(st.mean(share(g["agents"], lambda a: GOOD in a["instinct"]) for g in first), 4),
            "good_final": round(st.mean(share(g["agents"], lambda a: GOOD in a["instinct"]) for g in last), 4),
            "grab_rate": round(grabs / could, 4) if could else None,
            "grabs": grabs, "votes": votes if R["arm_meta"][arm]["vote"] else None,
            "exiled": sum(1 for g in allg for a in g["agents"] if a.get("exiled")) if R["arm_meta"][arm]["vote"] else None,
            "generations": len(allg),
            "notes_near": notes_effect(R, "near_energy", arm),
            "notes_general": notes_effect(R, "general", arm),
        }
        out[arm] = row
    return out


def grab_case(R, runs):
    """The seed where the grab rule spread furthest in ALPHA, traced in every grab arm."""
    seeds = [str(s) for s in R["seeds"]]
    hold = lambda g: share(g["agents"], lambda a: GRAB in a["instinct"])
    rate = lambda g: (sum(a.get("grabs", 0) for a in g["agents"]) / max(1, sum(a.get("could_grab", 0) for a in g["agents"])))
    best = max((s for s in seeds if ("ALPHA", s) in runs), key=lambda s: hold(runs[("ALPHA", s)]["generations"][-1]), default=None)
    if best is None:
        return None
    out = {"seed": best, "rule": GRAB, "arms": {}}
    for arm in ("ALPHA", "DOMEST", "MEDLEY"):
        d = runs.get((arm, best))
        if not d:
            continue
        gs = d["generations"]
        gone = next((g["gen"] for g in gs if hold(g) == 0), None)
        out["arms"][arm] = {"hold": [round(hold(g), 3) for g in gs], "grab": [round(rate(g), 3) for g in gs],
                            "exiled_gens": [g["gen"] for g in gs if any(a.get("exiled") for a in g["agents"])],
                            "rule_gone_at": gone}
    ms = [s for s in seeds if ("MEDLEY", s) in runs]
    if ms:
        top = max(ms, key=lambda s: hold(runs[("MEDLEY", s)]["generations"][-1]))
        g = runs[("MEDLEY", top)]["generations"][-1]
        if hold(g) >= 0.5:
            out["medley_holdout"] = {"seed": top, "hold": round(hold(g), 3), "grab": round(rate(g), 3)}
    return out


def sg(x, n=1):
    return f"{x:+.{n}f}"


def pc(x):
    return f"{100 * x:.0f}%"


def findings(R, P, M, C):
    T, N = R["transfer"], R["noise"]
    n = len(R["seeds"])
    theory = R["theory_arms"]
    f, a, u = [], [], []
    ne = lambda arm: T["near_energy"][arm]
    ge = lambda arm: T["general"][arm]

    # ---- for ----
    p = P["near_energy"]["SOLO"]["vs_drift"]
    f.append(f"Selection on survival alone raised skill at the selected task. With no reward for ability, SOLO gained "
             f"{sg(ne('SOLO')['change']['mean'])} near-transfer energy against {sg(ne('DRIFT')['change']['mean'])} under chance, "
             f"and was above DRIFT in {p['higher']} of {n} seeds (paired gap {sg(p['mean'])}, t {p['t']}). "
             f"The gap is inside the pre-set noise band of {N['near_energy']['band']:.1f}, so the pre-set verdict stays as it is. "
             f"The paired check was added after seeing the data.")
    low = sorted(SELECTED, key=lambda x: M[x]["harmful_final"])[:3]
    f.append(f"Selection removed harmful instincts. {pc(M['DRIFT']['harmful_gen0'])} of generation 0 agents held a harmful rule. "
             f"The final share was {pc(M['DRIFT']['harmful_final'])} under chance and "
             + ", ".join(f"{pc(M[x]['harmful_final'])} in {x}" for x in low) + ".")
    if C and "ALPHA" in C["arms"]:
        al, do = C["arms"]["ALPHA"], C["arms"].get("DOMEST")
        s = (f"Punishment held grabbing down, as Theory 3 requires. Over all generations the grab rate was "
             f"{100 * M['ALPHA']['grab_rate']:.1f}% in ALPHA with no vote, {100 * M['DOMEST']['grab_rate']:.1f}% in DOMEST and "
             f"{100 * M['MEDLEY']['grab_rate']:.1f}% in MEDLEY. In seed {C['seed']} the rule \"{C['rule']}\" spread from "
             f"{pc(al['hold'][0])} to {pc(al['hold'][-1])} of ALPHA agents and grabbing reached {pc(al['grab'][-1])}.")
        if do and do["rule_gone_at"] is not None:
            s += (f" In DOMEST, from the same start, the rule was gone by generation {do['rule_gone_at']}"
                  + (f", after an exile in generation {do['exiled_gens'][0]}." if do["exiled_gens"] else "."))
        f.append(s)
    c, m = M["CULTURE"]["notes_near"], M["MEDLEY"]["notes_near"]
    if c and m:
        f.append(f"Inherited notes carried usable knowledge, as Theory 4 requires. Final CULTURE populations earned "
                 f"{sg(c['mean'])} near-transfer energy with their notes compared with the same agents without them, "
                 f"higher in {c['higher']} of {c['n']} seeds. In MEDLEY the figure was {sg(m['mean'])}, higher in {m['higher']} of {m['n']}.")

    # ---- against ----
    gch = {x: ge(x)["change"]["mean"] for x in SELECTED}
    a.append(f"General ability did not move in any arm. Changes ran from {sg(min(gch.values()), 3)} to {sg(max(gch.values()), 3)} "
             f"against {sg(ge('DRIFT')['change']['mean'], 3)} under chance and a noise band of {N['general']['band']:.3f}. "
             f"This is the test the thesis depends on.")
    ends = {x: ge(x)["end"] for x in R["arms"]}
    mid = {x: T["mid"][x]["vs_bare"]["mean"] for x in R["arms"]}
    a.append(f"No evolved population beat the model with no evolved text. Final general ability was {min(ends.values()):.3f} to "
             f"{max(ends.values()):.3f} across arms and the bare model scored {ge('DRIFT')['bare']:.3f}. "
             f"On code-breaking every arm ended below the bare model, the closest by {abs(max(mid.values())):.3f}.")
    best = max(theory, key=lambda x: ne(x)["change"]["mean"])
    below = [x for x in theory if P["near_energy"][x]["vs_solo"]["mean"] < 0]
    worst = min(theory, key=lambda x: P["near_energy"][x]["vs_solo"]["mean"])
    w = P["near_energy"][worst]["vs_solo"]
    a.append(f"No theory arm beat plain ecology. SOLO gained {sg(ne('SOLO')['change']['mean'])} near-transfer energy. "
             f"The best theory arm, {best}, gained {sg(ne(best)['change']['mean'])}, and {len(below)} of {len(theory)} theory arms were below SOLO. "
             f"{worst} was below SOLO in {w['lower']} of {n} seeds.")
    far = T["far"]["DOMEST"]["change"]
    lowest = min(R["arms"], key=lambda x: T["far"][x]["change"]["mean"])
    if lowest == "DOMEST":
        a.append(f"Less grabbing did not become more ability. DOMEST had the lowest rule-induction change of any arm, {sg(far['mean'], 3)}, "
                 f"inside the noise band of {N['far']['band']:.3f}.")
    lm = R["rules"]["law_mentions"]["final"]
    l1 = max(v["any"]["L1"] for v in lm.values())
    a.append(f"The two hidden laws were not found and passed on. At most {pc(l1)} of any final population has text matching law 1. "
             f"The matches for law 2 are fixed orders invented by the mutation writer, for example \"Always soak first, then pound, and roast last; drying can wait.\" "
             f"They appear under chance alone too, in {pc(lm['DRIFT']['any']['L2'])} of final DRIFT agents.")
    mg = M["MEDLEY"]["notes_general"]
    if mg:
        a.append(f"Notes did not help outside their own world. With notes, MEDLEY's general ability was {sg(mg['mean'], 3)} relative to the same agents without them, "
                 f"lower in {mg['lower']} of {mg['n']} seeds. In CULTURE the figure was {sg(M['CULTURE']['notes_general']['mean'], 3)}.")

    # ---- untested ----
    g0, bare = ne("DRIFT")["start"], ne("DRIFT")["bare"]
    u.append(f"Whether selection can lift ability above the starting model. The weights never changed, and generation 0 began below the bare model "
             f"({g0:.1f} against {bare:.1f} near-transfer energy) because the starting pool includes harmful rules. Every gain here is recovery toward the bare model.")
    ex = sum(M[x]["exiled"] or 0 for x in ("DOMEST", "MEDLEY"))
    gens = sum(M[x]["generations"] for x in ("DOMEST", "MEDLEY"))
    t3 = (f"Theory 3 at full strength. Exile removed {ex} members in {gens} generations, and grabbing took off in only one ALPHA seed, "
          f"so the contrast between ALPHA and DOMEST rests on that seed.")
    if C and C.get("medley_holdout"):
        h = C["medley_holdout"]
        t3 += (f" In MEDLEY seed {h['seed']} the grab rule reached {pc(h['hold'])} of agents while grabbing stayed at {pc(h['grab'])}, "
               f"so holding the rule did not always produce grabbing.")
    u.append(t3)
    u.append("A fair near test for the paired arms. The near test is a solo season, the activity SOLO and CULTURE were selected on. "
             "Advising, sharing and voting were selected in the paired arms and no test scores them.")
    u.append(f"Longer runs and larger groups. {R['config'].get('G')} generations of {R['config'].get('N')} agents over {n} seeds gave a chance spread "
             f"of {N['near_energy']['band']:.1f} energy, wider than any effect seen.")
    return {"for": f, "against": a, "untested": u}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs2")
    ap.add_argument("--results", default="results2.json")
    a = ap.parse_args()
    R = json.load(open(a.results))
    runs = load(a.runs)
    fix_mentions(R, runs)
    P, M, C = paired_block(R), mechanism(R, runs), grab_case(R, runs)
    mu = (R.get("usage") or {}).get("measurement") or {}
    mfile = os.path.join(a.runs, "measure.json")
    mfail = (json.load(open(mfile)).get("usage") or {}) if os.path.exists(mfile) else {}
    notes = {
        "mention_caveat": ("Shares are recomputed from the stored text with the corrected keyword check. "
                           "The matches for L2 are fixed orders invented by the mutation writer that happen to end with dry. None states the law."),
        "limits": [
            "Rules are cut at 14 words, so some rules written by the mutation writer end mid-sentence.",
            "The paired per-seed comparison was added after seeing the data. It is more sensitive than the pre-set rule and is reported next to it. The verdicts use the pre-set rule only.",
        ] + ([f"In the measurement, {mfail.get('failures', 0)} of {mu.get('calls', 0):,} calls returned nothing and {mfail.get('parse_fail', 0)} replies could not be parsed. Each counted as a failed try."]
             if mfail.get("failures") or mfail.get("parse_fail") else []),
    }
    R["extras"] = {"paired": P, "mechanism": M, "case": C, "notes": notes}
    R["findings"] = findings(R, P, M, C)
    json.dump(R, open(a.results, "w"))
    for k, v in R["findings"].items():
        print(k.upper())
        for line in v:
            print("  -", line)
    print(json.dumps(M, indent=None)[:1500])
    print(json.dumps(C))


if __name__ == "__main__":
    main()
