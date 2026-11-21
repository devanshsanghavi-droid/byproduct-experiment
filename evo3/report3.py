"""Build the version 3 checks page from the fetched Modal results.

    python3 -m evo3.report3            # reads runs3/modal/*.json, writes checks3.json and report3.html

Two setups are reported: the 1.5B model on a GPU (main) and the 0.5B model on CPU (first attempt).
"""
import glob
import html
import json
import math
import os

THESIS = ("All three surviving theories say intelligence was a byproduct. A faithful simulation therefore scores "
          "agents only on staying alive and being copied, then measures general ability separately on held-out tests.")
# chance (random replies) and correct play, measured with the mock policy on the same test worlds
REF = {"gpu": {"near": (-1.42, 85.04), "near_plants": (0.079, 0.621), "near_tracks": (0.156, 1.0), "near_trade": (0.359, 1.0)},
       "cpu": {"near": (5.83, 93.88), "near_plants": (0.249, 0.802), "near_tracks": (0.174, 1.0), "near_trade": (0.339, 1.0)}}
HELD = {"code": 0.097, "rule": 0.497, "order": 0.25}
COLS = ["", "Near energy", "Plants", "Tracks", "Trade", "Code-breaking", "Rule induction", "Ordering", "General ability",
        "Retention"]
KEYS = [("near", 1), ("near_plants", 2), ("near_tracks", 2), ("near_trade", 2), ("code", 3), ("rule", 3), ("order", 3),
        ("general", 3), ("retention", 2)]


def load(name):
    p = f"runs3/modal/{name}.json"
    return json.load(open(p)) if os.path.exists(p) else None


def f(x, n=2, sign=False):
    if x is None:
        return "n/a"
    return f"{x:+.{n}f}" if sign else f"{x:.{n}f}"


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def se(xs):
    if len(xs) < 2:
        return None
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) / math.sqrt(len(xs))


def table(head, rows):
    h = "".join(f"<th>{c}</th>" for c in head)
    b = "".join("<tr" + (' class="ref"' if r and str(r[0]).startswith("~") else "") + ">"
                + "".join(f"<td>{str(c).lstrip('~')}</td>" for c in r) + "</tr>" for r in rows)
    return f'<div class="scroll"><table><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table></div>'


def srow(label, s, ref=False):
    return [("~" if ref else "") + label] + [f(s.get(k), n) for k, n in KEYS]


def ref_rows(kind):
    r = REF[kind]
    gen = sum(HELD.values()) / 3
    chance = ["~Chance (random replies)"] + [f(r[k][0], n) for k, n in KEYS[:4]] + [f(HELD["code"], 3), f(HELD["rule"], 3),
                                                                                  f(HELD["order"], 3), f(gen, 3), "0.00"]
    correct = ["~Correct play"] + [f(r[k][1], n) for k, n in KEYS[:4]] + ["n/a"] * 5
    return chance, correct


def chart(title, runs, chance, lo=-20, hi=60):
    """Mean energy per generation, DRIFT against SOLO, same worlds for both."""
    W, H, L, R, T, B = 340, 200, 34, 58, 12, 26
    G = max((len(r["generations"]) for r in runs.values()), default=8) or 8
    x = lambda g: L + (W - L - R) * (g / max(1, G - 1))
    y = lambda v: T + (H - T - B) * (1 - (max(lo, min(hi, v)) - lo) / (hi - lo))
    out = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{title}">']
    for v in range(lo, hi + 1, 20):
        out.append(f'<line x1="{L}" x2="{W - R}" y1="{y(v):.1f}" y2="{y(v):.1f}" class="grid"/>'
                   f'<text x="{L - 6}" y="{y(v) + 3.5:.1f}" class="tick" text-anchor="end">{v}</text>')
    out.append(f'<line x1="{L}" x2="{W - R}" y1="{y(chance):.1f}" y2="{y(chance):.1f}" class="chance"/>'
               f'<text x="{W - R + 4}" y="{y(chance) + 3.5:.1f}" class="tick">chance</text>')
    for g in range(G):
        out.append(f'<text x="{x(g):.1f}" y="{H - 8}" class="tick" text-anchor="middle">{g + 1}</text>')
    ends = []
    for arm in ("DRIFT", "SOLO"):
        r = runs.get(arm)
        if not r or not r["generations"]:
            continue
        pts = [(x(g["g"]), y(g["energy"])) for g in r["generations"]]
        out.append(f'<polyline class="line {arm.lower()}" points="' + " ".join(f"{a:.1f},{b:.1f}" for a, b in pts) + '"/>')
        out += [f'<circle class="dot {arm.lower()}" cx="{a:.1f}" cy="{b:.1f}" r="2.6"/>' for a, b in pts]
        ends.append([arm, pts[-1][0], pts[-1][1]])
    if len(ends) == 2 and abs(ends[0][2] - ends[1][2]) < 12:   # keep the two end labels apart
        up, down = sorted(ends, key=lambda e: e[2])
        mid = (up[2] + down[2]) / 2
        up[2], down[2] = mid - 6, mid + 6
    for arm, a, b in ends:
        out.append(f'<text x="{a + 7:.1f}" y="{b + 3.5:.1f}" class="lab {arm.lower()}">{arm}</text>')
    return f"<figure><figcaption>{title}</figcaption>" + "".join(out) + "</svg></figure>"


def pilot(pattern, base):
    """Per-seed final scores and paired gaps for one setup."""
    runs = {}
    for p in glob.glob(pattern):
        d = json.load(open(p))
        runs.setdefault(d["seed"], {})[d["arm"]] = d
    seeds = [s for s in sorted(runs) if all(a in runs[s] and runs[s][a].get("tests", {}).get("final") for a in ("DRIFT", "SOLO"))]
    P = {"runs": runs, "seeds": seeds, "rows": []}
    for s in seeds:
        fin = {a: runs[s][a]["tests"]["final"]["scores"] for a in ("DRIFT", "SOLO")}
        en = {a: [g["energy"] for g in runs[s][a]["generations"]] for a in ("DRIFT", "SOLO")}
        late = [a - b for a, b in zip(en["SOLO"], en["DRIFT"])][1:]
        P["rows"].append({"seed": s, "final": fin, "in_run": mean(late), "in_run_pos": sum(d > 0 for d in late),
                          "in_run_n": len(late)})
    for k in ("near", "general", "retention"):
        gaps = [r["final"]["SOLO"][k] - r["final"]["DRIFT"][k] for r in P["rows"]]
        P[k] = {"gaps": gaps, "mean": mean(gaps), "se": se(gaps), "pos": sum(g > 0 for g in gaps),
                "solo": mean([r["final"]["SOLO"][k] for r in P["rows"]]),
                "drift": mean([r["final"]["DRIFT"][k] for r in P["rows"]]), "base": (base or {}).get(k)}
    P["done"] = len(seeds) > 0 and len(seeds) == len(runs)
    P["pass"] = P["done"] and P["near"]["pos"] == len(seeds)
    return P


def pilot_tables(P, base, kind):
    chance, correct = ref_rows(kind)
    rows = [chance, srow("Unmodified model", base or {}, True)]
    for arm in ("DRIFT", "SOLO"):
        m = {k: mean([r["final"][arm].get(k) for r in P["rows"]]) for k, _ in KEYS}
        rows.append(srow(f"{arm}, mean of {len(P['seeds'])} seeds" + (" (noise floor)" if arm == "DRIFT" else ""), m, arm == "DRIFT"))
    summary = table(COLS, rows + [correct])
    per = table(["Seed", "Near: DRIFT", "Near: SOLO", "Gap", "General: DRIFT", "General: SOLO", "Gap",
                 "In-run energy gap, generations 2 on", "Generations SOLO led"],
                [[r["seed"], f(r["final"]["DRIFT"]["near"], 1), f(r["final"]["SOLO"]["near"], 1),
                  f(r["final"]["SOLO"]["near"] - r["final"]["DRIFT"]["near"], 1, True),
                  f(r["final"]["DRIFT"]["general"], 3), f(r["final"]["SOLO"]["general"], 3),
                  f(r["final"]["SOLO"]["general"] - r["final"]["DRIFT"]["general"], 3, True),
                  f(r["in_run"], 1, True), f"{r['in_run_pos']} of {r['in_run_n']}"] for r in P["rows"]]
                + [["~Mean", f(P["near"]["drift"], 1), f(P["near"]["solo"], 1),
                    f"{f(P['near']['mean'], 1, True)} (SE {f(P['near']['se'], 1)})",
                    f(P["general"]["drift"], 3), f(P["general"]["solo"], 3),
                    f"{f(P['general']['mean'], 3, True)} (SE {f(P['general']['se'], 3)})",
                    f(mean([r["in_run"] for r in P["rows"]]), 1, True), ""]])
    return summary, per


def paired(a, b):
    """Mean and standard error of item-by-item differences b - a."""
    d = [y - x for x, y in zip(a, b)]
    return mean(d), se(d)


def general_gap(a, b):
    """Change in general ability (mean of the three tests) between two item sets, with its standard error."""
    parts = [paired(a[k], b[k]) for k in ("code", "rule", "order")]
    return mean([m for m, _ in parts]), math.sqrt(sum((e or 0) ** 2 for _, e in parts)) / 3


def wide_refs():
    """Chance and correct play on the wider world's own test items (no model needed)."""
    import asyncio
    import random
    from evo3 import tests3, wide, world3 as W
    from evo3.local import LocalLLM
    mock = LocalLLM(mock=True)

    async def go():
        ch = (await tests3.battery(mock, W.WIDE))["scores"]
        lives = [await W.life(None, W.new_world(random.Random(90000 + i), W.WIDE), 90000 + i, oracle=True) for i in range(24)]
        co = {"near": mean([x["energy"] for x in lives]), "mid": 1.0, "near_wide": 1.0}
        for fam in W.families(W.WIDE):
            co["near_" + fam] = sum(x[fam]["wins"] for x in lives) / sum(x[fam]["rounds"] for x in lives)
        big = await mock.run([wide.heldout_life(mock, 95000 + i) for i in range(600)])
        for fam in wide.HELD:   # chance per held-out family from a larger sample
            ch["mid_" + fam] = sum(x[fam]["wins"] for x in big) / sum(x[fam]["rounds"] for x in big)
        ch["mid"] = mean([x["rate"] for x in big])
        return ch, co, list(wide.TRAIN), list(wide.HELD)
    return asyncio.run(go())


def wide_section():
    """Ceiling check on the wider world. Returns (html, summary) or ("", None) when the results are not there."""
    wj, cj = load("gpu-ceiling-wide"), load("gpu-ceiling-narrow-ctl")
    if not wj or not cj:
        return "", None
    ch, co, new, held = wide_refs()
    ws = {s["iters"]: s for s in wj["stages"]}
    cs = {s["iters"]: s for s in cj["stages"]}
    last = max(ws)
    base, mid_w, end_w, end_c = ws[0], ws.get(400), ws[last], cs[max(cs)]
    B, Wd, C = base["items"], end_w["items"], end_c["items"]
    g = {"mid_wide": paired(B["mid"], Wd["mid"]), "mid_ctl": paired(B["mid"], C["mid"]), "mid_breadth": paired(C["mid"], Wd["mid"]),
         "gen_wide": general_gap(B, Wd), "gen_ctl": general_gap(B, C), "gen_breadth": general_gap(C, Wd),
         "ret_wide": paired(B["retention"], Wd["retention"]), "ret_ctl": paired(B["retention"], C["retention"])}
    z = lambda k: (g[k][0] / g[k][1]) if g[k][1] else 0.0
    if z("gen_wide") > 2 and z("gen_breadth") > 2:
        outcome = "Pass"
    elif z("mid_wide") > 2 and z("mid_breadth") > 2:
        outcome = "Partial"
    else:
        outcome = "Fail"
    cols = ["", "Near energy (wider world)", "13 new families", "Held-out families", "Code-breaking", "Rule induction", "Ordering",
            "General ability", "Retention"]
    keys = [("near", 1), ("near_wide", 3), ("mid", 3), ("code", 3), ("rule", 3), ("order", 3), ("general", 3), ("retention", 2)]
    row = lambda label, sc, ref=False: [("~" if ref else "") + label] + [f(sc.get(k), n) for k, n in keys]
    chance = dict(ch, code=HELD["code"], rule=HELD["rule"], order=HELD["order"], retention=0.0)
    chance["general"] = (HELD["code"] + HELD["rule"] + HELD["order"]) / 3
    rows = [row("Chance (random replies)", chance, True), row("Unmodified 1.5B model (baseline)", base["scores"], True),
            row(f"Control: {max(cs):,} steps on correct play, 3-family world", end_c["scores"])]
    if mid_w:
        rows.append(row("Wider world: 400 steps on correct play", mid_w["scores"]))
    rows += [row(f"Wider world: {last:,} steps on correct play", end_w["scores"]), row("Correct play", co, True)]
    main_t = table(cols, rows)
    gap_t = table(["Change, paired item by item", "Held-out families", "SE", "General ability", "SE", "Retention", "SE"], [
        ["Wider world minus unmodified", f(g["mid_wide"][0], 3, True), f(g["mid_wide"][1], 3), f(g["gen_wide"][0], 3, True),
         f(g["gen_wide"][1], 3), f(g["ret_wide"][0], 3, True), f(g["ret_wide"][1], 3)],
        ["Control minus unmodified (noise floor for breadth)", f(g["mid_ctl"][0], 3, True), f(g["mid_ctl"][1], 3),
         f(g["gen_ctl"][0], 3, True), f(g["gen_ctl"][1], 3), f(g["ret_ctl"][0], 3, True), f(g["ret_ctl"][1], 3)],
        ["Wider world minus control (effect of breadth)", f(g["mid_breadth"][0], 3, True), f(g["mid_breadth"][1], 3),
         f(g["gen_breadth"][0], 3, True), f(g["gen_breadth"][1], 3), "", ""]])
    fam_rows = []
    for fam in ["plants", "tracks", "trade"] + new:
        k = "near_" + fam
        fam_rows.append([fam + ("" if fam in new else " (original)"), "in lives", f(ch.get(k)), f(base["scores"].get(k)),
                         f(end_c["scores"].get(k)), f(end_w["scores"].get(k))])
    for fam in held:
        k = "mid_" + fam
        fam_rows.append([fam, "held out", f(ch.get(k)), f(base["scores"].get(k)), f(end_c["scores"].get(k)), f(end_w["scores"].get(k))])
    fam_t = table(["Family", "Role", "Chance", "Unmodified", "Control (3-family training)", "Wider-world training"], fam_rows)

    def guesses(stage):
        out = []
        for e in (stage.get("replies") or {}).get("code", []):
            if e["user"].startswith("Guess 1 ") and out:
                break
            out.append(html.escape(e["out"].strip()[:90]))
        return out[:8]
    g0, g1 = guesses(base), guesses(end_w)
    rep_t = table(["Guess", "Unmodified model", "After wider-world training"],
                  [[str(i + 1), g0[i] if i < len(g0) else "", g1[i] if i < len(g1) else ""] for i in range(max(len(g0), len(g1)))])
    rest = [h for h in held if h != "reverse"]
    no_rev = {n: mean([sc["mid_" + h] for h in rest]) for n, sc in
              (("base", base["scores"]), ("control", end_c["scores"]), ("wide", end_w["scores"]))}
    rev = {n: sc["mid_reverse"] for n, sc in (("base", base["scores"]), ("control", end_c["scores"]), ("wide", end_w["scores"]))}
    conc_t = table(["Held-out families", "Unmodified", "Control", "Wider-world training", "Wider minus control"], [
        ["All 6", f(base["scores"]["mid"], 3), f(end_c["scores"]["mid"], 3), f(end_w["scores"]["mid"], 3),
         f(end_w["scores"]["mid"] - end_c["scores"]["mid"], 3, True)],
        ["reverse only", f(rev["base"], 3), f(rev["control"], 3), f(rev["wide"], 3), f(rev["wide"] - rev["control"], 3, True)],
        ["The other 5", f(no_rev["base"], 3), f(no_rev["control"], 3), f(no_rev["wide"], 3), f(no_rev["wide"] - no_rev["control"], 3, True)]])
    nw = "near_wide"
    summary = {"outcome": outcome, "steps": last, "mid_without_reverse": {k: round(v, 4) for k, v in no_rev.items()}, "examples": wj.get("examples"), "base": base["scores"], "wide": end_w["scores"],
               "control": end_c["scores"], "chance": {k: round(v, 4) for k, v in chance.items()},
               "correct_near": co["near"], "gaps": {k: [round(v[0], 4), round(v[1], 4) if v[1] else None] for k, v in g.items()},
               "usd": round(wj.get("usd", 0) + cj.get("usd", 0), 2)}
    sent = {"Pass": "General ability rose above both the unmodified model and the control.",
            "Partial": "Families never trained on improved beyond the control, but the three original tests did not.",
            "Fail": "Neither the held-out families nor the three original tests rose above the control by more than 2 standard errors."}[outcome]
    page = f"""<section><h2>Check 2 repeated on a wider world <span class="badge {outcome.lower()}">{outcome}</span></h2>
<p>The world was widened from 3 task families to 16, and 6 more families were kept out of every life and every training set.
The adapter was then trained on {wj.get('examples'):,} examples of correct play. A control adapter got the same number of steps and
examples from the original 3-family world. {sent}</p>
{main_t}
<p class="muted">Temperature 0, same items for every row. "13 new families" is the win rate on the families added to lives; for the
control row those families were never trained on. General ability is the mean of code-breaking, rule induction and ordering.</p>
{gap_t}
<p class="muted">SE is the standard error of the paired difference across test items. Rules written before the results: pass if general
ability ends more than 2 SE above both the unmodified model and the control; partial if only the held-out families do.</p>
<h3>Where the held-out gain sits</h3>{conc_t}
<p class="muted">Most of the gain over the unmodified model comes from one family (reverse), and it appears in the control as well.
On the other five families the wider-world adapter ends {f(no_rev["wide"] - no_rev["base"], 3, True)} against the unmodified model and
{f(no_rev["wide"] - no_rev["control"], 3, True)} against the control; no standard error is available for this split. The control's own
transfer inside the world was small: on the 13 families it never trained on, it moved from {f(base["scores"][nw], 3)} to
{f(end_c["scores"][nw], 3)}. Each arm is one training run, so the SE above covers test items only, not run-to-run variation.</p>
<h3>Win rate by family</h3>{fam_t}
<h3>First code-breaking item: the model's replies</h3>{rep_t}</section>"""
    return page, summary


def build():
    led = json.load(open("runs3/modal/ledger.json")) if os.path.exists("runs3/modal/ledger.json") else {}
    spent = sum(v["usd"] for v in led.values())
    gpu_spent = sum(v["usd"] for k, v in led.items() if k.split(":")[1].startswith("gpu-"))
    # ---- 1.5B on GPU ------------------------------------------------------------------
    gspeed, ghead, gceil = load("gpu-speed-15b"), load("gpu-headroom-15b"), load("gpu-ceiling-15b")
    gst = {s["iters"]: s["scores"] for s in (gceil or {}).get("stages", [])}
    gbase = gst.get(0, {})
    GP = pilot("runs3/modal/gpu-pilot-*.json", gbase)
    # ---- 0.5B on CPU ------------------------------------------------------------------
    cspeed, chead, cbase = load("speed-05b"), load("headroom-05b"), (load("base-05b-2") or {}).get("scores", {})
    cceil = (load("ceiling-retest") or {}).get("scores", {})
    CP = pilot("runs3/modal/pilot-*.json", cbase)

    R = {"thesis": THESIS, "spent": round(spent, 2), "gpu_spent": round(gpu_spent, 2),
         "gpu": {"base": gbase, "ceiling": gst.get(400, {}), "pilot": {k: GP[k] for k in ("seeds", "rows", "near", "general", "retention", "done", "pass")}},
         "cpu": {"base": cbase, "ceiling": cceil, "pilot": {k: CP[k] for k in ("seeds", "rows", "near", "general", "retention", "done", "pass")}}}
    wide_html, R["wide"] = wide_section()
    wide_lede = "By the rule set in advance, the full 7-arm run should not go ahead on this setup."
    if R["wide"]:
        gw = R["wide"]["gaps"]
        wide_lede = (f"On a world widened to 16 families the result is {R['wide']['outcome'].lower()}: families never trained on rose by "
                     f"{f(gw['mid_breadth'][0], 3, True)} (SE {f(gw['mid_breadth'][1], 3)}) over a control trained on the 3-family world, "
                     f"while the three original tests show no effect of breadth ({f(gw['gen_breadth'][0], 3, True)}, SE "
                     f"{f(gw['gen_breadth'][1], 3)}). Direct training is the upper bound for selection, so the full 7-arm run is still "
                     "not justified at this model size.")
    json.dump(R, open("checks3.json", "w"), indent=1)

    gs = (gspeed or {}).get("models", [{}])[0]
    cs = (cspeed or {}).get("models", [{}])[0]
    gc = gst.get(400, {})
    n = len(GP["seeds"])
    badge = lambda s: f'<span class="badge {s.split(",")[0].lower()}">{s}</span>'
    g3 = (f"SOLO above DRIFT in {GP['near']['pos']} of {n} seeds; mean gap {f(GP['near']['mean'], 1, True)} "
          f"(SE {f(GP['near']['se'], 1)})") if GP["done"] else "still running"
    if not GP["done"]:
        g3_badge = "Running"
    elif GP["pass"]:
        g3_badge = "Pass"
    else:   # the every-seed rule was missed; say so, and separate a near miss from a null
        near_miss = GP["near"]["pos"] >= n - 1 and (GP["near"]["se"] or 0) > 0 and GP["near"]["mean"] > 2 * GP["near"]["se"]
        g3_badge = "Partial" if near_miss else "Fail"
        g3 += ". The every-seed rule was not met"
    checks_t = table(["Check", "Pass rule (set in advance)", "1.5B model on GPU", "", "0.5B model on CPU", ""], [
        ["0. Speed", "gives the real run time",
         f"{f(gs.get('calls_per_s'), 1)} calls per second, {f(gs.get('train_s_per_iter'))} s per training step", badge("Done"),
         f"{f(cs.get('calls_per_s'))} calls per second, {f(cs.get('train_s_per_iter'), 1)} s per training step", badge("Done")],
        ["1. Headroom", "every task and test above chance, below 85% of maximum",
         "world: all 3 families above chance. Held-out: code-breaking above chance, rule induction and ordering at chance",
         badge("Partial"), "at chance on every task and test", badge("Fail")],
        ["2. Ceiling", "direct training on the world moves held-out scores",
         f"near energy {f(gbase.get('near'), 1)} to {f(gc.get('near'), 1)}; general ability {f(gbase.get('general'), 3)} to "
         f"{f(gc.get('general'), 3)}", badge("Fail"),
         f"near energy {f(cbase.get('near'), 1)} to {f(cceil.get('near'), 1)}; general ability {f(cbase.get('general'), 3)} to "
         f"{f(cceil.get('general'), 3)}", badge("Fail")],
        ["3. Pilot", "SOLO above DRIFT on near energy in every seed", g3,
         badge(g3_badge),
         f"SOLO above DRIFT in {CP['near']['pos']} of {len(CP['seeds'])} seeds; gaps "
         + ", ".join(f(g, 1, True) for g in CP["near"]["gaps"]), badge("Pass, weak")],
    ])

    why_t = table(["Worker (measured)", "Model", "Calls per second", "Seconds per training step", "Cost per hour", "Cost per 1,000 calls"], [
        ["8 CPU cores", "0.5B", f(cs.get("calls_per_s")), f(cs.get("train_s_per_iter"), 1), "$0.47",
         f"${0.4732 / 3.6 / cs.get('calls_per_s', 1):.3f}" if cs else "n/a"],
        [f"1 GPU ({(ghead or {}).get('model', {}).get('gpu') or 'L4 or A10'})", "1.5B", f(gs.get("calls_per_s"), 1),
         f(gs.get("train_s_per_iter")), "$1.37", f"${1.3702 / 3.6 / gs.get('calls_per_s', 1):.3f}" if gs else "n/a"]])

    def head_table(h, label):
        rows = []
        for r in (h or {}).get("families", []):
            lv = r["level"]
            name = {"plants": f"plants, {lv.get('steps')} step{'s' if lv.get('steps', 1) > 1 else ''} of {lv.get('actions')}",
                    "tracks": f"tracks, ring of {lv.get('ring')}", "trade": f"trade, {lv.get('offers')} offers"}[r["family"]]
            rows.append([name, f(r["random"]), f(r["model_t0.0"]), f(r["correct"]), f"{100 * r['invalid_t0.0']:.0f}%"])
        return table([label, "Chance", "Model", "Correct play", "Unreadable replies"], rows)

    gb = (ghead or {}).get("battery", {}).get("model", {})
    held_t = table(["Held-out test (1.5B model)", "Chance", "Model"],
                   [["code-breaking", f(HELD["code"], 3), f(gb.get("code"), 3)], ["rule induction", f(HELD["rule"], 3), f(gb.get("rule"), 3)],
                    ["ordering", f(HELD["order"], 3), f(gb.get("order"), 3)]])

    gch, gco = ref_rows("gpu")
    gceil_t = table(COLS, [gch, srow("Unmodified 1.5B model", gbase, True), srow("After 150 steps on correct play", gst.get(150, {})),
                           srow("After 400 steps on correct play", gc), gco])
    cch, cco = ref_rows("cpu")
    cceil_t = table(COLS, [cch, srow("Unmodified 0.5B model", cbase, True), srow("After 400 steps on correct play", cceil), cco])

    gsum, gper = pilot_tables(GP, gbase, "gpu") if GP["seeds"] else ("", "")
    csum, cper = pilot_tables(CP, cbase, "cpu") if CP["seeds"] else ("", "")
    gcharts = "".join(chart(f"Seed {s}: mean energy per generation", GP["runs"][s], REF["gpu"]["near"][0]) for s in sorted(GP["runs"]))

    kinds = {}
    for k, v in led.items():
        nm = k.split(":")[1]
        grp = ("GPU, 1.5B: " if nm.startswith("gpu-") else "CPU, 0.5B: ") + (
            "speed" if "speed" in nm else "headroom" if ("headroom" in nm or nm.startswith("base")) else
            "ceiling, wider world" if ("wide" in nm or "narrow" in nm) else "ceiling" if "ceiling" in nm else "pilot")
        kinds[grp] = kinds.get(grp, 0) + v["usd"]
    cost_t = table(["Item", "Cost (own upper estimate)"],
                   [[k, f"${v:.2f}"] for k, v in sorted(kinds.items())]
                   + [["~GPU subtotal", f"${gpu_spent:.2f}"], ["~Total", f"${spent:.2f}"]])

    if GP["done"]:
        verdict = (f"With the 1.5B model, selection on survival alone moved near-task energy by {f(GP['near']['mean'], 1, True)} "
                   f"against chance parents (SOLO {f(GP['near']['solo'], 1)}, DRIFT {f(GP['near']['drift'], 1)}, unmodified "
                   f"{f(gbase.get('near'), 1)}; SOLO ahead in {GP['near']['pos']} of {n} seeds). General ability did not follow: "
                   f"SOLO {f(GP['general']['solo'], 3)}, DRIFT {f(GP['general']['drift'], 3)}, unmodified {f(gbase.get('general'), 3)}.")
    else:
        verdict = "The GPU pilot is still running."
    page = f"""<title>Byproduct Experiment 3</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Space+Grotesk:wght@500;600&display=swap">
<style>
:root {{ --bg:#f6f4ee; --card:#fffdf8; --fg:#1c1b18; --muted:#6a665d; --rule:#dcd7ca; --accent:#1f6f54; --drift:#8a8578;
  --pass:#1f6f54; --fail:#b0432b; --run:#8a6a12; --passbg:#dcefe6; --failbg:#f6ddd6; --runbg:#f3e9c8; }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ --bg:#151513; --card:#1d1c19; --fg:#ebe8e0; --muted:#a09b90;
  --rule:#35332e; --accent:#62c49e; --drift:#8f8a7e; --pass:#7fd3b0; --fail:#f09a84; --run:#e2c46a; --passbg:#173428; --failbg:#3b1d16;
  --runbg:#3a3113; color-scheme: dark }} }}
:root[data-theme="dark"] {{ --bg:#151513; --card:#1d1c19; --fg:#ebe8e0; --muted:#a09b90; --rule:#35332e; --accent:#62c49e;
  --drift:#8f8a7e; --pass:#7fd3b0; --fail:#f09a84; --run:#e2c46a; --passbg:#173428; --failbg:#3b1d16; --runbg:#3a3113; color-scheme: dark }}
body {{ background:var(--bg); color:var(--fg); font:15px/1.55 Inter,system-ui,sans-serif; }}
main {{ max-width:1000px; margin:0 auto; padding-block:28px 56px; padding-left:max(16px, env(safe-area-inset-left, 0px));
  padding-right:max(16px, env(safe-area-inset-right, 0px)); display:flex; flex-direction:column; gap:26px; }}
h1 {{ font:600 clamp(20px,3.2vw,27px)/1.25 "Space Grotesk",Inter,sans-serif; margin:0; text-wrap:balance; }}
h2 {{ font:600 18px/1.3 "Space Grotesk",Inter,sans-serif; margin:0 0 8px; }}
h3 {{ font:600 14px/1.3 Inter,sans-serif; margin:16px 0 6px; }}
p {{ margin:0 0 8px; max-width:68ch; }} .lede {{ font-size:16px; }} .muted,figcaption,small {{ color:var(--muted); }}
.kicker {{ font:600 12px/1.3 Inter,sans-serif; letter-spacing:.08em; text-transform:uppercase; color:var(--accent); margin-bottom:10px; }}
.tiles {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr)); gap:12px; }}
.tile {{ background:var(--card); border:1px solid var(--rule); border-radius:10px; padding:12px 14px; min-width:0; }}
.tile b {{ display:block; font:600 22px/1.2 "Space Grotesk",Inter,sans-serif; font-variant-numeric:tabular-nums; }}
.tile span {{ color:var(--muted); font-size:13px; }}
section {{ min-width:0; }} .scroll + p {{ margin-top:10px; }}
.scroll {{ overflow-x:auto; border:1px solid var(--rule); border-radius:10px; background:var(--card); }}
table {{ border-collapse:collapse; width:100%; font-size:13.5px; font-variant-numeric:tabular-nums; }}
th,td {{ padding:8px 10px; text-align:left; border-bottom:1px solid var(--rule); vertical-align:top; }}
th {{ font-weight:600; font-size:12px; color:var(--muted); white-space:nowrap; }} tbody tr:last-child td {{ border-bottom:0; }}
tr.ref td {{ color:var(--muted); }} td:first-child {{ min-width:130px; }}
.badge {{ display:inline-block; padding:2px 9px; border-radius:99px; font-weight:600; font-size:12px; white-space:nowrap; }}
.badge.pass,.badge.done {{ color:var(--pass); background:var(--passbg); }} .badge.fail {{ color:var(--fail); background:var(--failbg); }}
.badge.running,.badge.partial {{ color:var(--run); background:var(--runbg); }}
.charts {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(270px,1fr)); gap:14px; margin-bottom:12px; }}
figure {{ margin:0; background:var(--card); border:1px solid var(--rule); border-radius:10px; padding:10px 10px 4px; min-width:0; }}
figcaption {{ font-size:12.5px; margin-bottom:4px; }} svg {{ width:100%; height:auto; display:block; }}
.grid {{ stroke:var(--rule); stroke-width:1; }} .chance {{ stroke:var(--muted); stroke-width:1; stroke-dasharray:3 3; }}
.tick {{ fill:var(--muted); font:10px Inter,sans-serif; }} .line {{ fill:none; stroke-width:2; }}
.line.solo {{ stroke:var(--accent); }} .line.drift {{ stroke:var(--drift); stroke-dasharray:5 3; }}
.dot.solo {{ fill:var(--accent); }} .dot.drift {{ fill:var(--drift); }}
.lab {{ font:600 10.5px Inter,sans-serif; }} .lab.solo {{ fill:var(--accent); }} .lab.drift {{ fill:var(--drift); }}
ul {{ margin:0; padding-left:20px; max-width:70ch; }} li {{ margin-bottom:5px; }}
</style>
<main>
<header>
  <div class="kicker">Byproduct experiment, version 3: weights evolve. Checks only.</div>
  <h1>{html.escape(THESIS)}</h1>
</header>
<p class="lede">{verdict}</p>
<p>The ceiling check failed for both models on the 3-family world: even training directly on correct answers did not lift the held-out tests. {wide_lede}</p>
<div class="tiles">
  <div class="tile"><b>{f(gbase.get('near'), 1)} to {f(gc.get('near'), 1)}</b><span>1.5B near energy when trained directly on correct play (chance {f(REF['gpu']['near'][0], 1)}, correct play {f(REF['gpu']['near'][1], 1)})</span></div>
  <div class="tile"><b>{f(gbase.get('general'), 3)} to {f(gc.get('general'), 3)}</b><span>1.5B general ability over the same training</span></div>
  <div class="tile"><b>{f(GP['near']['mean'], 1, True) if GP['done'] else 'running'}</b><span>pilot, 1.5B: final near energy, SOLO minus DRIFT, mean of {n} seeds</span></div>
  <div class="tile"><b>${spent:.2f}</b><span>total Modal spend (own upper estimate); GPU part ${gpu_spent:.2f}</span></div>
</div>
{wide_html}
<section><h2>The four checks, both setups</h2>{checks_t}</section>
<section><h2>Why a GPU</h2>
<p>A GPU is not required in principle. It is cheaper per unit of work: these models are large matrix multiplications, which a GPU runs in parallel. Measured on Modal:</p>
{why_t}
<p class="muted">On CPU only the 0.5B model was affordable, and it starts at chance. The 1.5B model ran out of memory on the 12 GB CPU worker. GPU cost is booked at the A10 rate for every second, so it is an upper estimate.</p></section>
<section><h2>Check 1: headroom</h2>
<p>Win rate per decision at temperature 0.</p>
{head_table(ghead, "Task (1.5B model on GPU)")}<p></p>{held_t}
<p class="muted">Rule induction and ordering start at chance for the 1.5B model, so they can show a gain but not a loss.</p>
<h3>0.5B model on CPU, strict reply format</h3>{head_table(chead, "Task (0.5B model on CPU)")}
<p class="muted">Most 0.5B replies were not in the requested format. Reply parsing was then made lenient (a bare answer counts) for every other number on this page.</p></section>
<section><h2>Check 2: ceiling</h2>
<p>The adapter was trained directly on 3,900 examples of correct play. This breaks the thesis rule on purpose: it is the most any selection scheme could achieve in this world.</p>
<h3>1.5B model on GPU</h3>{gceil_t}
<p class="muted">Code-breaking scores were identical item by item before and after training. General ability is the mean of code-breaking, rule induction and ordering.</p>
<h3>0.5B model on CPU (easier world)</h3>{cceil_t}</section>
<section><h2>Check 3: pilot</h2>
<p>Parents are the top quarter by energy (SOLO) or a random quarter (DRIFT). The adapter is then trained on the parents' whole-life transcripts. Both arms of a seed meet the same worlds. The world changes every generation, so a line moves with the world as well as with the weights; compare the two lines with each other.</p>
<h3>1.5B model on GPU: {n} seeds, 8 generations, 32 lives, 100 training steps per generation</h3>
<div class="charts">{gcharts}</div>
<p>Tests on the final weights, temperature 0, same items for every row:</p>{gsum}<p></p>{gper}
<p class="muted">SE is the standard error of the paired gap across seeds. The near-energy gap is {f(GP['near']['mean'] / GP['near']['se'], 1) if GP['near']['se'] else 'n/a'} standard errors from zero; the general-ability gap is {f(abs(GP['general']['mean']) / GP['general']['se'], 1) if GP['general']['se'] else 'n/a'}. General ability rose by about the same amount in both arms, so selection did not cause that rise.</p>
<h3>0.5B model on CPU: 2 seeds, 8 generations, 24 lives, 60 training steps per generation</h3>{csum}<p></p>{cper}</section>
<section><h2>What changed from the written design</h2><ul>
<li>Compute moved from the Mac to Modal: first CPU workers with a $10 limit, then one GPU worker once you approved it.</li>
<li>The 0.5B CPU runs used an easier world (plants need 1 step). The 1.5B GPU runs use the designed world.</li>
<li>Replies are parsed leniently: a bare answer without JSON counts.</li>
<li>After the first ceiling check failed, the world was widened to 16 families and the ceiling check was repeated with a control arm.</li>
<li>Only DRIFT and SOLO were built. The five social arms were not, because the ceiling check failed first.</li>
<li>Retention is a hand-made set of sums, capitals and comparisons, not a public benchmark.</li></ul></section>
<section><h2>Limits</h2><ul>
<li>Models of 0.5B and 1.5B parameters cannot stand in for a frontier model. A null result here does not transfer.</li>
<li>Two of the three held-out tests sit at chance for both models, so general ability rests mostly on code-breaking.</li>
<li>One adapter per population, so heredity is at population level.</li>
<li>The wider-world check is one training run per arm. Its standard errors cover test items only.</li>
<li>The 6 held-out families were written by the same author in the same style as the 13 trained ones, so they measure transfer to nearby tasks.</li>
<li>Costs are my own upper estimate. Modal's bill is the authority.</li></ul></section>
<section><h2>Cost</h2>{cost_t}</section>
</main>"""
    open("report3.html", "w").write(page)
    return R


if __name__ == "__main__":
    R = build()
    print(json.dumps({"spent": R["spent"], "gpu_spent": R["gpu_spent"],
                      "gpu_pilot": {k: R["gpu"]["pilot"][k] for k in ("seeds", "near", "general", "done", "pass")}})[:900])
    print(len(open("report3.html").read()), "bytes")
