"""Results page for the breadth experiment. Reads runs3/breadth_stats.json (made by evo3.breadth_stats).

    PYTHONPATH=. python3 -m evo3.breadth_page   -> breadth.html
"""
import json
import math
import os

S = json.load(open("runs3/breadth_stats.json"))
R = json.load(open("runs3/bbh_rescore.json"))
P = S["post"]
N = json.load(open("runs3/near_stats.json"))      # rerun with the rule fixed first (evo3.near_stats)
LEDGER = sum(v["usd"] for v in json.load(open("runs3/modal/ledger.json")).values())
SPENT, RERUN_SPENT, STRONG_SPENT = 2.66, 2.58, 7.89      # fixed: the ledger now also holds version 4
THESIS = ("All three surviving theories say intelligence was a byproduct. A faithful simulation therefore scores agents "
          "only on staying alive and being copied, then measures general ability separately on held-out tests.")
KS = sorted(int(k) for k in S["levels"])
LABEL = {"held": "Held-out families", "unseen": "Same-pool families not trained on, all levels", "seen": "Families trained on",
         "bbh": "Public test", "general": "General ability", "retention": "Retention"}
COLS = ["held", "unseen", "seen", "bbh", "general", "retention"]


def sg(x, d=3):
    return f"{x:+.{d}f}"


def cell(k, m):
    v = S["levels"][str(k)].get(m)
    if not v:
        return "<td></td>"
    se = f' <small>± {v["se"]:.3f}</small>' if v["se"] is not None else ""
    note = "<br><small>1 family per run</small>" if m == "unseen" and P["unseen_n"][str(k)] < 5 else ""
    return f"<td>{v['mean']:.3f}{se}{note}</td>"


def main_table():
    ch, base = S["chance"], S["base"]
    rows = ['<tr class="ref"><td>Chance (random replies)</td>'
            f'<td>{ch["held"]:.3f}</td><td colspan="2">{ch["pool"]:.3f}</td><td>{S["bbh_chance"]:.3f}</td><td>0.281</td><td>0.000</td></tr>',
            '<tr class="ref"><td>Untrained model (baseline)</td>'
            f'<td>{base["held"]:.3f}</td><td colspan="2">{base["pool"]:.3f}</td><td>{base["bbh"]:.3f}</td>'
            f'<td>{base["general"]:.3f}</td><td>{base["retention"]:.3f}</td></tr>']
    for k in KS:
        rows.append(f"<tr><td>Trained on {k} families</td>" + "".join(cell(k, m) for m in COLS) + "</tr>")
    return "\n".join(rows)


def fit_table():
    rows = []
    for m in COLS:
        f, g = S["fit"].get(m), S["top_minus_bottom"].get(m)
        if not f:
            continue
        verdict = "rises" if f["slope"] > 0 and f["t"] > 2 else "falls" if f["slope"] < 0 and f["t"] < -2 else "flat"
        rows.append(f"<tr><td>{LABEL[m]}</td><td>{sg(f['slope'], 4)}</td><td>{f['se']:.4f}</td><td>{f['t']:+.1f}</td>"
                    f"<td>{sg(g['gap'])} <small>± {g['se']:.3f}</small></td><td>{verdict}</td></tr>")
    return "\n".join(rows)


def chart(title, runs, means, base, chance, base_label="untrained", ks=None):
    """Value against number of families (log scale). runs: [(k, value)], means: {k: mean}. One dot per run."""
    vals = [v for _, v in runs] + [base] + ([chance] if chance is not None else [])
    lo, hi = math.floor(min(vals) * 10 - 1e-9) / 10, math.ceil(max(vals) * 10 + 1e-9) / 10
    ks = ks or KS
    x0, x1, y0, y1 = 46, 330, 196, 26
    X = lambda k: x0 + (math.log2(k) - 1) * (x1 - x0) / (math.log2(ks[-1]) - 1)
    Y = lambda v: y0 - (v - lo) * (y0 - y1) / (hi - lo)
    o = [f'<svg viewBox="0 0 350 232" role="img" aria-label="{title} against number of families trained on">']
    for i in range(round((hi - lo) * 10) + 1):
        t = lo + i / 10
        o.append(f'<line class="grid" x1="{x0}" y1="{Y(t):.1f}" x2="{x1 + 8}" y2="{Y(t):.1f}"/>')
        o.append(f'<text class="tick" x="{x0 - 6}" y="{Y(t) + 3.5:.1f}" text-anchor="end">{t:.1f}</text>')
    for k in ks:
        o.append(f'<text class="tick" x="{X(k):.1f}" y="212" text-anchor="middle">{k}</text>')
    o.append(f'<text class="tick" x="{(x0 + x1) / 2}" y="228" text-anchor="middle">task families trained on</text>')
    o.append(f'<line class="base" x1="{x0}" y1="{Y(base):.1f}" x2="{x1 + 8}" y2="{Y(base):.1f}"/>')
    below = sorted(means.items())[-1][1] > base      # keep the label on the side away from the data
    o.append(f'<text class="lab base" x="{x1 + 8}" y="{Y(base) + (12 if below else -4):.1f}" text-anchor="end">{base_label} {base:.3f}</text>')
    if chance is not None:
        o.append(f'<line class="chance" x1="{x0}" y1="{Y(chance):.1f}" x2="{x1 + 8}" y2="{Y(chance):.1f}"/>')
        o.append(f'<text class="lab" x="{x1 + 8}" y="{Y(chance) - 4:.1f}" text-anchor="end">chance {chance:.3f}</text>')
    pts = [(X(k), Y(v)) for k, v in sorted(means.items())]
    o.append('<polyline class="line" points="' + " ".join(f"{x:.1f},{y:.1f}" for x, y in pts) + '"/>')
    for k, v in runs:
        o.append(f'<circle class="run" cx="{X(k):.1f}" cy="{Y(v):.1f}" r="3"/>')
    for x, y in pts:
        o.append(f'<circle class="mean" cx="{x:.1f}" cy="{y:.1f}" r="4"/>')
    o.append("</svg>")
    return f"<figure><figcaption>{title}</figcaption>" + "\n".join(o) + "</figure>"


def score_chart(m, title, chance):
    runs = [(r["k"], r[m]) for r in S["runs"] if r["k"] > 0 and r[m] is not None]
    return chart(title, runs, {k: S["levels"][str(k)][m]["mean"] for k in KS}, S["base"][m], chance)


def gain_chart():
    ks = [k for k in KS if P["unseen_n"][str(k)] >= 30]
    runs = [(k, v) for k in ks for v in P["unseen_gain_levels"][str(k)]["runs"]]
    return chart("Same-pool families a run did not train on: gain over the untrained model", runs,
                 {k: P["unseen_gain_levels"][str(k)]["mean"] for k in ks}, 0.0, None, "no gain")


def gain_rows():
    rows = []
    for k in KS:
        v, n = P["unseen_gain_levels"][str(k)], P["unseen_n"][str(k)]
        note = " <small>left out of the fit</small>" if n < 30 else ""
        rows.append(f"<tr><td>{k}</td><td>{n}{note}</td><td>{sg(v['mean'])} <small>± {v['se']:.3f}</small></td></tr>")
    return "\n".join(rows)


def base_rows():
    names = {"held": "Held-out families", "unseen": "Same-pool families not trained on", "bbh": "Public test, strict scoring",
             "general": "General ability", "retention": "Retention"}
    return "\n".join(f"<tr><td>{names[m]}</td><td>{sg(P['trained_minus_base'][m]['mean'])}</td><td>{P['trained_minus_base'][m]['se']:.3f}</td></tr>"
                     for m in names)


def rescore_rows():
    names = {"base": "Untrained model", "k02-r1": "Trained on 2 families (run 1)", "k16-r2": "Trained on 16 families (run 2)",
             "k64-r2": "Trained on 64 families (run 2)"}
    return "\n".join(f"<tr{' class=\"ref\"' if k == 'base' else ''}><td>{names[k]}</td><td>{R[k]['strict']:.3f}</td><td>{R[k]['lenient']:.3f}</td>"
                     f"<td>{R[k]['reply_kinds'].get('letter', 0):.0%}</td><td>{R[k]['reply_kinds'].get('option text', 0):.0%}</td></tr>" for k in names)


def family_table(data, names=None):
    ks = ["0"] + [str(k) for k in KS]
    head = "<tr><th>Family</th><th>Untrained</th>" + "".join(f"<th>{k}</th>" for k in KS) + "<th>Change, untrained to " + str(KS[-1]) + "</th></tr>"
    rows = []
    for h in sorted(data, key=lambda h: data[h][ks[-1]] - data[h]["0"], reverse=True):
        v = data[h]
        rows.append(f"<tr><td>{(names or {}).get(h, h)}</td>" + "".join(f"<td>{v[k]:.2f}</td>" for k in ks)
                    + f"<td>{sg(v[ks[-1]] - v['0'], 2)}</td></tr>")
    return f"<thead>{head}</thead><tbody>" + "\n".join(rows) + "</tbody>"


# ---- rerun with the rule fixed first -------------------------------------------------------------
NKS = sorted(int(k) for k in N["levels"])
NLAB = {"near": "Reserved same-pool families (rule 1)", "diff": "Reserved minus held-out, per run (rule 2)",
        "held": "Held-out families", "rest_gain": "Earlier comparison repeated: gain on non-reserved families not trained on",
        "seen": "Families trained on", "near_valid": "Valid replies on reserved families"}


def reading(f):
    return "rises" if f["slope"] > 0 and f["t"] > 2 else "falls" if f["slope"] < 0 and f["t"] < -2 else "flat"


def near_table():
    b, c = N["base"], N["chance"]
    pm = lambda v: f'{v["mean"]:.3f} <small>± {v["se"]:.3f}</small>'
    rows = [f'<tr class="ref"><td>Chance (random replies)</td><td>{c["near"]:.3f}</td><td></td><td>{c["held"]:.3f}</td><td></td><td></td></tr>',
            f'<tr class="ref"><td>Untrained model (baseline)</td><td>{b["near"]:.3f} <small>± {b["near_item_se"]:.3f} item noise</small></td><td></td>'
            f'<td>{b["held"]:.3f}</td><td></td><td>{b["near_valid"]:.3f}</td></tr>']
    for k in NKS:
        lv = N["levels"][str(k)]
        rows.append(f'<tr><td>Trained on {k} families</td><td>{pm(lv["near"])}</td><td>{sg(lv["near"]["minus_base"])}</td>'
                    f'<td>{pm(lv["held"])}</td><td>{pm(lv["seen"])}</td><td>{lv["near_valid"]["mean"]:.3f}</td></tr>')
    return "\n".join(rows)


def near_fit_rows():
    return "\n".join(f"<tr><td>{NLAB[m]}</td><td>{sg(N['fit'][m]['slope'], 4)}</td><td>{N['fit'][m]['se']:.4f}</td>"
                     f"<td>{N['fit'][m]['t']:+.1f}</td><td>{reading(N['fit'][m])}</td></tr>" for m in NLAB if m in N["fit"])


def near_chart(m, title, chance):
    runs = [(r["k"], r[m]) for r in N["runs"] if r["k"] > 0]
    return chart(title, runs, {k: N["levels"][str(k)][m]["mean"] for k in NKS}, N["base"][m], chance, ks=NKS)


def near_family_table():
    data = N["by_family"]
    ks = ["0"] + [str(k) for k in NKS]
    head = "<tr><th>Reserved family</th><th>Untrained</th>" + "".join(f"<th>{k}</th>" for k in NKS) + f"<th>Change, untrained to {NKS[-1]}</th></tr>"
    rows = [f"<tr><td>{h}</td>" + "".join(f"<td>{data[h][k]:.2f}</td>" for k in ks) + f"<td>{sg(data[h][ks[-1]] - data[h]['0'], 2)}</td></tr>"
            for h in sorted(data, key=lambda h: data[h][ks[-1]] - data[h]["0"], reverse=True)]
    return f"<thead>{head}</thead><tbody>" + "\n".join(rows) + "</tbody>"


def near_lede():
    f, d, h, g = N["fit"]["near"], N["fit"]["diff"], N["fit"]["held"], N["top_minus_bottom"]
    lo, hi = N["levels"][str(NKS[0])]["near"], N["levels"][str(NKS[-1])]["near"]
    if N["rule1"]:
        a = (f"Confirmed. On 16 pool families that no run trained on, the score rose by {sg(f['slope'])} for each doubling of the "
             f"number of families trained on (standard error {f['se']:.3f}, {f['t']:.1f} standard errors above flat), from "
             f"{lo['mean']:.3f} at {NKS[0]} families to {hi['mean']:.3f} at {NKS[-1]}; the untrained model scores {N['base']['near']:.3f}.")
    else:
        a = (f"Not confirmed. On 16 pool families that no run trained on, the score changed by {sg(f['slope'])} for each doubling of "
             f"the number of families trained on (standard error {f['se']:.3f}, {f['t']:+.1f} standard errors from flat), "
             f"{lo['mean']:.3f} at {NKS[0]} families and {hi['mean']:.3f} at {NKS[-1]}; the untrained model scores {N['base']['near']:.3f}.")
    b = (f"In the same runs the held-out families, built on skills the pool never uses, changed by {sg(h['slope'])} per doubling "
         f"(standard error {h['se']:.3f}). The gap between the two scores "
         + (f"widened by {sg(d['slope'])} per doubling (standard error {d['se']:.3f}), so the near-far split is confirmed as well."
            if N["rule2"] else f"changed by {sg(d['slope'])} per doubling (standard error {d['se']:.3f}), so the near-far split is not confirmed."))
    return a, b


def lede():
    f, t = S["fit"]["held"], P["trained_minus_base"]["held"]
    g, gl = P["unseen_gain_fit"], P["unseen_gain_levels"]
    a = (f"No, by the rule set in advance. With the training budget fixed, the score on 18 task families built on skills the "
         f"training pool never uses changed by {sg(f['slope'])} for each doubling of the number of families trained on "
         f"(standard error {f['se']:.3f}). Training at any breadth lifted those families by {sg(t['mean'])} "
         f"(standard error {t['se']:.3f}) over the untrained model, and more families added nothing further.")
    b = (f"Breadth did help one kind of unseen task: families from the same pool that a run happened not to train on. "
         f"Their gain over the untrained model grew from {sg(gl[str(KS[0])]['mean'])} at {KS[0]} families to "
         f"{sg(gl[str(g['max_k'])]['mean'])} at {g['max_k']} ({sg(g['slope'])} per doubling, standard error {g['se']:.3f}). "
         "This comparison was chosen after seeing the data.")
    c = ("My reading: breadth buys transfer to tasks that resemble the training set, and none to tasks that need a different skill. "
         "General ability and the public test stayed flat.")
    return a, b, c


# ---- two strengthening experiments (SPEC3.md, 2026-11-15) -------------------------------------------
def load(path):
    return json.load(open(path)) if os.path.exists(path) else None


XA, XB = load("runs3/n2_stats.json"), load("runs3/m3_stats.json")
XNAME = {"nr": "Confirmed rerun: 1.5B model, reserved set 1", "n2": "A. Second reserved set: 1.5B model, reserved set 2",
         "m3": "B. Larger model: 3B model, reserved set 1"}


def xks(D):
    return sorted(int(k) for k in D["levels"])


def xsummary_rows():
    rows = []
    for D in (N, XA, XB):
        if not D:
            continue
        ks, f = xks(D), D["fit"]
        lo, hi = D["levels"][str(ks[0])]["near"], D["levels"][str(ks[-1])]["near"]
        se = lambda m: f'{sg(f[m]["slope"])} <small>± {f[m]["se"]:.3f}</small>'
        ok = D["rule1"]
        rows.append(f'<tr><td>{XNAME[D.get("prefix", "nr")]}</td><td>{D["n_runs"]}</td><td>{D["chance"]["near"]:.3f}</td><td>{D["base"]["near"]:.3f}</td>'
                    f'<td>{lo["mean"]:.3f}</td><td>{hi["mean"]:.3f}</td><td>{se("near")}<br><small>{f["near"]["t"]:+.1f} standard errors</small></td>'
                    f'<td>{se("held")}</td><td>{se("diff")}</td>'
                    f'<td><span class="badge {"pass" if ok else "fail"}">{D["outcome"]}</span></td></tr>')
    return "\n".join(rows)


def xtable(D):
    b, c = D["base"], D["chance"]
    pm = lambda v: f'{v["mean"]:.3f} <small>± {v["se"]:.3f}</small>'
    rows = [f'<tr class="ref"><td>Chance (random replies)</td><td>{c["near"]:.3f}</td><td></td><td>{c["held"]:.3f}</td><td></td><td></td></tr>',
            f'<tr class="ref"><td>Untrained model (baseline)</td><td>{b["near"]:.3f} <small>± {b["near_item_se"]:.3f} item noise</small></td><td></td>'
            f'<td>{b["held"]:.3f}</td><td></td><td>{b["near_valid"]:.3f}</td></tr>']
    for k in xks(D):
        lv = D["levels"][str(k)]
        rows.append(f'<tr><td>Trained on {k} families <small>({lv["near"]["n"]} runs)</small></td><td>{pm(lv["near"])}</td><td>{sg(lv["near"]["minus_base"])}</td>'
                    f'<td>{pm(lv["held"])}</td><td>{pm(lv["seen"])}</td><td>{lv["near_valid"]["mean"]:.3f}</td></tr>')
    head = ('<thead><tr><th>Row</th><th>Reserved families<br><small>16 families, 512 questions, never trained on</small></th><th>Reserved, minus untrained</th>'
            '<th>Held-out families<br><small>18 families, 576 questions</small></th><th>Families trained on</th><th>Valid replies on reserved</th></tr></thead>')
    return f'<div class="scroll"><table>{head}<tbody>' + "\n".join(rows) + "</tbody></table></div>"


def xchart(D, m, title):
    ks = xks(D)
    runs = [(r["k"], r[m]) for r in D["runs"] if r["k"] > 0]
    return chart(title, runs, {k: D["levels"][str(k)][m]["mean"] for k in ks}, D["base"][m], D["chance"][m], ks=ks)


def xfamily(D):
    data, ks = D["by_family"], xks(D)
    cols = ["0"] + [str(k) for k in ks]
    head = "<tr><th>Reserved family</th><th>Untrained</th>" + "".join(f"<th>{k}</th>" for k in ks) + f"<th>Change, untrained to {ks[-1]}</th></tr>"
    rows = [f"<tr><td>{h}</td>" + "".join(f"<td>{data[h][k]:.2f}</td>" for k in cols) + f"<td>{sg(data[h][cols[-1]] - data[h]['0'], 2)}</td></tr>"
            for h in sorted(data, key=lambda h: data[h][cols[-1]] - data[h]["0"], reverse=True)]
    return f'<div class="scroll"><table><thead>{head}</thead><tbody>' + "\n".join(rows) + "</tbody></table></div>"


def xsentence(D, what):
    f, h, d, ks = D["fit"]["near"], D["fit"]["held"], D["fit"]["diff"], xks(D)
    lo, hi = D["levels"][str(ks[0])]["near"], D["levels"][str(ks[-1])]["near"]
    verdict = "Replicated" if D["rule1"] else D["outcome"]
    split = ("the gap to the held-out families widened by " if D["rule2"] else "the gap to the held-out families changed by ") + \
            f"{sg(d['slope'])} per doubling (standard error {d['se']:.3f})"
    return (f"<strong>{what}</strong> {verdict}. The reserved-family score changed by {sg(f['slope'])} per doubling of the families trained on "
            f"(standard error {f['se']:.3f}, {f['t']:+.1f} standard errors from flat), from {lo['mean']:.3f} at {ks[0]} families to "
            f"{hi['mean']:.3f} at {ks[-1]}; untrained {D['base']['near']:.3f}. Held-out families: {sg(h['slope'])} per doubling "
            f"(standard error {h['se']:.3f}); {split}.")


def strong_section():
    if not (XA or XB):
        return ""
    o = ['<section>', '<h2>Two checks on the confirmed result: a second reserved set and a larger model</h2>',
         '<p>The confirmed rerun rested on one set of 16 reserved families and one model size. Each check below changes one of those '
         'and nothing else. The rules were written before either was run and are the same two rules as the rerun.</p>']
    if XA:
        o.append("<p>" + xsentence(XA, "A, second reserved set.") + "</p>")
    if XB:
        o.append("<p>" + xsentence(XB, "B, model twice the size.") + "</p>")
    o.append('<div class="scroll"><table><thead><tr><th>Experiment</th><th>Trained runs</th><th>Chance on reserved</th><th>Untrained</th>'
             '<th>Trained on 2 families</th><th>Trained on 32 families</th><th>Reserved families: change per doubling (rule 1)</th>'
             '<th>Held-out families: change per doubling</th><th>Reserved minus held-out: change per doubling (rule 2)</th><th>Pre-set outcome</th></tr></thead>'
             f'<tbody>{xsummary_rows()}</tbody></table></div>')
    o.append('<p class="muted">Fraction correct at temperature 0. ± is the standard error of the fitted slope across runs. '
             'Reserved set 2 shares no family with reserved set 1, so the scores of rows 1 and 2 are on different questions and their levels are not comparable; the slopes are.</p>')
    o.append('<div class="charts">')
    if XA:
        o += [xchart(XA, "near", "A: second reserved set, 1.5B model"), xchart(XA, "held", "A: held-out families, 1.5B model")]
    if XB:
        o += [xchart(XB, "near", "B: reserved set 1, 3B model"), xchart(XB, "held", "B: held-out families, 3B model")]
    o.append('</div><p class="muted">Grey dots are single runs. Green dots and the line are the level means. The vertical axes differ between panels.</p>')
    if XA:
        o += ["<p><strong>A, level by level.</strong> Reserved set 2: " + ", ".join(XA["reserved"]) + ".</p>", xtable(XA), xfamily(XA)]
    if XB:
        o += ["<p><strong>B, level by level.</strong> Same families, seeds and questions as the confirmed rerun; only the model changed.</p>", xtable(XB),
              '<p class="muted"><strong>Deviation from the written plan.</strong> 14 of the 20 trained 3B runs first stopped with a GPU out-of-memory error. '
              'The plan said failed runs are not replaced. They were run again with the same families, seeds and settings, because the failures followed prompt length '
              'and would have left mostly 2-family runs. In those 14 runs each 8-example training step went through the model as two groups of 4 with the gradients summed, '
              'which is the same gradient up to rounding. The other 6 trained runs used a single pass. The change was written into the plan before any score of this check was read.</p>']
        t = XB.get("twin")
        if t:
            o.append(f'<p class="muted">Each 3B run against its 1.5B twin trained on the same families ({t["n"]} pairs): reserved {sg(t["near"]["mean"])} '
                     f'(standard error {t["near"]["se"]:.3f}), held-out {sg(t["held"]["mean"])} (standard error {t["held"]["se"]:.3f}). '
                     f'The reserved difference changed by {sg(t["near_fit"]["slope"])} per doubling (standard error {t["near_fit"]["se"]:.3f}). '
                     + (f'Untrained 3B minus untrained 1.5B: reserved {sg(t["base_near"])}, held-out {sg(t["base_held"])}.' if "base_near" in t else "") + '</p>')
        o.append(xfamily(XB))
    o.append("</section>")
    return "\n".join(o)


def strong_short():
    bits = []
    for D, name in ((XA, "a second, separate reserved set"), (XB, "a model twice the size (3B)")):
        if D:
            f = D["fit"]["near"]
            bits.append(f"{name}: {sg(f['slope'])} per doubling (standard error {f['se']:.3f}), pre-set outcome \"{D['outcome']}\"")
    return ("<p><strong>Two checks added on 2026-11-15:</strong> " + "; ".join(bits) + ".</p>") if bits else ""


A, B2, C = lede()
NA, NB = near_lede()
nbadge = "pass" if N["rule1"] else "fail"
badge = "pass" if S["outcome"] == "Curve" else "fail"
ch = S["chance"]
HTML = f"""<title>Breadth Curve Experiment</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Space+Grotesk:wght@500;600&display=swap">
<style>
/* Layout: one reading column. Verdict, main table with baseline rows, two curves to scale, then detail tables. Tokens match the earlier pages. */
:root {{ --bg:#f6f4ee; --card:#fffdf8; --fg:#1c1b18; --muted:#6a665d; --rule:#dcd7ca; --accent:#1f6f54; --drift:#8a8578;
  --pass:#1f6f54; --fail:#b0432b; --run:#8a6a12; --passbg:#dcefe6; --failbg:#f6ddd6; --runbg:#f3e9c8; }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ --bg:#151513; --card:#1d1c19; --fg:#ebe8e0; --muted:#a09b90;
  --rule:#35332e; --accent:#62c49e; --drift:#8f8a7e; --pass:#7fd3b0; --fail:#f09a84; --run:#e2c46a; --passbg:#173428; --failbg:#3b1d16;
  --runbg:#3a3113; color-scheme: dark }} }}
:root[data-theme="dark"] {{ --bg:#151513; --card:#1d1c19; --fg:#ebe8e0; --muted:#a09b90; --rule:#35332e; --accent:#62c49e;
  --drift:#8f8a7e; --pass:#7fd3b0; --fail:#f09a84; --run:#e2c46a; --passbg:#173428; --failbg:#3b1d16; --runbg:#3a3113; color-scheme: dark }}
body {{ background:var(--bg); color:var(--fg); font:15px/1.55 Inter,system-ui,sans-serif; }}
main {{ max-width:1000px; margin:0 auto; padding-block:28px 56px; padding-left:max(16px, env(safe-area-inset-left, 0px));
  padding-right:max(16px, env(safe-area-inset-right, 0px)); display:flex; flex-direction:column; gap:30px; }}
h1 {{ font:600 clamp(20px,3.2vw,27px)/1.25 "Space Grotesk",Inter,sans-serif; margin:0; text-wrap:balance; }}
h2 {{ font:600 18px/1.3 "Space Grotesk",Inter,sans-serif; margin:0 0 10px; text-wrap:balance; }}
p {{ margin:0 0 8px; max-width:68ch; }} .lede {{ font-size:16.5px; }} .muted,figcaption,small {{ color:var(--muted); }}
.kicker {{ font:600 12px/1.3 Inter,sans-serif; letter-spacing:.08em; text-transform:uppercase; color:var(--accent); margin-bottom:10px; }}
section {{ min-width:0; }} .scroll + p, .charts + p {{ margin-top:10px; }}
.scroll {{ overflow-x:auto; border:1px solid var(--rule); border-radius:10px; background:var(--card); }}
table {{ border-collapse:collapse; width:100%; font-size:13.5px; font-variant-numeric:tabular-nums; }}
th,td {{ padding:8px 10px; text-align:left; border-bottom:1px solid var(--rule); vertical-align:top; }}
th {{ font-weight:600; font-size:12px; color:var(--muted); }} tbody tr:last-child td {{ border-bottom:0; }}
tr.ref td {{ color:var(--muted); }} td:first-child {{ min-width:150px; }} td small {{ font-size:12px; white-space:nowrap; }}
.badge {{ display:inline-block; padding:2px 9px; border-radius:99px; font-weight:600; font-size:12px; white-space:nowrap; vertical-align:2px; }}
.badge.fail {{ color:var(--fail); background:var(--failbg); }} .badge.pass {{ color:var(--pass); background:var(--passbg); }}
.charts {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(290px,1fr)); gap:14px; }}
figure {{ margin:0; background:var(--card); border:1px solid var(--rule); border-radius:10px; padding:12px 12px 6px; min-width:0; }}
figcaption {{ font-size:13px; margin-bottom:6px; font-weight:600; color:var(--fg); }} svg {{ width:100%; height:auto; display:block; }}
.grid {{ stroke:var(--rule); stroke-width:1; }} .chance {{ stroke:var(--muted); stroke-width:1; stroke-dasharray:3 3; }}
.base {{ stroke:var(--run); stroke-width:1.4; stroke-dasharray:6 3; }} text.base {{ stroke:none; fill:var(--run); }}
.tick {{ fill:var(--muted); font:10.5px Inter,sans-serif; }} .lab {{ fill:var(--muted); font:10.5px Inter,sans-serif; }}
tr.ref td small {{ color:var(--muted); }}
.line {{ fill:none; stroke:var(--accent); stroke-width:2; }} .mean {{ fill:var(--accent); }} .run {{ fill:var(--drift); }}
ul {{ margin:0; padding-left:20px; max-width:70ch; }} li {{ margin-bottom:6px; }} code {{ font-size:13px; }}
</style>
<main>
<header>
  <div class="kicker">Byproduct experiment. Follow-up: does breadth of training buy transfer?</div>
  <h1>{THESIS}</h1>
</header>

<section>
  <p class="lede">{A} <span class="badge {badge}">Pre-set outcome: {S["outcome"]}</span></p>
  <p>{B2}</p>
  <p><strong>Rerun with the rule fixed first:</strong> {NA} <span class="badge {nbadge}">Pre-set outcome: {N["outcome"]}</span></p>
  {strong_short()}
  <p>{C}</p>
  <p class="muted">Setup: Qwen2.5-1.5B, {S["n_runs"]} training runs. Each run trained on K one-question task families drawn at random from a pool of {S["pool"]},
  with the same budget at every K: 1,200 steps on 9,600 examples of correct answers. The rule was written before the runs:
  a rising line more than 2 standard errors above flat counts as a curve.</p>
</section>

<section>
  <h2>Scores by number of families trained on</h2>
  <div class="scroll"><table>
    <thead><tr><th>Row</th><th>Held-out families<br><small>18 families, 576 questions, never trained on</small></th>
<th>Same-pool families not trained on</th><th>Families trained on</th><th>Public test, strict scoring<br><small>11 BIG-Bench Hard tasks</small></th>
    <th>General ability<br><small>3 original tests</small></th><th>Retention</th></tr></thead>
    <tbody>
    {main_table()}
    </tbody>
  </table></div>
  <p class="muted">Fraction correct at temperature 0 on the same items in every row. Trained rows are the mean of 3 runs, each with its own
  random set of families; ± is the standard error across those 3 runs. For the two reference rows the pool score covers all {S["pool"]} pool families.</p>
</section>

<section>
  <h2>The curves</h2>
  <div class="charts">
    {score_chart("held", "Held-out families: skills the pool never uses", ch["held"])}
    {gain_chart()}
    {score_chart("general", "General ability (3 original tests)", 0.281)}
    {score_chart("retention", "Retention (unrelated skills kept)", None)}
  </div>
  <p class="muted">Grey dots are single runs. Green dots and the line are the mean of 3 runs. The horizontal axis doubles at each step.
  The vertical axes differ between panels.</p>
</section>

<section>
  <h2>Is the line rising?</h2>
  <div class="scroll"><table>
    <thead><tr><th>Measure</th><th>Change per doubling of families</th><th>Standard error</th><th>Slope ÷ standard error</th>
    <th>{KS[-1]} families minus {KS[0]} families</th><th>Reading</th></tr></thead>
    <tbody>
    {fit_table()}
    </tbody>
  </table></div>
  <p class="muted">Least-squares line over the {S["n_runs"]} trained runs. "Rises" or "falls" needs more than 2 standard errors; anything less reads as flat. The same-pool row includes the 64-family level,
  where only 1 family per run is left unseen; the next section gives that line without it.</p>
</section>

<section>
  <h2>Added after seeing the data: transfer inside the pool</h2>
  <p>The plan listed "pool families a run did not train on" as a reported score, with no pass rule. Two choices below were made after
  seeing the results: comparing each run with the untrained model on the same families, and leaving out the 64-family level, where only 1 family per run is left.</p>
  <div class="scroll"><table>
    <thead><tr><th>Families trained on</th><th>Same-pool families left unseen, per run</th><th>Gain over the untrained model on those families</th></tr></thead>
    <tbody>
    {gain_rows()}
    </tbody>
  </table></div>
  <p>Line through levels 2 to {P["unseen_gain_fit"]["max_k"]}: {sg(P["unseen_gain_fit"]["slope"])} per doubling, standard error {P["unseen_gain_fit"]["se"]:.3f},
  which is {P["unseen_gain_fit"]["t"]:.1f} standard errors above flat.</p>
  <p class="muted">Why the two unseen sets may differ: the pool was written in four themes (arithmetic, symbols, lists, logic), so its families share
  sub-skills, and the writers reported near-duplicates. The held-out set was built on skills the pool was told not to use. This explanation is mine and was not tested.</p>
</section>

<section>
  <h2>Rerun with the rule fixed first: reserved families</h2>
  <p>{NA} <span class="badge {nbadge}">Pre-set outcome: {N["outcome"]}</span></p>
  <p>{NB}</p>
  <p class="muted">Design, written into the plan before any rerun result: 16 of the {S["pool"]} pool families were drawn once by a fixed seed and kept out of
  every training set. {N["n_runs"]} new runs, 4 at each of {", ".join(str(k) for k in NKS)} families, each drew its families from the other 49 with new seeds.
  Same model and budget. Rule 1: the line of the reserved-family score against the number of families must rise by more than 2 standard errors.
  Rule 2: the same for the reserved score minus the held-out score, run by run.</p>
  <div class="scroll"><table>
    <thead><tr><th>Row</th><th>Reserved same-pool families<br><small>16 families, 512 questions, never trained on</small></th><th>Reserved, minus untrained</th>
    <th>Held-out families<br><small>18 families, 576 questions</small></th><th>Families trained on</th><th>Valid replies on reserved</th></tr></thead>
    <tbody>
    {near_table()}
    </tbody>
  </table></div>
  <p class="muted">Fraction correct at temperature 0 on the same items in every row. Trained rows are the mean of 4 runs; ± is the standard error across those runs.
  The untrained row is one model, so its ± is the item-sampling noise of 512 questions.</p>
  <div class="charts">
    {near_chart("near", "Rerun: reserved same-pool families", N["chance"]["near"])}
    {near_chart("held", "Rerun: held-out families (skills the pool never uses)", N["chance"]["held"])}
  </div>
  <p class="muted">Grey dots are single runs. Green dots and the line are the mean of 4 runs.</p>
  <div class="scroll"><table>
    <thead><tr><th>Measure</th><th>Change per doubling of families</th><th>Standard error</th><th>Slope ÷ standard error</th><th>Reading</th></tr></thead>
    <tbody>
    {near_fit_rows()}
    </tbody>
  </table></div>
  <p class="muted">Least-squares lines over the {N["n_runs"]} rerun runs. {NKS[-1]} families minus {NKS[0]} families on the reserved score:
  {sg(N["top_minus_bottom"]["gap"])} (standard error {N["top_minus_bottom"]["se"]:.3f}). All {N["n_runs"]} trained runs minus the untrained model:
  reserved {sg(N["trained_minus_base"]["near"]["mean"])} (standard error {N["trained_minus_base"]["near"]["se"]:.3f}), held-out
  {sg(N["trained_minus_base"]["held"]["mean"])} (standard error {N["trained_minus_base"]["held"]["se"]:.3f}).</p>
  <div class="scroll"><table>
    {near_family_table()}
  </table></div>
  <p class="muted">Reserved families one by one, mean of 4 runs per level. Some have a close relative among the trainable families
  (nextsign and nextto with neighbor; whichsign with resign); the plan lists this as a known limit.</p>
</section>

{strong_section()}

<section>
  <h2>All trained runs against the untrained model</h2>
  <div class="scroll"><table>
    <thead><tr><th>Measure</th><th>Mean of {S["n_runs"]} trained runs minus untrained</th><th>Standard error across runs</th></tr></thead>
    <tbody>
    {base_rows()}
    </tbody>
  </table></div>
</section>

<section>
  <h2>The public-test drop is a scoring artefact</h2>
  <p>Strict scoring counts a multiple-choice reply as right only if it names the option letter. Trained models often reply with the option's text.
  I recorded raw replies for four models and rescored them, also accepting a reply equal to the text of exactly one option.</p>
  <div class="scroll"><table>
    <thead><tr><th>Model</th><th>Strict score</th><th>Rescored</th><th>Multiple-choice replies that are a letter</th><th>That are an option's text</th></tr></thead>
    <tbody>
    {rescore_rows()}
    </tbody>
  </table></div>
  <p class="muted">440 items per model, so one score has a standard error of about 0.023. After rescoring, no model differs from the untrained one by 2 standard errors.
  Only 4 of the 19 models were rescored, so the per-level public-test numbers above remain strict.</p>
</section>

<section>
  <h2>Held-out families one by one</h2>
  <div class="scroll"><table>{family_table(S["held_by_family"])}</table></div>
  <p class="muted">Columns are the number of families trained on; values are means of 3 runs, 32 questions per family. Sorted by change.</p>
</section>

<section>
  <h2>Public test task by task</h2>
  <div class="scroll"><table>{family_table(S["bbh_by_task"], {t: t.replace("_", " ") for t in S["bbh_by_task"]})}</table></div>
  <p class="muted">40 items per task from BIG-Bench Hard, written by other authors. Several tasks resemble pool families
  (counting, navigation, swaps, boolean logic), so this is a transfer test written by others, and only partly a far test.</p>
</section>

<section>
  <h2>Limits</h2>
  <ul>
    <li>One model size (1.5B). Three runs per level, so each standard error rests on 3 values.</li>
    <li>At low K, which families were drawn is mixed with how many. Each run drew its own set to average over this.</li>
    <li>The 52 new pool families and 12 new held-out families were written by AI subagents in one style, then machine-checked. A family that trains a skill close to a held-out one would inflate the curve; the per-family table shows where the gain sits.</li>
    <li>The first same-pool result was analysed after seeing the data. The rerun fixed its rule first; its outcome is "{N["outcome"]}". {"Two later checks repeated it on a second reserved set and on a 3B model; their outcomes are in the section above. All runs use one model family (Qwen2.5) and one task style." if (XA or XB) else "It covers one reserved set of 16 families and one model size."}</li>
    <li>The model was trained on correct answers directly. This measures what breadth can buy at most; it is not a selection experiment.</li>
  </ul>
</section>

<section>
  <h2>Cost</h2>
  <p>${SPENT:.2f} on Modal for this experiment (own upper estimate), against a $10 limit: ${S["usd"]:.2f} for the 19 runs, the rest for a smoke test and the four reply recordings. Rules and design: SPEC3.md, "Breadth curve".</p>
  <p>Rerun: ${RERUN_SPENT:.2f} (own upper estimate) for 21 runs and a smoke test, against about $3 approved. Rules: SPEC3.md, "Near-transfer rerun".</p>
  {f'<p>Two checks: ${STRONG_SPENT:.2f} (own upper estimate), against under $10 approved. Rules: SPEC3.md, "Two strengthening experiments".</p>' if (XA or XB) else ""}
</section>
</main>
"""
open("breadth.html", "w").write(HTML)
print(len(HTML), "bytes; outcome", S["outcome"])
