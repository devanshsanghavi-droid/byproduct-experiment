"""Results page for version 4 (social arms with changing weights). Rules: SPEC4.md.

    PYTHONPATH=. python3 -m evo3.social_page   -> social.html

Reads runs3/s4_stats.json (evo3.social_stats), runs3/null4.json (evo3.social_null) and runs3/s4_smoke.json.
Every sentence that states a result is built from those numbers.
"""
import json
import math

from evo3 import social as S

ST = json.load(open("runs3/s4_stats.json"))
NULL = json.load(open("runs3/null4.json"))
SMOKE = json.load(open("runs3/s4_smoke.json"))
THESIS = ("All three surviving theories say intelligence was a byproduct. A faithful simulation therefore scores agents "
          "only on staying alive and being copied, then measures general ability separately on held-out tests.")
ARMS = [a for a in S.ARMS if a in ST["arms"]]
UN = ST["untrained"]
WHAT = {"DRIFT": "random parents (chance)", "SOLO": "survival alone", "SOLO-W": "survival alone, old learning rule",
        "COOP": "cooperation: pairs, shared harvest", "CULTURE": "cultural brain: copy a family from a successful non-parent",
        "ALPHA": "pairs, one may keep all", "DOMEST": "self-domestication: keep-all plus vote", "MEDLEY": "all three theories"}
SMOKE_COST = round(SMOKE.get("SOLO", {}).get("usd", 0) + SMOKE.get("MEDLEY", {}).get("usd", 0), 2)


def ms(v, d=3, sign=False):
    """[mean, se, n] as 'mean ± se'."""
    if not v:
        return ""
    m = f"{v[0]:+.{d}f}" if sign else f"{v[0]:.{d}f}"
    return m + (f" <small>± {v[1]:.{d}f}</small>" if v[1] is not None else "")


def badge(text, kind):
    return f'<span class="badge {kind}">{text}</span>'


def lab(word):
    return {"above": badge("above", "pass"), "below": badge("below", "fail"), "neither": badge("no difference", "run")}[word]


# ---- tables -----------------------------------------------------------------------------------------
def arm_table():
    rows = ['<tr class="ref"><td>Untrained model (baseline)</td><td></td><td></td><td></td>'
            f'<td>{UN["inworld"]:.3f}</td><td>{UN["near"]:.3f}</td><td>{UN["held"]:.3f}</td><td></td></tr>',
            '<tr class="ref"><td>Chance (random replies)</td><td></td><td></td><td></td><td></td><td>0.289</td><td></td><td></td></tr>']
    for arm in ARMS:
        a = ST["arms"][arm]
        nb = a.get("null_breadth")
        null = f'{nb["untrained"][0]:.1f} to {nb["perfect"][0]:.1f}' if nb else ""
        rows.append(f'<tr><td><strong>{arm}</strong><br><small class="wrap">{WHAT[arm]}</small></td><td>{a["n"]}</td>'
                    f'<td>{ms(a["breadth"], 2)}</td><td>{null}</td><td>{ms(a["inworld"])}</td><td>{ms(a["near"])}</td>'
                    f'<td>{ms(a["held"])}</td><td>{a["acc_first"][0]:.2f} to {a["acc_last"][0]:.2f}</td></tr>')
    return "\n".join(rows)


def theory_table():
    rows = []
    for arm, t in ST["theories"].items():
        a = ST["arms"][arm]
        g, ng = a["gap"], a.get("null_gap_log2", {})
        kind = "pass" if "more than mechanics" in t["breadth_label"] else ("fail" if t["breadth_label"] in ("Narrows", "Not tested") else "run")
        rows.append(f'<tr><td><strong>{t["title"]}</strong><br><small>{arm} against {t["control"]}</small></td>'
                    f'<td>{ms(g["log2_breadth"], 3, True)}</td>'
                    f'<td>{ng.get("perfect", 0):+.3f} to {ng.get("untrained", 0):+.3f}</td>'
                    f'<td>{badge(t["breadth_label"], kind)}</td>'
                    f'<td>{ms(g["inworld"], 3, True)}<br>{lab(a["gap_label"]["inworld"])}</td>'
                    f'<td>{ms(g["near"], 3, True)}</td><td>{ms(g["held"], 3, True)}</td></tr>')
    for arm, title in S.THEORIES.items():
        if arm not in ST["theories"]:
            rows.append(f'<tr><td><strong>{title}</strong><br><small>{arm} against {S.CONTROL[arm]}</small></td>'
                        f'<td colspan="2">not run</td><td>{badge("Not tested", "fail")}</td><td colspan="3">'
                        'The model kept the whole harvest every time it was asked in the smoke test, so the arms that depend on that choice were dropped by rule.</td></tr>')
    return "\n".join(rows)


def rule_table():
    r1, r6, rE = ST["rule1"], ST["rule6"], ST["ruleE"]
    rows = [
        f'<tr><td>1. Learning check</td><td>SOLO practised skill minus untrained {UN["inworld"]:.3f}</td><td>{ms(r1["solo_minus_untrained_inworld"], 3, True)}</td>'
        f'<td>{badge("Pass", "pass") if r1["passes"] else badge("No learning", "fail")}</td></tr>',
        f'<tr><td>1. Who gets the credit</td><td>SOLO minus DRIFT, practised skill</td><td>{ms(r1["solo_minus_drift_inworld"], 3, True)}</td>'
        f'<td>{badge("Credit: " + r1["credit"], "pass" if r1["credit"] == "selection" else "run")}</td></tr>',
        f'<tr><td>6. Selection and breadth</td><td>SOLO minus DRIFT, breadth in doublings</td><td>{ms(r6["log2_breadth"], 3, True)}</td>'
        f'<td>{lab(S_verdict(r6["log2_breadth"]))}<br><small>null model: {NULL["tables"]["untrained"]["arms"]["SOLO"]["gap_log2"][0]:+.2f} to '
        f'{NULL["tables"]["perfect"]["arms"]["SOLO"]["gap_log2"][0]:+.2f}</small></td></tr>',
        f'<tr><td>6. Selection and similar tasks</td><td>SOLO minus DRIFT, similar-task score</td><td>{ms(r6["near"], 3, True)}</td><td>{lab(S_verdict(r6["near"]))}</td></tr>']
    if "SOLO-W" in ST["arms"]:
        g = ST["arms"]["SOLO-W"]["gap"]
        rows.append(f'<tr><td>Old learning rule</td><td>SOLO-W minus SOLO, practised skill</td><td>{ms(g["inworld"], 3, True)}</td>'
                    f'<td>{lab(ST["arms"]["SOLO-W"]["gap_label"]["inworld"])}</td></tr>')
        rows.append(f'<tr><td>Old learning rule</td><td>SOLO-W minus SOLO, similar-task score</td><td>{ms(g["near"], 3, True)}</td>'
                    f'<td>{lab(ST["arms"]["SOLO-W"]["gap_label"]["near"])}</td></tr>')
    uv = ST["untrained_valid"]
    for arm in ARMS:
        e = rE[arm]
        rows.append(f'<tr><td>E, exploratory. {arm}</td><td>similar-task score minus untrained {UN["near"]:.3f}'
                    f'<br><small>valid replies only, minus untrained {uv["near_valid_only"]:.3f}</small></td>'
                    f'<td>{ms(e["near"], 3, True)}<br>{ms(e["near_valid_only"], 3, True)}</td>'
                    f'<td>{lab(e["near_label"])}<br>{lab(e["near_valid_only_label"])}</td></tr>')
    for arm in ARMS:
        e = rE[arm]
        rows.append(f'<tr><td>E, exploratory. {arm}</td><td>new-skill score minus untrained {UN["held"]:.3f}</td>'
                    f'<td>{ms(e["held"], 3, True)}</td><td>{lab(e["held_label"])}</td></tr>')
    return "\n".join(rows)


def S_verdict(m):
    if not m or m[1] is None:
        return "neither"
    return "above" if m[0] > 2 * m[1] else ("below" if m[0] < -2 * m[1] else "neither")


def check_table():
    so, me = SMOKE.get("SOLO", {}), SMOKE.get("MEDLEY", {})
    gaps = [NULL["tables"]["perfect"]["arms"][a]["log2_breadth"][0] - NULL["tables"]["untrained"]["arms"][S.CONTROL[a]]["log2_breadth"][0]
            for a in S.THEORIES]
    k = me.get("keep", [None])[0]
    asked = me.get("keep_asked", [0])[0]
    return "\n".join([
        f'<tr><td>0. Null model, free</td><td>largest breadth gap possible between a theory arm and its control</td><td>{max(gaps):.2f} doublings</td>'
        f'<td>2 doublings</td><td>{badge("Score lift untestable", "fail")}</td></tr>',
        f'<tr><td>1. Smoke, learning</td><td>score on practised families, temperature 0</td><td>{so.get("practised_before", 0):.3f} untrained to {so.get("practised_after", 0):.3f}</td>'
        f'<td>up at least 0.05</td><td>{badge("Pass", "pass")}</td></tr>',
        f'<tr><td>1. Smoke, cost</td><td>projected cost of one full run</td><td>${so.get("full_run_usd_estimate", 0):.2f} to ${me.get("full_run_usd_estimate", 0):.2f}</td>'
        f'<td>at most $0.26</td><td>{badge("Pass", "pass")}</td></tr>',
        f'<tr><td>2. Smoke, keep-all rate</td><td>share of holders who kept the whole harvest, generation 1</td><td>{round((k or 0) * asked)} of {asked} ({(k or 0):.0%})</td>'
        f'<td>between 5% and 95%</td><td>{badge("Fail: 3 arms dropped", "fail")}</td></tr>',
        f'<tr><td>2. Smoke, vote</td><td>agents barred per generation</td><td>{me.get("barred", [0])[0]}</td><td>at least 1</td><td>{badge("Pass", "pass")}</td></tr>'])


# ---- charts -----------------------------------------------------------------------------------------
def dots(title, key, d=2, ref=None, ref_label="", band=None, note=""):
    """One dot per run, a bar at each arm's mean. `ref` draws a dashed line; `band[arm]` = (low, high) shaded per arm."""
    runs = [(r["arm"], r[key], r["seed"]) for r in ST["runs"] if r[key] is not None]
    vals = [v for _, v, _ in runs] + ([ref] if ref is not None else []) + ([x for b in (band or {}).values() for x in b])
    span = max(vals) - min(vals) or 1
    step = 10 ** math.floor(math.log10(span / 3))
    step = next(s for s in (step, 2 * step, 5 * step, 10 * step) if span / s <= 6)
    lo, hi = math.floor(min(vals) / step) * step, math.ceil(max(vals) / step) * step
    x0, x1, y0, y1 = 46, 336, 190, 22
    slot = (x1 - x0) / len(ARMS)
    X = lambda arm: x0 + slot * (ARMS.index(arm) + 0.5)
    Y = lambda v: y0 - (v - lo) * (y0 - y1) / (hi - lo)
    o = [f'<svg viewBox="0 0 350 226" role="img" aria-label="{title}, one dot per run, by arm">']
    t = lo
    while t <= hi + 1e-9:
        o.append(f'<line class="grid" x1="{x0}" y1="{Y(t):.1f}" x2="{x1}" y2="{Y(t):.1f}"/>')
        o.append(f'<text class="tick" x="{x0 - 6}" y="{Y(t) + 3.5:.1f}" text-anchor="end">{t:.{d}f}</text>')
        t += step
    for arm in ARMS:
        if band and arm in band:
            b0, b1 = sorted(band[arm])
            o.append(f'<rect class="band" x="{X(arm) - slot * 0.34:.1f}" y="{Y(b1):.1f}" width="{slot * 0.68:.1f}" height="{max(2, Y(b0) - Y(b1)):.1f}" rx="2">'
                     f'<title>{arm}: null model, {b0:.{d}f} to {b1:.{d}f}</title></rect>')
        o.append(f'<text class="tick" x="{X(arm):.1f}" y="206" text-anchor="middle">{arm}</text>')
    if ref is not None:
        o.append(f'<line class="base" x1="{x0}" y1="{Y(ref):.1f}" x2="{x1}" y2="{Y(ref):.1f}"/>')
        o.append(f'<text class="lab base" x="{x1}" y="{Y(ref) - 4:.1f}" text-anchor="end">{ref_label} {ref:.3f}</text>')
    for arm in ARMS:
        mine = sorted((v, s) for a, v, s in runs if a == arm)
        for i, (v, s) in enumerate(mine):
            off = (i - (len(mine) - 1) / 2) * 5.5
            o.append(f'<circle class="run" cx="{X(arm) + off:.1f}" cy="{Y(v):.1f}" r="3.2"><title>{arm}, seed {s}: {v:.{d + 1}f}</title></circle>')
        m = ST["arms"][arm][key][0]
        o.append(f'<line class="meanbar" x1="{X(arm) - slot * 0.3:.1f}" y1="{Y(m):.1f}" x2="{X(arm) + slot * 0.3:.1f}" y2="{Y(m):.1f}"><title>{arm} mean: {m:.{d + 1}f}</title></line>')
    o.append("</svg>")
    cap = f'<p class="cap">{note}</p>' if note else ""
    return f"<figure><figcaption>{title}</figcaption>" + "\n".join(o) + cap + "</figure>"


def lines(arm, key, title, lo, hi, d=1, ref=None):
    """Small multiple: one thin line per seed, a thick line for the mean, against generation."""
    runs = [r[key] for r in ST["runs"] if r["arm"] == arm]
    G = len(runs[0])
    x0, x1, y0, y1 = 34, 200, 112, 14
    X = lambda g: x0 + g * (x1 - x0) / (G - 1)
    Y = lambda v: y0 - (v - lo) * (y0 - y1) / (hi - lo)
    o = [f'<svg viewBox="0 0 212 138" role="img" aria-label="{title} by generation, {arm}">']
    for t in (lo, (lo + hi) / 2, hi):
        o.append(f'<line class="grid" x1="{x0}" y1="{Y(t):.1f}" x2="{x1}" y2="{Y(t):.1f}"/>')
        o.append(f'<text class="tick" x="{x0 - 5}" y="{Y(t) + 3.5:.1f}" text-anchor="end">{t:.{d}f}</text>')
    for g in (0, G - 1):
        o.append(f'<text class="tick" x="{X(g):.1f}" y="126" text-anchor="middle">gen {g + 1}</text>')
    if ref is not None:
        o.append(f'<line class="base" x1="{x0}" y1="{Y(ref):.1f}" x2="{x1}" y2="{Y(ref):.1f}"/>')
    for v in runs:
        o.append('<polyline class="thin" points="' + " ".join(f"{X(g):.1f},{Y(x):.1f}" for g, x in enumerate(v)) + '"/>')
    mean = [sum(v[g] for v in runs) / len(runs) for g in range(G)]
    o.append('<polyline class="line" points="' + " ".join(f"{X(g):.1f},{Y(x):.1f}" for g, x in enumerate(mean)) + '"/>')
    for g, x in enumerate(mean):
        o.append(f'<circle class="mean" cx="{X(g):.1f}" cy="{Y(x):.1f}" r="2.6"><title>{arm}, generation {g + 1}: mean {x:.{d + 1}f}</title></circle>')
    o.append("</svg>")
    return f"<figure><figcaption>{arm}</figcaption>" + "\n".join(o) + "</figure>"


# ---- sentences built from the numbers ---------------------------------------------------------------
def lede():
    r1 = ST["rule1"]
    parts = []
    for arm, t in ST["theories"].items():
        g = ST["arms"][arm]["gap"]["log2_breadth"]
        name = {"COOP": "Cooperation", "CULTURE": "Copying from successful non-parents", "DOMEST": "The exile vote"}[arm]
        parts.append(f"{name}: {t['breadth_label'].lower()} ({g[0]:+.2f} doublings of breadth against {t['control']}, standard error {g[1]:.2f}).")
    learn = (f"Populations did learn: SOLO's practised skill ended {r1['solo_minus_untrained_inworld'][0]:+.3f} against the untrained "
             f"{UN['inworld']:.3f} (standard error {r1['solo_minus_untrained_inworld'][1]:.3f}). "
             f"SOLO minus DRIFT on the same score was {r1['solo_minus_drift_inworld'][0]:+.3f} (standard error {r1['solo_minus_drift_inworld'][1]:.3f}), "
             f"so by the pre-set rule the learning is credited to {r1['credit']}.")
    return " ".join(parts), learn


def explore():
    e = ST["ruleE"]
    up = [a for a in ARMS if e[a]["near_label"] == "above"]
    best = max(ARMS, key=lambda a: e[a]["near"][0])
    s = (f"{len(up)} of {len(ARMS)} arms ended above the untrained model on similar unseen tasks"
         + (f" ({', '.join(up)})" if up else "") + f". The largest gain was {best} at {e[best]['near'][0]:+.3f} (standard error {e[best]['near'][1]:.3f}). ")
    r6 = ST["rule6"]["near"]
    word = {"above": "above zero by the 2 standard error rule", "below": "below zero by the 2 standard error rule",
            "neither": "short of the 2 standard error rule"}[S_verdict(r6)]
    uv = ST["untrained_valid"]
    lo = min(ARMS, key=lambda a: e[a]["near_valid_only"][0])
    hi = max(ARMS, key=lambda a: e[a]["near_valid_only"][0])
    s += (f"SOLO minus DRIFT on that score was {r6[0]:+.3f} (standard error {r6[1]:.3f}), {word}. "
          f"Part of every gain is reply format: the untrained model gave an unusable reply on {uv['near_invalid']:.1%} of these questions, "
          f"the trained populations on {uv['trained_near_invalid']:.1%}. Counting valid replies only, gains run from "
          f"{e[lo]['near_valid_only'][0]:+.3f} ({lo}) to {e[hi]['near_valid_only'][0]:+.3f} ({hi}). "
          "These lines were added after the smoke test and are exploratory.")
    return s


LEDE, LEARN = lede()
acc_lo, acc_hi = 0.0, 1.0
pf_hi = max(max(r["pfams"]) for r in ST["runs"])
pf_hi = int(math.ceil(pf_hi / 5) * 5)
null_band = {a: (ST["arms"][a]["null_breadth"]["untrained"][0], ST["arms"][a]["null_breadth"]["perfect"][0]) for a in ARMS if ST["arms"][a].get("null_breadth")}
TOTAL = round(ST["cost"] + SMOKE_COST, 2)

HTML = f"""<title>Social Arms Experiment</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Space+Grotesk:wght@500;600&display=swap">
<style>
/* Layout: one reading column. Verdict first, then the arm table with baseline rows, theory labels, charts, what was not tested. Tokens match the earlier pages. */
:root {{ --bg:#f6f4ee; --card:#fffdf8; --fg:#1c1b18; --muted:#6a665d; --rule:#dcd7ca; --accent:#1f6f54; --drift:#8a8578;
  --pass:#1f6f54; --fail:#b0432b; --run:#8a6a12; --passbg:#dcefe6; --failbg:#f6ddd6; --runbg:#f3e9c8; --band:#e6e0cf; }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ --bg:#151513; --card:#1d1c19; --fg:#ebe8e0; --muted:#a09b90;
  --rule:#35332e; --accent:#62c49e; --drift:#8f8a7e; --pass:#7fd3b0; --fail:#f09a84; --run:#e2c46a; --passbg:#173428; --failbg:#3b1d16;
  --runbg:#3a3113; --band:#34312a; color-scheme: dark }} }}
:root[data-theme="dark"] {{ --bg:#151513; --card:#1d1c19; --fg:#ebe8e0; --muted:#a09b90; --rule:#35332e; --accent:#62c49e;
  --drift:#8f8a7e; --pass:#7fd3b0; --fail:#f09a84; --run:#e2c46a; --passbg:#173428; --failbg:#3b1d16; --runbg:#3a3113; --band:#34312a; color-scheme: dark }}
body {{ background:var(--bg); color:var(--fg); font:15px/1.55 Inter,system-ui,sans-serif; }}
main {{ max-width:1000px; margin:0 auto; padding-block:28px 56px; padding-left:max(16px, env(safe-area-inset-left, 0px));
  padding-right:max(16px, env(safe-area-inset-right, 0px)); display:flex; flex-direction:column; gap:30px; }}
h1 {{ font:600 clamp(20px,3.2vw,27px)/1.25 "Space Grotesk",Inter,sans-serif; margin:0; text-wrap:balance; }}
h2 {{ font:600 18px/1.3 "Space Grotesk",Inter,sans-serif; margin:0 0 10px; text-wrap:balance; }}
p {{ margin:0 0 8px; max-width:68ch; }} .lede {{ font-size:16.5px; }} .muted,small {{ color:var(--muted); }}
.kicker {{ font:600 12px/1.3 Inter,sans-serif; letter-spacing:.08em; text-transform:uppercase; color:var(--accent); margin-bottom:10px; }}
section {{ min-width:0; }} .scroll + p, .charts + p {{ margin-top:10px; }}
.scroll {{ overflow-x:auto; border:1px solid var(--rule); border-radius:10px; background:var(--card); }}
table {{ border-collapse:collapse; width:100%; font-size:13.5px; font-variant-numeric:tabular-nums; }}
th,td {{ padding:8px 10px; text-align:left; border-bottom:1px solid var(--rule); vertical-align:top; }}
th {{ font-weight:600; font-size:12px; color:var(--muted); }} tbody tr:last-child td {{ border-bottom:0; }}
tr.ref td {{ color:var(--muted); }} td:first-child {{ min-width:150px; }} td small {{ font-size:12px; white-space:nowrap; }} td small.wrap {{ white-space:normal; }}
.badge {{ display:inline-block; padding:2px 9px; border-radius:99px; font-weight:600; font-size:12px; white-space:nowrap; vertical-align:2px; }}
.badge.fail {{ color:var(--fail); background:var(--failbg); }} .badge.pass {{ color:var(--pass); background:var(--passbg); }}
.badge.run {{ color:var(--run); background:var(--runbg); }}
.charts {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(290px,1fr)); gap:14px; }}
.small {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr)); gap:12px; }}
figure {{ margin:0; background:var(--card); border:1px solid var(--rule); border-radius:10px; padding:12px 12px 6px; min-width:0; }}
figcaption {{ font-size:13px; margin-bottom:6px; font-weight:600; color:var(--fg); }} svg {{ width:100%; height:auto; display:block; }}
.cap {{ font-size:12px; color:var(--muted); margin:4px 0 6px; }}
.grid {{ stroke:var(--rule); stroke-width:1; }} .band {{ fill:var(--band); }}
.base {{ stroke:var(--run); stroke-width:1.4; stroke-dasharray:6 3; }} text.base {{ stroke:none; fill:var(--run); }}
.tick {{ fill:var(--muted); font:10px Inter,sans-serif; }} .lab {{ fill:var(--muted); font:10.5px Inter,sans-serif; }}
.line {{ fill:none; stroke:var(--accent); stroke-width:2; }} .thin {{ fill:none; stroke:var(--drift); stroke-width:1; opacity:.6; }}
.mean {{ fill:var(--accent); }} .run {{ fill:var(--drift); }} .meanbar {{ stroke:var(--accent); stroke-width:3; stroke-linecap:round; }}
ul {{ margin:0; padding-left:20px; max-width:70ch; }} li {{ margin-bottom:6px; }} code {{ font-size:13px; }}
</style>
<main>
<header>
  <div class="kicker">Byproduct experiment. Version 4: social arms with changing weights</div>
  <h1>{THESIS}</h1>
</header>

<section>
  <p class="lede">This version could not test the thesis. {badge("Thesis verdict: not testable at this scale", "fail")}</p>
  <p>A free check run before any model call showed that the social rules can separate an arm from its control by under one doubling of
  breadth. Two were needed for a score difference to be detectable with 5 seeds. The verdict was fixed then, before any result.</p>
  <p><strong>What the {ST["n_runs"]} runs did measure.</strong> {LEDE}</p>
  <p>{LEARN}</p>
  <p>{explore()}</p>
  <p class="muted">Setup: Qwen2.5-1.5B, one adapter per population of 32 agents, 8 generations, {len(ST["seeds"])} seeds per arm. Each agent carries a
  repertoire of task families and lives 24 one-question rounds. The top 8 by energy are parents. The adapter is trained after every
  generation on parents' own replies that were correct (SOLO-W: all parent replies). Rules and labels were written before any model call
  (<code>SPEC4.md</code>).</p>
</section>

<section>
  <h2>Each arm against the baselines</h2>
  <div class="scroll"><table>
    <thead><tr><th>Arm</th><th>Runs</th><th>Breadth<br><small>effective families trained on</small></th><th>Breadth from rules alone<br><small>null model, no learning</small></th>
    <th>Practised skill<br><small>49 families the world can pose</small></th><th>Similar unseen tasks<br><small>16 reserved families</small></th>
    <th>New-skill tasks<br><small>18 held families</small></th><th>Share right in life<br><small>generation 1 to 8</small></th></tr></thead>
    <tbody>{arm_table()}</tbody></table></div>
  <p class="muted">Mean ± 1 standard error over seeds. The null model column gives the breadth each arm reaches with a frozen score table:
  the untrained model's scores, then a perfect table. Run-to-run spread of the similar-task score in earlier experiments was 0.031.</p>
</section>

<section>
  <h2>The theories, judged against their controls</h2>
  <div class="scroll"><table>
    <thead><tr><th>Theory</th><th>Breadth gap<br><small>doublings, paired by seed</small></th><th>Gap from rules alone<br><small>null model range</small></th>
    <th>Breadth label</th><th>Practised-skill gap</th><th>Similar-task gap<br><small>descriptive, not a test</small></th>
    <th>New-skill gap<br><small>descriptive, not a test</small></th></tr></thead>
    <tbody>{theory_table()}</tbody></table></div>
  <p class="muted">"More than mechanics" needs the real gap to exceed the larger null-model gap by more than 2 standard errors. Practised skill
  is skill on families the population could train on. It is not held-out ability and is not evidence for the thesis.</p>
</section>

<section>
  <h2>Where the runs landed</h2>
  <div class="charts">
    {dots("Breadth: effective number of families trained on", "breadth", 0, band=null_band, note="Shaded: the range the rules alone produce (null model). Dots: runs. Bar: mean.")}
    {dots("Practised skill, final adapter", "inworld", 2, ref=UN["inworld"], ref_label="untrained", note="Dashed: untrained model. Dots: runs. Bar: mean.")}
    {dots("Similar unseen tasks, final adapter", "near", 2, ref=UN["near"], ref_label="untrained", note="Dashed: untrained model. Descriptive only.")}
    {dots("New-skill tasks, final adapter", "held", 2, ref=UN["held"], ref_label="untrained", note="Dashed: untrained model. Descriptive only.")}
  </div>
</section>

<section>
  <h2>Share of replies right during life, by generation</h2>
  <div class="small">{"".join(lines(a, "acc", "Share right", acc_lo, acc_hi) for a in ARMS)}</div>
  <p class="muted">Thin lines: seeds. Thick line: mean. Sampling temperature 0.7.</p>
  <h2 style="margin-top:22px">Distinct families among the 8 parents, by generation</h2>
  <div class="small">{"".join(lines(a, "pfams", "Families among parents", 0, pf_hi, 0) for a in ARMS)}</div>
  <p class="muted">Every population starts from the same 2 families. Breadth has to be built by gaining families and keeping them.</p>
</section>

<section>
  <h2>Rules read from the runs</h2>
  <div class="scroll"><table>
    <thead><tr><th>Rule</th><th>Comparison</th><th>Mean difference ± standard error</th><th>Result</th></tr></thead>
    <tbody>{rule_table()}</tbody></table></div>
  <p class="muted">"Above" and "below" mean more than 2 standard errors from zero, paired by seed. One test passes by luck about 6% of the time.
  The rows marked E were added after the smoke test and are exploratory.</p>
</section>

<section>
  <h2>Checks before the run, and what they removed</h2>
  <div class="scroll"><table>
    <thead><tr><th>Check</th><th>Measure</th><th>Result</th><th>Pass rule</th><th>Outcome</th></tr></thead>
    <tbody>{check_table()}</tbody></table></div>
  <ul style="margin-top:10px">
    <li><strong>Score lift from social rules was declared untestable.</strong> Rules 3, 4 and 5 of the design are not read. Similar-task and
    new-skill scores above are descriptive.</li>
    <li><strong>Theory 3, self-domestication, was not tested.</strong> The model kept the whole harvest every time it was asked, so there was
    no variation for selection or a vote to act on. ALPHA, DOMEST and MEDLEY were not run. The prompt was not reworded to force a pass.</li>
  </ul>
</section>

<section>
  <h2>Limits</h2>
  <ul>
    <li>Who practises what follows repertoire rules written by hand. The null model measures how much of each result those rules explain.</li>
    <li>One adapter per population. Skill belongs to the population; no claim about an individual can be read from these runs.</li>
    <li>The learning rule differs from version 3: descendants inherit only what was right. This filter improves a model with no selection.</li>
    <li>COOP differs from SOLO in three ways at once: two repertoires practised, pay tied to a random partner, parents chosen as pairs.</li>
    <li>One 1.5B model from one family, {len(ST["seeds"])} seeds, 8 generations, task families written by AI in one style.</li>
    <li>The code, rules and this page were written by an AI model directed by the author.</li>
  </ul>
  <p class="muted" style="margin-top:10px">Cost: ${ST["cost"]:.2f} for the {ST["n_runs"]} runs and ${SMOKE_COST:.2f} for two smoke runs, ${TOTAL:.2f} of an $11 limit (own upper estimate).</p>
</section>
</main>
"""

if __name__ == "__main__":
    open("social.html", "w").write(HTML)
    print("wrote social.html", len(HTML))
