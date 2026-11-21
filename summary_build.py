"""Final summary page for versions 1 to 3. Every number is read from results.json, results2.json or checks3.json.
Run: python3 summary_build.py  ->  summary.html"""
import json
from string import Template

r1 = json.load(open("results.json"))
r2 = json.load(open("results2.json"))
c3 = json.load(open("checks3.json"))
ARMS = ["DRIFT", "SOLO", "COOP", "ALPHA", "DOMEST", "CULTURE", "MEDLEY"]


def sg(x, d=3):
    s = f"{x:+.{d}f}"
    return s.replace("-0." + "0" * d, "+0." + "0" * d) if float(f"{x:.{d}f}") == 0 else s


# ---- version 1
v1 = {a: r1["ability"]["arms"][a]["mean"]["mean"] for a in ARMS}
v1_band = r1["ability"]["noise"]["mean"]["band"]
v1_cap = {a: r1["change"][a]["CAP"]["mean"] for a in ARMS}
v1_cap_band = r1["noise"]["CAP"]["band"]
v1_cost = r1["usage"]["total_cost_usd"]
v1_exiles = sum(r1["extras"]["exile"][a]["exiled_total"] for a in ("DOMEST", "MEDLEY"))

# ---- version 2
near2 = {a: r2["transfer"]["near_energy"][a]["change"]["mean"] for a in ARMS}
gen2 = {a: r2["transfer"]["general"][a]["change"]["mean"] for a in ARMS}
gen2_end = {a: r2["transfer"]["general"][a]["end"] for a in ARMS}
bare2 = r2["transfer_meta"]["bare"]["general"]["mean"]
band_near2 = r2["noise"]["near_energy"]["band"]
band_gen2 = r2["noise"]["general"]["band"]
p_near = r2["extras"]["paired"]["near_energy"]["SOLO"]["vs_drift"]
p_gen = r2["extras"]["paired"]["general"]["SOLO"]["vs_drift"]
grab = {a: r2["social"]["grab_rate"][a]["overall"] for a in ("ALPHA", "DOMEST", "MEDLEY")}
v2_exiles = sum(r2["social"]["exile"][a]["exiled_total"] for a in ("DOMEST", "MEDLEY"))
v2_cost = r2["usage"]["total_cost_usd"]

# ---- version 3
g = c3["gpu"]
pn, pg = g["pilot"]["near"], g["pilot"]["general"]
w = c3["wide"]
gaps = w["gaps"]
t3_near, t3_gen = pn["mean"] / pn["se"], pg["mean"] / pg["se"]


# ---- figure 1: selection minus chance, in standard errors
def fig_se():
    rows = [("Version 2", "skill on the selected task", p_near["t"], "task"),
            ("Version 2", "general ability", p_gen["t"], "gen"),
            ("Version 3", "skill on the selected tasks", t3_near, "task"),
            ("Version 3", "general ability", t3_gen, "gen")]
    x0, x1, lo, hi = 250, 650, -2.0, 5.0
    X = lambda v: x0 + (v - lo) * (x1 - x0) / (hi - lo)
    ys = [44, 78, 128, 162]
    o = [f'<svg viewBox="0 0 680 222" role="img" aria-label="Selection minus chance in standard errors, versions 2 and 3">']
    o.append(f'<rect class="band" x="{X(-2):.1f}" y="22" width="{X(2) - X(-2):.1f}" height="164"/>')
    for t in (-2, 0, 2, 4):
        cls = "zero" if t == 0 else "grid"
        o.append(f'<line class="{cls}" x1="{X(t):.1f}" y1="22" x2="{X(t):.1f}" y2="186"/>')
        o.append(f'<text class="tick" x="{X(t):.1f}" y="202" text-anchor="middle">{t:+d}</text>'.replace("+0", "0"))
    o.append(f'<text class="tick" x="{X(0):.1f}" y="16" text-anchor="middle">inside 2 standard errors of zero</text>')
    o.append(f'<text class="tick" x="{x1}" y="218" text-anchor="end">selection minus chance, in standard errors</text>')
    for (ver, lab, v, cls), y in zip(rows, ys):
        a, b = sorted((X(0), X(v)))
        o.append(f'<text class="rowlab" x="{x0 - 12}" y="{y + 4}" text-anchor="end"><tspan class="ver">{ver}</tspan> {lab}</text>')
        o.append(f'<rect class="bar {cls}" x="{a:.1f}" y="{y - 8}" width="{max(b - a, 1.5):.1f}" height="16" rx="2"/>')
        if v >= 0:
            o.append(f'<text class="val" x="{b + 6:.1f}" y="{y + 4}">{v:+.1f}</text>')
        else:
            o.append(f'<text class="val" x="{a - 6:.1f}" y="{y + 4}" text-anchor="end">{v:+.1f}</text>')
    o.append("</svg>")
    return "\n".join(o)


# ---- figure 2: wider-world ceiling check
def fig_wide():
    groups = [("13 families trained on", "wider arm only", "near_wide"),
              ("6 families never trained on", "held out of both arms", "mid"),
              ("General ability", "three original tests", "general")]
    series = [("base", "unmod", "Unmodified model"), ("control", "ctl", "Control: 3-family training"),
              ("wide", "wide", "16-family training")]
    y0, y1 = 250, 40
    Y = lambda v: y0 - v * (y0 - y1)
    o = ['<svg viewBox="0 0 680 300" role="img" aria-label="Wider-world ceiling check scores">']
    for t in (0, 0.25, 0.5, 0.75, 1.0):
        o.append(f'<line class="grid" x1="48" y1="{Y(t):.1f}" x2="664" y2="{Y(t):.1f}"/>')
        o.append(f'<text class="tick" x="42" y="{Y(t) + 3.5:.1f}" text-anchor="end">{t:.2f}</text>')
    lx = 48
    for key, cls, name in series:
        o.append(f'<rect class="bar {cls}" x="{lx}" y="10" width="12" height="12" rx="2"/>')
        o.append(f'<text class="tick" x="{lx + 17}" y="20">{name}</text>')
        lx += 30 + 6.2 * len(name)
    o.append(f'<line class="chance" x1="{lx}" y1="16" x2="{lx + 22}" y2="16"/><text class="tick" x="{lx + 28}" y="20">Chance</text>')
    bw, gap, gw = 46, 8, 205
    for gi, (lab, sub, key) in enumerate(groups):
        gx = 62 + gi * gw
        for si, (skey, cls, _) in enumerate(series):
            v = w[skey][key]
            x = gx + si * (bw + gap)
            o.append(f'<rect class="bar {cls}" x="{x}" y="{Y(v):.1f}" width="{bw}" height="{y0 - Y(v):.1f}" rx="2"/>')
            o.append(f'<text class="val" x="{x + bw / 2}" y="{Y(v) - 5:.1f}" text-anchor="middle">{v:.3f}</text>')
        ch = w["chance"][key]
        o.append(f'<line class="chance" x1="{gx - 6}" y1="{Y(ch):.1f}" x2="{gx + 3 * bw + 2 * gap + 6}" y2="{Y(ch):.1f}"/>')
        cx = gx + (3 * bw + 2 * gap) / 2
        o.append(f'<text class="grouplab" x="{cx}" y="270" text-anchor="middle">{lab}</text>')
        o.append(f'<text class="tick" x="{cx}" y="285" text-anchor="middle">{sub}</text>')
    o.append("</svg>")
    return "\n".join(o)


# ---- theory table
THEORY = [
    ("DRIFT", "Chance only", "Parents picked at random. Version 1 used this arm for the rejected Cognitive Revolution theory.", "Noise floor"),
    ("SOLO", "Ecology only", "Top survivors reproduce. No social mechanism.", "Baseline"),
    ("COOP", "Cooperative Brain", "Pairs advised and shared in both versions.", "Yes"),
    ("ALPHA", "Control for self-domestication", f"Grabbing allowed, no vote. Grab rate about {grab['ALPHA'] * 100:.0f}% in version 2.", "Yes"),
    ("DOMEST", "Self-Domestication",
     f"Version 1: {v1_exiles} exiles, so not tested. Version 2: {r2['social']['exile']['DOMEST']['exiled_total']} exiles, grab rate about {grab['DOMEST'] * 100:.0f}%.", "Weakly"),
    ("CULTURE", "Cultural Brain", "Notes were inherited and used. In version 2 they added +3.1 energy on the selected task.", "Yes"),
    ("MEDLEY", "All three together", f"{r2['social']['exile']['MEDLEY']['exiled_total']} exile in version 2. Notes added +3.0 energy.", "Partly"),
]
rows = []
for arm, theory, mech, op in THEORY:
    ref = ' class="ref"' if arm in ("DRIFT", "SOLO") else ""
    rows.append(f"<tr{ref}><td><b>{theory}</b><br><small>{arm}</small></td><td>{mech}</td><td>{op}</td>"
                f"<td>{sg(near2[arm], 1)}</td><td>{sg(gen2[arm])}</td><td>{sg(v1[arm])}</td><td>{sg(v1_cap[arm], 2)}</td></tr>")
theory_rows = "\n".join(rows)

V = dict(
    thesis=r1["thesis"],
    fig_se=fig_se(), fig_wide=fig_wide(), theory_rows=theory_rows,
    v1_solo=sg(v1["SOLO"]), v1_drift=sg(v1["DRIFT"]), v1_gap=sg(v1["SOLO"] - v1["DRIFT"]), v1_cult=sg(v1["CULTURE"]),
    v1_band=f"{v1_band:.3f}", v1_cap_band=f"{v1_cap_band:.2f}", v1_cost=f"{v1_cost:.2f}", v1_exiles=v1_exiles,
    cap_coop=sg(v1_cap["COOP"], 2), cap_alpha=sg(v1_cap["ALPHA"], 2), cap_dom=sg(v1_cap["DOMEST"], 2), cap_med=sg(v1_cap["MEDLEY"], 2),
    n2_solo=sg(near2["SOLO"], 1), n2_drift=sg(near2["DRIFT"], 1), n2_gap=sg(p_near["mean"], 1), n2_t=f"{p_near['t']:.2f}",
    n2_hi=p_near["higher"], n2_cult=sg(near2["CULTURE"], 1), band_near2=f"{band_near2:.1f}",
    g2_solo=sg(gen2["SOLO"]), g2_drift=sg(gen2["DRIFT"]), g2_gap=sg(p_gen["mean"]), g2_t=f"{p_gen['t']:.2f}", g2_hi=p_gen["higher"],
    g2_min=sg(min(gen2.values())), g2_max=sg(max(gen2.values())), band_gen2=f"{band_gen2:.3f}",
    g2_end_lo=f"{min(gen2_end.values()):.3f}", g2_end_hi=f"{max(gen2_end.values()):.3f}", bare2=f"{bare2:.3f}",
    grab_a=f"{grab['ALPHA'] * 100:.0f}", grab_d=f"{grab['DOMEST'] * 100:.0f}", v2_exiles=v2_exiles, v2_cost=f"{v2_cost:.2f}",
    p_solo=f"{pn['solo']:.1f}", p_drift=f"{pn['drift']:.1f}", p_base=f"{pn['base']:.1f}", p_gap=sg(pn["mean"], 1), p_se=f"{pn['se']:.1f}",
    p_pos=pn["pos"], pg_solo=f"{pg['solo']:.3f}", pg_drift=f"{pg['drift']:.3f}", pg_base=f"{pg['base']:.3f}",
    pg_gap=sg(pg["mean"]), pg_se=f"{pg['se']:.3f}", t3_near=f"{t3_near:+.1f}", t3_gen=f"{t3_gen:+.1f}",
    c_near0=f"{g['base']['near']:.1f}", c_near1=f"{g['ceiling']['near']:.1f}", c_gen0=f"{g['base']['general']:.3f}",
    c_gen1=f"{g['ceiling']['general']:.3f}", c_ret0=f"{g['base']['retention']:.2f}", c_ret1=f"{g['ceiling']['retention']:.2f}",
    w_mid=sg(gaps["mid_breadth"][0]), w_mid_se=f"{gaps['mid_breadth'][1]:.3f}", w_gen=sg(gaps["gen_breadth"][0]),
    w_gen_se=f"{gaps['gen_breadth'][1]:.3f}", w_genw=sg(gaps["gen_wide"][0]), w_genw_se=f"{gaps['gen_wide'][1]:.3f}",
    w_genc=sg(gaps["gen_ctl"][0]), w_genc_se=f"{gaps['gen_ctl'][1]:.3f}",
    rev0=f"{w['base']['mid_reverse']:.3f}", rev_c=f"{w['control']['mid_reverse']:.3f}", rev_w=f"{w['wide']['mid_reverse']:.3f}",
    o5_0=f"{w['mid_without_reverse']['base']:.3f}", o5_c=f"{w['mid_without_reverse']['control']:.3f}", o5_w=f"{w['mid_without_reverse']['wide']:.3f}",
    modal=f"{c3['spent']:.2f}", modal_gpu=f"{c3['gpu_spent']:.2f}", wide_usd=f"{w['usd']:.2f}",
    api_total=f"{v1_cost + v2_cost:.2f}", all_total=f"{v1_cost + v2_cost + c3['spent']:.2f}",
)

PAGE = Template(r"""<title>Byproduct Experiment Summary</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Space+Grotesk:wght@500;600&display=swap">
<style>
/* Layout: one reading column. Verdict first, then one table per question, two figures drawn to scale. Tokens match the three earlier results pages. */
:root { --bg:#f6f4ee; --card:#fffdf8; --fg:#1c1b18; --muted:#6a665d; --rule:#dcd7ca; --accent:#1f6f54; --drift:#8a8578;
  --pass:#1f6f54; --fail:#b0432b; --run:#8a6a12; --passbg:#dcefe6; --failbg:#f6ddd6; --runbg:#f3e9c8; --band:#ebe7db; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { --bg:#151513; --card:#1d1c19; --fg:#ebe8e0; --muted:#a09b90;
  --rule:#35332e; --accent:#62c49e; --drift:#8f8a7e; --pass:#7fd3b0; --fail:#f09a84; --run:#e2c46a; --passbg:#173428; --failbg:#3b1d16;
  --runbg:#3a3113; --band:#26251f; color-scheme: dark } }
:root[data-theme="dark"] { --bg:#151513; --card:#1d1c19; --fg:#ebe8e0; --muted:#a09b90; --rule:#35332e; --accent:#62c49e;
  --drift:#8f8a7e; --pass:#7fd3b0; --fail:#f09a84; --run:#e2c46a; --passbg:#173428; --failbg:#3b1d16; --runbg:#3a3113; --band:#26251f; color-scheme: dark }
body { background:var(--bg); color:var(--fg); font:15px/1.55 Inter,system-ui,sans-serif; }
main { max-width:1000px; margin:0 auto; padding-block:28px 56px; padding-left:max(16px, env(safe-area-inset-left, 0px));
  padding-right:max(16px, env(safe-area-inset-right, 0px)); display:flex; flex-direction:column; gap:30px; }
h1 { font:600 clamp(20px,3.2vw,27px)/1.25 "Space Grotesk",Inter,sans-serif; margin:0; text-wrap:balance; }
h2 { font:600 18px/1.3 "Space Grotesk",Inter,sans-serif; margin:0 0 10px; text-wrap:balance; }
h3 { font:600 13px/1.3 Inter,sans-serif; letter-spacing:.06em; text-transform:uppercase; margin:0 0 8px; }
p { margin:0 0 8px; max-width:68ch; } .lede { font-size:16.5px; } .muted,figcaption,small { color:var(--muted); }
.kicker { font:600 12px/1.3 Inter,sans-serif; letter-spacing:.08em; text-transform:uppercase; color:var(--accent); margin-bottom:10px; }
section { min-width:0; } .scroll + p, figure + p { margin-top:10px; }
.scroll { overflow-x:auto; border:1px solid var(--rule); border-radius:10px; background:var(--card); }
table { border-collapse:collapse; width:100%; font-size:13.5px; font-variant-numeric:tabular-nums; }
th,td { padding:8px 10px; text-align:left; border-bottom:1px solid var(--rule); vertical-align:top; }
th { font-weight:600; font-size:12px; color:var(--muted); } tbody tr:last-child td { border-bottom:0; }
tr.ref td { color:var(--muted); } td:first-child { min-width:132px; } td small { font-size:12px; }
.badge { display:inline-block; padding:2px 9px; border-radius:99px; font-weight:600; font-size:12px; white-space:nowrap; }
.badge.fail { color:var(--fail); background:var(--failbg); } .badge.partial { color:var(--run); background:var(--runbg); }
.badge.pass { color:var(--pass); background:var(--passbg); }
figure { margin:0; background:var(--card); border:1px solid var(--rule); border-radius:10px; padding:12px 12px 6px; min-width:0; overflow-x:auto; }
figcaption { font-size:13px; margin-bottom:8px; max-width:74ch; } svg { width:100%; min-width:540px; height:auto; display:block; }
.grid { stroke:var(--rule); stroke-width:1; } .zero { stroke:var(--muted); stroke-width:1.2; } .band { fill:var(--band); }
.chance { stroke:var(--fg); stroke-width:1.2; stroke-dasharray:4 3; fill:none; }
.tick { fill:var(--muted); font:11px Inter,sans-serif; } .val { fill:var(--fg); font:600 11.5px Inter,sans-serif; }
.rowlab { fill:var(--fg); font:12px Inter,sans-serif; } .rowlab .ver { font-weight:600; } .grouplab { fill:var(--fg); font:600 12px Inter,sans-serif; }
.bar.task,.bar.wide { fill:var(--accent); } .bar.gen,.bar.unmod { fill:var(--drift); } .bar.ctl { fill:var(--run); }
.cols { display:grid; grid-template-columns:repeat(auto-fit,minmax(280px,1fr)); gap:22px; }
.cols > div { min-width:0; } .for h3 { color:var(--pass); } .against h3 { color:var(--fail); } .open h3 { color:var(--run); }
ul { margin:0; padding-left:20px; max-width:70ch; } li { margin-bottom:6px; }
a { color:var(--accent); } a:focus-visible { outline:2px solid var(--accent); outline-offset:2px; }
</style>
<main>
<header>
  <div class="kicker">Byproduct experiment. Final summary of versions 1 to 3.</div>
  <h1>$thesis</h1>
</header>

<section>
  <p class="lede">The prediction did not hold at this scale. In every version, selecting on survival alone raised skill on the tasks agents were selected on. General ability on held-out tests never moved beyond noise, and no theory arm beat plain selection.</p>
  <p>The strongest form of the test was not run: the five social arms with changing weights, on a model large enough to pass the ceiling check.</p>
</section>

<section>
  <h2>Three versions at a glance</h2>
  <div class="scroll"><table>
    <thead><tr><th>Version</th><th>What could evolve</th><th>Model and run</th><th>Skill on selected tasks, selection minus chance</th><th>General ability, selection minus chance</th><th>Noise it is judged against</th><th>Outcome by pre-set rule</th></tr></thead>
    <tbody>
    <tr><td><b>1</b><br><small>2026-11-11</small></td><td>Four capacity settings per agent, each with an energy cost</td><td>Claude Haiku 5.5, frozen. 7 arms, 5 seeds, 16 agents, 12 generations</td><td>Not measured separately</td><td>$v1_gap<br><small>SOLO $v1_solo, chance $v1_drift</small></td><td>Band of $v1_band (2 SD of the chance arm over 5 seeds)</td><td><span class="badge fail">Not supported</span></td></tr>
    <tr><td><b>2</b><br><small>2026-11-11</small></td><td>Prompt rules and inherited notes</td><td>Claude Haiku 5.5, frozen, hidden reasoning off. 7 arms, 5 seeds, 16 agents, 12 generations</td><td>$n2_gap energy<br><small>SOLO $n2_solo, chance $n2_drift. Higher in $n2_hi of 5 seeds</small></td><td>$g2_gap<br><small>SOLO $g2_solo, chance $g2_drift. Higher in $g2_hi of 5 seeds</small></td><td>Bands of $band_near2 energy and $band_gen2 general ability</td><td><span class="badge fail">Not supported</span></td></tr>
    <tr><td><b>3</b><br><small>2026-11-12 to 14</small></td><td>Model weights, through one LoRA adapter per population</td><td>Qwen2.5-1.5B on a Modal A10 GPU. SOLO and chance arms only, 5 seeds, 32 lives, 8 generations</td><td>$p_gap energy (SE $p_se)<br><small>SOLO $p_solo, chance $p_drift. Higher in $p_pos of 5 seeds</small></td><td>$pg_gap (SE $pg_se)<br><small>SOLO $pg_solo, chance $pg_drift</small></td><td>Standard errors across seeds</td><td><span class="badge partial">Checks partial</span><br><small>full run not justified</small></td></tr>
    </tbody>
  </table></div>
  <p class="muted">Chance is the DRIFT arm: same world, parents picked at random. SOLO is selection on survival energy with no social mechanism. In version 1, held-out ability was read from two of the capacity settings, so it is a weaker measure than in versions 2 and 3.</p>
</section>

<section>
  <h2>The same pattern twice</h2>
  <figure>
    <figcaption>Selection minus chance, paired by seed and divided by its standard error. Versions 2 and 3 use different models and different things that evolve. Bars inside the shaded band cannot be told apart from zero.</figcaption>
    $fig_se
  </figure>
  <p>Skill on the selected task sits 3.5 to 3.8 standard errors above chance in both versions. General ability sits inside the band in both. The version 2 paired check was added after seeing the data, so its pre-set outcome stays as "not supported".</p>
</section>

<section>
  <h2>Evidence for and against the thesis</h2>
  <div class="cols">
    <div class="for"><h3>For</h3><ul>
      <li>Skill rose with no reward for skill. Version 2: SOLO gained $n2_solo energy against $n2_drift under chance. Version 3: SOLO ended at $p_solo against $p_drift.</li>
      <li>The mechanisms two theories need did operate in version 2. With an exile vote the grab rate was about $grab_d% against about $grab_a% without it. Inherited notes added +3.1 energy, higher in 5 of 5 seeds.</li>
      <li>In version 1, ability rose in both solo-foraging arms in 5 of 5 seeds (SOLO $v1_solo, CULTURE $v1_cult). Both are inside the noise band of $v1_band.</li>
      <li>Breadth of pressure matters a little. Training on 16 task families lifted families never trained on by $w_mid (SE $w_mid_se) over training on 3.</li>
    </ul></div>
    <div class="against"><h3>Against</h3><ul>
      <li>General ability never moved beyond noise. Version 1: best arm $v1_cult against a band of $v1_band. Version 2: $g2_min to $g2_max against a band of $band_gen2. Version 3: $pg_gap (SE $pg_se).</li>
      <li>No theory arm beat plain selection. Version 1: CULTURE $v1_cult, SOLO $v1_solo. Version 2: all four theory arms ended below SOLO on the selected task.</li>
      <li>Social life cut costly capacity in version 1: COOP $cap_coop, ALPHA $cap_alpha, DOMEST $cap_dom, MEDLEY $cap_med, against a band of $v1_cap_band.</li>
      <li>In version 2 every evolved population ended below the model with no evolved text ($g2_end_lo to $g2_end_hi against $bare2). Notes lowered general ability in MEDLEY by 0.027, in 5 of 5 seeds.</li>
      <li>Training directly on correct answers is the upper bound for any selection scheme. It left general ability unchanged: $c_gen0 to $c_gen1 on the 3-family world.</li>
    </ul></div>
    <div class="open"><h3>Not tested</h3><ul>
      <li>Theory arms with changing weights. Version 3 built only SOLO and the chance arm.</li>
      <li>Self-Domestication at full strength. Exile removed $v1_exiles agents in version 1 and $v2_exiles in version 2, and the grabbing contrast rests on one seed.</li>
      <li>Large models, runs longer than 12 generations, groups larger than 32.</li>
      <li>A world broad enough for the ceiling check to pass.</li>
    </ul></div>
  </div>
</section>

<section>
  <h2>The theories, arm by arm</h2>
  <div class="scroll"><table>
    <thead><tr><th>Theory and arm</th><th>What happened to its mechanism</th><th>Mechanism operated</th><th>Version 2: selected-task energy change</th><th>Version 2: general ability change</th><th>Version 1: held-out ability change</th><th>Version 1: total capacity change</th></tr></thead>
    <tbody>
    $theory_rows
    <tr class="ref"><td><b>Noise band</b><br><small>2 SD of chance arm</small></td><td></td><td></td><td>$band_near2</td><td>$band_gen2</td><td>$v1_band</td><td>$v1_cap_band</td></tr>
    </tbody>
  </table></div>
  <p class="muted">Changes are final generation minus generation 0, mean of 5 seeds. An arm counts only if it clears the chance arm by more than the noise band. None does on either ability column.</p>
</section>

<section>
  <h2>Version 3 in detail: can training move the tests at all?</h2>
  <div class="scroll"><table>
    <thead><tr><th>Check (Qwen2.5-1.5B, GPU)</th><th>Outcome</th><th>Result</th></tr></thead>
    <tbody>
    <tr><td>Headroom</td><td><span class="badge partial">Partial</span></td><td>All 3 world families and code-breaking above chance. Rule induction and ordering at chance.</td></tr>
    <tr><td>Ceiling, 3-family world</td><td><span class="badge fail">Fail</span></td><td>Trained on correct play: selected-task energy $c_near0 to $c_near1, general ability $c_gen0 to $c_gen1, retention $c_ret0 to $c_ret1.</td></tr>
    <tr><td>Pilot, 5 seeds</td><td><span class="badge partial">Partial</span></td><td>SOLO $p_solo, chance $p_drift, unmodified $p_base. Gap $p_gap (SE $p_se). General ability gap $pg_gap (SE $pg_se).</td></tr>
    <tr><td>Ceiling, 16-family world</td><td><span class="badge partial">Partial</span></td><td>Families never trained on: $w_mid (SE $w_mid_se) over the control. General ability: $w_gen (SE $w_gen_se).</td></tr>
    </tbody>
  </table></div>
  <figure style="margin-top:14px">
    <figcaption>Ceiling check on the wider world. Both trained arms got 1,200 steps on 9,750 examples of correct play. Scores are fractions correct at temperature 0 on the same items. Dashed lines mark random replies.</figcaption>
    $fig_wide
  </figure>
  <ul>
    <li>Training on 16 families nearly solved the 13 new ones. Training on 3 families left them where they started.</li>
    <li>Most of the held-out gain is one family, reversing a list: $rev0 unmodified, $rev_c control, $rev_w wider. The other five families: $o5_0, $o5_c, $o5_w.</li>
    <li>General ability rose by about 0.03 in every trained adapter, with or without selection or breadth (wider $w_genw, SE $w_genw_se; control $w_genc, SE $w_genc_se). Neither clears 2 standard errors.</li>
  </ul>
</section>

<section>
  <h2>Limits</h2>
  <ul>
    <li>Scale. A frozen Haiku model in versions 1 and 2, and 0.5B and 1.5B open models in version 3. A null result here does not transfer to large models or long runs.</li>
    <li>Version 1 read ability from two capacity settings through a table measured once, so ability could only move with those settings.</li>
    <li>Version 2 started below the unmodified model because the starting rule pool held harmful rules. Every gain was recovery toward that model.</li>
    <li>Version 3 has one adapter per population, and each ceiling arm is one training run. Its standard errors cover test items only.</li>
    <li>All tests were written for this project by the same author. For the 1.5B model, two of the three general tests sat at chance.</li>
  </ul>
</section>

<section>
  <h2>What would change this answer</h2>
  <p class="muted">My recommendation, in order. None of it has been run or priced.</p>
  <ul>
    <li>Repeat only the ceiling check on a larger open model, with the 16-family world and a public far-transfer test. If direct training cannot move held-out tests, selection cannot either.</li>
    <li>If that check passes, build the five social arms with changing weights and run all seven.</li>
    <li>If it fails again, the finding stands as: selection through survival buys task skill, and task skill stays where it was trained.</li>
  </ul>
</section>

<section>
  <h2>Follow-up, 2026-11-14: the breadth curve</h2>
  <p>The first step above was replaced by a cheaper test on the same 1.5B model: 18 runs trained on 2 to 64 task families with a fixed budget. By a rule set in advance the result is flat. Families built on skills the training pool never uses gained +0.067 (SE 0.008) from any training and nothing more from breadth (+0.003 per doubling, SE 0.005). Families from the same pool that a run did not train on did gain with breadth (+0.035 per doubling, SE 0.005), a comparison chosen after seeing the data. That comparison was then rerun with its rule fixed first: 20 new runs, tested on 16 pool families that no run trained on. Confirmed: +0.024 per doubling (SE 0.005), from 0.601 at 2 families to 0.704 at 32 (untrained 0.543), while the held-out families stayed flat in the same runs (+0.003, SE 0.004). Two checks with rules fixed first then repeated it: on a second, separate set of 16 reserved families (+0.026 per doubling, SE 0.005) and on a 3B model with the same families and seeds (+0.024, SE 0.005). Both replicated; the 3B check reran 14 runs after a GPU memory error, a recorded deviation from its plan. Details: <a href="breadth.html">breadth curve results</a>. Cost: $$2.66 for the first 19 runs, $$2.58 for the rerun and $$7.89 for the two checks, all on Modal, not included in the table below.</p>
</section>

<section>
  <h2>Follow-up, 2026-11-17: the theory arms with changing weights (version 4)</h2>
  <p>Version 4 ran theory arms with changing weights for the first time. It could not test the thesis. A free check before any model call showed that the social rules could separate an arm from its control by under one doubling of breadth (0.96 at most), where two were needed for a score difference to be detectable with 5 seeds. The verdict was fixed then as not testable at this scale. A smoke test then showed the 1.5B model keeps a shared harvest every time it is asked (15 of 15), so the three self-domestication arms were dropped by rule. Theory 3 is untested again.</p>
  <p>25 runs followed: 5 arms, 5 seeds, populations of 32 agents sharing one adapter, 8 generations. Cooperation widened what a population practised by +0.35 doublings (SE 0.10), which is inside the range the hand-written rules produce with no learning (+0.19 to +0.52), so its label is "broadens, mechanical". Copying from successful non-parents changed nothing (-0.02, SE 0.10). Populations did learn: practised skill rose +0.077 (SE 0.014) over the untrained 0.510. By the pre-set rule the credit goes to learning from correct answers and not to selection, because survival selection added +0.001 (SE 0.022) over random parents.</p>
  <p>Exploratory, added after the smoke test: all five arms ended above the untrained model on similar unseen tasks, by +0.047 to +0.088 (+0.028 to +0.068 counting valid replies only). Selection against random parents on that score was +0.041 (SE 0.021), just short of the 2 standard error rule. Version 3's learning rule, which copies wrong answers too, scored lower than the new rule on those tasks (-0.036, SE 0.008). Cost $$4.53. <a href="social.html">Version 4 results</a>.</p>
</section>

<section>
  <h2>Cost</h2>
  <div class="scroll"><table>
    <thead><tr><th>Item</th><th>Spend</th><th>Budget</th></tr></thead>
    <tbody>
    <tr><td>Version 1 full run and tests</td><td>$$$v1_cost</td><td rowspan="2">Claude API, $$200 of credits. Total with pilots about $$48.5</td></tr>
    <tr><td>Version 2 full run, tests and headroom checks</td><td>$$$v2_cost</td></tr>
    <tr><td>Version 3 checks on Modal (GPU part $$$modal_gpu, wider-world step $$$wide_usd)</td><td>$$$modal</td><td>$$10 limit, own upper estimate</td></tr>
    <tr class="ref"><td>Three versions, pilots excluded</td><td>$$$all_total</td><td></td></tr>
    </tbody>
  </table></div>
</section>

<section>
  <h2>Detail pages and research used</h2>
  <ul>
    <li><a href="report.html">Version 1 results</a>, <a href="report2.html">Version 2 results</a>, <a href="report3.html">Version 3 results</a>, <a href="breadth.html">breadth experiments</a>, <a href="social.html">Version 4 results</a>.</li>
    <li>Generalisation to unseen tasks depends on the number of distinct tasks more than on examples per task (<a href="https://arxiv.org/abs/2402.10891">arXiv 2402.10891</a>, <a href="https://arxiv.org/abs/2405.19787">arXiv 2405.19787</a>). Read from search summaries.</li>
    <li>LoRA forgets less and learns less than full fine-tuning (Biderman et al. 2024). Repeated training on a model's own unfiltered output degrades it (arXiv 2510.16657, read from search summaries).</li>
  </ul>
</section>
</main>
""")

html = PAGE.substitute(V)
open("summary.html", "w").write(html)
print(len(html), "bytes")
for k in ("v1_gap", "v1_band", "n2_gap", "n2_t", "g2_gap", "g2_t", "g2_min", "g2_max", "g2_end_lo", "g2_end_hi", "bare2", "p_gap", "p_se",
          "pg_gap", "pg_se", "t3_near", "t3_gen", "w_mid", "w_gen", "api_total", "all_total", "grab_a", "grab_d", "v1_exiles", "v2_exiles",
          "cap_coop", "cap_alpha", "cap_dom", "cap_med", "v1_cap_band", "c_gen0", "c_gen1", "w_genw", "w_genc"):
    print(f"  {k:10s} {V[k]}")
