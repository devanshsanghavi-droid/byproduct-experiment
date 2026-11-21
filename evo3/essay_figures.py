"""Standalone figures for the public write-up. Reads runs3/breadth_stats.json, runs3/near_stats.json, checks3.json.

    PYTHONPATH=. python3 -m evo3.essay_figures   -> figures/fig1_ceiling.png, fig2_breadth_flat.png, fig3_near_far.png
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

S, N, C = (json.load(open(f)) for f in ("runs3/breadth_stats.json", "runs3/near_stats.json", "checks3.json"))
BG, INK, INK2, GRID, BLUE, ORANGE = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6", "#eb6834"
plt.rcParams.update({"font.family": "Helvetica Neue", "font.size": 11, "text.color": INK, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "figure.facecolor": BG, "axes.facecolor": BG,
                     "savefig.facecolor": BG, "axes.edgecolor": GRID})


def style(ax):
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def head(fig, title, sub):
    fig.text(0.02, 0.965, title, fontsize=14, fontweight="bold", va="top")
    fig.text(0.02, 0.895, sub, fontsize=10.5, color=INK2, va="top")


def kaxis(ax, ks):
    ax.set_xscale("log", base=2)
    ax.set_xticks(ks)
    ax.set_xticklabels([str(k) for k in ks])
    ax.minorticks_off()
    ax.set_xlabel("Task families trained on (same training budget at every level)")


def fig1():
    b, c = C["gpu"]["base"], C["gpu"]["ceiling"]
    fig, axes = plt.subplots(1, 2, figsize=(8, 4.3))
    head(fig, "Training on correct answers lifts the trained skill and nothing else",
         "Qwen2.5-1.5B before and after 1,200 training steps on correct play in a 3-task world")
    for ax, key, title, fmt, top in ((axes[0], "near", "Trained skill: food energy in the world", "{:.1f}", 80),
                                     (axes[1], "general", "General ability: 3 held-out tests", "{:.3f}", 0.5)):
        vals = [b[key], c[key]]
        bars = ax.bar(["Untrained", "Trained"], vals, width=0.5, color=BLUE)
        for r, v in zip(bars, vals):
            ax.text(r.get_x() + r.get_width() / 2, v + top * 0.02, fmt.format(v), ha="center", fontsize=11)
        ax.set_ylim(0, top)
        ax.set_title(title, fontsize=11, loc="left", color=INK2)
        style(ax)
    axes[1].axhline(0.281, color=INK2, lw=1, ls=(0, (3, 3)))
    axes[1].set_xlim(-0.45, 2.15)
    axes[1].text(1.32, 0.281 + 0.008, "chance 0.281", ha="left", fontsize=9.5, color=INK2)
    fig.subplots_adjust(top=0.74, bottom=0.09, left=0.07, right=0.98, wspace=0.25)
    fig.savefig("figures/fig1_ceiling.png", dpi=200)


def fig2():
    ks = sorted(int(k) for k in S["levels"])
    fig, ax = plt.subplots(figsize=(8, 4.5))
    f = S["fit"]["held"]
    head(fig, "More kinds of practice did not help on tasks that need a new skill",
         f"Score on 18 held-out task families. Change per doubling: {f['slope']:+.3f} (standard error {f['se']:.3f}). 18 runs, Qwen2.5-1.5B")
    runs = [(r["k"], r["held"]) for r in S["runs"] if r["k"] > 0]
    ax.scatter([k for k, _ in runs], [v for _, v in runs], s=22, color=BLUE, alpha=0.35, linewidths=0, label="single run")
    means = [S["levels"][str(k)]["held"]["mean"] for k in ks]
    ax.plot(ks, means, color=BLUE, lw=2, marker="o", ms=7, mec=BG, mew=2, label="mean of 3 runs")
    for y, lab in ((S["base"]["held"], "untrained model"), (S["chance"]["held"], "chance")):
        ax.axhline(y, color=INK2, lw=1, ls=(0, (3, 3)))
        ax.text(ks[-1] * 1.12, y + 0.008, f"{lab} {y:.3f}", ha="right", fontsize=9.5, color=INK2)
    ax.set_ylim(0.15, 0.6)
    ax.set_xlim(ks[0] / 1.2, ks[-1] * 1.15)
    ax.set_ylabel("Fraction correct")
    kaxis(ax, ks)
    style(ax)
    ax.legend(frameon=False, loc="upper left", fontsize=9.5, ncol=2)
    fig.subplots_adjust(top=0.8, bottom=0.13, left=0.09, right=0.98)
    fig.savefig("figures/fig2_breadth_flat.png", dpi=200)


def fig3():
    ks = sorted(int(k) for k in N["levels"])
    fig, ax = plt.subplots(figsize=(8, 5.1))
    a, h = N["fit"]["near"], N["fit"]["held"]
    head(fig, "Broader training helps on similar unseen tasks, not on new skills",
         f"Change per doubling of families: similar tasks {a['slope']:+.3f} (standard error {a['se']:.3f}), new-skill tasks {h['slope']:+.3f} ({h['se']:.3f}).\n"
         "20 runs on Qwen2.5-1.5B, rule fixed in advance.")
    for key, col, lab, dy in (("near", BLUE, "Similar tasks: 16 reserved families, never trained on", 0.03),
                              ("held", ORANGE, "New-skill tasks: 18 held-out families", -0.045)):
        runs = [(r["k"], r[key]) for r in N["runs"] if r["k"] > 0]
        ax.scatter([k for k, _ in runs], [v for _, v in runs], s=22, color=col, alpha=0.35, linewidths=0)
        means = [N["levels"][str(k)][key]["mean"] for k in ks]
        ax.plot(ks, means, color=col, lw=2, marker="o", ms=7, mec=BG, mew=2, label=lab)
        ax.text(ks[-1] * 1.07, means[-1], f"{means[-1]:.3f}", va="center", fontsize=10.5)
        base = N["base"][key]
        ax.plot([ks[0] / 1.5, ks[0] / 1.5], [base, base], marker="D", ms=7, color=col, mec=BG, mew=1.5, ls="none")
        ax.plot([ks[0] / 1.5, ks[0]], [base, means[0]], color=col, lw=1.2, ls=(0, (3, 3)))
        ax.text(ks[0] / 1.5, base + dy, f"{base:.3f}", ha="center", va="center", fontsize=10.5)
    ax.plot([], [], marker="D", ms=7, color=INK2, mec=BG, ls="none", label="untrained model")
    ax.scatter([], [], s=22, color=INK2, alpha=0.4, linewidths=0, label="single run (lines join means of 4)")
    ax.set_ylim(0.3, 0.8)
    ax.set_xlim(ks[0] / 1.8, ks[-1] * 1.3)
    ax.set_ylabel("Fraction correct")
    kaxis(ax, ks)
    ax.set_xticks([ks[0] / 1.5] + ks)
    ax.set_xticklabels(["0"] + [str(k) for k in ks])
    style(ax)
    ax.legend(frameon=False, loc="upper left", fontsize=9.5, ncol=2, columnspacing=1.2)
    fig.subplots_adjust(top=0.78, bottom=0.115, left=0.09, right=0.98)
    fig.savefig("figures/fig3_near_far.png", dpi=200)


def fig4():
    """The confirmed result and its two checks, as three panels on one shared scale."""
    import os
    sets = [("Confirmed run\n1.5B model, reserved set 1", N)]
    for title, path in (("Check A: second reserved set\n1.5B model, reserved set 2", "runs3/n2_stats.json"),
                        ("Check B: larger model\n3B model, reserved set 1", "runs3/m3_stats.json")):
        if os.path.exists(path):
            sets.append((title, json.load(open(path))))
    if len(sets) < 2:
        return
    fig, axes = plt.subplots(1, len(sets), figsize=(10.5, 4.9), sharey=True)
    head(fig, "The result holds on a second set of tasks and on a model twice the size"
         if len(sets) == 3 and all(d["rule1"] for _, d in sets) else "The confirmed result and its checks",
         "Blue: similar tasks never trained on. Orange: new-skill tasks. Change per doubling of families is printed in each panel.")
    for ax, (title, D) in zip(axes, sets):
        ks = sorted(int(k) for k in D["levels"])
        for key, col in (("near", BLUE), ("held", ORANGE)):
            runs = [(r["k"], r[key]) for r in D["runs"] if r["k"] > 0]
            ax.scatter([k for k, _ in runs], [v for _, v in runs], s=18, color=col, alpha=0.35, linewidths=0)
            means = [D["levels"][str(k)][key]["mean"] for k in ks]
            ax.plot(ks, means, color=col, lw=2, marker="o", ms=6, mec=BG, mew=1.5)
            base = D["base"][key]
            ax.plot([ks[0] / 1.6], [base], marker="D", ms=6, color=col, mec=BG, mew=1.2, ls="none")
            ax.plot([ks[0] / 1.6, ks[0]], [base, means[0]], color=col, lw=1.1, ls=(0, (3, 3)))
        a, h = D["fit"]["near"], D["fit"]["held"]
        ax.text(0.03, 0.97, f"similar {a['slope']:+.3f} (standard error {a['se']:.3f})\nnew-skill {h['slope']:+.3f} ({h['se']:.3f})",
                transform=ax.transAxes, va="top", fontsize=9.5, color=INK2)
        ax.set_title(title, fontsize=10.5, loc="left", color=INK)
        ax.set_xscale("log", base=2)
        ax.set_xticks([ks[0] / 1.6] + ks)
        ax.set_xticklabels(["0"] + [str(k) for k in ks])
        ax.minorticks_off()
        ax.set_xlim(ks[0] / 2, ks[-1] * 1.25)
        style(ax)
    lo = min(min(r[k] for r in D["runs"] for k in ("near", "held")) for _, D in sets) - 0.03
    hi = max(max(r[k] for r in D["runs"] for k in ("near", "held")) for _, D in sets)
    axes[0].set_ylim(int(lo * 10) / 10, int(hi * 10 + 2) / 10)
    axes[0].set_ylabel("Fraction correct")
    fig.text(0.5, 0.02, "Task families trained on (0 = untrained model; same training budget at every level)", ha="center", fontsize=10, color=INK2)
    fig.subplots_adjust(top=0.74, bottom=0.13, left=0.07, right=0.985, wspace=0.08)
    fig.savefig("figures/fig4_checks.png", dpi=200)


if __name__ == "__main__":
    fig1(), fig2(), fig3(), fig4()
    print("wrote figures/fig1_ceiling.png, fig2_breadth_flat.png, fig3_near_far.png, fig4_checks.png (if the checks exist)")
