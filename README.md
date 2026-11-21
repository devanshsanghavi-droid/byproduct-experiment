# Byproduct experiment

> All three surviving theories say intelligence was a byproduct. A faithful simulation
> therefore scores agents only on staying alive and being copied, then measures general
> ability separately on held-out tests.

This repository holds the code, the written designs and every stored result of a small
experiment built on that thesis. The question: can evolutionary or social pressure make a
language model more capable in general?

**Short answer from these runs: no general gain from evolutionary or social pressure was found.**
Selection raised skill on the tasks it selected for. Broader training raised scores on similar
unseen tasks. In the breadth experiments, tasks that need a new skill stayed flat as breadth grew.
The one version that ran theory arms with changing weights could not test the thesis at its scale.

## Disclosure

The code, the design documents (`SPEC*.md`), the analysis and this README were written by
Claude (Anthropic's AI model) in Claude Code, directed by the repository author. The author
chose the question, approved each run and its budget, and is responsible for the claims.
The task families in version 3 were also written by AI.

The design documents say each rule was written before its results. They were written in
the same working sessions as the runs. This repository has no commit history from that
time, so that ordering cannot be checked from git.

## Results

Scores are the share of test items answered correctly, at temperature 0. "SE" is one
standard error.

### Versions 1 and 2: frozen model, evolving settings or text (Claude Haiku 5.5)

| Version | What evolved | Runs | General ability | Verdict by the pre-set rule |
|---|---|---|---|---|
| 1 | Capacity dials on a frozen model | 7 arms x 5 seeds, 12 generations | No arm beyond the chance-only noise band | Not supported |
| 2 | Prompts and notes | 7 arms x 5 seeds, 12 generations | 0.49 to 0.52 in every arm, bare model 0.554 | Not supported |

In version 2, selection did remove harmful starting rules (40% of agents to 6% under plain
selection, 41% under chance). No theory arm beat plain selection on its own.

### Version 3: weights change (Qwen2.5, LoRA rank 8)

| Check | Result |
|---|---|
| Ceiling, 1.5B model: train directly on correct play | In-world score 21.9 to 68.6. General ability 0.325 to 0.324. |
| Pilot, 5 seeds: selection against random copying | In-world score +8.4 (SE 2.4). General ability gap -0.005. |
| Wider world, 3 to 16 task families | Held-out families +0.054 (SE 0.016). Three far tests +0.000 (SE 0.018). |

The seven-arm run with changing weights was never run, because the ceiling check failed.

### Breadth experiments: does training on more task families help unseen tasks?

Each run has the same training budget, spread over 2 to 32 families. "Similar tasks" are
16 families from the same pool that no run trained on. "New-skill tasks" are 18 families
built on skills the pool was not allowed to use. "Per doubling" is the slope of score on
log2 of the number of training families.

| Experiment | Trained runs | Chance | Untrained | 2 families | 32 families | Similar tasks, per doubling | New-skill tasks, per doubling | Outcome by the pre-set rule |
|---|---|---|---|---|---|---|---|---|
| Rerun: 1.5B, reserved set 1 | 20 | 0.289 | 0.543 | 0.601 | 0.704 | +0.024 (SE 0.005) | +0.003 (SE 0.004) | Confirmed |
| Check A: 1.5B, reserved set 2 | 20 | 0.270 | 0.484 | 0.567 | 0.682 | +0.026 (SE 0.005) | +0.003 (SE 0.005) | Replicated |
| Check B: 3B, reserved set 1 | 20 | 0.289 | 0.633 | 0.696 | 0.799 | +0.024 (SE 0.005) | +0.006 (SE 0.006) | Replicated |

The two reserved sets use different questions, so compare their slopes, not their levels.

The first breadth run (18 runs, 2 to 64 families) had the pre-set outcome "Flat" on the
new-skill tasks: +0.003 per doubling (SE 0.005). The similar-task effect was first seen
there after the fact (+0.035 per doubling), which is why the rerun above fixed its rule in
advance.

### Version 4: social arms with changing weights

A population of 32 agents shares one adapter. Each agent carries an inherited repertoire of task
families, variety pays, and the adapter is trained on parents' own correct replies. Design, rules and
every amendment are in `SPEC4.md`, written before any model call.

**Verdict, fixed before any model run: not testable at this scale.** A free null model (the same
simulation with a frozen score table and no learning) showed the social rules can separate an arm
from its control by under one doubling of breadth. Two were needed. A smoke test then showed the
model keeps a shared harvest every time it is asked (15 of 15), so the three self-domestication arms
were dropped by rule and Theory 3 is untested.

| Arm | Runs | Breadth | Breadth from rules alone | Practised skill | Similar unseen tasks | New-skill tasks |
|---|---|---|---|---|---|---|
| Untrained model | | | | 0.510 | 0.543 | 0.406 |
| DRIFT, random parents | 5 | 4.02 (0.13) | 4.3 to 4.8 | 0.586 (0.016) | 0.590 (0.017) | 0.452 (0.016) |
| SOLO, survival alone | 5 | 5.20 (0.52) | 5.9 to 9.8 | 0.587 (0.014) | 0.631 (0.008) | 0.467 (0.009) |
| SOLO-W, old learning rule | 5 | 5.72 (0.62) | 6.3 to 9.8 | 0.550 (0.012) | 0.595 (0.011) | 0.432 (0.007) |
| COOP, cooperation | 5 | 6.61 (0.56) | 8.3 to 11.1 | 0.588 (0.010) | 0.612 (0.005) | 0.465 (0.009) |
| CULTURE, cultural brain | 5 | 5.09 (0.41) | 5.7 to 10.3 | 0.601 (0.012) | 0.614 (0.009) | 0.468 (0.012) |

Standard errors in brackets. Breadth is the effective number of task families trained on.

| Comparison | Mean difference (SE) | Result by the pre-set rule |
|---|---|---|
| COOP minus SOLO, breadth in doublings (rules alone: +0.19 to +0.52) | +0.354 (0.096) | Broadens, mechanical |
| CULTURE minus SOLO, breadth in doublings (rules alone: -0.04 to +0.07) | -0.019 (0.102) | No breadth change |
| SOLO practised skill minus untrained | +0.077 (0.014) | Populations learn |
| SOLO minus DRIFT, practised skill | +0.001 (0.022) | Credit goes to learning from correct answers, not selection |
| SOLO minus DRIFT, similar unseen tasks | +0.041 (0.021) | No difference (1.9 SE) |
| SOLO-W minus SOLO, similar unseen tasks | -0.036 (0.008) | Old rule is lower |
| Each arm's similar-task score minus untrained (exploratory) | +0.047 to +0.088 | Above in 5 of 5 arms |

The exploratory row was added after the smoke test. Part of it is reply format: unusable replies fell
from 3.7% to 0.3%. Counting valid replies only, the gains are +0.028 to +0.068.

## Deviations and limits

- **Check B deviation.** 14 of the 20 trained 3B runs failed with a GPU out-of-memory
  error. The plan said failed runs are not replaced. They were rerun with the same names,
  families, seeds and settings, with each 8-example training step computed as two groups
  of 4 and the gradients summed. This was recorded in `SPEC3.md` before any check B score
  was read. 6 trained runs used the single pass and 14 used the split.
- **One model family.** Every weight-changing run used Qwen2.5 (1.5B or 3B).
- **One task style.** All task families are one-question text tasks written by AI.
- **Small samples.** 4 runs per level in the breadth experiments.
- **"Similar" is defined by construction.** Similar tasks are families from the same pool,
  with no independent measure of similarity.
- **A post hoc finding started the line.** The similar-task effect was noticed after the
  first breadth run, then tested with rules fixed in advance.
- **Version 4 did not test the thesis.** Its design could not separate arms far enough, and that was
  declared before any model run. Self-domestication was not tested in any version with changing weights.
- **Version 4's repertoire rules are hand-written.** The null model measures how much of each result
  those rules explain.

## Layout

| Path | Contents |
|---|---|
| `SPEC.md`, `SPEC2.md`, `SPEC3.md` | Designs, verdict rules, status notes and deviations for versions 1, 2 and 3 |
| `evo/`, `runs/`, `results.json` | Version 1 code, raw runs and analysed results |
| `evo2/`, `runs2/`, `results2.json` | Version 2 code, raw runs and analysed results |
| `evo3/` | Version 3 code: world, task families (`fam_*.py`), training, Modal app, statistics, pages |
| `runs3/modal/` | One JSON file per version 3 run, plus `ledger.json` (estimated spend) |
| `runs3/job_*.json` | The exact job lists that were submitted |
| `runs3/*_stats.json` | Statistics for the breadth experiments |
| `checks3.json` | Version 3 check results |
| `SPEC4.md` | Version 4 design, independent review, check results, amendments and results |
| `evo3/social*.py` | Version 4: simulation, null model, statistics, page, job lists |
| `runs3/modal/s4-*.json`, `runs3/s4_stats.json`, `runs3/null4*.json` | Version 4 runs, statistics and null-model results |
| `figures/` | Four figures made by `evo3/essay_figures.py` |
| `docs/` | Result pages as standalone HTML. Start at `docs/index.html`. |
| `summary_build.py` | Builds the summary page from the three result files |
| `tools/wrap_pages.py` | Turns the built page fragments into the standalone pages in `docs/` |

## Reproduce the analysis (free, no model calls)

Needs Python 3.11 or later. Only the figures need a package (`pip install -r requirements.txt`).

```bash
export PYTHONPATH=.

# Version 1 and 2 results from the raw runs
python3 -m evo.analyze --runs runs --out results.json
python3 -m evo.extras --runs runs --results results.json
python3 -m evo2.analyze --runs runs2 --out results2.json --headroom-dir runs2
python3 -m evo2.extras --runs runs2 --results results2.json

# Breadth statistics
python3 -m evo3.breadth_stats     # first breadth run
python3 -m evo3.near_stats        # rerun, reserved set 1
python3 -m evo3.near_stats n2     # check A
python3 -m evo3.near_stats m3     # check B
python3 -m evo3.bbh_rescore       # public test, rescored

# Version 4
python3 -m evo3.social_null --seeds 200 --keep 1.0 --vote 0.97   # free null model, no model calls
python3 -m evo3.social_stats smoke
python3 -m evo3.social_stats

# Pages and figures
python3 -m evo.build_report --results results.json --out report.html
python3 -m evo2.build_report --results results2.json --out report2.html
python3 -m evo3.breadth_page
python3 -m evo3.social_page
python3 summary_build.py
python3 -m evo3.essay_figures
python3 tools/wrap_pages.py
```

These commands were run on a copy of this repository. The statistics files, the null-model
file, the breadth page, the version 4 page, the summary page and the figures came out identical to
the stored ones.
`results.json`, `results2.json` and the version 1 and 2 pages differ only in their
`generated` date.

`python3 -m evo3.report3` rebuilds `checks3.json` and the version 3 page. It reads the
spend from `runs3/modal/ledger.json`, which now holds the total for all later runs
($24.69), so the rebuilt page shows that figure where the stored page shows $7.02.

## Rerun the experiments (costs money)

| Part | Needs | What it cost here |
|---|---|---|
| Versions 1 and 2 | An Anthropic API key in `ANTHROPIC_API_KEY`, `pip install anthropic` | About $48.5 |
| Version 3, breadth experiments and version 4 | A Modal account, `pip install modal` | $24.69 (own upper estimate) |

Total compute for the project was about $73.

```bash
export PYTHONPATH=.
modal deploy -m evo3.modal_app            # creates app "evo3", volume "evo3-runs", dict "evo3-board"
python3 -m evo3.submit runs3/job_near.json
python3 -m evo3.submit --fetch            # status, spend, results into runs3/modal/
```

`evo3/remote.py` holds the spend guard (`CAP`). It counts all spend in the ledger, so set
it for your own account before you submit anything. GPU runs used Modal L4 or A10 cards.
Exact scores can vary with the card and library versions; the image in
`evo3/modal_app.py` does not pin versions.

## Third-party material

- **Models.** No weights are stored here. Qwen2.5-0.5B and 1.5B Instruct are under
  Apache 2.0. Qwen2.5-3B-Instruct is under the Qwen research licence, which does not allow
  commercial use.
- **Public test.** `runs3/modal/bbh-dump-*.json` contain 40 items per task from BIG-Bench
  Hard with model replies. The items come from
  [suzgunmirac/BIG-Bench-Hard](https://github.com/suzgunmirac/BIG-Bench-Hard); see that
  repository for its licence and citation.

## Licence

MIT for the code and the results in this repository. See `LICENSE`.
