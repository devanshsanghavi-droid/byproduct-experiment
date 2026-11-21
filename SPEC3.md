# Version 3: weights evolve (design only, nothing built)

Thesis under test, unchanged: "All three surviving theories say intelligence was a byproduct. A faithful
simulation therefore scores agents only on staying alive and being copied, then measures general ability
separately on held-out tests."

## Why version 2 was replaced

Version 2 evolved prompt text around a frozen model. Selection removed harmful text and recovered the
near-task score toward the unmodified model. General ability stayed flat in all 7 arms and below the
unmodified model. Frozen weights cap everything at the starting model, so the real question (can
evolutionary pressure make AI more capable) was not tested. Version 3 lets the weights change.

## What evolves

| Channel | What it is | Inheritance | Variation |
|---|---|---|---|
| Weights | one LoRA adapter per population on a small open-weights model | next generation's adapter = current adapter trained on survivors' whole-life transcripts | sampling at temperature 1.0 during each life |
| Note (CULTURE, MEDLEY only) | up to 40 words written at end of life | read by the next generation | written from experience |

- The only signal is who survives. Survivors' transcripts are copied whole, failed tries included.
  No step is ever filtered by task success, and no test score reaches training.
- Generation 0 is the unmodified model in every arm. Any gain is a gain over the unmodified model.
  Version 2's "every gain is recovery" problem cannot occur.
- Instinct text and its seed rules are removed.

## Model and machine

- Machine: Apple M4 Pro, 48 GB memory. Local fine-tuning with mlx-lm (`mlx_lm.lora`). Not installed yet.
- Model: Qwen2.5-1.5B-Instruct, 4-bit MLX build (open licence, no account needed). Fallbacks: the 0.5B
  and 3B sizes. Check 1 picks the size. Newer small models may exist; confirm at build time.
- Claude API cost: $0. The $151 of remaining credits is not needed.

## World (wider than version 2)

One task cannot select for general ability. Each life now has 24 decisions across 3 task families.

| Family | Skill | Status |
|---|---|---|
| Plants | ordered search with feedback, two hidden laws | built (evo2/world.py) |
| Tracks | predict where an animal appears next from a hidden movement rule | to build |
| Trade | pick the best of 3 offers by arithmetic on quantities and values | to build |

Energy: +4 per success, -1 otherwise, as before. Instances are redrawn every generation.

## Arms (same 7 as version 2)

| Arm | Tests | Parents | Extra |
|---|---|---|---|
| DRIFT | chance, and what repeated self-training does by itself | random 8 of 32 | none |
| SOLO | ecology only | top 8 of 32 by energy | none |
| COOP | Theory 2 | top 8 | pairs: one advises, one acts, shared harvest |
| ALPHA | control for 3 | top 8 | pairs, actor may grab |
| DOMEST | Theory 3 | top 8, exiled excluded | as ALPHA plus exile vote |
| CULTURE | Theory 4 | top 8 | notes |
| MEDLEY | 2 + 3 + 4 | top 8, exiled excluded | pairs, grab, vote, notes |

Every arm trains for the same number of steps on the same number of examples per generation.
Starting values, to be fixed in the pilot and then frozen: 32 lives, 10 generations, 5 seeds, LoRA
rank 8, 150 training steps, examples drawn from the last 3 generations of survivors.

## Measurement

All tests run at temperature 0 on a fixed item set, identical for every population. None is used for
selection.

| Test | Distance | Status |
|---|---|---|
| near | fresh world, same 3 families | extend evo2/world.py |
| code-breaking | same kind of search, new surface | built (evo2/far.py) |
| rule induction | different task | built |
| ordering from clues | different task | built; was at maximum for Haiku, recheck |
| retention | slice of a public reasoning benchmark, detects general forgetting | to choose |

General ability = mean of the held-out tests that pass Check 1.

Reference rows: unmodified model (also generation 0), DRIFT, and CEILING (an adapter trained directly
on correct play; breaks the thesis rule on purpose and is used only as an upper bound).

## Checks before the full run (each can stop the project)

| Check | What it does | Pass rule | Estimated time |
|---|---|---|---|
| 0. Speed | install, download, time one generation and one fine-tune | gives the real run time | 30 min |
| 1. Headroom | unmodified model on every test | above chance and below 85% of maximum | 30 min |
| 2. Ceiling | train directly on correct play in the world, then run all tests | near rises; report whether held-out moves at all | 1 to 2 h |
| 3. Pilot | DRIFT and SOLO, 2 seeds, 6 generations | SOLO above DRIFT on near energy in both seeds | 3 to 5 h |

Check 2 is the decisive one. If direct training on the world does not move held-out scores, selection
cannot either, and the full run would report noise. In that case: widen the world or stop.

Full run: 35 runs. Estimated 15 to 40 hours of laptop time. Not measured; Check 0 replaces this estimate.

## Verdict rules (set before any data)

Paired by seed from the start: each arm's change minus the reference arm's change, same seed.

1. Selection raises general ability: SOLO or any theory arm ends above the unmodified model and above
   DRIFT in all 5 seeds (sign test, one-sided p = 0.031).
2. Thesis supported: at least 2 theory arms end above the unmodified model, above DRIFT and above SOLO
   on general ability, each in all 5 seeds.
3. Thesis not supported: no theory arm passes rule 2.
4. Reported in every case: retention change (forgetting), and the share of the CEILING gain reached.

## Research grounding (web search; Paper Lantern tools were not available in this session)

- Repeated training on a model's own unfiltered output degrades it. Filtering by an external check
  reduces the damage; model self-review does not. Survival energy is an external check, and DRIFT is
  the unfiltered case, so DRIFT is expected to fall. (arxiv 2510.16657; alphaxiv 2606.28438; read from
  search snippets, not in full.)
- Naive recursive self-training gave no downstream gain in one study; reported gains traced to
  human-designed filtering. This predicts a flat general score here. (EACL 2026 Findings 213.)
- LoRA forgets less out-of-domain ability than full fine-tuning and also learns less; narrow data and
  high learning rates raise forgetting. Hence low rank, a retention test and a 3-family world.
  (Biderman et al. 2024, "LoRA Learns Less and Forgets Less".)

## Limits known in advance

- One adapter per population. Individuals differ only by sampling, so heredity is at population level.
  Upgrade if results warrant: 4 lineages with separate adapters.
- Inheritance copies behaviour acquired in life. Biological genes do not work this way.
- A 1.5B model is far from frontier scale. A null result does not transfer to large models.
- My estimate, not a measurement: near energy rises under selection; general ability most likely stays
  flat or falls; chance that rule 2 passes is roughly 10 to 20%.

## Status, 2026-11-12: checks run on Modal CPU (results page: report3.html)

Compute moved to Modal CPU workers with a $10 limit at the user's instruction. Spend: $4.03 (own upper estimate).

| Check | Outcome | Result |
|---|---|---|
| 0. Speed | done | 0.5B model: 2.75 calls per second, 5.3 s per training step on 8 cores; about $0.07 per generation |
| 1. Headroom | fail | 0.5B model at chance on every family and test; 1.5B (measured on the Mac) above chance on tracks only |
| 2. Ceiling | fail | direct training on correct play: near energy 8.4 to 81.2, general ability 0.228 to 0.178 |
| 3. Pilot | pass, weak | final near energy SOLO minus DRIFT: +5.3 and +3.4 in the 2 seeds; general ability not different |

Deviations from the design above: Qwen2.5-0.5B on PyTorch CPU (not 1.5B on MLX); easier world (plants 1 step, ring of 6,
3 trade offers); lenient reply parsing; 24 lives, top 6, 8 generations, 60 training steps; only DRIFT and SOLO built;
retention is a hand-made set. Code: evo3/ (world3, tests3, sim3, torch_llm, remote, modal_app, submit, live_modal, report3).
By the rule set before the data, the full 7-arm run should not go ahead on this setup.

## Status, 2026-11-13: checks rerun with the 1.5B model on a Modal GPU

The user approved a GPU if its cost stayed under $10. GPU spend: $2.38. Total Modal spend: $6.41 (own upper estimate).

| Check | Outcome | Result (Qwen2.5-1.5B, designed world) |
|---|---|---|
| 0. Speed | done | 32.4 calls per second, 0.38 s per training step on one A10; about 4 times cheaper per call than the CPU worker |
| 1. Headroom | partial | all 3 world families above chance; code-breaking above chance; rule induction and ordering at chance |
| 2. Ceiling | fail | direct training on correct play: near energy 21.9 to 68.6 (correct play 85.0), general ability 0.325 to 0.324, retention 0.82 to 0.72 |
| 3. Pilot, 5 seeds | partial | final near energy SOLO 29.7, DRIFT 21.3, gap +8.4 (SE 2.4), 4 of 5 seeds; general ability gap -0.005 (SE 0.008) |

Reading: survival-only selection through weights raises skill on the selected tasks. Nothing reaches the held-out tests, and
direct training on correct answers does not either. By the pre-set rule the 7-arm run should not go ahead on this world.

## Wider world, 2026-11-14: ceiling check only (rules written before the results arrived)

The user approved widening the world and rerunning only the ceiling check on the GPU.

- World: the 3 original families plus 13 new one-question families (count, tally, path, series, sort, gate, decode,
  oddone, minmax, swap, share, pick, countif), 3 questions each per life. 65 decisions per life, maximum energy 260.
- New test "held-out families": 6 families that are in no life and no training set (taboo, rank, days, reverse, elim,
  change), 576 questions. The three original held-out tests and retention are unchanged.
- Trained arm: 1,200 steps on 9,750 examples of correct play in the wider world.
- Control arm: 1,200 steps on 9,750 examples of correct play in the original 3-family world, same tests. It separates
  the effect of breadth from the effect of any training at all.
- Research grounding (web search; Paper Lantern tools not available): generalisation to unseen tasks depends on how many
  distinct tasks are seen much more than on examples per task (arxiv 2402.10891, 2405.19787; read from search summaries).

Pass rules:
1. Pass: general ability (the three original tests) ends more than 2 standard errors above the unmodified model and
   above the control arm.
2. Partial: rule 1 fails, but the held-out families end more than 2 standard errors above both.
3. Fail: neither.

Results (Qwen2.5-1.5B, Modal A10, one training run per arm; results page: report3.html). Outcome by the rules above: Partial.

| Row | Near energy (wider world) | 13 new families | Held-out families | General ability | Retention |
|---|---|---|---|---|---|
| Chance | -6.5 | 0.174 | 0.274 | 0.281 | 0.00 |
| Unmodified model | 78.3 | 0.485 | 0.507 | 0.329 | 0.83 |
| Control (3-family training, 1,200 steps) | 137.3 | 0.510 | 0.589 | 0.359 | 0.76 |
| Wider-world training, 1,200 steps | 201.0 | 0.920 | 0.642 | 0.359 | 0.80 |
| Correct play | 241.0 | 1.000 | 1.000 | n/a | n/a |

- Effect of breadth (wider minus control, paired by test item): held-out families +0.054 (SE 0.016); general ability +0.000 (SE 0.018).
- Most of the held-out gain over the unmodified model is one family (reverse: 0.094, control 0.708, wider 0.823). On the other five
  families: unmodified 0.590, control 0.565, wider 0.606.
- The control's transfer to the 13 families it never trained on: 0.485 to 0.510.
- Code-breaking replies now change with training (0.206, 0.075 after 400 steps, 0.263 after 1,200), so the test responds to the weights.
- Cost: $0.61. Total Modal spend $7.02 (own upper estimate), GPU part $2.99. App stopped.
- Reading: breadth gives a small transfer to nearby unseen families and none to the three original tests. Direct training is the
  upper bound for selection, so the 7-arm run is still not justified at this model size.

## Breadth curve, 2026-11-14 (design and rules written before any result)

The user approved one more experiment under a new $10 Modal limit: does transfer to unseen task families rise with the
number of distinct families trained on, when the training budget is fixed?

- Pool: 65 one-question families that can be trained on (the 13 existing ones plus 52 new ones written by subagents and
  passed through evo3/famcheck.py). Held out: 18 families that are in no training set (the 6 existing plus 12 new).
- Levels: K = 2, 4, 8, 16, 32, 64 families. 3 runs per level. Each run draws its own K families at random from the pool
  and has its own training seed. Plus 1 untrained baseline. 19 jobs.
- Budget per run, identical at every level: 1,200 steps, batch 8, 9,600 examples of correct play split equally among the
  K families, learning rate 2e-4, LoRA rank 8, Qwen2.5-1.5B on a Modal GPU. The three multi-step families (plants, tracks,
  trade) are left out of training so that K is the only thing that varies.
- Tests, temperature 0, same items in every run:
  - held-out families, 18 x 32 = 576 questions (primary);
  - pool families, 65 x 8 questions, split per run into families it trained on and families it did not;
  - public test: 11 tasks x 40 items from BIG-Bench Hard (boolean_expressions, navigate, object_counting,
    tracking_shuffled_objects_three_objects, logical_deduction_three_objects, date_understanding, web_of_lies,
    temporal_sequences, reasoning_about_colored_objects, penguins_in_a_table, multistep_arithmetic_two), fetched inside the
    Modal image from the benchmark authors' repository;
  - the three original tests (general ability) and retention.
- Rules:
  1. Curve: least-squares line of the held-out score on log2(K) over the 18 trained runs. Pass if the slope is more than
     2 standard errors above zero.
  2. Far curve: the same test on the public test score and on general ability, reported separately.
  3. Flat: rule 1 fails.
  Reported in every case: K=64 mean minus K=2 mean with its standard error from the 3 runs per level, each level against
  the untrained baseline, retention, and the per-family table.
- Limits known in advance: the new families were written by AI subagents in one style; with 3 draws per level, "which
  families" is mixed with "how many" at low K; one model size; several public tasks resemble pool families (counting,
  navigation, swaps, boolean logic), so the public test is a transfer test written by others, not a pure far test.
- Spend: own ledger stops at $16.00 in total, which is $8.98 of new spend on top of the $7.02 already used.

Results, 2026-11-14 (results page: breadth.html; numbers: runs3/breadth_stats.json). Outcome by rule 1: Flat.

| Families trained on | Held-out families (18) | Same-pool families not trained on | Families trained on | General ability | Retention |
|---|---|---|---|---|---|
| Chance | 0.207 | 0.221 | | 0.281 | 0.000 |
| Untrained | 0.417 | 0.517 (all 65) | | 0.330 | 0.822 |
| 2 | 0.457 | 0.544 | 1.000 | 0.313 | 0.793 |
| 4 | 0.500 | 0.581 | 0.969 | 0.330 | 0.856 |
| 8 | 0.476 | 0.586 | 0.979 | 0.342 | 0.863 |
| 16 | 0.491 | 0.626 | 0.953 | 0.347 | 0.815 |
| 32 | 0.499 | 0.644 | 0.914 | 0.333 | 0.863 |
| 64 | 0.477 | 1 family per run | 0.871 | 0.298 | 0.915 |

- Rule 1 (held-out families): slope +0.003 per doubling (SE 0.005). Flat. All 18 trained runs minus untrained: +0.067 (SE 0.008).
- Rule 2: public test slope -0.004 (SE 0.006), general ability -0.002 (SE 0.006). Neither rises.
- Retention rose with breadth: +0.017 per doubling (SE 0.007). Trained-family score fell as the budget was spread: -0.024 (SE 0.003).
- Added after seeing the data: gain on same-pool families a run did not train on, against the untrained model on the same
  families, levels 2 to 32: +0.031, +0.064, +0.073, +0.121, +0.177; slope +0.035 per doubling (SE 0.005). The 64 level is left
  out because 1 family per run remains. Not a pre-set rule; needs a rerun with the rule fixed first.
- Added after seeing the data: the strict public-test score fell for trained models (-0.070, SE 0.009) because they answer
  multiple-choice items with the option's text, not its letter. Raw replies recorded for 4 models and rescored
  (evo3/bbh_rescore.py): untrained 0.370, K=2 0.391, K=16 0.368, K=64 0.339 (SE about 0.023 each). No real drop.
- Reading (mine): breadth buys transfer to tasks that resemble the training set and none to tasks that need a skill the
  training set never uses. The held-out set was built on skills the pool was forbidden to use.
- Cost: $2.66 (own upper estimate) of the second $10 limit. Ledger total $9.68. App stopped.

## Near-transfer rerun, 2026-11-14 (rules written before any rerun result)

Purpose: the same-pool gain above was found after seeing the data. This rerun tests it with the rule fixed first.

- Reserved families: 16 of the 65 pool families, drawn once with seed 20261114 (`breadth.reserved()`), never trained on
  by any rerun run: beadpair, countif, gap, morecuts, needs, nextsign, nextto, notchnum, onlyin, path, sooner, sort, swap,
  verify, whichsign, whohas. The draw was made before any rerun result and is not changed afterwards.
- Runs: 1 untrained baseline + 4 runs at each of K = 2, 4, 8, 16, 32 (20 trained runs). Each run draws its K families
  from the other 49 pool families (`breadth.subset2(k, draw)`, seed 50000 + 1000 K + draw) and uses training seed
  7000 + 100 K + draw. None of these subsets or seeds was used before. Model, budget and settings as above: Qwen2.5-1.5B,
  1,200 steps, batch 8, 9,600 examples split equally, learning rate 2e-4.
- Tests, temperature 0, same items in every run:
  - reserved families, 16 x 32 = 512 questions (primary);
  - held-out families, 18 x 32 = 576 questions (same items as above);
  - pool families, 65 x 8 questions (same items as above), for a repeat of the earlier comparison.
  The public test and the three original tests are not run.
- Rules:
  1. Near curve: least-squares line of the reserved-family score on log2(K) over the 20 trained runs. Confirmed if the
     slope is more than 2 standard errors above zero. Not confirmed otherwise.
  2. Split: per run, reserved-family score minus held-out score; the same line fitted to that difference. The near-far
     split is confirmed if rule 1 passes and this slope is more than 2 standard errors above zero.
  Reported in every case: each level against the untrained baseline with the standard error from 4 runs; K=32 mean minus
  K=2 mean; the held-out slope; the share of valid replies on reserved families; the per-family table; the earlier
  comparison repeated on the 49 non-reserved families (gain on families a run did not train on, against the untrained
  model on the same families).
- Limits known in advance: one reserved set, so the result is about these 16 families; some reserved families have a
  close relative among the trainable ones (nextsign, nextto and neighbor; whichsign and resign), which is part of what
  "similar task" means here; 4 runs per level; one model size; families written by AI subagents in one style.
- Spend: own ledger stops at $13.00 in total, which is $3.32 for this rerun on top of the $9.68 already used.

Rerun results, 2026-11-14 (numbers: runs3/near_stats.json; page: breadth.html). Outcome: Confirmed, and the near-far split is confirmed.

| Families trained on | Reserved same-pool families (16) | Held-out families (18) | Families trained on |
|---|---|---|---|
| Chance | 0.289 | 0.207 | |
| Untrained | 0.543 | 0.406 | |
| 2 | 0.601 (SE 0.011) | 0.478 | 1.000 |
| 4 | 0.652 (SE 0.019) | 0.489 | 1.000 |
| 8 | 0.666 (SE 0.025) | 0.484 | 0.965 |
| 16 | 0.691 (SE 0.016) | 0.481 | 0.947 |
| 32 | 0.704 (SE 0.015) | 0.497 | 0.943 |

- Rule 1 (reserved families): +0.024 per doubling (SE 0.005, 4.6 standard errors). Confirmed.
- Rule 2 (reserved minus held-out, per run): +0.022 per doubling (SE 0.005, 4.1 standard errors). Split confirmed.
- Held-out slope +0.003 (SE 0.004). K=32 minus K=2 on reserved: +0.103 (SE 0.019). All trained runs minus untrained:
  reserved +0.120 (SE 0.011), held-out +0.080 (SE 0.006).
- Valid replies on reserved families: 0.963 untrained, 0.98 to 1.00 trained, no slope (-0.000, SE 0.001). So the first
  training step includes a reply-format gain; the rise with K does not.
- Earlier comparison repeated on the 49 non-reserved families: +0.019 per doubling (SE 0.004). The slope is smaller than the
  post hoc +0.035.
- Cost: $2.58 (own upper estimate). Ledger total $12.26. App stopped.

## Two strengthening experiments, 2026-11-15 (rules written before any result of either)

Purpose: the confirmed near-transfer result rests on one reserved set and one model size. Each experiment below changes
one of those and nothing else. The rules are the same two rules as the rerun above, applied to each experiment on its own.

### A. Second reserved set (task names n2-)

- Reserved families: 16 drawn once with seed 20261115 from the 49 pool families that were NOT reserved in the rerun
  (`breadth.reserved2()`), so the two reserved sets share no family: begins, bigger, count, everyother, first, fuse,
  handoff, liar, minmax, pick, queue, round, syllog, topitem, trips, twostep.
- Runs: 1 untrained baseline + 4 runs at each of K = 2, 4, 8, 16, 32 (20 trained runs). Each run draws its K families
  from the 49 pool families outside this reserved set (`breadth.subset3(k, draw)`, seed 60000 + 1000 K + draw), training
  seed 8000 + 100 K + draw. Model and settings unchanged: Qwen2.5-1.5B, 1,200 steps, batch 8, 9,600 examples split
  equally, learning rate 2e-4.
- Tests as in the rerun: reserved families 16 x 32 = 512 questions, held-out families 576 questions, pool families
  65 x 8 questions; temperature 0; same items in every run.

### B. Larger model (task names m3-)

- Model: Qwen/Qwen2.5-3B-Instruct (about twice the 1.5B model; released under the Qwen research licence). Everything
  else is copied from the confirmed rerun: reserved set 1 (`breadth.reserved()`), the same 20 family subsets
  (`breadth.subset2`), the same training seeds (7000 + 100 K + draw), the same budget and settings, the same test items.
  So each 3B run has a 1.5B twin trained on the same families.
- Runs: 1 untrained baseline + 4 runs at each of K = 2, 4, 8, 16, 32. Budget fallback fixed now: a 20-step smoke run
  is timed first; if the projected cost of one full 3B run is above $0.30, only draws 1 to 3 are run at each level
  (15 trained runs). The smoke run is not a result and decides nothing else.

### Rules (each experiment judged alone)

1. Near curve: least-squares line of the reserved-family score on log2(K) over the trained runs. Replicated if the slope
   is more than 2 standard errors above zero. Not replicated otherwise.
2. Split: the same line fitted to the per-run reserved minus held-out score. The split is replicated if rule 1 passes
   and this slope is more than 2 standard errors above zero.

Reported in every case and deciding nothing: each level against the untrained baseline; top level minus bottom level;
the held-out slope; the share of valid replies; the per-family table. For B also the per-twin difference between the
3B and 1.5B reserved scores and its slope on log2(K), and whether training at 3B changes the held-out score.

Readings fixed now:
- A replicated: the near-transfer result is not a property of the first 16 reserved families.
- A not replicated: the confirmed result is about those 16 families only, and the write-up must say so.
- B replicated: the result holds at twice the model size. B not replicated: it is a result about the 1.5B model.
- Runs that fail or are stopped by the budget are not replaced; the line is fitted to the runs that finished, and the
  count is reported. An experiment with fewer than 10 finished trained runs is reported as "not completed".

Limits known in advance: still one task style written by AI subagents; one model family (Qwen2.5); 3 or 4 runs per
level; the 3B model uses a different licence from the 1.5B model, which matters only for reuse of its weights.

Spend: the user approved under $10 on 2026-11-15. Ledger total before these runs: $12.26. Own ledger stops at $21.75
in total, which is $9.49 for both experiments.

Smoke run for B (2026-11-15, before any full run): 20 steps and half-size tests on an A10 took 68 s; projected cost of one
full 3B run $0.23, below the $0.30 line, so all 4 draws per level are run (20 trained runs + 1 baseline).

Deviation for B (written 2026-11-15, after 14 of the 20 trained 3B runs stopped with a GPU out-of-memory error during
their first training steps, and before any B score was read or any B statistic computed):
- The "not replaced" rule above was meant for runs lost to chance or to the budget. These failures are a memory limit
  of the 22 GB card. They hit runs by prompt length, so dropping them would leave mostly 2-family runs and bias the line.
- The failed runs are therefore run again under the same names, families, training seeds and settings. The only change is
  inside one training step: the 8 examples are passed through the model as two groups of 4 and the two gradients are
  summed before the update. The summed gradient is the same quantity as before (sum of reply-token losses over all 8
  examples divided by their token count), up to floating-point rounding. Runs that already finished are kept as they are.
- Experiment A finished all 21 runs before this note and is not affected.
- The write-up must report this as a deviation from the written plan.

Results of the two strengthening experiments, 2026-11-15 (numbers: runs3/n2_stats.json, runs3/m3_stats.json; page: breadth.html).

| Experiment | Trained runs | Untrained | 2 families | 32 families | Reserved, per doubling (rule 1) | Held-out, per doubling | Reserved minus held-out (rule 2) | Outcome |
|---|---|---|---|---|---|---|---|---|
| Confirmed rerun: 1.5B, reserved set 1 | 20 | 0.543 | 0.601 | 0.704 | +0.024 (SE 0.005, t 4.6) | +0.003 (SE 0.004) | +0.022 (SE 0.005, t 4.1) | Confirmed |
| A: 1.5B, reserved set 2 | 20 | 0.484 | 0.567 | 0.682 | +0.026 (SE 0.005, t 5.7) | +0.003 (SE 0.005) | +0.022 (SE 0.005, t 4.4) | Replicated, split replicated |
| B: 3B, reserved set 1 | 20 | 0.633 | 0.696 | 0.799 | +0.024 (SE 0.005, t 4.7) | +0.006 (SE 0.006) | +0.018 (SE 0.005, t 3.3) | Replicated, split replicated |

- Chance on reserved: 0.289 (set 1), 0.270 (set 2). Valid replies have no slope in either check.
- B twins: each 3B run minus its 1.5B twin, reserved +0.093 (SE 0.007), held-out +0.077 (SE 0.010); the reserved
  difference has no slope on log2(K) (-0.000, SE 0.005). So the larger model scores higher everywhere and gains the
  same amount per doubling.
- B training step: 6 trained runs used a single 8-example pass; 14 used two 4-example passes with summed gradients
  (the deviation above). No run needed a further split. 20 of the 21 B runs ran on an A10 card, 1 on an L4.
- Cost (own upper estimate): A $2.54, B $5.05, smoke tests and failed starts $0.30; $7.89 in total against under $10
  approved. Ledger total $20.15. App stopped.
