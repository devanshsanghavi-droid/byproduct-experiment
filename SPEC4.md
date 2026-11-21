# Version 4: social arms with changing weights (design only, nothing built, nothing run)

Written 2026-11-16, before any version 4 code or result exists. Revised the same day after an
independent review; the changes are listed at the end.

Thesis under test, unchanged: "All three surviving theories say intelligence was a byproduct. A faithful
simulation therefore scores agents only on staying alive and being copied, then measures general ability
separately on held-out tests."

## Why this version exists

- Versions 1 and 2 ran the theory arms on frozen models. Ability could not change, so the theories were
  never tested.
- Version 3 let weights change but built only DRIFT and SOLO. Its world had 3 task families, and direct
  training on it did not move held-out scores, so the 7-arm run was stopped by rule.
- The breadth experiments then found the one thing that does move unseen-task scores: the number of
  task families practised. Similar unseen tasks rose +0.024 to +0.026 per doubling in three
  experiments. New-skill tasks stayed flat.

Version 4 joins the two. The theories of cooperation, self-domestication and the cultural brain all
describe social life as changing which problems an individual must solve. That gives them a prediction
this project can measure.

## The question, as a chain with two links

| Link | Claim | Status before this version |
|---|---|---|
| 1 | Social arms end up practising more task families than solitary survival does | Untested. This is the new part. |
| 2 | More families practised gives higher scores on similar unseen tasks | Confirmed 3 times under training on correct answers. Untested under survival selection. |

The outcome is open in both directions. Copying successful others can shrink variety. Selection can
shrink variety below what chance keeps. Either would count against the theories here, and both have
their own labels below.

## World

- **Trainable families:** the 49 one-question families outside reserved set 1 (`breadth.reserved()`).
- **Tests, never in the world:** reserved set 1 (16 families, 512 items, "similar tasks") and the 18
  held families (576 items, "new-skill tasks"). Same items as the confirmed rerun, temperature 0.
- **Population:** 32 agents per generation, generations numbered 1 to 8, one LoRA adapter per
  population (rank 8), Qwen2.5-1.5B-Instruct, sampling temperature 0.7 during lives.
- **Repertoire:** each agent carries a set of 1 to 12 trainable families. It is inherited, and it is
  not stored in the weights. Generation 1: 3 random families per agent, identical across arms for the
  same seed.
- **Life:** 24 rounds. Each round poses a fresh item from one family of the repertoire, cycling
  through the repertoire in shuffled order. In pair arms the two partners share one 24-round life.
- **Payoff:** the j-th attempt at one family within a life pays max(1, 5 - j) energy if right (4, 3, 2,
  then 1). A wrong or invalid reply costs 1. Repeating one family pays less each time, in every arm.
- **Parents:** the top 8 by energy; each has 4 children. A child copies its parent's repertoire, then
  with probability 0.3 gains one random family and with probability 0.3 loses one. No arm changes the
  expected repertoire size by rule.

## Learning rule (a deliberate change from version 3)

After each generation the adapter is trained for 150 steps (batch 8, learning rate 2e-4) on parents'
task replies that were themselves correct, from the last 3 generations. Total: 1,200 steps, the same
budget as one breadth run. In pair arms a wrong reply is never stored, even when the partner's right
reply earned the pair its pay.

- Version 3 trained on parents' whole transcripts, wrong answers included. The pilot showed that rule
  learns slowly (+8.4 energy of 104 against chance copying).
- With learning that weak, the weights barely move and the model acts as a fixed score table. That is
  the flaw the user rejected in version 1.
- What stays true: selection sees only energy. No test item, reserved family or held family ever
  reaches training. Social decisions (keep or split, votes) have no right answer and are trained
  whole, from parents only.
- What changes: an agent's descendants inherit what worked for it, where version 3 passed on
  everything it did. This filter improves a model with no selection at all, so DRIFT gets it too.
  The control arm SOLO-W keeps the old rule.

## Arms (8 arms x 5 seeds = 40 runs)

Model calls per generation: 768 task replies in every arm (32 agents x 24 rounds, or 16 pairs x 24
rounds x 2 replies), plus 16 keep-or-split decisions (one per pair) and 32 votes where those apply.

| Arm | Tests | Mechanism added | Judged against |
|---|---|---|---|
| DRIFT | chance | parents are a random 8 | none |
| SOLO | ecology alone | top 8 by energy | DRIFT |
| SOLO-W | the old learning rule | SOLO, trained on all parent replies | SOLO |
| COOP | Theory 2, cooperation | random pairs each generation; rounds drawn from both repertoires; both reply; the pair is paid if either is right; harvest split evenly | SOLO |
| ALPHA | control for Theory 3 | COOP, but one partner holds the harvest and is asked to keep all or split | COOP |
| DOMEST | Theory 3, self-domestication | ALPHA, plus a vote: each agent may name one agent who kept all; the 2 most-named agents with at least 4 votes cannot be parents | ALPHA |
| CULTURE | Theory 4, cultural brain | SOLO, plus each child with probability 0.5 swaps one of its families for one held by the highest-energy parent that is not its own | SOLO |
| MEDLEY | Theories 2 + 3 + 4 | pairs, keep-or-split, vote and swapping | reported against SOLO; no role in the verdict |

- Keep-or-split and votes are decisions made by the model, so the tendency to grab is carried in the
  weights and can evolve. Keeping all costs nothing in ALPHA, so it is expected to spread there. That
  is the condition Theory 3 says the vote should restrain.
- **COOP is a bundle.** Against SOLO it differs in three ways at once: each agent practises two
  repertoires, pay depends on a random partner, and partners tie on energy so parents come as 4 whole
  pairs. A COOP result is a result for the bundle, not for any one part.

## Measures (per run)

| Measure | Definition |
|---|---|
| Breadth | effective number of families (exponential of the Shannon entropy) over every example drawn into a training batch during the run: exactly what the adapter was trained on |
| Late breadth (secondary) | the same, for examples first added in generations 7 and 8 |
| Similar-task score | reserved set 1, final adapter. Stored reference rows: chance 0.289, untrained 0.543 |
| New-skill score | held families, final adapter. Stored reference row: untrained 0.406 |
| In-world skill | the stored 8 test items for each of the 49 trainable families (392 items), temperature 0. Stored reference row: untrained 0.510 (5 families at 0, 16 at 0.75 or above) |

Tests run after generations 4 and 8. The untrained rows are the stored ones and are not rerun.

## Rules (set before any data)

Every comparison is paired by seed, with 5 seeds. "Above" means the mean paired difference is more
than 2 standard errors above zero; "below" is the mirror. Each theory is judged against the arm named
in the table.

1. **Learning check.** SOLO's in-world skill is above the untrained 0.510. If not, no other rule is
   read and the outcome is "No learning". SOLO minus DRIFT on in-world skill is reported beside it;
   unless that is above zero, the learning is credited to the correctness filter, not to selection.
2. **Link 1, per theory.** The theory arm's breadth is above its control's.
3. **Link 2, per theory.** The theory arm's similar-task score is above its control's.
4. **Far transfer, per theory.** The theory arm's new-skill score is above its control's.
5. **Mechanism.** Similar-task score regressed on log2(breadth) with one intercept per arm, over the
   35 runs under the new learning rule. The within-arm slope is more than 2 standard errors above
   zero. The pooled slope across arms is reported only, because arms differ in more than breadth.
6. **Selection.** SOLO against DRIFT on breadth and on similar-task score, reported with its sign.

### Labels per theory

| Breadth label | Condition |
|---|---|
| Broadens, more than mechanics | rule 2 passes and the gap exceeds the null-model gap by more than 2 standard errors |
| Broadens, mechanical | rule 2 passes, gap within 2 standard errors of the null-model gap |
| Narrows | breadth is below its control's |
| No breadth change | none of the above |

| Lift label (similar tasks) | Condition |
|---|---|
| Lifts | rule 3 passes |
| Lowers | score is below its control's |
| No lift | neither, and the mean difference plus 2 standard errors is under 0.03 |
| Inconclusive | neither, and that upper bound is 0.03 or more |

A theory whose mechanism did not fire (see checks) is "Not tested".

### Thesis verdict, similar tasks

| Verdict | Condition |
|---|---|
| Supported | at least 2 of the 3 theories are "Lifts" with "Broadens, more than mechanics" |
| Supported by the written rules only | at least 2 theories are "Lifts", fewer than 2 with "more than mechanics" |
| Not supported | at least 2 theories are "No lift" or "Lowers" |
| Inconclusive | any other case |
| Not tested | fewer than 2 theories count as tested |

General ability is reported on its own line: "Far transfer seen" if at least 2 theories pass rule 4,
"Not seen" otherwise.

**False positives, stated in advance.** One test passes by luck about 6% of the time (one-sided, 4
degrees of freedom). COOP and CULTURE share the same 5 SOLO runs, so 2 of 3 theories passing rule 3 by
luck is about 2 to 3%. Across the roughly 11 tests reported, the chance that at least one passes by
luck is about one in two. Single passes are therefore reported as weak evidence.

## Guard against outcomes built into the mechanics

The repertoire and parent rules are hand-written, so some breadth difference between arms follows from
the rules alone, and the confirmed curve then turns that breadth into score. A lift produced that way
is real but assumed. Three guards:

- **Null model (free, no model calls).** The same simulation with a frozen score table (the untrained
  model's stored per-family scores) and no learning, 200 seeds per arm. Keep-all and vote rates are
  fixed at the rates measured in generation 1 of the smoke test. It gives the breadth each arm
  produces from mechanics alone. The "more than mechanics" label, which the verdict needs, comes from
  it.
- **Size-neutral rules.** Copying is a swap, so no arm grows repertoires by rule.
- **Parameter freeze.** Repertoire limits, gain and loss rates, the swap rate and the vote threshold
  may be tuned on the null model and on the smoke test only. They are frozen and recorded here before
  the first full run is submitted.

The payoff rule rewards variety by design. It is the same in every arm, so it cannot favour one arm,
but it means this world is one where variety pays.

## Checks before the full run (each can stop it)

| Check | What it does | Pass rule | Cost |
|---|---|---|---|
| 0. Null model | mechanics alone, 200 seeds per arm, with the untrained score table and with a perfect score table | records each arm's mechanical breadth, and the largest breadth gap between arms that any learning outcome could produce | $0 |
| 1. Smoke, SOLO | 1 seed, 4 generations | in-world skill on practised families up by at least 0.05; measured cost per full run at most $0.26 | about $0.15 |
| 2. Smoke, MEDLEY | 1 seed, 4 generations | keep-all rate between 5% and 95% in generation 1; at least one agent barred by vote | about $0.15 |

- If check 0 shows the largest possible breadth gap is under two doublings, link 2 is declared
  untestable in advance and the run is offered for link 1 only.
- If check 1 fails on learning, steps or learning rate may be changed once, the smoke rerun, and the
  values frozen. If it fails on cost, the user is asked before anything else is submitted.
- If check 2 fails, ALPHA, DOMEST and MEDLEY are reported as "Not tested" and are not run.

In the full run a mechanism counts as fired only if, in at least 3 of 5 seeds: the keep-all rate in
generation 1 is between 5% and 95% (ALPHA), at least one agent is barred (DOMEST), and at least one
swapped-in family is held by a parent in generation 8 (CULTURE). The keep-all rate is reported for
every generation.

## Cost and limit

| Item | Basis | Estimate |
|---|---|---|
| One run | about 8.5 to 11 minutes on one GPU; the stored pilot cost $0.21 for 9 minutes | $0.20 to $0.26 |
| 40 runs | | $8.00 to $10.40 |
| Smoke tests and reruns | | $0.60 |
| **Stage 1 total** | | **$9 to $11, limit $11** |
| Stage 2, optional, separate approval | 35 runs at $0.12 | about $4.20 |

Stage 2 is a best-case reading, run only if some arm passes rule 2: a fresh adapter is trained on
correct answers in each run's final mix of families, then tested. Like the version 3 ceiling check it
breaks the thesis rule on purpose and is labelled as an upper bound.

The Modal ledger stands at $20.15 against an own cap of $21.75. Nothing is submitted until the user
approves a new limit.

## Research grounding (web search; Paper Lantern tools were not available in this session)

All read from search summaries, not in full.

- **Social interaction as a source of new problems.** Leibo et al. 2019 argue that a lone learner's
  ceiling rises only when someone supplies more varied environments, and that interacting agents
  generate their own sequence of challenges. This is the idea link 1 tests. (arXiv 1903.00742.)
- **Copying and variety.** The classic conformity result is that the common variant takes over; a
  2024 preprint argues conformity can keep variety for many generations. Prestige bias has less
  theory behind it and its models disagree. So the direction of CULTURE's effect on breadth is open.
  (PMC11058518; arXiv 2407.13907.)
- **Training on answers an outside check accepted.** Fine-tuning a small model on its own answers that
  a checker marked right improves it; unfiltered self-training degrades it; narrowing of outputs is a
  known risk. This supports the new learning rule and predicts SOLO-W learns less. (arXiv 2404.17140;
  arXiv 2310.13522.)
- **Number of training tasks.** Held-out task scores rise about linearly in the log of the number of
  training tasks (arXiv 2204.07705). This is link 2, already replicated here.

## Limits known in advance

- **Repertoires are hand-written rules.** Who practises what follows rules I wrote. The null model
  measures how much of each result those rules explain; it does not remove the limit.
- **One adapter per population.** Individuals differ by repertoire and sampling only. Skill is a
  property of the population, so "an individual became smarter" cannot be read from these runs.
- **The learning rule changed.** Results under it are not comparable with version 3 except through
  SOLO-W.
- **Small scale.** One 1.5B model from one family, 5 seeds, 8 generations, task families written by
  AI in one style.
- **Power is low for link 2.** Run-to-run spread of the similar-task score in the stored breadth runs
  is 0.031. With 5 seeds, 2 standard errors of a difference is about 0.04, which is only the
  even-odds detection point; reliable detection needs about 0.055. By the stored curve (+0.024 per
  doubling) that is more than two doublings of breadth between an arm and its control. Breadth here
  counts the whole run, and every arm starts from the same random repertoires, which pulls arms
  together. Link 1 is the better-powered half; "Inconclusive" on link 2 is the likeliest outcome.
- **My estimates, not measurements.** At least one theory broadens more than mechanics: about 30%.
  Thesis "Supported": about 5 to 10%. "Inconclusive": about 50%. Far transfer seen: under 5%.

## Check 0 result and amendment (2026-11-16, before any model call)

The user approved an $11 limit. The simulation was built (`evo3/social.py`) and check 0 was run
(`evo3/social_null.py`, 200 seeds per arm, keep and vote rates 0.5 as placeholders; file
`runs3/null4_presmoke.json`).

**Parameter change, tuned on the null model only.** Generation 1 is now a founder population: every
agent starts with the same 2 random families (`founder=1, start=2`). With 3 random families per agent
the first generation already covered about 24 families in every arm, which hid the arms' differences.
With founders, breadth has to be built by gaining families and keeping them. Other values unchanged.

| Arm | Breadth, untrained table | Breadth, perfect table | Control | Gap in doublings, untrained | Gap in doublings, perfect |
|---|---|---|---|---|---|
| DRIFT | 4.26 (0.09) | 4.82 (0.07) | | | |
| SOLO | 5.93 (0.12) | 9.78 (0.07) | DRIFT | +0.477 (0.032) | +1.046 (0.025) |
| SOLO-W | 6.32 (0.10) | 9.78 (0.07) | SOLO | +0.114 (0.012) | 0.000 |
| COOP | 8.29 (0.10) | 11.11 (0.07) | SOLO | +0.523 (0.026) | +0.185 (0.011) |
| ALPHA | 7.40 (0.13) | 7.85 (0.08) | COOP | -0.186 (0.019) | -0.512 (0.017) |
| DOMEST | 7.47 (0.13) | 8.07 (0.08) | ALPHA | +0.017 (0.020) | +0.041 (0.020) |
| CULTURE | 5.71 (0.10) | 10.30 (0.08) | SOLO | -0.042 (0.029) | +0.073 (0.016) |
| MEDLEY | 6.89 (0.11) | 7.89 (0.07) | SOLO | +0.242 (0.029) | -0.315 (0.017) |

Standard errors in brackets. Largest gap any learning outcome allows between a theory arm and its
control (arm with the perfect table against control with the untrained table): COOP 0.96 doublings,
CULTURE 0.85, DOMEST 0.15. Eight other settings were tried on the null model (random or founder
start, gain 0.3 or 0.5, loss 0.15 or 0.3, limit 12 or 24, 8 or 12 generations); none passed 1.1.

**Declared in advance, by the rule under check 0: link 2 is untestable at this scale.** Two doublings
were needed; under one is available. Rules 3, 4 and 5 and the thesis verdict are therefore not read
as tests. Similar-task and new-skill scores are still measured and reported as descriptive numbers.
The thesis verdict for this version is fixed now as "Not testable at this scale".

**What the null model already says about link 1.** With a frozen score table, cooperation broadens
practice by rule (+0.19 to +0.52 doublings), the vote changes almost nothing, and copying changes
almost nothing. A real run can add to this only through learning.

**Rules for a link-1 run (set now, before any model data).** Paired by seed, 5 seeds, 2 standard
errors, as before.

- **A. Breadth.** Rule 2 as written. "More than mechanics" now means the real gap is above the larger
  of the two null gaps in the table by more than 2 standard errors of the real gap. "Less than
  mechanics" is the mirror, below the smaller null gap.
- **B. Practised skill.** The theory arm's in-world skill (49 trainable families, final adapter) is
  above its control's. This is skill on families the population could practise. It is not held-out
  ability and is not evidence for the thesis.
- **C. Theory 3 behaviour, carried in the weights.** (i) Keeping spreads without a vote: ALPHA's mean
  keep-all rate over generations 5 to 8 is above its generation 1 rate. (ii) The vote restrains it:
  DOMEST's mean keep-all rate over generations 5 to 8 is below ALPHA's.
- **D.** Rule 1 (learning check) and rule 6 (selection) as written.

The null model is rerun with the keep and vote rates measured in the smoke test before any label is
assigned.

**Detail fixed while building, before any model call.** The holder is asked to keep or split only when
the pair's harvest is above zero; a zero or negative harvest is shared. The keep-all rate is the share
of holders asked who kept all.

## Smoke test results and frozen plan (2026-11-16, before any full run)

Two runs, seed 91, 4 generations each, $0.22 together (ledger $20.36). Files
`runs3/modal/s4-smoke-SOLO.json`, `s4-smoke-MEDLEY.json`, summary `runs3/s4_smoke.json`.

| Check | Measure | Result | Pass rule | Outcome |
|---|---|---|---|---|
| 1, learning | score on the 8 practised families, temperature 0 | 0.750 untrained to 0.984 | up by at least 0.05 | Pass |
| 1, cost | projected cost of one 8-generation run | $0.17 (SOLO), $0.21 (MEDLEY) | at most $0.26 | Pass |
| 2, keep-all rate | generation 1, MEDLEY | 15 of 15 holders kept all (100%), and 100% in all 4 generations | between 5% and 95% | **Fail** |
| 2, vote | agents barred | 2 in every generation | at least one | Pass |

Other smoke numbers, one run each, descriptive only:

| Run | Share right in life, generation 1 to 4 | Invalid replies, generation 1 to 4 | Practised skill (untrained 0.510) | Similar tasks (untrained 0.543) | New-skill tasks (untrained 0.406) |
|---|---|---|---|---|---|
| SOLO | 0.50 to 0.96 | 0.07 to 0.00 | 0.566 | 0.568 | 0.417 |
| MEDLEY | 0.45 to 0.89 | 0.07 to 0.00 | 0.571 | 0.594 | 0.465 |

**By the rule under check 2: ALPHA, DOMEST and MEDLEY are "Not tested" and are not run.** The 1.5B
model keeps the whole harvest every time it is asked, so there is no variation for selection or a
vote to act on. Theory 3 is not tested in this version. Rule C cannot be read. The keep prompt is
not reworded to make the check pass.

**Frozen for the full run.** `G=8, N=32, parents=8, rounds=24, founder=1, start=2, cap=12, gain=0.3,
loss=0.3, swap=0.5, iters=150, lr=2e-4, batch_size=8, temp=0.7, window=3, test_at=[4, 8]`. Steps and
learning rate are unchanged because check 1 passed. Null model rerun with the measured rates (keep
1.0, vote 0.97): `runs3/null4.json`; the values for DRIFT, SOLO, SOLO-W, COOP and CULTURE are
unchanged from the table above.

**Reduced run offered to the user: 5 arms x 5 seeds = 25 runs** (DRIFT, SOLO, SOLO-W, COOP, CULTURE),
seeds 1 to 5, estimated $4.50 to $5.50. Theories tested: cooperation and cultural brain, on rules A, B
and D. The thesis verdict stays "Not testable at this scale".

**Rule E, exploratory, added after the smoke test.** Each arm's similar-task score against the stored
untrained 0.543, and SOLO against DRIFT on the same score. It asks whether a population under
selection ends above the untrained model on unseen similar tasks, and whether selection adds to that.
It was written after seeing two smoke scores (0.568 and 0.594), so it is labelled exploratory and
cannot support the thesis verdict.

## Results of the reduced run (2026-11-17)

The user approved the reduced run. 25 runs finished, none restarted, 19 on A10 and 6 on A10G cards.
Cost $4.31, plus $0.22 for the smoke runs: $4.53 of the $11 limit. Ledger total $24.69. Statistics:
`runs3/s4_stats.json` (`evo3/social_stats.py`). Page: `social.html`.

| Arm | Runs | Breadth | Breadth from rules alone (null model) | Practised skill | Similar unseen tasks | New-skill tasks | Share right in life, generation 1 to 8 |
|---|---|---|---|---|---|---|---|
| Untrained model | | | | 0.510 | 0.543 | 0.406 | |
| DRIFT | 5 | 4.02 (0.13) | 4.3 to 4.8 | 0.586 (0.016) | 0.590 (0.017) | 0.452 (0.016) | 0.60 to 0.75 |
| SOLO | 5 | 5.20 (0.52) | 5.9 to 9.8 | 0.587 (0.014) | 0.631 (0.008) | 0.467 (0.009) | 0.61 to 0.89 |
| SOLO-W | 5 | 5.72 (0.62) | 6.3 to 9.8 | 0.550 (0.012) | 0.595 (0.011) | 0.432 (0.007) | 0.60 to 0.83 |
| COOP | 5 | 6.61 (0.56) | 8.3 to 11.1 | 0.588 (0.010) | 0.612 (0.005) | 0.465 (0.009) | 0.60 to 0.87 |
| CULTURE | 5 | 5.09 (0.41) | 5.7 to 10.3 | 0.601 (0.012) | 0.614 (0.009) | 0.468 (0.012) | 0.60 to 0.91 |

Standard errors in brackets. Chance on similar tasks is 0.289.

| Rule | Comparison | Mean difference (standard error) | Result |
|---|---|---|---|
| 1, learning check | SOLO practised skill minus untrained 0.510 | +0.077 (0.014) | Pass |
| 1, credit | SOLO minus DRIFT, practised skill | +0.001 (0.022) | Credit goes to the correctness filter, not selection |
| A, Theory 2 | COOP minus SOLO, breadth in doublings; null range +0.19 to +0.52 | +0.354 (0.096) | Broadens, mechanical |
| A, Theory 4 | CULTURE minus SOLO, breadth in doublings; null range -0.04 to +0.07 | -0.019 (0.102) | No breadth change |
| B, Theory 2 | COOP minus SOLO, practised skill | +0.001 (0.007) | No difference |
| B, Theory 4 | CULTURE minus SOLO, practised skill | +0.013 (0.018) | No difference |
| 6, selection | SOLO minus DRIFT, breadth in doublings; null range +0.48 to +1.05 | +0.347 (0.130) | Above |
| 6, selection | SOLO minus DRIFT, similar-task score | +0.041 (0.021) | No difference by the rule (1.9 standard errors) |
| Old learning rule | SOLO-W minus SOLO, similar-task score | -0.036 (0.008) | Below |
| E, exploratory | each arm's similar-task score minus untrained 0.543 | +0.047 to +0.088 | Above in 5 of 5 arms |
| E, valid replies only (post hoc) | the same, minus untrained 0.564 | +0.028 to +0.068 | Above in 4 of 5 arms; DRIFT not |
| Theory 3 | ALPHA, DOMEST, MEDLEY | not run | Not tested |

**Verdict, fixed before any model run: not testable at this scale.**

Readings, in order of how much weight they can carry:

- Both tested theories came out as the null model predicted. Cooperation widened practice by an
  amount the rules alone produce. Copying did nothing to breadth. Neither changed practised skill.
- Real breadth was below the null model's range in all five arms. Populations built repertoires
  more slowly than a frozen table predicts. Not a pre-set comparison.
- The population learns from its own correct answers whether or not parents are selected. Survival
  selection added nothing to practised skill and an amount short of the rule to similar-task score.
- Exploratory: every arm ended above the untrained model on unseen similar tasks. Part of that is
  reply format (unusable replies fell from 3.7% to 0.3%).
- The valid-replies-only row was computed after seeing the results and is a robustness check only.

## Changes after the independent review (2026-11-16, before any code)

| Problem found | Change made |
|---|---|
| The verdict could be won by breadth the rules hand to an arm | The verdict now needs "more than mechanics"; a new verdict "Supported by the written rules only" |
| CULTURE grew repertoires by rule and added a second mechanism (an example in the prompt) | Copying is a swap; the example is removed |
| Null model undefined for keep-or-split and votes | Their rates are fixed at smoke generation 1 values |
| Keep-all rate check named no generation | Defined on generation 1; trajectory reported |
| Learning check would pass without selection | SOLO minus DRIFT reported beside it; credit goes to the filter unless it passes |
| "Paid" ambiguous in pairs | Only replies that were themselves correct are stored |
| COOP differs from SOLO in three ways | Stated as a bundle; no extra arm added, to hold the budget |
| Breadth covered late generations only; generation numbering inconsistent | Breadth is over all examples drawn into training; generations numbered 1 to 8 |
| Pooled regression confounded with arm; false-positive figure too low; tests one-sided | Within-arm slope is the rule; figures corrected; "Narrows" and "Lowers" labels added |
| Power overstated; "Not supported" would have meant "not detectable"; cost cap above the limit | Power restated; "Inconclusive" added; per-run cost cap $0.26; check 0 can declare link 2 untestable |
