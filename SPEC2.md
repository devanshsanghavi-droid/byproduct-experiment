# Version 2: text evolves (draft, not yet built beyond the world and the headroom check)

Thesis under test, unchanged: "All three surviving theories say intelligence was a byproduct. A faithful
simulation therefore scores agents only on staying alive and being copied, then measures general ability
separately on held-out tests."

## Why version 1 was replaced

Version 1 evolved four dials that limited a frozen model, and read "ability" off a table indexed by two
of those dials. Nothing in it could make the model more capable, and the ability score was circular.

## What evolves

| Channel | What it is | Inheritance | Variation | Limit |
|---|---|---|---|---|
| Instinct | up to 6 short rules placed in the agent's prompt | from one parent | blind: delete a rule, copy a rule from another survivor, or a model-written new or reworded rule that never sees outcomes | 14 words per rule |
| Note | text the agent writes at the end of its life from its own records | read by offspring (parent's note plus the top earner's note) | written from experience | 40 words |

The capacity dials are gone. Every agent sees all of its own records and may write a short plan.

## World (built: evo2/world.py)

- 3 plants per generation, 18 rounds, each plant needs 3 ordered steps from 4 actions.
- Feedback after each try: how many steps are right and in place. +4 for a full success, -1 otherwise.
- Recipes and plant names are redrawn every generation, so specific recipes are not worth inheriting.
- Two hidden laws hold in every generation: no action repeats in a recipe; "dry" is never first.
  These are the only world knowledge worth inheriting. 18 of 64 recipes are possible.

## Arms

| Arm | Tests | Foraging | Grab | Exile vote | Notes | Parents |
|---|---|---|---|---|---|---|
| DRIFT | chance only; also shows what the mutation writer adds by itself | solo | no | no | no | random half |
| SOLO | ecology only | solo | no | no | no | top half by energy |
| COOP | Theory 2 | pairs re-formed each round; one advises, one acts; shared harvest | no | no | no | top half |
| ALPHA | control for 3 | pairs; actor may grab the harvest | yes | no | no | top half |
| DOMEST | Theory 3 | as ALPHA; members see who grabbed from them; 2 votes exile | yes | yes | no | top half, exiled excluded |
| CULTURE | Theory 4 | solo | no | no | yes | top half |
| MEDLEY | 2 + 3 + 4 | pairs | yes | yes | yes | top half, exiled excluded |

Grabbing and voting are choices made by the agent under its instincts, so both dispositions can evolve.

## Measurement (built: evo2/far.py)

Every score is a real model call made with the agent's evolved text in the prompt. None is used for selection.

| Test | Distance | What it is |
|---|---|---|
| near | same world laws, new recipes | a solo season in a fresh world |
| mid | same kind of search, new surface, no laws | 4-letter code-breaking |
| far 1 | different task | rule induction with self-chosen probes |
| far 2 | different task | ordering from shuffled clues |

Reference rows: the bare model with no evolved text, the generation 0 population, and DRIFT.

## Headroom check, run 2026-11-11 (evo2/headroom.py, 24 repeats, same task instances for every row)

Hidden reasoning on (default):

| Prompt text | near energy | mid code | far rule | far order |
|---|---|---|---|---|
| none (bare model) | 35.8 | 0.467 | 0.776 | 1.00 |
| general method rules | 32.6 | 0.483 | 0.776 | 1.00 |
| method + the two world laws | 39.7 | 0.467 | 0.786 | 1.00 |
| harmful rules | 34.7 | 0.467 | 0.776 | 1.00 |
| standard error | about 1.7 | about 0.03 | about 0.055 | 0 |

Hidden reasoning off:

| Prompt text | near energy | mid code | far rule | far order |
|---|---|---|---|---|
| none (bare model) | 30.5 | 0.429 | 0.667 | 1.00 |
| general method rules | 28.7 | 0.412 | 0.724 | 1.00 |
| method + the two world laws | 36.8 | 0.388 | 0.635 | 1.00 |
| code-breaking algorithm | 29.1 | 0.396 | 0.641 | 0.98 |
| harmful rules | 14.3 | 0.254 | 0.661 | 0.98 |
| standard error | about 2.0 to 3.7 | about 0.03 to 0.05 | about 0.045 | 0 to 0.02 |

Reading:

1. No helpful text raised any transfer score above the bare model, with reasoning on or off.
2. World knowledge helps locally: the two laws add about 4 to 6 energy on the near test.
3. With reasoning off, harmful text hurts the near and mid tests, so selection has something to remove.
4. The ordering test is at its maximum and cannot show a gain.

Consequence: in this setup evolved text can add local knowledge and can remove damage. It has shown no
room to make the frozen model better at unseen tasks than it already is. A run that hopes for that gain
would most likely report noise. Raising general ability needs the weights to change.

## Options

A. Knowledge accumulation (recommended). Run the arms above with reasoning off. Ask which pressures make
   a population accumulate useful world knowledge fastest. Keep mid and far tests as a check, with the
   stated expectation that they stay flat or only recover up to the bare model.
B. Recovery from damage. Seed generation 0 with mixed good and harmful instincts and measure how far
   selection restores transfer scores toward the bare model. Ceiling is the bare model.
C. Weights evolve. Fine-tune a small open-weights model on survivors each generation. This is the only
   design that can raise general ability. It cannot be done with the Claude API.
