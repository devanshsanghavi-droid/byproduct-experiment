# Byproduct experiment - shared spec

## Thesis (use as the headline sentence of every write-up)

All three surviving theories say intelligence was a byproduct. A faithful
simulation therefore scores agents only on staying alive and being copied,
then measures general ability separately on held-out tests.

The thesis may turn out wrong. Reports state results for and against it.

## What evolves

Model weights are frozen. What evolves is a 4-gene genome that gates how much
of the model's cognition an agent may use. Capacity genes cost energy, as
brains do. See `evo/genome.py` (already written, do not change its interface).

| Gene | Levels 0-4 map to | Costly |
|---|---|---|
| MEM | own past trial records visible: 0, 2, 4, 8, 16 (most recent kept) | yes |
| TALK | max words per message or gossip line: 0, 3, 6, 12, 24 | yes |
| LORE | max words of inherited notes read / written: 0, 10, 25, 50, 100 | yes |
| AGG | probability of grabbing after a joint success: 0, .2, .4, .6, .8 | no |

- Upkeep = `cost` x (MEM + TALK + LORE) energy per lifetime. Default cost 1.5.
- Fitness = net energy = gross foraging energy - upkeep. Nothing else is scored.
- Mutation is blind and done in code: each gene +-1 with probability 0.25.
- Generation 0 genomes are uniform random and identical across arms for a seed.

## World (identical in every arm for a given seed)

- 4 plants with nonsense names generated from the seed (e.g. "zuvo").
- 4 actions: soak, pound, roast, dry.
- Each plant has a hidden recipe: an ordered pair of actions (repeats allowed, 16 options).
- A lifetime is R rounds. Each round each foraging unit gets one random in-season plant.
- Correct recipe: +4 energy per agent. Wrong: -1 per agent.
- At each new generation one random plant's recipe is re-randomized.
- Prompts state rules factually. They never coach strategy.

## Arms

| Arm | Theory | Foraging | Grab | Vote | Notes | Parents chosen by |
|---|---|---|---|---|---|---|
| DRIFT | 1, chance alone | solo | - | - | - | random half |
| SOLO | baseline, ecology only | solo | - | - | - | top half by net energy |
| COOP | 2, cooperation | pairs | - | - | - | top half |
| ALPHA | control for 3 | pairs | yes | - | - | top half |
| DOMEST | 3, self-domestication | pairs | yes | yes | - | top half |
| CULTURE | 4, cultural brain | solo | - | - | yes | top half |
| MEDLEY | 2 + 3 + 4 | pairs | yes | yes | yes | top half |

Each chosen parent has 2 offspring. Population size N is even and constant.

### Solo foraging (1 call per agent per round)

Agent sees: rules, its MEM-limited records, inherited notes if the arm has
them, the in-season plant. Replies JSON `{"step1": "...", "step2": "..."}`.
Feedback: right/wrong for each step. Record stored in memory.

### Pair foraging (obligate; up to 2 calls per agent per round)

1. Matching: mutual preferences first, then one-sided, then random. Preferences
   come from the previous round's action reply (`prefer_partner`, an agent id or null).
2. Roles: random. "first" picks step 1, "second" picks step 2.
3. Message phase: each agent writes one message to its partner, clipped in code
   to TALK words. TALK 0 means no call and an empty message.
4. Action phase: agent sees partner's message, replies JSON
   `{"step": "...", "prefer_partner": "a3" | null, "gossip": "..." | null}`.
   `gossip` is only used in arms with Vote, clipped to TALK words, posted to a
   public board (everyone sees the last 10 lines).
5. Outcome: both right gives a pot of 8. Split views: "first" learns only whether
   step 2 was right, "second" learns only whether step 1 was right. Both see the
   joint prep and the pass/fail outcome.
6. Grab (arms with Grab, decided in code from AGG, no call): after a success each
   agent grabs with its grab probability. One grabs: 7 / 1. Both grab: 2 / 2.
   Neither: 4 / 4. The victim's memory record says who grabbed.

### Vote (arms with Vote; 1 call per agent per lifetime)

At the end of the lifetime each agent sees its MEM-limited records, the gossip
board, and the id list, and replies `{"exile": "a7" | "none"}`. An agent named
by at least N/4 voters is exiled: it has no offspring. At most N/4 exiles per
generation, most votes first. If too few eligible parents remain, the survivors
fill the offspring slots.

### Notes (arms with Notes; 1 call per agent per lifetime)

At the end of the lifetime each agent with LORE > 0 writes a note, clipped to
its LORE words. Each newborn receives its parent's note and the note of the
previous generation's top net-energy agent, each clipped to the newborn's LORE words.

## Agent prompt framing

System prompt: the rules above, in plain words, as they apply to that arm.
State the goal as "end the season with as much energy as you can" (plus "and
avoid exile" where Vote applies). Do not mention the genome, evolution,
intelligence, or the experiment. Agent ids are a0..a{N-1} within a generation.

## Code layout

| File | Owner | Purpose |
|---|---|---|
| `evo/llm.py` | done | Async Claude wrapper, cost cap, mock mode |
| `evo/genome.py` | done | Genes, level tables, mutation, upkeep, word clipping |
| `evo/sim.py` | agent A | World, all 7 arms, evolution loop, checkpoints |
| `evo/heldout.py` | agent B | Ability surface on tasks never used for selection |
| `evo/analyze.py` | agent C | runs/*.json + heldout.json -> results.json |
| `evo/build_report.py`, `report_template.html` | agent C | results.json -> report.html |

Every script must run end to end with `--mock` (zero API spend).

## CLI contracts

```
python3 -m evo.sim --arms all --seeds 1,2,3 --N 12 --R 12 --G 10 --cost 1.5 \
    --model claude-haiku-5-5 --budget 25 --concurrency 16 --out runs [--mock] [--resume]
python3 -m evo.sim --pilot ...        # N=8 R=8 G=3 seeds=1, all arms
python3 -m evo.heldout --model claude-haiku-5-5 --budget 8 --reps 24 --out runs/heldout.json [--mock]
python3 -m evo.analyze --runs runs --out results.json
python3 -m evo.build_report --results results.json --out report.html
```

sim.py runs every (arm, seed) concurrently on one shared `LLM` instance, prints
one progress line per finished generation, rewrites the run file after every
generation, and with `--resume` skips finished runs and continues partial ones.
On `BudgetExceeded` it saves what it has and exits cleanly.

## Run file: `runs/{ARM}_s{seed}.json`

```json
{
  "arm": "COOP", "seed": 1, "model": "claude-haiku-5-5", "mock": false,
  "config": {"N": 12, "R": 12, "G": 10, "cost": 1.5},
  "complete": true,
  "generations": [
    {"gen": 0,
     "agents": [
       {"id": "a3", "parent": null,
        "genome": {"MEM": 2, "TALK": 1, "LORE": 0, "AGG": 3},
        "gross": 14, "upkeep": 4.5, "net": 9.5,
        "successes": 5, "trials": 12,
        "grabs": 0, "grabbed": 0, "votes_against": 0, "exiled": false,
        "note": ""}
     ],
     "mean": {"MEM": 2.1, "TALK": 1.9, "LORE": 2.0, "AGG": 2.2,
              "gross": 9.0, "net": 0.1, "success_rate": 0.31},
     "exiled": 0, "parse_fail": 0}
  ],
  "final_population": [{"genome": {"MEM": 3, "TALK": 2, "LORE": 0, "AGG": 1}, "parent": "a2"}],
  "usage": {"calls": 1440, "cost_usd": 0.31},
  "samples": {"messages": [], "gossip": [], "notes": []}
}
```

`final_population` is the unplayed newborn generation after the last selection.
`samples` holds up to 20 real examples of each kind, with generation and agent id.

## Held-out ability surface: `runs/heldout.json`

Tasks never used for selection, so scores are the byproduct measure. Ability is
measured per gene level, then looked up for each population.

| Test | Depends on | Task | Score 0-1 |
|---|---|---|---|
| H1 code-breaking | MEM | Hidden 3-symbol code from 4 symbols, per-position feedback, 8 turns, only MEM-limited history visible | 1 if solved, scaled down by turns used |
| H2 stranger relay | MEM, TALK | Tested agent and a fixed partner (MEM 2, TALK 2) must pass a hidden 4-attribute target across TALK-limited messages, both roles | fraction of attributes right |
| H3 rule induction | MEM | Hidden rule over number triples, 6 probes with yes/no feedback, MEM-limited history, then classify 8 test triples | accuracy |

```json
{"model": "...", "mock": false, "reps": 24,
 "H1": {"0": {"mean": 0.1, "sd": 0.2, "n": 24}, "1": {}, "2": {}, "3": {}, "4": {}},
 "H3": {"0": {}, "...": {}},
 "H2": {"0,0": {"mean": 0.25, "sd": 0.1, "n": 12}, "0,1": {}, "...": {}},
 "usage": {"calls": 0, "cost_usd": 0}}
```

H2 keys are "MEM,TALK". An agent's ability on a test is the surface value at its genes.

## Report requirements

- Headline sentence is the thesis above.
- Every measurement appears in a table with the DRIFT row (noise floor) and the SOLO row (baseline).
- Show per-seed values, not only means. Three seeds is a small sample; say so.
- State the limits: frozen weights, the model already knows human strategy, held-out
  scores are a function of the evolved capacity genes by construction.
