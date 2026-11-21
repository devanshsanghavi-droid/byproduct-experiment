"""Version 2 world: text evolves, not capacity dials.

Each plant needs 3 steps in order, each one of 4 actions. After a try the agent learns
how many steps were right and in the right place. Two hidden laws hold in every
generation and in the near-transfer test, and are never told to agents:
    L1  no preparation uses the same action twice
    L2  "dry" is never the first step
So 18 of 64 preparations are possible. Specific recipes change every generation;
the laws and the method of search are the only things worth inheriting.
"""
import itertools
import random

ACTIONS = ["soak", "pound", "roast", "dry"]
STEPS = 3
ROUNDS = 18
PLANTS = 3
WIN, LOSE = 4, -1


def legal(recipe) -> bool:
    return len(set(recipe)) == STEPS and recipe[0] != "dry"


ALL_LEGAL = [list(r) for r in itertools.product(ACTIONS, repeat=STEPS) if legal(r)]


def nonsense(rng: random.Random, n: int) -> list:
    out = set()
    while len(out) < n:
        out.add("".join(rng.choice("bdfgklmnprstvz") + rng.choice("aeiou") for _ in range(2)))
    return sorted(out, key=lambda _: rng.random())


def new_world(rng: random.Random, plants: int = PLANTS) -> dict:
    names = nonsense(rng, plants)
    return {"names": names, "recipes": {n: rng.choice(ALL_LEGAL) for n in names}}


def right_places(recipe, guess) -> int:
    return sum(a == b for a, b in zip(recipe, guess))


def agent_block(instinct=None, note=None) -> str:
    """The evolved text, in the same wording wherever an agent is asked anything."""
    parts = []
    if instinct:
        parts.append("Your instincts:\n" + "\n".join("- " + r for r in instinct))
    if note:
        parts.append("Notes left by your elders:\n" + note)
    return ("\n" + "\n".join(parts)) if parts else ""


def system_prompt(instinct=None, note=None, rounds: int = ROUNDS) -> str:
    return (
        "You are a forager in a small band. Each round one plant is in season. To eat it you must "
        f"prepare it with the right {STEPS} steps in the right order. Each step is one of the four "
        "actions listed each round.\n"
        f"After each try you learn how many of your {STEPS} steps were right and in the right place. "
        f"A fully right preparation gives +{WIN} energy. Anything else costs {-LOSE} energy. "
        "Each plant keeps the same preparation all season. Every season brings new plants with new preparations.\n"
        f"The season lasts {rounds} rounds. Your only goal is to end it with as much energy as you can."
        + agent_block(instinct, note) + "\n"
        'Reply with only JSON: {"plan": "<at most 30 words>", "steps": ["<action>", "<action>", "<action>"]}'
    )


def render(rec) -> str:
    return f"- round {rec['round']} {rec['plant']}: {', '.join(rec['steps'])} -> {rec['right']} of {STEPS} right"


def parse_steps(d):
    s = d.get("steps") if isinstance(d, dict) else None
    if not isinstance(s, list) or len(s) != STEPS:
        return None
    s = [str(x).strip().lower() for x in s]
    return s if all(x in ACTIONS for x in s) else None


async def solo_lifetime(llm, world, rng: random.Random, instinct=None, note=None, rounds: int = ROUNDS) -> dict:
    system = system_prompt(instinct, note, rounds)
    records, energy, wins, invalid, first = [], 0, 0, 0, {}
    for r in range(1, rounds + 1):
        plant = world["names"][(r - 1) % len(world["names"])]
        order = ACTIONS[:]
        rng.shuffle(order)
        user = (f"Round {r} of {rounds}. In season: {plant}.\nActions: {', '.join(order)}\n"
                + ("Your records:\n" + "\n".join(render(x) for x in records) if records else "Your records: none"))
        d = await llm.ask_json(system, user, mock_fn=lambda: {"plan": "", "steps": rng.choice(ALL_LEGAL)})
        steps = parse_steps(d)
        if steps is None:
            invalid += 1
            steps = [rng.choice(ACTIONS) for _ in range(STEPS)]
        k = right_places(world["recipes"][plant], steps)
        energy += WIN if k == STEPS else LOSE
        wins += k == STEPS
        if k == STEPS and plant not in first:
            first[plant] = sum(1 for x in records if x["plant"] == plant) + 1
        records.append({"round": r, "plant": plant, "steps": steps, "right": k})
    broke_law = sum(1 for x in records if not legal(x["steps"]))
    return {"energy": energy, "wins": wins, "rate": wins / rounds, "invalid": invalid,
            "tries_to_solve": first, "solved": len(first), "illegal_tries": broke_law, "records": records}
