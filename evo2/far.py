"""Transfer tests for version 2. Every score comes from real model calls made with the
agent's evolved text in the prompt. Nothing here is ever used for selection.

    mid   code-breaking  same kind of search as the world, new surface, no hidden laws
    far1  rule induction probe a hidden number rule, then classify unseen cases
    far2  ordering       one-shot deduction from shuffled clues
"""
import random

from evo.heldout import RULES, random_triple, sample_triple, as_triple, as_bool
from evo2.world import agent_block

CODE_LEN, CODE_SYMBOLS, CODE_TURNS = 4, "ABCDE", 10
PROBES, TESTS = 6, 8
RUNNERS = 8


async def code_breaking(llm, rng: random.Random, instinct=None, note=None) -> float:
    code = [rng.choice(CODE_SYMBOLS) for _ in range(CODE_LEN)]
    system = (f"A hidden code has {CODE_LEN} positions. Each position holds one of the letters "
              f"{', '.join(CODE_SYMBOLS)}. Letters may repeat.\nAfter each guess you learn how many positions "
              f"are exactly right. You have {CODE_TURNS} guesses. Find the code in as few guesses as you can."
              + agent_block(instinct, note) + "\n"
              'Reply with only JSON: {"plan": "<at most 30 words>", "guess": "<' + str(CODE_LEN) + ' letters>"}')
    hist = []
    for t in range(1, CODE_TURNS + 1):
        user = (f"Guess {t} of {CODE_TURNS}.\n"
                + ("Your guesses so far:\n" + "\n".join(f"- {g} -> {k} of {CODE_LEN} right" for g, k in hist)
                   if hist else "Your guesses so far: none"))
        d = await llm.ask_json(system, user, mock_fn=lambda: {"plan": "", "guess": "".join(rng.choice(CODE_SYMBOLS) for _ in range(CODE_LEN))})
        g = "".join(ch for ch in str(d.get("guess", "")).upper() if ch in CODE_SYMBOLS)[:CODE_LEN]
        if len(g) != CODE_LEN:
            g = "".join(rng.choice(CODE_SYMBOLS) for _ in range(CODE_LEN))
        k = sum(a == b for a, b in zip(code, g))
        if k == CODE_LEN:
            return 1.0 - (t - 1) / CODE_TURNS   # earlier solve scores higher; unsolved scores 0
        hist.append((g, k))
    return 0.0


async def rule_induction(llm, rng: random.Random, instinct=None, note=None) -> float:
    name, rule = rng.choice(RULES)
    first = sample_triple(rng, rule, True)
    system = ("A hidden rule sorts triples of whole numbers from 1 to 12 into those that fit and those that do not.\n"
              f"You may test {PROBES} triples of your own, one at a time, and are told whether each fits. "
              f"Then you must judge {TESTS} new triples. You score one point per correct judgment."
              + agent_block(instinct, note))
    seen = [(first, True)]
    for p in range(1, PROBES + 1):
        user = (f"Test {p} of {PROBES}.\nKnown so far:\n"
                + "\n".join(f"- {t}: {'fits' if f else 'does not fit'}" for t, f in seen)
                + '\nReply with only JSON: {"plan": "<at most 30 words>", "triple": [a, b, c]}')
        d = await llm.ask_json(system, user, mock_fn=lambda: {"plan": "", "triple": random_triple(rng)})
        t = as_triple(d.get("triple")) or random_triple(rng)
        seen.append((t, bool(rule(t))))
    tests = [sample_triple(rng, rule, i % 2 == 0) for i in range(TESTS)]
    rng.shuffle(tests)
    user = ("All tests are used.\nKnown:\n" + "\n".join(f"- {t}: {'fits' if f else 'does not fit'}" for t, f in seen)
            + "\nJudge these triples in order:\n" + "\n".join(f"{i + 1}. {t}" for i, t in enumerate(tests))
            + '\nReply with only JSON: {"plan": "<at most 30 words>", "fits": [true or false for each, in order]}')
    d = await llm.ask_json(system, user, mock_fn=lambda: {"plan": "", "fits": [rng.random() < 0.5 for _ in tests]})
    ans = d.get("fits") if isinstance(d.get("fits"), list) else []
    right = 0
    for i, t in enumerate(tests):
        a = as_bool(ans[i]) if i < len(ans) else None
        right += (a is not None and a == bool(rule(t)))
    return right / TESTS


async def ordering(llm, rng: random.Random, instinct=None, note=None) -> float:
    names = rng.sample(["Ana", "Ben", "Cal", "Dee", "Eli", "Fay", "Gus", "Hal", "Ivy", "Jon"], RUNNERS)
    clues = []
    for a, b in zip(names, names[1:]):
        clues.append(f"{a} finished just ahead of {b}." if rng.random() < 0.5 else f"{b} finished just behind {a}.")
    rng.shuffle(clues)
    k = rng.randrange(RUNNERS)
    place = ["1st", "2nd", "3rd", "4th", "5th", "6th", "7th", "8th"][k]
    system = (f"{RUNNERS} runners finished a race with no ties. Work out the order from the clues."
              + agent_block(instinct, note) + "\n"
              'Reply with only JSON: {"plan": "<at most 30 words>", "answer": "<name>"}')
    user = "Clues:\n" + "\n".join("- " + c for c in clues) + f"\nWho finished {place}?"
    d = await llm.ask_json(system, user, mock_fn=lambda: {"plan": "", "answer": rng.choice(names)})
    return float(str(d.get("answer", "")).strip().lower() == names[k].lower())
