"""Wider world for version 3: many small one-question task families.

TRAIN families are part of a life when the world config lists them (cfg["wide"]). HELD families
are never part of any life and never trained on; they are used only as a test ("mid").
Research grounding (web search): generalisation to unseen tasks depends on the number of distinct
tasks seen far more than on examples per task (arxiv 2402.10891, 2405.19787). Hence many families
with few rounds each.

Each generator returns {"user", "answer", "kind", ...}. Kinds: int, yesno, word, list.
"""
import json
import random
import re

from evo2.world import nonsense

WIN, LOSE = 4, -1
ITEMS = ["nut", "fish", "hide", "root", "shell", "berry", "egg", "reed"]
COLORS = ["red", "blue", "green", "white", "black"]
SHAPES = ["round", "square", "long", "flat"]
TREES = ["oak", "pine", "birch", "elm", "ash"]
FMT = {"int": '{"answer": <number>}', "yesno": '{"answer": "yes" or "no"}', "word": '{"answer": "<one word>"}',
       "list": '{"answer": ["<first>", "<second>", ...]}'}


def q(user, answer, kind="int", **kw):
    return dict(user=user, answer=answer, kind=kind, **kw)


def yn(b):
    return "yes" if b else "no"


# ---- families used in lives (trained on) ------------------------------------------------
def g_count(r):
    kinds, n = r.sample(ITEMS, 3), r.randint(5, 8)
    seq, t = [r.choice(kinds) for _ in range(n)], r.choice(kinds)
    return q(f"Catch today: {', '.join(seq)}. How many times does {t} appear?", seq.count(t), lo=0, hi=n)


def g_tally(r):
    x = v = r.randint(4, 12)
    ops = []
    for _ in range(3):
        d = r.randint(1, 5)
        if r.random() < 0.5 and v - d >= 0:
            v -= d
            ops.append(f"take out {d}")
        else:
            v += d
            ops.append(f"put in {d}")
    return q(f"A pit holds {x} stones. Then: {', '.join(ops)}. How many stones are in the pit now?", v, lo=0, hi=27)


def g_path(r):
    while True:
        moves = [(r.choice(["north", "south", "east", "west"]), r.randint(1, 4)) for _ in range(3)]
        axis = r.choice(["north", "east"])
        opp = {"north": "south", "east": "west"}[axis]
        v = sum(n for d, n in moves if d == axis) - sum(n for d, n in moves if d == opp)
        if v >= 0:
            break
    walk = ", then ".join(f"{n} step{'s' if n > 1 else ''} {d}" for d, n in moves)
    return q(f"You leave camp and walk {walk}. How many steps {axis} of camp are you now?", v, lo=0, hi=12)


def g_series(r):
    kind = r.choice(["add", "mul", "grow"])
    if kind == "add":
        a, d = r.randint(1, 9), r.randint(2, 6)
        s = [a + d * i for i in range(5)]
    elif kind == "mul":
        a, m = r.randint(1, 4), r.choice([2, 3])
        s = [a * m ** i for i in range(5)]
    else:
        a, s = r.randint(1, 9), []
        for i in range(5):
            a += i
            s.append(a)
    return q(f"Marks on the wall: {', '.join(map(str, s[:4]))}. What number comes next?", s[4], lo=1, hi=60)


def g_sort(r):
    names = r.sample(ITEMS, r.choice([3, 4]))
    w = dict(zip(names, r.sample(range(1, 20), len(names))))
    up = r.random() < 0.5
    return q("Weights: " + ", ".join(f"{n} {w[n]}" for n in names) + ". List them from "
             + ("lightest to heaviest." if up else "heaviest to lightest."),
             sorted(names, key=lambda n: w[n], reverse=not up), "list")


CONDS = ["the rope is tight", "the stone is wet", "the wind is high", "the fire is lit", "the gate is shut"]


def g_gate(r):
    a, b = r.sample(CONDS, 2)
    va, vb = r.random() < 0.5, r.random() < 0.5
    form, val = r.choice([(f"{a} and {b}", va and vb), (f"{a} or {b}", va or vb),
                          (f"{a} and it is not true that {b}", va and not vb),
                          (f"it is not true that {a}", not va)])
    return q(f"The trap springs if {form}. Today: {a}: {yn(va)}; {b}: {yn(vb)}. Does the trap spring?", yn(val), "yesno")


def g_decode(r):
    signs, means = nonsense(r, 3), r.sample(ITEMS, 3)
    msg = [r.randrange(3) for _ in range(3)]
    return q("Signs: " + ", ".join(f"{s} means {m}" for s, m in zip(signs, means)) + ". Translate this message, "
             f"sign by sign: {' '.join(signs[i] for i in msg)}", [means[i] for i in msg], "list")


def g_oddone(r):
    a, b = r.sample(COLORS, 2)
    n = r.randint(4, 6)
    k = r.randrange(n)
    return q("Beads in a row: " + ", ".join(b if i == k else a for i in range(n))
             + ". Which position, counting from 1, holds the bead that differs?", k + 1, lo=1, hi=n)


def g_minmax(r):
    xs = r.sample(range(2, 40), r.choice([4, 5]))
    name, i = r.choice([("largest", -1), ("smallest", 0), ("second largest", -2), ("second smallest", 1)])
    return q(f"Fires counted from the hill on each night: {', '.join(map(str, xs))}. What is the {name} count?",
             sorted(xs)[i], lo=2, hi=39)


def g_swap(r):
    cups = r.sample(ITEMS, 3)
    start, steps = list(cups), []
    for _ in range(2):
        i, j = r.sample(range(3), 2)
        cups[i], cups[j] = cups[j], cups[i]
        steps.append(f"swap cups {min(i, j) + 1} and {max(i, j) + 1}")
    k = r.randrange(3)
    return q(f"Cups 1, 2, 3 hold: {', '.join(start)}. First {steps[0]}. Then {steps[1]}. What is in cup {k + 1} now?",
             cups[k], "word", options=start)


def g_share(r):
    k, each, rem = r.randint(2, 6), r.randint(2, 7), r.randint(0, 1)
    rem = r.randrange(k) if rem else 0
    n = k * each + rem
    if r.random() < 0.5:
        return q(f"{n} fish are shared among {k} hunters, the same whole number each, as many as possible. "
                 "How many fish does each hunter get?", each, lo=1, hi=12)
    return q(f"{n} fish are shared among {k} hunters, the same whole number each, as many as possible. "
             "How many fish are left over?", rem, lo=0, hi=5)


def g_pick(r):
    while True:
        names = r.sample(ITEMS, 3)
        w = dict(zip(names, r.sample(range(1, 20), 3)))
        t, more = r.randint(2, 18), r.random() < 0.5
        hit = [n for n in names if (w[n] > t if more else w[n] < t)]
        if len(hit) == 1 and t not in w.values():
            break
    return q("Weights: " + ", ".join(f"{n} {w[n]}" for n in names) + f". Which one weighs {'more' if more else 'less'} than {t}?",
             hit[0], "word", options=names)


def g_countif(r):
    xs, t = [r.randint(1, 20) for _ in range(5)], r.randint(5, 15)
    more = r.random() < 0.5
    xs = [x + 1 if x == t else x for x in xs]
    return q(f"Numbers: {', '.join(map(str, xs))}. How many of them are {'greater' if more else 'smaller'} than {t}?",
             sum(x > t if more else x < t for x in xs), lo=0, hi=5)


# ---- held-out families (never in a life, never trained on) --------------------------------
def h_taboo(r):
    by_color = r.random() < 0.5
    bad_c, ok_c1, ok_c2 = r.sample(COLORS, 3)
    s1, s2, s3 = r.sample(SHAPES, 3)
    if by_color:   # one colour makes people sick, whatever the shape
        sick, fine = [f"{bad_c} {s1}", f"{bad_c} {s2}"], [f"{ok_c1} {s1}", f"{ok_c2} {s2}"]
        safe = r.random() < 0.5
        probe = f"{ok_c1 if safe else bad_c} {s3}"
    else:          # one shape makes people sick, whatever the colour
        sick, fine = [f"{bad_c} {s1}", f"{ok_c1} {s1}"], [f"{bad_c} {s2}", f"{ok_c1} {s3}"]
        safe = r.random() < 0.5
        probe = f"{ok_c2} {s2 if safe else s1}"
    return q(f"People got sick after eating these fruits: {', '.join(sick)}. People were fine after these: "
             f"{', '.join(fine)}. Is a {probe} fruit safe to eat?", yn(safe), "yesno")


def h_rank(r):
    order = r.sample(TREES, 4)   # tallest first
    facts = [f"The {a} is taller than the {b}." if r.random() < 0.5 else f"The {b} is shorter than the {a}."
             for a, b in zip(order, order[1:])]
    r.shuffle(facts)
    top = r.random() < 0.5
    return q(" ".join(facts) + f" Which tree is the {'tallest' if top else 'shortest'}?", order[0 if top else -1],
             "word", options=order)


def h_days(r):
    k, a = r.randint(3, 6), r.randint(1, 5)
    hit = r.random() < 0.5
    b = a + k * r.randint(2, 4) + (0 if hit else r.randint(1, k - 1))
    return q(f"A caravan comes every {k} days. It came on day {a}. Does it come on day {b}?", yn(hit), "yesno")


def h_reverse(r):
    words = nonsense(r, r.choice([3, 4]))
    return q(f"Say these words in reverse order: {', '.join(words)}", words[::-1], "list")


def h_elim(r):
    pots = r.sample(COLORS, 4)
    ans = r.choice(pots)
    nots = [p for p in pots if p != ans]
    r.shuffle(nots)
    return q(f"The honey is in one of four pots: {', '.join(pots)}. "
             + " ".join(f"It is not in the {p} pot." for p in nots) + " Which pot holds the honey?", ans, "word", options=pots)


def h_change(r):
    c, n = r.randint(2, 6), r.randint(2, 4)
    x = c * n + r.randint(1, 12)
    return q(f"You have {x} shells. A knife costs {c} shells. You buy {n} knives. How many shells are left?",
             x - c * n, lo=0, hi=20)


TRAIN = {"count": (g_count, "You are a forager counting your catch."),
         "tally": (g_tally, "You keep the store of your camp and track what goes in and out."),
         "path": (g_path, "You are a scout who keeps track of where you are."),
         "series": (g_series, "You read the marks left by earlier travellers. The marks follow a pattern."),
         "sort": (g_sort, "You load the sledge and must put things in order of weight."),
         "gate": (g_gate, "You set traps. A trap springs exactly when its condition holds."),
         "decode": (g_decode, "You read messages written in signs."),
         "oddone": (g_oddone, "You check strings of beads for the one that differs."),
         "minmax": (g_minmax, "You keep watch and compare counts."),
         "swap": (g_swap, "You watch cups being moved and keep track of what each one holds."),
         "share": (g_share, "You divide the catch among the hunters."),
         "pick": (g_pick, "You choose goods by weight."),
         "countif": (g_countif, "You sort numbers against a limit.")}
HELD = {"taboo": (h_taboo, "You decide which fruits are safe from what happened to others."),
        "rank": (h_rank, "You compare trees from what you are told."),
        "days": (h_days, "You plan around visitors who come on a fixed cycle."),
        "reverse": (h_reverse, "You repeat words back in a new order."),
        "elim": (h_elim, "You search for hidden food."),
        "change": (h_change, "You trade with shells.")}
ALL = dict(TRAIN, **HELD)


def system_for(fam, kind) -> str:
    return f"{ALL[fam][1]} Right answer: +{WIN} energy. Wrong: {LOSE}.\nReply with only JSON: {FMT[kind]}"


def items_for(fam, rng, n) -> list:
    return [ALL[fam][0](rng) for _ in range(n)]


def random_answer(it, rng):
    k = it["kind"]
    if k == "int":
        return rng.randint(it["lo"], it["hi"])
    if k == "yesno":
        return rng.choice(["yes", "no"])
    if k == "word":
        return rng.choice(it["options"])
    return rng.sample(it["answer"], len(it["answer"]))


def check(it, d):
    """(right, valid). A bare answer without JSON counts, as everywhere in version 3."""
    if not isinstance(d, dict):
        return False, False
    got, k = d.get("answer", d.get("_raw")), it["kind"]
    if got is None:
        return False, False
    if k == "int":
        m = re.search(r"-?\d+", str(got))
        return (bool(m) and int(m.group(0)) == it["answer"]), bool(m)
    if k == "list":
        if isinstance(got, str):
            got = [x for x in re.split(r"[,\s]+", got.strip("[] .")) if x]
        if not isinstance(got, list):
            return False, False
        got = [str(x).strip().strip("\"'.").lower() for x in got]
        return got == it["answer"], bool(got)
    if isinstance(got, bool):
        got = yn(got)
    text = str(got).lower()
    opts = ["yes", "no"] if k == "yesno" else it["options"]
    found = [o for o in opts if re.search(rf"\b{re.escape(o)}\b", text)]
    return found == [it["answer"]], len(found) == 1


async def family_life(fam, llm, items, rng, oracle=False, log=None) -> dict:
    wins = invalid = 0
    for it in items:
        system = system_for(fam, it["kind"])
        if oracle:
            d = {"answer": it["answer"]}
            if log is not None:
                log.append({"system": system, "user": it["user"], "out": json.dumps(d)})
        else:
            d = await llm.ask_json(system, it["user"], mock_fn=lambda: {"answer": random_answer(it, rng)}, max_tokens=32)
        right, valid = check(it, d)
        wins += right
        invalid += not valid
    return {"energy": WIN * wins + LOSE * (len(items) - wins), "wins": wins, "rounds": len(items), "invalid": invalid}


async def heldout_life(llm, seed, rounds=4, oracle=False) -> dict:
    """One test item: `rounds` questions from every held-out family. Returns win rate per family."""
    import asyncio
    rng = random.Random(seed)
    sets = {f: items_for(f, rng, rounds) for f in HELD}
    parts = await asyncio.gather(*[family_life(f, llm, sets[f], random.Random(seed * 13 + i), oracle)
                                   for i, f in enumerate(HELD)])
    out = dict(zip(HELD, parts))
    out["rate"] = sum(p["wins"] for p in parts) / sum(p["rounds"] for p in parts)
    return out
