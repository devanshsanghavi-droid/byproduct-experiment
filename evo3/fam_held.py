"""Held-out task families, second set. Never part of any life or training set; used only as a test.

Each needs a skill that no trainable family is allowed to need (the list is in SPEC3.md, breadth curve section).
"""
from evo2.world import nonsense
from evo3.wide import ITEMS, q, yn

PEOPLE = ["tom", "sam", "lee", "ada", "max", "zoe", "kim", "roy"]
ANIMALS = ["ox", "elk", "hare", "otter", "badger", "buffalo", "antelope", "wolverine"]   # 2 to 9 letters, all different


def h_dedup(r):
    kinds = r.sample(ITEMS, 3)
    while True:
        seq = [r.choice(kinds) for _ in range(r.randint(5, 6))]
        first = list(dict.fromkeys(seq))
        if 2 <= len(first) < len(seq):
            break
    return q(f"Baskets in the order they came in: {', '.join(seq)}. List each kind once, in the order it first appeared.",
             first, "list")


def h_fromend(r):
    words = nonsense(r, r.randint(4, 6))
    k = r.choice([2, 3])
    return q(f"Huts along the river, from upstream to downstream: {', '.join(words)}. "
             f"Which hut is {'second' if k == 2 else 'third'} from the downstream end?", words[-k], "word", options=words)


def h_majority(r):
    while True:
        kinds = r.sample(ITEMS, 3)
        seq = [r.choice(kinds) for _ in range(r.randint(6, 8))]
        c = sorted(((seq.count(k), k) for k in kinds), reverse=True)
        if c[0][0] > c[1][0]:
            break
    return q(f"Things carried into camp today: {', '.join(seq)}. Which kind was carried in most often?", c[0][1],
             "word", options=kinds)


def h_between(r):
    lo = r.randint(2, 20)
    hi = lo + r.randint(3, 12)
    inside = r.random() < 0.5
    x = r.randint(lo + 1, hi - 1) if inside else r.choice([r.randint(max(0, lo - 8), lo - 1), r.randint(hi + 1, hi + 8)])
    return q(f"The ford is safe when the water mark is above {lo} and below {hi}. Today the mark is {x}. Is the ford safe?",
             yn(inside), "yesno")


def h_rotate(r):
    words = nonsense(r, r.choice([3, 4, 5]))
    return q(f"Watch order tonight: {', '.join(words)}. Tomorrow the first watcher moves to the end and everyone else "
             "moves up one place. What is tomorrow's order?", words[1:] + words[:1], "list")


def h_parity(r):
    xs = [r.randint(1, 9) for _ in range(3)]
    return q(f"Three hunters bring {xs[0]}, {xs[1]} and {xs[2]} fish. The catch can be split into two equal piles "
             "only if the total is even. Is the total even?", yn(sum(xs) % 2 == 0), "yesno")


def h_rule(r):
    kind = r.choice(["add", "mul", "both"])
    a = r.randint(2, 9) if kind == "add" else r.choice([2, 3, 4]) if kind == "mul" else r.randint(1, 4)
    f = (lambda x: x + a) if kind == "add" else (lambda x: x * a) if kind == "mul" else (lambda x: 2 * x + a)
    xs = r.sample(range(1, 9), 4)
    shown = " ".join(f"Pay {x}, get {f(x)}." for x in xs[:3])
    return q(f"A trader follows one hidden rule. {shown} By the same rule, you pay {xs[3]}. How many do you get?",
             f(xs[3]), lo=1, hi=40)


def h_overlap(r):
    a, b = r.sample(ITEMS, r.randint(3, 5)), r.sample(ITEMS, r.randint(3, 5))
    return q(f"North camp has: {', '.join(a)}. South camp has: {', '.join(b)}. How many kinds do both camps have?",
             len(set(a) & set(b)), lo=0, hi=5)


def h_ring(r):
    n = r.randint(5, 8)
    s, k = r.randint(1, n), r.randint(2, n + 3)
    return q(f"{n} stones stand in a circle, numbered 1 to {n} in order. You start at stone {s} and step forward {k} "
             f"stones. After stone {n} comes stone 1. Which stone do you stand at?", (s - 1 + k) % n + 1, lo=1, hi=n)


def h_toll(r):
    lim, a = r.randint(8, 14), r.randint(2, 4)
    loads = [x + 1 if x == lim else x for x in (r.randint(3, 20) for _ in range(3))]
    return q(f"At the bridge, a load heavier than {lim} pays {a} shells and any other load pays 1 shell. Three loads "
             f"cross, weighing {loads[0]}, {loads[1]} and {loads[2]}. How many shells are paid in all?",
             sum(a if x > lim else 1 for x in loads), lo=3, hi=12)


def h_kin(r):
    a, b, c, d = r.sample(PEOPLE, 4)       # d is the parent of a, a of b, b of c
    facts = [f"{x.title()} is the parent of {y.title()}." for x, y in ((d, a), (a, b), (b, c))]
    r.shuffle(facts)
    rel, of, ans = r.choice([("grandparent", c, a), ("grandparent", b, d), ("grandchild", a, c), ("grandchild", d, b)])
    return q(" ".join(facts) + f" Who is the {rel} of {of.title()}?", ans, "word", options=[a, b, c, d])


def h_longest(r):
    ws = r.sample(ANIMALS, 4)
    most = r.random() < 0.5
    return q(f"Animal names carved on the post: {', '.join(ws)}. Which name has the {'most' if most else 'fewest'} letters?",
             (max if most else min)(ws, key=len), "word", options=ws)


FAMILIES = {"dedup": (h_dedup, "You tidy the list of what came into camp."),
            "fromend": (h_fromend, "You know the huts along the river."),
            "majority": (h_majority, "You keep track of what the camp gathers most."),
            "between": (h_between, "You judge whether the river can be crossed."),
            "rotate": (h_rotate, "You set the order of the night watch."),
            "parity": (h_parity, "You decide whether a catch can be split in two."),
            "rule": (h_rule, "You work out how a trader deals from past trades."),
            "overlap": (h_overlap, "You compare the stores of two camps."),
            "ring": (h_ring, "You walk around a circle of standing stones."),
            "toll": (h_toll, "You collect the toll at the bridge."),
            "kin": (h_kin, "You know who is related to whom."),
            "longest": (h_longest, "You read names carved on a post.")}
