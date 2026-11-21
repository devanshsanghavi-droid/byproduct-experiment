"""Task families: arithmetic and quantities.

Thirteen one-question families, each built on a different computation:
gap (a - b, "how many more"), undo (recover the starting amount), mean (average), part (unit fraction of),
trips (division rounded up), trade (proportion at a stated rate), plot (perimeter / area), stakes (gaps between
markers in a line), round (nearest ten), double (doubling / halving chain), bigger (compare two sums),
verify (check a claimed total), topup (same change applied to every basket).
"""
from evo3.wide import ITEMS, q, yn

PLURAL = {"nut": "nuts", "fish": "fish", "hide": "hides", "root": "roots", "shell": "shells", "berry": "berries",
          "egg": "eggs", "reed": "reeds"}
NAMES = ["Ana", "Bo", "Kes", "Mira", "Tol", "Una", "Dav", "Lio"]
CAMPS = ["hill", "river", "lake", "cave", "north", "south"]
HOLDERS = ["basket", "pit", "sack", "pot"]


def things(r):
    return PLURAL[r.choice(ITEMS)]


def g_gap(r):
    a, b = r.sample(range(3, 21), 2)
    x, y = r.sample(NAMES, 2)
    t = things(r)
    past, base = r.choice([("gathered", "gather"), ("brought back", "bring back"), ("found", "find")])
    top, low = (x, y) if a > b else (y, x)
    ask = (f"How many more {t} did {top} {base} than {low}?" if r.random() < 0.5
           else f"How many fewer {t} did {low} {base} than {top}?")
    return q(f"{x} {past} {a} {t}. {y} {past} {b} {t}. {ask}", abs(a - b), "int", lo=1, hi=17)


def g_undo(r):
    s, h, t = r.randint(3, 15), r.choice(HOLDERS), things(r)
    if r.random() < 0.5:
        k = r.randint(2, 9)
        change, now = f"{k} more were put in", s + k
    else:
        k = r.randint(1, s - 1)
        change, now = f"{k} {'was' if k == 1 else 'were'} taken out", s - k
    return q(f"A {h} held some {t}. Then {change}. Now the {h} holds {now}. How many {t} did it hold at first?",
             s, "int", lo=3, hi=15)


def g_mean(r):
    n = r.choice([3, 4])
    while True:
        xs = [r.randint(1, 15) for _ in range(n)]
        if sum(xs) % n == 0 and len(set(xs)) > 1:
            break
    t = things(r)
    per, lead = r.choice([("day", f"{t.capitalize()} gathered on each of {n} days"),
                          ("hunter", f"{t.capitalize()} carried by each of {n} hunters"),
                          ("basket", f"{t.capitalize()} in each of {n} baskets")])
    return q(f"{lead}: {', '.join(map(str, xs))}. What is the average number of {t} per {per}?",
             sum(xs) // n, "int", lo=1, hi=15)


FRACS = [("half", 2), ("a third", 3), ("a quarter", 4), ("a fifth", 5)]


def g_part(r):
    word, d = r.choice(FRACS)
    k, t, who = r.randint(2, 8), things(r), r.choice(NAMES)
    past, base = r.choice([("takes", "take"), ("dries", "dry"), ("gives away", "give away")])
    return q(f"The camp store holds {d * k} {t}. {who} {past} {word} of them. How many {t} does {who} {base}?",
             k, "int", lo=1, hi=10)


LOADS = [("stones", "wall"), ("logs", "fire"), ("hides", "drying rack"), ("baskets", "river camp")]


def g_trips(r):
    c, trips = r.randint(2, 6), r.randint(2, 7)
    n = c * trips - r.randrange(c)
    load, place = r.choice(LOADS)
    who = r.choice(NAMES)
    return q(f"{who} can carry at most {c} {load} in one trip. {n} {load} must be carried to the {place}. "
             f"How many trips does {who} need to carry them all?", trips, "int", lo=1, hi=10)


def g_trade(r):
    x, y = r.sample(ITEMS, 2)
    unit, a = r.randint(2, 5), r.randint(2, 4)
    c = r.choice([n for n in range(1, 7) if n != a])
    ask = f"does 1 {x}" if c == 1 else f"do {c} {PLURAL[x]}"
    return q(f"{a} {PLURAL[x]} trade for {a * unit} {PLURAL[y]}. At the same rate, how many {PLURAL[y]} {ask} "
             "trade for?", c * unit, "int", lo=2, hi=30)


GROUNDS = ["drying ground", "garden plot", "fish yard", "sleeping floor"]


def g_plot(r):
    w, h, g = r.randint(2, 7), r.randint(2, 7), r.choice(GROUNDS)
    base = f"A {g} has four straight sides and square corners. It is {w} steps long and {h} steps wide. "
    if r.random() < 0.5:
        return q(base + "How many steps is one walk all the way around its edge?", 2 * (w + h), "int", lo=4, hi=49)
    return q(base + "How many square mats, each 1 step long and 1 step wide, cover it exactly?", w * h, "int",
             lo=4, hi=49)


MARKERS = [("stake", "stakes"), ("post", "posts"), ("torch", "torches"), ("marker stone", "marker stones"),
           ("drying pole", "drying poles")]


def g_stakes(r):
    n, d = r.randint(3, 7), r.randint(2, 5)
    one, many = r.choice(MARKERS)
    if r.random() < 0.5:
        return q(f"{n} {many} stand in a straight line. Each one is {d} steps from the next. "
                 f"How many steps is it from the first {one} to the last?", (n - 1) * d, "int", lo=3, hi=30)
    return q(f"Some {many} stand in a straight line. Each one is {d} steps from the next. It is {(n - 1) * d} steps "
             f"from the first {one} to the last. How many {many} are there?", n, "int", lo=3, hi=30)


SIGHTS = ["birds", "deer", "fires", "tracks", "boats"]


def g_round(r):
    n = r.choice([v for v in range(11, 100) if v % 10 not in (0, 5)])
    who, seen = r.choice(NAMES), r.choice(SIGHTS)
    ask = r.choice(["What is that count rounded to the nearest ten?",
                    "Which multiple of ten is closest to that count?"])
    return q(f"{who} counted {n} {seen} from the hill. {ask}", (n + 5) // 10 * 10, "int", lo=10, hi=100)


def g_double(r):
    v = s = r.randint(2, 6)
    steps = []
    for _ in range(r.choice([2, 3])):
        if v % 2 == 0 and r.random() < 0.45:   # only halve when the heap splits exactly
            v //= 2
            steps.append("halved")
        else:
            v *= 2
            steps.append("doubled")
    t = r.choice(["stones", "shells", "nuts", "logs"])
    return q(f"A heap holds {s} {t}. The heap is {', then '.join(steps)}. How many {t} does the heap hold now?",
             v, "int", lo=1, hi=48)


def g_bigger(r):
    c1, c2 = r.sample(CAMPS, 2)
    while True:
        a, b, c, d = (r.randint(2, 9) for _ in range(4))
        if a + b != c + d:
            break
    t, more = things(r), r.random() < 0.5
    win = c1 if (a + b > c + d) == more else c2
    return q(f"Two camps gather {t} in the morning and again in the evening. The {c1} camp gathers {a} and then {b}. "
             f"The {c2} camp gathers {c} and then {d}. Which camp gathers {'more' if more else 'fewer'} {t} in all?",
             win, "word", options=[c1, c2])


def g_verify(r):
    xs = [r.randint(2, 9) for _ in range(r.choice([2, 3]))]
    right = r.random() < 0.5
    claim = sum(xs) + (0 if right else r.choice([-2, -1, 1, 2]))
    who, t = r.choice(NAMES), things(r)
    piles = " and ".join([", ".join(f"{x} {t}" for x in xs[:-1]), f"{xs[-1]} {t}"])
    return q(f"{who} puts piles together and says: {piles} make {claim} {t} in all. Is {who} right?",
             yn(right), "yesno")


def g_topup(r):
    n = r.choice([3, 4])
    xs, k, t = r.sample(range(6, 16), n), r.randint(2, 5), things(r)
    add = r.random() < 0.5
    change = f"{k} more {t} are put into each basket" if add else f"{k} {t} are taken out of each basket"
    return q(f"{'Three' if n == 3 else 'Four'} baskets hold these numbers of {t}: {', '.join(map(str, xs))}. "
             f"Then {change}. "
             f"List how many {t} each basket holds now, in the same order.",
             [str(x + k if add else x - k) for x in xs], "list")


FAMILIES = {"gap": (g_gap, "You compare what two foragers brought back and say how far apart they are."),
            "undo": (g_undo, "You work out how much a store held before it was changed."),
            "mean": (g_mean, "You find the average of a few counts."),
            "part": (g_part, "You measure out a named fraction of the camp store."),
            "trips": (g_trips, "You plan how many trips it takes to carry a whole load."),
            "barter": (g_trade, "You trade goods at a fixed rate."),
            "plot": (g_plot, "You measure plots of ground by pacing them."),
            "stakes": (g_stakes, "You set markers in a line and pace the distance between them."),
            "round": (g_round, "You report counts rounded to the nearest ten."),
            "double": (g_double, "You follow a heap as it is doubled and halved."),
            "bigger": (g_bigger, "You judge which of two camps has the larger total."),
            "verify": (g_verify, "You check the totals that other people claim."),
            "topup": (g_topup, "You update every basket by the same amount.")}
