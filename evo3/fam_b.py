"""Task families: symbols, marks and codes."""
from evo2.world import nonsense          # nonsense(r, n) -> n made-up words
from evo3.wide import COLORS, ITEMS, SHAPES, TREES, q, yn


def g_begins(r):
    """Prefix test: does the row open with a given sign (or a given pair of signs)?"""
    row = nonsense(r, r.choice([4, 5]))
    hit = r.random() < 0.5
    if r.random() < 0.5:
        probe = [row[0]] if hit else [r.choice(row[1:])]
        what = f"the sign {probe[0]}"
    else:
        if hit:
            probe = row[:2]
        else:
            i = r.randrange(1, len(row) - 1)
            probe = r.choice([row[i:i + 2], [row[1], row[0]], [row[0], row[2]]])
        what = f"the two signs {probe[0]} {probe[1]}, in that order"
    return q(f"Signs carved on the bark, in order: {' '.join(row)}. Does the carving begin with {what}?",
             yn(hit), "yesno")


def g_resign(r):
    """Replace every occurrence of one sign with another, leave the rest."""
    pool = nonsense(r, 4)
    x, y = pool[0], pool[3]
    while True:
        row = [r.choice(pool[:3]) for _ in range(r.choice([4, 5]))]
        if 1 <= row.count(x) <= 3 and len(set(row)) >= 2:
            break
    return q(f"Row of signs on the bark: {' '.join(row)}. Carve the row again with every {x} changed to {y}. "
             "Leave the other signs as they are.", [y if s == x else s for s in row], "list")


def g_whichsign(r):
    """Reverse lookup in a sign table: given the meaning, name the sign."""
    signs, means = nonsense(r, 4), r.sample(ITEMS, 4)
    t = r.randrange(4)
    return q("Sign table: " + ", ".join(f"{s} stands for {m}" for s, m in zip(signs, means))
             + f". Which sign stands for {means[t]}?", signs[t], "word", options=signs)


def g_twostep(r):
    """Two chained lookups: mark -> clan, clan -> bead colour."""
    marks, clans, cols = nonsense(r, 3), r.sample(TREES, 3), r.sample(COLORS, 3)
    order = r.sample(range(3), 3)
    t = r.randrange(3)
    return q("Each mark belongs to a clan: " + ", ".join(f"{m} to the {c} clan" for m, c in zip(marks, clans))
             + ". Each clan wears one bead colour: " + ", ".join(f"the {clans[i]} clan {cols[i]}" for i in order)
             + f". Which bead colour goes with the mark {marks[t]}?", cols[t], "word", options=cols)


def g_samerow(r):
    """Are two rows of signs identical?"""
    pool = nonsense(r, 6)
    n = r.choice([4, 5])
    a = pool[:n]
    b = list(a)
    same = r.random() < 0.5
    if not same:
        i = r.randrange(n)
        if r.random() < 0.5:
            b[i] = pool[5]
        else:
            j = (i + 1) % n if i + 1 < n else i - 1
            b[i], b[j] = b[j], b[i]
    return q(f"First row of signs: {' '.join(a)}. Second row of signs: {' '.join(b)}. "
             "Are the two rows exactly the same, sign for sign?", yn(same), "yesno")


def g_mismatch(r):
    """Count the places where two equal-length bead strings differ."""
    n = r.choice([5, 6])
    a = [r.choice(COLORS) for _ in range(n)]
    b = list(a)
    k = r.randint(0, 4)
    for i in r.sample(range(n), k):
        b[i] = r.choice([c for c in COLORS if c != a[i]])
    return q(f"Two strings of beads, each read left to right. First: {', '.join(a)}. Second: {', '.join(b)}. "
             "In how many places do the two strings hold a different colour?", k, lo=0, hi=4)


def g_nextsign(r):
    """Name the sign that directly follows a given sign."""
    row = nonsense(r, 5)
    i = r.randrange(4)
    return q(f"Signs along the path, in order: {', '.join(row)}. Which sign comes right after {row[i]}?",
             row[i + 1], "word", options=row)


def g_fuse(r):
    """Apply one rewrite rule once: two neighbouring signs become one new sign."""
    a, b, c, f1, f2 = nonsense(r, 5)
    while True:
        fill = [r.choice([f1, f2, a, b]) for _ in range(r.choice([2, 3]))]
        p = r.randrange(len(fill) + 1)
        row = fill[:p] + [a, b] + fill[p:]
        if sum(x == a and y == b for x, y in zip(row, row[1:])) == 1:
            break
    return q(f"Rule: where the sign {a} is followed at once by the sign {b}, carve the one sign {c} in place of "
             f"the two. Row: {' '.join(row)}. Carve the row again after using the rule.",
             fill[:p] + [c] + fill[p:], "list")


def g_beadpair(r):
    """Is there a bead of one colour with a bead of another colour right after it?"""
    cols = r.sample(COLORS, 3)
    a, b = cols[0], cols[1]
    hit = r.random() < 0.5
    while True:
        row = [r.choice(cols) for _ in range(6)]
        found = any(x == a and y == b for x, y in zip(row, row[1:]))
        if found == hit and a in row and b in row:
            break
    return q(f"Beads on a string, in order: {', '.join(row)}. Is there a {a} bead with a {b} bead right after it?",
             yn(hit), "yesno")


def g_bundle(r):
    """Entries like '3 fish': add up the numbers that carry one label."""
    kinds = r.sample(ITEMS, 3)
    n = r.choice([4, 5])
    while True:
        ent = [(r.randint(1, 6), r.choice(kinds)) for _ in range(n)]
        t = r.choice(kinds)
        if 2 <= sum(k == t for _, k in ent) < n:
            break
    return q("Carved on the bark, each entry a number and a thing: " + ", ".join(f"{c} {k}" for c, k in ent)
             + f". What is the total for {t}?", sum(c for c, k in ent if k == t), lo=2, hi=24)


MARKS = [("notch", "notches"), ("cut", "cuts"), ("stroke", "strokes"), ("line", "lines")]


def g_morecuts(r):
    """Two tallies written out stroke by stroke: which one is longer or shorter?"""
    t1, t2 = r.sample(TREES, 2)
    one, many = r.choice(MARKS)
    a, b = r.sample(range(2, 8), 2)
    more = r.random() < 0.5
    ans = t1 if (a > b) == more else t2
    return q(f"The {t1} stick shows: {' '.join([one] * a)}. The {t2} stick shows: {' '.join([one] * b)}. "
             f"Which stick has {'more' if more else 'fewer'} {many}?", ans, "word", options=[t1, t2])


def g_topitem(r):
    """Table of thing and number: name the thing with the largest (or smallest) number."""
    names = r.sample(ITEMS, 4)
    c = dict(zip(names, r.sample(range(2, 30), 4)))
    top = r.random() < 0.5
    return q("Counts in the store: " + ", ".join(f"{n} {c[n]}" for n in names)
             + f". Which thing has the {'largest' if top else 'smallest'} count?",
             (max if top else min)(names, key=lambda n: c[n]), "word", options=names)


def g_notchnum(r):
    """Read a number written with a big sign (5 or 10) and a small sign (1)."""
    big, small = nonsense(r, 2)
    v = r.choice([5, 10])
    nb, ns = r.randint(1, 3), r.randint(0, 4)
    return q(f"On count sticks the sign {big} stands for {v} and the sign {small} stands for 1. "
             f"A stick shows: {' '.join([big] * nb + [small] * ns)}. What number does the stick show?",
             v * nb + ns, lo=5, hi=34)


FAMILIES = {"begins": (g_begins, "You check how a carving on bark opens."),
            "resign": (g_resign, "You recarve rows of signs, changing one sign for another."),
            "whichsign": (g_whichsign, "You keep the sign table and find the sign for a thing."),
            "twostep": (g_twostep, "You follow a mark to its clan and the clan to its bead colour."),
            "samerow": (g_samerow, "You compare two rows of signs."),
            "mismatch": (g_mismatch, "You lay two strings of beads side by side and count where they differ."),
            "nextsign": (g_nextsign, "You follow the signs set along a path."),
            "fuse": (g_fuse, "You recarve a row of signs by a stated rule."),
            "beadpair": (g_beadpair, "You look for one bead followed by another on a string."),
            "bundle": (g_bundle, "You read bark entries and add up one kind of thing."),
            "morecuts": (g_morecuts, "You compare the marks cut on two sticks."),
            "topitem": (g_topitem, "You read the store counts and find the thing that stands out."),
            "notchnum": (g_notchnum, "You read numbers from the signs on count sticks.")}
