"""Task families: lists, sets and positions.

Thirteen one-question families about rows, lines, pairs of lists and a small grid. Every row is listed
from its start (or from the left) and every position is counted from the start, never from the end.
Items inside one question are always distinct, so each question has exactly one right answer.
"""
from evo3.wide import COLORS, ITEMS, TREES, q

GOODS = ITEMS + ["bone", "flint", "rope", "pot", "net", "spear", "drum", "mat"]
HUES = COLORS + ["grey", "brown", "yellow"]
WOODS = TREES + ["willow", "alder", "yew"]
NAMES = ["tam", "ori", "kel", "suna", "bram", "ila", "dov", "mesa", "nix", "runa"]
ORD = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh"]
STEP = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}


def g_nth(r):
    """Position -> item."""
    row = r.sample(HUES, r.randint(4, 7))
    k = r.randrange(len(row))
    return q(f"Huts stand in a row along the river. Starting from the first hut, their colours are: {', '.join(row)}. "
             f"What colour is the {ORD[k]} hut, counting from the start?", row[k], "word", options=row)


def g_whereis(r):
    """Item -> position."""
    row = r.sample(GOODS, r.randint(4, 7))
    k = r.randrange(len(row))
    return q(f"Drying racks stand in a row, numbered from 1 at the start. In order they hold: {', '.join(row)}. "
             f"What is the number of the rack that holds the {row[k]}?", k + 1, lo=1, hi=7)


def g_insert(r):
    """Put a new item right after (or right before) a named one; give the whole new row."""
    n = r.randint(3, 5)
    *row, new = r.sample(WOODS, n + 1)
    k, after = r.randrange(n), r.random() < 0.5
    out = row[:k + after] + [new] + row[k + after:]
    return q(f"Logs lie in a row by the fire, in this order: {', '.join(row)}. {'An' if new[0] in 'aeiou' else 'A'} {new} "
             f"log is put down right "
             f"{'after' if after else 'before'} the {row[k]} log. List the logs in their new order.", out, "list")


def g_remove(r):
    """Take a named item out; give what is left in the same order."""
    row = r.sample(GOODS, r.randint(4, 6))
    x = r.choice(row)
    return q(f"Goods are laid out on a flat rock in this order: {', '.join(row)}. A fox carries off the {x}. "
             "List what remains, in the same order.", [g for g in row if g != x], "list")


def g_onlyin(r):
    """Set difference: in the first list but not in the second."""
    a = r.sample(GOODS, r.randint(4, 5))
    only = r.choice([2, 3])
    shared = r.sample(a, len(a) - only)
    b = shared + r.sample([g for g in GOODS if g not in a], r.randint(1, 2))
    r.shuffle(b)
    return q(f"The river camp has: {', '.join(a)}. The hill camp has: {', '.join(b)}. Which things does the river camp "
             "have that the hill camp does not have? List them in the order given for the river camp.",
             [g for g in a if g not in b], "list")


def g_neighbor(r):
    """The item directly left or right of a named one."""
    row = r.sample(HUES, r.randint(4, 6))
    right = r.random() < 0.5
    k = r.randrange(0, len(row) - 1) if right else r.randrange(1, len(row))
    return q(f"Shells of different colours lie in a row. From left to right they are: {', '.join(row)}. What colour is "
             f"the shell immediately to the {'right' if right else 'left'} of the {row[k]} shell?",
             row[k + 1 if right else k - 1], "word", options=row)


def g_queue(r):
    """Who is at the front of a line after two comings and goings."""
    names = r.sample(NAMES, 6)
    line, spare = names[:r.choice([3, 4])], names[4:]
    start, seen, steps = list(line), list(line), []
    for _ in range(2):
        e = r.choice(["front", "front", "named", "back", "push"])
        if e == "front":
            line.pop(0)
            steps.append("the hunter at the front drinks and leaves")
        elif e == "named":
            x = r.choice(line)
            line.remove(x)
            steps.append(f"{x} gives up and leaves the line")
        else:
            x = spare.pop()
            seen.append(x)
            if e == "back":
                line.append(x)
                steps.append(f"{x} comes and joins at the back")
            else:
                line.insert(0, x)
                steps.append(f"{x} comes and pushes in at the very front")
    return q(f"Hunters wait in a line at the spring. Starting from the front they are: {', '.join(start)}. "
             f"First {steps[0]}. Then {steps[1]}. Who is at the front of the line now?", line[0], "word", options=seen)


def g_everyother(r):
    """Take, skip, take, skip ... starting with the first item."""
    row = r.sample(GOODS, r.randint(5, 7))
    return q(f"Baskets stand in a row. In order they hold: {', '.join(row)}. You take from the first basket, skip the "
             "next one, take from the one after that, and go on like this to the end of the row. "
             "List what you take, in order.", row[::2], "list")


def g_weave(r):
    """Merge two lists by alternating, first list first."""
    n, m = r.choice([(2, 2), (3, 3), (3, 2)])
    things = r.sample(GOODS, n + m)
    a, b = things[:n], things[n:]
    out = [x for i in range(n) for x in ([a[i]] + b[i:i + 1])]
    return q("Two hunters empty their bags by turns, one thing at a time, and the first hunter starts. "
             f"First bag, in order: {', '.join(a)}. Second bag, in order: {', '.join(b)}. "
             "List the things in the order they come out.", out, "list")


def g_grid(r):
    """Two single steps on a small grid, then say what is there."""
    rows, cols = r.choice([(2, 3), (3, 2), (2, 4), (4, 2)])
    cells = r.sample(GOODS, rows * cols)
    grid = [cells[i * cols:(i + 1) * cols] for i in range(rows)]
    while True:
        y, x = r.randrange(rows), r.randrange(cols)
        m1, m2 = r.choice(list(STEP)), r.choice(list(STEP))
        y1, x1 = y + STEP[m1][0], x + STEP[m1][1]
        y2, x2 = y1 + STEP[m2][0], x1 + STEP[m2][1]
        if 0 <= y1 < rows and 0 <= x1 < cols and 0 <= y2 < rows and 0 <= x2 < cols and (y2, x2) != (y, x):
            break
    lay = " ".join(f"Row {i + 1}: {', '.join(g)}." for i, g in enumerate(grid))
    return q("Baskets stand in a grid. The rows are given from top to bottom, and each row from left to right. "
             f"{lay} Start at the basket that holds the {grid[y][x]}. Move one basket {m1}, then one basket {m2}. "
             "What does the basket you reach hold?", grid[y2][x2], "word", options=cells)


def g_first(r):
    """Which of three named items comes earliest in a row."""
    row = r.sample(WOODS, r.randint(5, 7))
    a, b, c = three = r.sample(row, 3)
    return q(f"A path leaves camp and passes these trees in this order: {', '.join(row)}. Walking out from camp, "
             f"which of these trees do you reach first: the {a}, the {b} or the {c}?",
             min(three, key=row.index), "word", options=three)


def g_middle(r):
    """The centre item of an odd-length row."""
    row = r.sample(NAMES, r.choice([3, 5, 5, 7, 7]))
    return q(f"Hunters sit side by side on a long log. In order along the log they are: {', '.join(row)}. "
             "Who sits in the middle, with the same number of hunters on each side?",
             row[len(row) // 2], "word", options=row)


def g_inbetween(r):
    """How many items stand between two named ones."""
    row = r.sample(HUES, r.randint(5, 7))
    a, b = r.sample(row, 2)
    return q(f"Painted posts stand in a row along the path. In order their colours are: {', '.join(row)}. "
             f"How many posts stand between the {a} post and the {b} post?",
             abs(row.index(a) - row.index(b)) - 1, lo=0, hi=5)


FAMILIES = {"nth": (g_nth, "You walk along the huts and say what stands at a given place in the row."),
            "whereis": (g_whereis, "You look after the drying racks and know which rack holds each thing."),
            "insert": (g_insert, "You tend the fire and keep the logs in order when a new one is laid down."),
            "remove": (g_remove, "You guard the goods laid out on the rock and say what is left when something is taken."),
            "onlyin": (g_onlyin, "You compare two camps and find what one has that the other does not."),
            "neighbor": (g_neighbor, "You know which shell lies right beside which in a row."),
            "queue": (g_queue, "You watch the line at the spring and know whose turn is next."),
            "everyother": (g_everyother, "You gather from a row of baskets, taking one and skipping one."),
            "weave": (g_weave, "You watch two hunters lay out their goods by turns and keep the order."),
            "grid": (g_grid, "You find your way among baskets set out in rows and columns."),
            "first": (g_first, "You know the trees along the path and which one a walker meets first."),
            "middle": (g_middle, "You notice who sits at the centre of a row."),
            "inbetween": (g_inbetween, "You judge how many posts stand between two posts in a row.")}
