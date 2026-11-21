"""Task families: logic, states, time and space."""
from evo2.world import nonsense
from evo3.wide import COLORS, ITEMS, TREES, q, yn

DIRS = ["north", "east", "south", "west"]
LEFT = {"north": "west", "west": "south", "south": "east", "east": "north"}
RIGHT = {v: k for k, v in LEFT.items()}
OPP = {"north": "south", "south": "north", "east": "west", "west": "east"}
PLACES = ["hut", "well", "pit", "fire", "ford", "cave", "rock", "oak"]
NAMES = ["mara", "oren", "tuli", "sana", "doro", "keb", "lino", "vasa"]
THINGS = ["knife", "rope", "bowl", "spear", "drum", "net"]
EVENTS = ["hunt", "storm", "feast", "dance", "trade", "rain", "race", "song"]
STATES = [("the well", "dry"), ("the fire", "lit"), ("the gate", "shut"), ("the river", "high"),
          ("the path", "muddy"), ("the basket", "full"), ("the net", "torn"), ("the ice", "thin")]
MATS = ["flint", "rope", "hide", "reed", "clay", "bone", "resin", "wood", "sinew", "bark"]
GOODS = ["spear", "raft", "drum", "pot", "tent", "trap", "sled", "bow"]
PLURAL = {"nut": "nuts", "fish": "fish", "hide": "hides", "root": "roots", "shell": "shells", "berry": "berries",
          "egg": "eggs", "reed": "reeds"}
NUM = {3: "Three", 4: "Four", 5: "Five"}


def hrs(n):
    return f"{n} hour{'' if n == 1 else 's'}"


def g_turn(r):
    """Facing direction after one or two quarter turns."""
    place, start = r.choice(PLACES), r.choice(DIRS)
    turns = [r.choice(["left", "right"]) for _ in range(r.choice([1, 2]))]
    face = start
    for t in turns:
        face = (LEFT if t == "left" else RIGHT)[face]
    said = ", then ".join(f"a quarter turn to your {t}" for t in turns)
    return q(f"You stand by the {place} facing {start}. You make {said}. "
             "Which way do you face now: north, east, south or west?", face, "word", options=DIRS)


def g_syllog(r):
    """One-step deduction from an all/no statement, valid or not."""
    a, b = nonsense(r, 2)
    if r.random() < 0.5:
        rule, fact, ask = r.choice([
            (f"All {a}s are {b}s.", f"is a {a}", f"is a {b}"),
            (f"All {a}s are {b}s.", f"is not a {b}", f"is not a {a}"),
            (f"No {a}s are {b}s.", f"is a {a}", f"is not a {b}")])
        ok = True
    else:
        rule, fact, ask = r.choice([
            (f"All {a}s are {b}s.", f"is a {b}", f"is a {a}"),
            (f"All {a}s are {b}s.", f"is not a {a}", f"is not a {b}"),
            (f"No {a}s are {b}s.", f"is not a {a}", f"is a {b}")])
        ok = False
    return q(f"{rule} The animal in the trap {fact}. From this alone, is it certain that the animal {ask}?",
             yn(ok), "yesno")


def g_vessel(r):
    """Final state of a container; filling and emptying only work while it is open."""
    pot = r.choice(["jar", "pot", "gourd", "flask"])
    ask_fill = r.random() < 0.6
    want = r.choice(["full", "empty"] if ask_fill else ["open", "shut"])
    while True:
        is_open, is_full = r.random() < 0.5, r.random() < 0.5
        start = f"{'open' if is_open else 'shut'} and {'full' if is_full else 'empty'}"
        acts = []
        for _ in range(r.choice([3, 4])):
            act = r.choice([x for x in ["open", "shut", "fill", "empty"] if not acts or x != acts[-1]])
            acts.append(act)
            if act in ("open", "shut"):
                is_open = act == "open"
            elif is_open:
                is_full = act == "fill"
        got = ("full" if is_full else "empty") if ask_fill else ("open" if is_open else "shut")
        used = any(x in (("fill", "empty") if ask_fill else ("open", "shut")) for x in acts)
        if got == want and used:
            break
    return q(f"A {pot} starts {start}. While it is shut, trying to fill or empty it changes nothing. Then, in order, "
             f"someone tries to: {', '.join(x + ' it' for x in acts)}. Is the {pot} "
             f"{'full or empty' if ask_fill else 'open or shut'} at the end?", want, "word",
             options=["full", "empty"] if ask_fill else ["open", "shut"])


def g_handoff(r):
    """Who holds a thing after it is handed along, with a second thing as a distractor."""
    names = r.sample(NAMES, 4)
    x, y = r.sample(THINGS, 2)
    hold = {x: names[0], y: names[1]}
    text = f"{names[0].capitalize()} holds the {x} and {names[1].capitalize()} holds the {y}."
    for _ in range(3):
        t = r.choice([x, y])
        to = r.choice([n for n in names if n != hold[t]])
        text += f" {hold[t].capitalize()} gives the {t} to {to.capitalize()}."
        hold[t] = to
    t = r.choice([x, y])
    return q(f"{text} Who holds the {t} now?", hold[t], "word", options=names)


def g_sooner(r):
    """Which of two events came first or later, with times counted before and after noon."""
    a, b = r.sample(EVENTS, 2)
    while True:
        ta, tb = [(r.choice(["before", "after"]), r.randint(1, 5)) for _ in range(2)]
        if ta != tb:
            break
    when = {a: -ta[1] if ta[0] == "before" else ta[1], b: -tb[1] if tb[0] == "before" else tb[1]}
    first = r.random() < 0.5
    ans = (min if first else max)([a, b], key=when.get)
    return q(f"The {a} began {hrs(ta[1])} {ta[0]} noon. The {b} began {hrs(tb[1])} {tb[0]} noon. "
             f"Which began {'first' if first else 'later'}, the {a} or the {b}?", ans, "word", options=[a, b])


def g_liar(r):
    """What is true, given a statement from someone who always tells the truth or always says the opposite."""
    honest, liar = [n.capitalize() for n in r.sample(NAMES, 2)]
    intro = [f"{honest} always tells the truth.", f"{liar} always says the opposite of the truth."]
    r.shuffle(intro)
    subj, prop = r.choice(STATES)
    lies, neg = r.random() < 0.5, r.random() < 0.5
    truth = (not neg) != lies
    return q(f"{' '.join(intro)} {liar if lies else honest} says: {subj} is {'not ' if neg else ''}{prop}. "
             f"Is {subj} {prop}?", yn(truth), "yesno")


def g_needs(r):
    """Whether both needed materials are among the things present."""
    good = r.choice(GOODS)
    a, b, *rest = r.sample(MATS, 6)
    n = r.choice([3, 4])
    ok = r.random() < 0.5
    keep = [a, b] if ok else r.choice([[a], [b], [a], [b], []])
    have = keep + rest[:n - len(keep)]
    r.shuffle(have)
    return q(f"To make a {good} you need both {a} and {b}. The hut holds: {', '.join(have)}. "
             f"Does the hut hold everything you need to make a {good}?", yn(ok), "yesno")


def g_compass(r):
    """Compass direction between two places from one north/south fact and one east/west fact."""
    a, b, c = r.sample(PLACES, 3)
    ns, ew = r.choice(["north", "south"]), r.choice(["east", "west"])
    d1, d2 = (ns, ew) if r.random() < 0.5 else (ew, ns)
    flip = r.randrange(3)
    s1 = f"The {a} is due {OPP[d1]} of the {b}." if flip == 1 else f"The {b} is due {d1} of the {a}."
    s2 = f"The {b} is due {OPP[d2]} of the {c}." if flip == 2 else f"The {c} is due {d2} of the {b}."
    return q(f"{s1} {s2} Seen from the {a}, in which direction is the {c}: northeast, northwest, southeast "
             "or southwest?", ns + ew, "word", options=["northeast", "northwest", "southeast", "southwest"])


def g_every(r):
    """All / at least one / none, checked over a short inventory."""
    n = r.choice([3, 4])
    cols = r.sample(COLORS, n)
    t, o1, o2 = r.sample(ITEMS, 3)
    quant, ok = r.choice(["all", "some", "none"]), r.random() < 0.5
    if quant == "all":
        k = n if ok else r.randint(0, n - 1)
    elif quant == "some":
        k = r.randint(1, n) if ok else 0
    else:
        k = 0 if ok else r.randint(1, n)
    holds = [t] * k + [r.choice([o1, o2]) for _ in range(n - k)]
    r.shuffle(holds)
    claim = {"all": f"all of the baskets hold {PLURAL[t]}", "some": f"at least one of the baskets holds {PLURAL[t]}",
             "none": f"none of the baskets hold {PLURAL[t]}"}[quant]
    return q(f"{NUM[n]} baskets: " + ", ".join(f"the {c} one holds {PLURAL[h]}" for c, h in zip(cols, holds))
             + f". Is it true that {claim}?", yn(ok), "yesno")


def g_nextto(r):
    """The neighbour of a named thing in a row laid out along a compass line."""
    kind, pool = r.choice([("hut", COLORS), ("tree", TREES)])
    row = r.sample(pool, r.choice([4, 5]))
    a, b = r.choice([("west", "east"), ("north", "south")])
    i = r.randrange(len(row) - 1)
    ref, ans, d = (row[i], row[i + 1], b) if r.random() < 0.5 else (row[i + 1], row[i], a)
    return q(f"{NUM[len(row)]} {kind}s stand in a row from {a} to {b}: {', '.join(row)}. "
             f"Which {kind} is directly {d} of the {ref}{' hut' if kind == 'hut' else ''}?", ans, "word", options=row)


def g_hours(r):
    """Start, length and end of a spell of time counted in hours after dawn: find the missing one."""
    s, d = r.randint(1, 6), r.randint(1, 6)
    e = s + d
    text, ans = r.choice([
        (f"The fire was lit {hrs(s)} after dawn and burned for {hrs(d)}. How many hours after dawn did it go out?", e),
        (f"You set out {hrs(s)} after dawn and walk for {hrs(d)}. How many hours after dawn do you arrive?", e),
        (f"The hunters left {hrs(s)} after dawn and came back {hrs(e)} after dawn. How many hours were they away?", d),
        (f"The rain began {hrs(s)} after dawn and stopped {hrs(e)} after dawn. How many hours did it rain?", d),
        (f"The walk to the lake takes {hrs(d)}. You reached the lake {hrs(e)} after dawn. "
         "How many hours after dawn did you set out?", s),
        (f"The fire burned for {hrs(d)} and went out {hrs(e)} after dawn. How many hours after dawn was it lit?", s)])
    return q(text, ans, "int", lo=1, hi=12)


def g_whohas(r):
    """Look up one trait of a person identified by another trait."""
    names, cols, tools = r.sample(NAMES, 3), r.sample(COLORS, 3), r.sample(THINGS, 3)
    facts = " ".join(f"{n.capitalize()} wears a {c} cloak and carries the {t}." for n, c, t in zip(names, cols, tools))
    k = r.randrange(3)
    ask, ans, opts = r.choice([
        (f"What does the hunter in the {cols[k]} cloak carry?", tools[k], tools),
        (f"What colour is the cloak of the hunter who carries the {tools[k]}?", cols[k], cols),
        (f"Who carries the {tools[k]}?", names[k], names),
        (f"Who wears the {cols[k]} cloak?", names[k], names)])
    return q(f"{facts} {ask}", ans, "word", options=opts)


def g_brim(r):
    """Adding to a container with a limit: what is inside, and what does not fit."""
    box, thing = r.choice([("basket", "fish"), ("pouch", "shells"), ("pit", "stones"), ("sack", "roots"),
                           ("pot", "eggs")])
    cap = r.randint(5, 12)
    a = r.randint(1, cap - 1)
    b = r.randint(1, cap - a) if r.random() < 0.5 else r.randint(cap - a + 1, cap - a + 6)
    inside = min(cap, a + b)
    ask, ans = r.choice([(f"How many {thing} are in the {box} now?", inside),
                         (f"How many {thing} are left outside?", a + b - inside)])
    return q(f"A {box} can hold at most {cap} {thing}. It already holds {a}. You try to put in {b} more; any that "
             f"do not fit are left outside. {ask}", ans, "int", lo=0, hi=12)


FAMILIES = {
    "turn": (g_turn, "You are a scout who keeps track of which way you face."),
    "syllog": (g_syllog, "You judge what is certain about a trapped animal from what is known of its kind."),
    "vessel": (g_vessel, "You look after the camp's vessels and track whether each is open or shut, full or empty."),
    "handoff": (g_handoff, "You watch tools pass from hand to hand and keep track of who holds what."),
    "sooner": (g_sooner, "You keep the order of the day's events, told as hours before or after noon."),
    "liar": (g_liar, "You work out what is true from the words of people who either never lie or always lie."),
    "needs": (g_needs, "You check the stores before work starts to see whether everything needed is there."),
    "compass": (g_compass, "You are a pathfinder who works out where places lie from each other."),
    "every": (g_every, "You inspect the baskets and judge claims about what all, some or none of them hold."),
    "nextto": (g_nextto, "You know the camp's rows of huts and trees and which one stands beside which."),
    "hours": (g_hours, "You keep time by counting hours after dawn."),
    "whohas": (g_whohas, "You know the hunters by their cloaks and by what they carry."),
    "brim": (g_brim, "You fill the camp's stores and know that nothing holds more than its limit."),
}
