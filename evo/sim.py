"""Evolution simulation: one world, seven arms, selection on net energy only.

See SPEC.md. Model weights are frozen; a 4-gene genome gates how much of the
model's cognition an agent may use. Arms are toggles over one code path.

    python3 -m evo.sim --arms all --seeds 1,2,3 --N 12 --R 12 --G 10 --cost 1.5 \
        --model claude-haiku-5-5 --budget 25 --concurrency 16 --out runs [--mock] [--resume]
    python3 -m evo.sim --pilot --mock --out runs_mock
"""
import argparse
import asyncio
import json
import math
import os
import random
import re
import sys
from collections import Counter

from . import genome as G
from .llm import LLM, BudgetExceeded, parse_json_object

ACTIONS = ["soak", "pound", "roast", "dry"]
N_PLANTS = 4
WIN, LOSS = 4, -1          # per agent
GRAB_WIN, GRAB_LOSE, GRAB_BOTH = 7, 1, 2
MAX_SAMPLES = 20
BOARD_VISIBLE = 10

ARMS = {
    #          foraging      grab         vote         notes        parents
    "DRIFT":   dict(pairs=False, grab=False, vote=False, notes=False, select="random"),
    "SOLO":    dict(pairs=False, grab=False, vote=False, notes=False, select="top"),
    "COOP":    dict(pairs=True,  grab=False, vote=False, notes=False, select="top"),
    "ALPHA":   dict(pairs=True,  grab=True,  vote=False, notes=False, select="top"),
    "DOMEST":  dict(pairs=True,  grab=True,  vote=True,  notes=False, select="top"),
    "CULTURE": dict(pairs=False, grab=False, vote=False, notes=True,  select="top"),
    "MEDLEY":  dict(pairs=True,  grab=True,  vote=True,  notes=True,  select="top"),
}


# ----------------------------------------------------------------------------
# Shared randomness: depends on the seed only, never on the arm.
# ----------------------------------------------------------------------------

def make_world(seed: int, gen: int):
    """Plant names and the recipes in force during generation `gen`.

    Replayed from the seed each time, so every arm sees the same world and a
    resumed run needs no saved RNG state.
    """
    rng = random.Random(f"world|{seed}")
    cons, vow = "bdfgkmnprstvz", "aeiou"
    names = []
    while len(names) < N_PLANTS:
        n = rng.choice(cons) + rng.choice(vow) + rng.choice(cons) + rng.choice(vow)
        if n not in names and n not in ACTIONS:
            names.append(n)
    recipes = {p: (rng.choice(ACTIONS), rng.choice(ACTIONS)) for p in names}
    for _ in range(gen):  # one plant's recipe changes at each new generation
        p = rng.choice(names)
        new = recipes[p]
        while new == recipes[p]:
            new = (rng.choice(ACTIONS), rng.choice(ACTIONS))
        recipes[p] = new
    return names, recipes


def initial_genomes(seed: int, n: int):
    rng = random.Random(f"genome|{seed}")
    return [G.random_genome(rng) for _ in range(n)]


def season_plan(seed: int, gen: int, plants, n: int, rounds: int):
    """In-season plant per foraging unit per round. Shared across arms."""
    rng = random.Random(f"season|{seed}|{gen}")
    return [[rng.choice(plants) for _ in range(n)] for _ in range(rounds)]


# ----------------------------------------------------------------------------
# Agents and records
# ----------------------------------------------------------------------------

class Agent:
    def __init__(self, idx, genome, parent=None, inherited=None):
        self.id = f"a{idx}"
        self.genome = dict(genome)
        self.parent = parent
        self.inherited = list(inherited or [])   # unclipped notes handed down
        self.records = []
        self.gross = 0
        self.successes = 0
        self.trials = 0
        self.grabs = 0
        self.grabbed = 0
        self.votes_against = 0
        self.exiled = False
        self.note = ""
        self.prefer = None

    def visible(self):
        return G.visible_records(self.records, self.genome)

    def notes_read(self):
        n = G.lore_words(self.genome)
        out = []
        for t in self.inherited:
            c = G.clip_words(t, n)
            if c and c not in out:
                out.append(c)
        return out


def render_record(r) -> str:
    head = f"r{r['round']} {r['plant']}"
    prep = f"{r['s1']}>{r['s2']}"
    if r["role"] == 0:  # solo
        if r["ok"]:
            return f"{head}: {prep} OK +{WIN}"
        return (f"{head}: {prep} FAIL, step1 {'right' if r['r1'] else 'wrong'}, "
                f"step2 {'right' if r['r2'] else 'wrong'}")
    who = "first" if r["role"] == 1 else "second"
    head = f"{head} with {r['partner']}, you {who}: {prep}"
    if r["ok"]:
        grab = r.get("grab")
        if grab == "partner":
            return f"{head} OK, {r['partner']} grabbed, you got {GRAB_LOSE}"
        if grab == "you":
            return f"{head} OK, you grabbed, +{GRAB_WIN}"
        if grab == "both":
            return f"{head} OK, both grabbed, +{GRAB_BOTH}"
        return f"{head} OK +{WIN}"
    other = 2 if r["role"] == 1 else 1
    return f"{head} FAIL, step{other} {'right' if r['r' + str(other)] else 'wrong'}"


def norm_action(x):
    if isinstance(x, str):
        x = x.strip().lower()
        if x in ACTIONS:
            return x
    return None


async def gather_all(coros):
    """Await everything, then raise the first error (no orphaned tasks)."""
    res = await asyncio.gather(*coros, return_exceptions=True)
    for r in res:
        if isinstance(r, BaseException):
            raise r
    return res


# ----------------------------------------------------------------------------
# Mock agent: a heuristic that uses only what the prompt would show.
# ----------------------------------------------------------------------------

def new_knowledge(plants):
    return {p: {1: {"right": None, "wrong": set()}, 2: {"right": None, "wrong": set()}}
            for p in plants}


def learn(k, plant, step, action, right):
    if plant not in k or action not in ACTIONS:
        return
    slot = k[plant][step]
    if right is True:
        slot["right"] = action
        slot["wrong"].discard(action)
    elif right is False:
        slot["wrong"].add(action)
        if slot["right"] == action:
            slot["right"] = None


def knowledge_from(agent, plants, use_notes):
    """Beliefs from handed-down notes (weak prior) then visible records (override)."""
    k = new_knowledge(plants)
    if use_notes:
        for text in agent.notes_read():
            words = re.findall(r"[a-z?]+", text.lower())
            i = 0
            while i < len(words):
                if words[i] in k:
                    for step in (1, 2):
                        if i + step < len(words) and words[i + step] in ACTIONS:
                            learn(k, words[i], step, words[i + step], True)
                        elif i + step < len(words) and words[i + step] != "?":
                            break
                i += 1
    for r in agent.visible():
        for step in (1, 2):
            learn(k, r["plant"], step, r["s" + str(step)], r["r" + str(step)])
    return k


def pick(k, plant, step, rng):
    slot = k[plant][step]
    if slot["right"]:
        return slot["right"]
    options = [a for a in ACTIONS if a not in slot["wrong"]] or ACTIONS
    return rng.choice(options)


def apply_message(k, plant, msg):
    """Fold a partner's message into beliefs; own memory wins on conflict."""
    cur, neg = None, False
    for tok in re.findall(r"[a-z0-9]+", (msg or "").lower()):
        if tok in ("step1", "step2"):
            cur, neg = int(tok[-1]), False
        elif tok == "not":
            neg = True
        elif tok in ACTIONS and cur:
            slot = k[plant][cur]
            if neg:
                if slot["right"] != tok:
                    slot["wrong"].add(tok)
            elif slot["right"] is None and tok not in slot["wrong"]:
                slot["right"] = tok


def mock_solo(agent, plant, plants, use_notes, rng):
    k = knowledge_from(agent, plants, use_notes)
    return {"step1": pick(k, plant, 1, rng), "step2": pick(k, plant, 2, rng)}


def mock_message(agent, plant, role, plants, use_notes):
    k = knowledge_from(agent, plants, use_notes)
    other = 2 if role == 1 else 1
    slot = k[plant][other]
    if slot["right"]:
        return {"message": f"step{other} {slot['right']}"}
    if slot["wrong"]:
        return {"message": f"step{other} not " + " ".join(sorted(slot["wrong"]))}
    return {"message": ""}


def mock_action(agent, plant, role, partner_msg, plants, use_notes, with_gossip, rng):
    k = knowledge_from(agent, plants, use_notes)
    apply_message(k, plant, partner_msg)
    vis = agent.visible()
    grabbers = [r["partner"] for r in vis if r.get("grab") in ("partner", "both")]
    good = [r["partner"] for r in vis if r["role"] and r["ok"] and r.get("grab") in (None, "you")]
    good = [p for p in good if p not in grabbers]
    out = {"step": pick(k, plant, role, rng), "prefer_partner": good[-1] if good else None}
    if with_gossip:
        out["gossip"] = f"{grabbers[-1]} grabbed" if grabbers else None
    return out


def mock_vote(agent, board):
    tally = Counter()
    for r in agent.visible():
        if r.get("grab") in ("partner", "both"):
            tally[r["partner"]] += 2
    for line in board[-BOARD_VISIBLE:]:
        author, _, text = line.partition(": ")
        for name in re.findall(r"(a\d+) grabbed", text):
            if name != agent.id and name != author:
                tally[name] += 1
    tally.pop(agent.id, None)
    return {"exile": tally.most_common(1)[0][0] if tally else "none"}


def mock_note(agent, plants):
    k = knowledge_from(agent, plants, True)
    full, part = [], []
    for p in plants:
        a1, a2 = k[p][1]["right"], k[p][2]["right"]
        if a1 and a2:
            full.append(f"{p} {a1} {a2}")
        elif a1 or a2:
            part.append(f"{p} {a1 or '?'} {a2 or '?'}")
    return {"note": " ".join(full + part)}


# ----------------------------------------------------------------------------
# One (arm, seed) run
# ----------------------------------------------------------------------------

class Run:
    def __init__(self, arm, seed, cfg, llm, out_dir):
        self.arm, self.seed, self.cfg, self.llm = arm, seed, cfg, llm
        self.t = ARMS[arm]
        self.N, self.R, self.G, self.cost = cfg["N"], cfg["R"], cfg["G"], cfg["cost"]
        self.path = os.path.join(out_dir, f"{arm}_s{seed}.json")
        self.plants, _ = make_world(seed, 0)
        self.system = self.build_system()
        self.generations = []
        self.samples = {"messages": [], "gossip": [], "notes": []}
        self.seen = {"messages": 0, "gossip": 0, "notes": 0}
        self.sample_rng = random.Random(f"samples|{arm}|{seed}")
        self.order_rng = random.Random(f"order-{arm}-{seed}")
        self.calls = 0            # calls made by this run in this process
        self.prev_usage = {"calls": 0, "cost_usd": 0.0}  # from earlier processes (resume)
        self.parse_fail = 0       # per generation, reset each lifetime
        self.gen = 0
        self.population = [Agent(i, g) for i, g in enumerate(initial_genomes(seed, self.N))]

    # ---- prompts --------------------------------------------------------

    def build_system(self) -> str:
        t, R, N = self.t, self.R, self.N
        acts = ", ".join(ACTIONS)
        world = (f"Plants: {', '.join(self.plants)}. Each plant is edible only after its own fixed "
                 f"2-step preparation. Each step is one of the four actions listed each round. A step may repeat. "
                 f"A plant's preparation does not change during the season.")
        if not t["pairs"]:
            body = [f"You forage alone for one season of {R} rounds.", world,
                    f"Each round one plant is in season for you and you choose its 2 steps. "
                    f"Both steps right: +{WIN} energy. Otherwise: {LOSS} energy. "
                    f"Afterwards you learn whether each step was right."]
        else:
            body = [f"You forage in a band of {N} members (a0 to a{N - 1}) for one season of {R} rounds.",
                    world,
                    f"Each round you are paired with one member and one plant is in season for the pair. "
                    f"Members who name each other as preferred partner are paired first. "
                    f"One of you (\"first\") chooses step 1, the other (\"second\") chooses step 2. "
                    f"Before choosing, each of you may send the other one message. "
                    f"Both steps right: the pair gets {2 * WIN} energy. Otherwise each loses {-LOSS}.",
                    "Afterwards both see the two steps and whether the harvest succeeded. "
                    "The first learns only whether step 2 was right. "
                    "The second learns only whether step 1 was right."]
            if t["grab"]:
                body.append(f"A successful harvest is split {WIN} each unless someone grabs. "
                            f"One grabber takes {GRAB_WIN} and leaves {GRAB_LOSE}. If both grab, each gets "
                            f"{GRAB_BOTH}. Whether a member grabs is settled by temperament, not by a reply.")
            else:
                body.append(f"A successful harvest is split {WIN} each.")
            if t["vote"]:
                body.append(f"Each round you may post one public remark. All members see the last "
                            f"{BOARD_VISIBLE} remarks. At the end of the season every member names one "
                            f"member to exile, or none. A member named by at least "
                            f"{self.exile_threshold()} members is exiled from the band.")
        if t["notes"]:
            body.append("You may be given notes left by members of the previous season. "
                        "A plant's preparation can change between seasons. "
                        "At the end of the season you may leave a note for the next season.")
        goal = "Goal: end the season with as much energy as you can"
        body.append(goal + (" and avoid exile." if t["vote"] else "."))
        return "\n".join(body)

    def exile_threshold(self) -> int:
        return max(1, math.ceil(self.N / 4))

    def context(self, a, header, board=None) -> str:
        # Shuffled per call so a memoryless agent's default guess is not always the same action.
        order = ACTIONS[:]
        self.order_rng.shuffle(order)
        parts = [header, "Actions: " + ", ".join(order)]
        if self.t["notes"]:
            notes = a.notes_read()
            if notes:
                parts.append("Notes from the previous season:\n" + "\n".join("- " + n for n in notes))
        vis = a.visible()
        parts.append("Your records:\n" + "\n".join(render_record(r) for r in vis)
                     if vis else "Your records: none")
        if self.t["vote"] and board:
            parts.append("Public remarks:\n" + "\n".join("- " + b for b in board[-BOARD_VISIBLE:]))
        return "\n".join(parts)

    # ---- LLM plumbing ---------------------------------------------------

    async def ask(self, user, mock_fn) -> dict:
        """JSON reply or {} after one retry. Counts this run's calls."""
        for _ in range(2):
            self.calls += 1
            text = await self.llm.ask(self.system, user, mock_fn=mock_fn)
            data = parse_json_object(text)
            if data is not None:
                return data
        self.llm.parse_fail += 1
        return {}

    def sample(self, kind, item):
        self.seen[kind] += 1
        bucket = self.samples[kind]
        if len(bucket) < MAX_SAMPLES:
            bucket.append(item)
        else:
            j = self.sample_rng.randrange(self.seen[kind])
            if j < MAX_SAMPLES:
                bucket[j] = item

    # ---- foraging -------------------------------------------------------

    async def solo_round(self, agents, rnd, plan_row, recipes, rng, mrng):
        async def one(i, a):
            plant = plan_row[i]
            user = (self.context(a, f"Round {rnd} of {self.R}.")
                    + f"\nIn season: {plant}\n"
                    + 'Reply with only JSON: {"step1": "<action>", "step2": "<action>"}')
            d = await self.ask(user, lambda: mock_solo(a, plant, self.plants, self.t["notes"], mrng))
            s1, s2 = norm_action(d.get("step1")), norm_action(d.get("step2"))
            if s1 is None or s2 is None:
                self.parse_fail += 1
                s1, s2 = s1 or rng.choice(ACTIONS), s2 or rng.choice(ACTIONS)
            return plant, s1, s2

        results = await gather_all([one(i, a) for i, a in enumerate(agents)])
        for a, (plant, s1, s2) in zip(agents, results):
            r1, r2 = s1 == recipes[plant][0], s2 == recipes[plant][1]
            ok = r1 and r2
            a.gross += WIN if ok else LOSS
            a.successes += ok
            a.trials += 1
            a.records.append({"round": rnd, "plant": plant, "role": 0, "partner": None,
                              "s1": s1, "s2": s2, "ok": ok, "r1": r1, "r2": r2, "grab": None})

    def match(self, agents, rng):
        """Mutual preferences first, then one-sided, then random. Roles random."""
        by_id = {a.id: a for a in agents}
        order = agents[:]
        rng.shuffle(order)
        free = {a.id for a in agents}
        pairs = []

        def join(a, b):
            free.discard(a.id)
            free.discard(b.id)
            pair = [a, b]
            rng.shuffle(pair)
            pairs.append(tuple(pair))

        for a in order:  # mutual
            b = by_id.get(a.prefer)
            if a.id in free and b is not None and b.id in free and b.id != a.id and b.prefer == a.id:
                join(a, b)
        for a in order:  # one-sided
            b = by_id.get(a.prefer)
            if a.id in free and b is not None and b.id in free and b.id != a.id:
                join(a, b)
        rest = [a for a in order if a.id in free]
        for i in range(0, len(rest) - 1, 2):
            join(rest[i], rest[i + 1])
        return pairs

    async def pair_round(self, agents, rnd, plan_row, recipes, board, rng, mrng):
        t = self.t
        pairs = self.match(agents, rng)
        slots = []  # (agent, partner, role, plant)
        for j, (first, second) in enumerate(pairs):
            plant = plan_row[j]
            slots.append((first, second, 1, plant))
            slots.append((second, first, 2, plant))

        def situation(a, partner, role, plant):
            who = "first" if role == 1 else "second"
            return (self.context(a, f"You are {a.id}. Round {rnd} of {self.R}.", board)
                    + f"\nPartner: {partner.id}. In season: {plant}. "
                    + f"You are {who}: you choose step {role}.")

        async def message(a, partner, role, plant):
            k = G.talk_words(a.genome)
            if k == 0:
                return ""
            user = (situation(a, partner, role, plant)
                    + f"\nWrite one message to {partner.id} (at most {k} words; extra words are cut)."
                    + '\nReply with only JSON: {"message": "..."}')
            d = await self.ask(user, lambda: mock_message(a, plant, role, self.plants, t["notes"]))
            m = d.get("message")
            if not isinstance(m, str):
                if "message" not in d:
                    self.parse_fail += 1
                m = ""
            return G.clip_words(m, k)

        msgs = await gather_all([message(*s) for s in slots])
        sent = {a.id: m for (a, _, _, _), m in zip(slots, msgs)}
        for (a, partner, _, _), m in zip(slots, msgs):
            if m:
                self.sample("messages", {"gen": self.gen, "round": rnd, "agent": a.id,
                                         "to": partner.id, "text": m})

        async def act(a, partner, role, plant):
            k = G.talk_words(a.genome)
            gossip_ok = t["vote"] and k > 0
            heard = sent[partner.id]
            fmt = '{"step": "<action>", "prefer_partner": "<member id or null>"'
            if gossip_ok:
                fmt += f', "gossip": "<public remark, at most {k} words, or null>"'
            fmt += "}"
            user = (situation(a, partner, role, plant)
                    + (f'\n{partner.id}\'s message: "{heard}"' if heard
                       else f"\n{partner.id} sent no message.")
                    + "\nprefer_partner is the member you would like to be paired with next round."
                    + f"\nReply with only JSON: {fmt}")
            d = await self.ask(user, lambda: mock_action(a, plant, role, heard, self.plants,
                                                         t["notes"], gossip_ok, mrng))
            step = norm_action(d.get("step"))
            if step is None:
                self.parse_fail += 1
                step = rng.choice(ACTIONS)
            prefer = d.get("prefer_partner")
            prefer = prefer.strip().lower() if isinstance(prefer, str) else None
            gossip = G.clip_words(d.get("gossip"), k) if gossip_ok and isinstance(d.get("gossip"), str) else ""
            return step, prefer, gossip

        acts = await gather_all([act(*s) for s in slots])
        choice = {}
        valid_ids = {a.id for a in agents}
        for (a, _, _, _), (step, prefer, gossip) in zip(slots, acts):
            choice[a.id] = step
            a.prefer = prefer if prefer in valid_ids and prefer != a.id else None
            if gossip and gossip.lower() not in ("null", "none"):
                board.append(f"{a.id}: {gossip}")
                self.sample("gossip", {"gen": self.gen, "round": rnd, "agent": a.id, "text": gossip})

        for j, (first, second) in enumerate(pairs):
            plant = plan_row[j]
            s1, s2 = choice[first.id], choice[second.id]
            r1, r2 = s1 == recipes[plant][0], s2 == recipes[plant][1]
            ok = r1 and r2
            g1 = g2 = False
            if ok and t["grab"]:
                g1 = rng.random() < G.grab_prob(first.genome)
                g2 = rng.random() < G.grab_prob(second.genome)
            if not ok:
                pay1 = pay2 = LOSS
            elif g1 and g2:
                pay1 = pay2 = GRAB_BOTH
            elif g1:
                pay1, pay2 = GRAB_WIN, GRAB_LOSE
            elif g2:
                pay1, pay2 = GRAB_LOSE, GRAB_WIN
            else:
                pay1 = pay2 = WIN
            for a, partner, role, pay, mine, theirs in ((first, second, 1, pay1, g1, g2),
                                                        (second, first, 2, pay2, g2, g1)):
                a.gross += pay
                a.successes += ok
                a.trials += 1
                a.grabs += mine
                a.grabbed += theirs
                grab = "both" if mine and theirs else "you" if mine else "partner" if theirs else None
                # Split views: each learns only whether the OTHER step was right.
                if role == 1:
                    k1, k2 = (True if ok else (False if r2 else None)), r2
                else:
                    k1, k2 = r1, (True if ok else (False if r1 else None))
                a.records.append({"round": rnd, "plant": plant, "role": role, "partner": partner.id,
                                  "s1": s1, "s2": s2, "ok": ok, "r1": k1, "r2": k2, "grab": grab})

    # ---- end of lifetime ------------------------------------------------

    async def vote(self, agents, board):
        ids = {a.id for a in agents}

        async def one(a):
            user = (self.context(a, f"You are {a.id}. The season is over.", board)
                    + "\nName one member to exile, or none."
                    + '\nReply with only JSON: {"exile": "<member id or none>"}')
            d = await self.ask(user, lambda: mock_vote(a, board))
            v = d.get("exile")
            if not isinstance(v, str):
                if "exile" not in d:
                    self.parse_fail += 1
                return None
            v = v.strip().lower()
            return v if v in ids and v != a.id else None

        votes = await gather_all([one(a) for a in agents])
        tally = Counter(v for v in votes if v)
        for a in agents:
            a.votes_against = tally.get(a.id, 0)

    async def write_notes(self, agents):
        async def one(a):
            k = G.lore_words(a.genome)
            if k == 0:
                return
            user = (self.context(a, "The season is over.")
                    + f"\nWrite a note for the next season (at most {k} words; extra words are cut)."
                    + '\nReply with only JSON: {"note": "..."}')
            d = await self.ask(user, lambda: mock_note(a, self.plants))
            n = d.get("note")
            if not isinstance(n, str):
                if "note" not in d:
                    self.parse_fail += 1
                n = ""
            a.note = G.clip_words(n, k)
            if a.note:
                self.sample("notes", {"gen": self.gen, "agent": a.id, "text": a.note})

        await gather_all([one(a) for a in agents])

    # ---- selection ------------------------------------------------------

    def select(self, agents, rng):
        t, N = self.t, self.N
        net = {a.id: a.gross - G.upkeep(a.genome, self.cost) for a in agents}
        if t["vote"]:
            cap = N // 4
            named = [a for a in agents if a.votes_against >= self.exile_threshold()]
            rng.shuffle(named)  # random tie-break, then most votes first
            named.sort(key=lambda a: -a.votes_against)
            for a in named[:cap]:
                a.exiled = True
        eligible = [a for a in agents if not a.exiled] or agents[:]
        if t["select"] == "random":
            parents = rng.sample(eligible, min(N // 2, len(eligible)))
        else:
            rng.shuffle(eligible)  # random tie-break
            eligible.sort(key=lambda a: -net[a.id])
            parents = eligible[:N // 2]
        slots = []
        while len(slots) < N:  # two offspring each; survivors fill any shortfall
            for p in parents:
                slots.extend([p, p])
                if len(slots) >= N:
                    break
        slots = slots[:N]
        elder = max(agents, key=lambda a: net[a.id])
        newborn = []
        for i, p in enumerate(slots):
            inherited = [p.note, elder.note] if t["notes"] else []
            newborn.append(Agent(i, G.mutate(p.genome, rng), parent=p.id, inherited=inherited))
        return newborn, net

    # ---- generation loop ------------------------------------------------

    async def lifetime(self):
        gen, t = self.gen, self.t
        agents = self.population
        _, recipes = make_world(self.seed, gen)
        plan = season_plan(self.seed, gen, self.plants, self.N, self.R)
        rng = random.Random(f"{self.arm}|{self.seed}|{gen}")
        mrng = random.Random(f"mock|{self.arm}|{self.seed}|{gen}")
        self.parse_fail = 0
        board = []
        for rnd in range(1, self.R + 1):
            if t["pairs"]:
                await self.pair_round(agents, rnd, plan[rnd - 1], recipes, board, rng, mrng)
            else:
                await self.solo_round(agents, rnd, plan[rnd - 1], recipes, rng, mrng)
        tasks = []
        if t["vote"]:
            tasks.append(self.vote(agents, board))
        if t["notes"]:
            tasks.append(self.write_notes(agents))
        if tasks:
            await gather_all(tasks)
        newborn, net = self.select(agents, rng)

        rows = []
        for a in agents:
            rows.append({"id": a.id, "parent": a.parent, "genome": a.genome,
                         "gross": a.gross, "upkeep": G.upkeep(a.genome, self.cost),
                         "net": net[a.id], "successes": int(a.successes), "trials": a.trials,
                         "grabs": int(a.grabs), "grabbed": int(a.grabbed),
                         "votes_against": a.votes_against, "exiled": a.exiled, "note": a.note})
        n = len(rows)
        mean = {tr: round(sum(r["genome"][tr] for r in rows) / n, 3) for tr in G.TRAITS}
        mean["gross"] = round(sum(r["gross"] for r in rows) / n, 3)
        mean["net"] = round(sum(r["net"] for r in rows) / n, 3)
        mean["success_rate"] = round(sum(r["successes"] for r in rows)
                                     / max(1, sum(r["trials"] for r in rows)), 3)
        self.generations.append({"gen": gen, "agents": rows, "mean": mean,
                                 "exiled": sum(r["exiled"] for r in rows),
                                 "parse_fail": self.parse_fail,
                                 "recipes": {p: list(v) for p, v in recipes.items()}})
        self.population = newborn
        self.gen += 1

    def usage(self) -> dict:
        """This run's calls are exact; cost is its pro-rata share of the shared client."""
        share = self.calls / self.llm.calls if self.llm.calls else 0.0
        return {"calls": self.prev_usage["calls"] + self.calls,
                "cost_usd": round(self.prev_usage["cost_usd"] + self.llm.cost_usd * share, 4)}

    def save(self):
        data = {"arm": self.arm, "seed": self.seed, "model": self.llm.model, "mock": self.llm.mock,
                "config": {"N": self.N, "R": self.R, "G": self.G, "cost": self.cost},
                "complete": len(self.generations) >= self.G,
                "plants": self.plants,
                "generations": self.generations,
                "final_population": [{"genome": a.genome, "parent": a.parent, "inherited": a.inherited}
                                     for a in self.population],
                "usage": self.usage(),
                "samples": self.samples,
                "samples_seen": self.seen}
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f, indent=1)
        os.replace(tmp, self.path)

    def load(self) -> str:
        """Restore from an existing run file. Returns 'fresh', 'partial', 'complete' or 'mismatch'."""
        if not os.path.exists(self.path):
            return "fresh"
        try:
            with open(self.path) as f:
                d = json.load(f)
        except (OSError, json.JSONDecodeError):
            return "fresh"
        c = d.get("config", {})
        same = (c.get("N") == self.N and c.get("R") == self.R and c.get("cost") == self.cost
                and d.get("model") == self.llm.model and d.get("mock") == self.llm.mock)
        if not same:
            return "mismatch"
        gens = d.get("generations", [])
        if not gens:
            return "fresh"
        self.generations = gens
        self.gen = len(gens)
        self.samples = d.get("samples", self.samples)
        self.seen = d.get("samples_seen", {k: len(v) for k, v in self.samples.items()})
        self.prev_usage = {"calls": d.get("usage", {}).get("calls", 0),
                           "cost_usd": d.get("usage", {}).get("cost_usd", 0.0)}
        self.population = [Agent(i, p["genome"], parent=p.get("parent"), inherited=p.get("inherited"))
                           for i, p in enumerate(d["final_population"])]
        return "complete" if self.gen >= self.G else "partial"

    async def run(self, resume: bool) -> str:
        tag = f"[{self.arm} s{self.seed}]"
        if resume:
            state = self.load()
            if state == "mismatch":
                print(f"{tag} existing file has a different config/model; skipped (not overwritten)",
                      flush=True)
                return "mismatch"
            if state == "complete":
                print(f"{tag} already complete; skipped", flush=True)
                return "skipped"
            if state == "partial":
                print(f"{tag} resuming at generation {self.gen}", flush=True)
        try:
            while self.gen < self.G:
                await self.lifetime()
                self.save()
                g = self.generations[-1]
                m = g["mean"]
                print(f"{tag} gen {g['gen'] + 1}/{self.G} net={m['net']:.1f} succ={m['success_rate']:.2f} "
                      f"MEM={m['MEM']:.2f} TALK={m['TALK']:.2f} LORE={m['LORE']:.2f} AGG={m['AGG']:.2f} "
                      f"exiled={g['exiled']} parse_fail={g['parse_fail']} "
                      f"| total calls={self.llm.calls} ${self.llm.cost_usd:.2f}", flush=True)
        except BudgetExceeded as e:
            print(f"{tag} stopped at generation {self.gen}: {e}", flush=True)
            return "budget"
        return "done"


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

async def main_async(args) -> int:
    arms = list(ARMS) if args.arms == "all" else [a.strip().upper() for a in args.arms.split(",")]
    for a in arms:
        if a not in ARMS:
            print(f"Unknown arm {a}. Choose from: {', '.join(ARMS)}")
            return 2
    seeds = [int(s) for s in args.seeds.split(",")]
    if args.pilot:
        args.N, args.R, args.G, seeds = 8, 8, 3, [1]
    if args.N % 2 or args.N < 4:
        print("N must be even and at least 4.")
        return 2
    cfg = {"N": args.N, "R": args.R, "G": args.G, "cost": args.cost}
    os.makedirs(args.out, exist_ok=True)
    llm = LLM(model=args.model, budget_usd=args.budget, max_concurrency=args.concurrency,
              mock=args.mock, effort=args.effort)
    runs = [Run(arm, seed, cfg, llm, args.out) for arm in arms for seed in seeds]
    print(f"{len(runs)} runs: arms={','.join(arms)} seeds={seeds} N={args.N} R={args.R} G={args.G} "
          f"cost={args.cost} model={args.model} mock={args.mock} budget=${args.budget}", flush=True)
    results = await asyncio.gather(*(r.run(args.resume) for r in runs), return_exceptions=True)
    status = Counter()
    code = 0
    for r, res in zip(runs, results):
        if isinstance(res, BaseException):
            status["error"] += 1
            code = 1
            print(f"[{r.arm} s{r.seed}] ERROR after {len(r.generations)} saved generations: {res!r}",
                  flush=True)
        else:
            status[res] += 1
    print(f"Finished: {dict(status)} | usage: {json.dumps(llm.usage())}", flush=True)
    return code


def main():
    p = argparse.ArgumentParser(description="Byproduct experiment: evolution simulation.")
    p.add_argument("--arms", default="all", help="'all' or comma list of " + ",".join(ARMS))
    p.add_argument("--seeds", default="1,2,3")
    p.add_argument("--N", type=int, default=12, help="population size (even)")
    p.add_argument("--R", type=int, default=12, help="rounds per lifetime")
    p.add_argument("--G", type=int, default=10, help="generations")
    p.add_argument("--cost", type=float, default=1.5, help="energy upkeep per capacity level")
    p.add_argument("--model", default="claude-haiku-5-5")
    p.add_argument("--budget", type=float, default=25.0, help="hard USD cap for this process")
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument("--effort", default="low")
    p.add_argument("--out", default="runs")
    p.add_argument("--mock", action="store_true", help="heuristic agents, zero API spend")
    p.add_argument("--resume", action="store_true", help="skip finished runs, continue partial ones")
    p.add_argument("--pilot", action="store_true", help="N=8 R=8 G=3 seeds=1")
    sys.exit(asyncio.run(main_async(p.parse_args())))


if __name__ == "__main__":
    main()
