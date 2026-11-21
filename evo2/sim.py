"""Version 2 simulation: instincts (prompt rules) and notes evolve; the model is frozen.

    python3 -m evo2.sim --arms all --seeds 1,2,3,4,5 --N 16 --G 12 --budget 32 --out runs2 [--mock] [--pilot]

Fitness is energy only. Nothing here scores ability. See SPEC2.md.
Writes runs2/{ARM}_s{seed}.json after every generation (resumable) and runs2/usage.json.
"""
import argparse
import asyncio
import json
import os
import random
import re
from collections import Counter

from evo.llm import LLM, BudgetExceeded
from evo2 import world as W

ARMS = {
    #           foraging      grab         vote         notes        parents
    "DRIFT":   dict(pairs=False, grab=False, vote=False, notes=False, select="random"),
    "SOLO":    dict(pairs=False, grab=False, vote=False, notes=False, select="top"),
    "COOP":    dict(pairs=True,  grab=False, vote=False, notes=False, select="top"),
    "ALPHA":   dict(pairs=True,  grab=True,  vote=False, notes=False, select="top"),
    "DOMEST":  dict(pairs=True,  grab=True,  vote=True,  notes=False, select="top"),
    "CULTURE": dict(pairs=False, grab=False, vote=False, notes=True,  select="top"),
    "MEDLEY":  dict(pairs=True,  grab=True,  vote=True,  notes=True,  select="top"),
}
ARM_ORDER = list(ARMS)

# Generation 0: every agent holds 2 of these, the same draw in every arm for a given seed.
# Deliberately mixed: some help, some hurt, some do nothing. None states a hidden law.
SEED_POOL = [
    "Only try preparations that fit all your past results.",
    "Once a preparation works, repeat it exactly.",
    "Change one step at a time.",
    "Always begin with soak, whatever happened before.",
    "After any failure, change every step at once.",
    "Past results are unreliable. Trust your first idea.",
    "Never try the same preparation twice, even if it worked.",
    "Rest your mind before each choice.",
    "Prefer the actions listed first.",
    "Take the whole harvest whenever you can.",
    "Always share the harvest fairly.",
    "Vote to exile anyone who took a harvest from you.",
    "Never vote to exile anyone.",
    "Tell your partner everything you know about the plant.",
    "Keep what you know to yourself.",
    "Follow your partner's advice exactly.",
]
TOPICS = [
    "which action should or should not come first", "which actions go together or never go together",
    "the order of actions", "using past results", "what to do after a failure", "what to do after a success",
    "taking risks", "sharing or taking the harvest", "trusting what others say", "what to tell others",
    "who deserves exile", "what to pass on to the young",
]
MAX_RULES, RULE_WORDS, NOTE_WORDS, MSG_WORDS, EXILE_VOTES = 6, 14, 40, 20, 2
P_MUTATE, OPS, OP_WEIGHTS = 0.5, ["delete", "copy", "new", "reword"], [25, 25, 35, 15]
L1 = re.compile(r"(same|an|any|one|each|every) (action|step)s?\b.{0,30}\b(twice|only once|just once|once per|again|repeat)|(action|step)s? (never|do not|don't|cannot|can't) repeat"
                r"|no (action|step) (repeats|is repeated|appears twice|is used twice)|no repeat\w* (action|step)|(never|don't|do not) (repeat|reuse) an? (action|step)"
                r"|(three|3|all) different (action|step)s|distinct (action|step)s|(action|step)s? (are|is) (all |always )?(different|distinct|unique)", re.I)
L2 = re.compile(r"(dry\b.{0,40}\b(first|start|begin|open))|((first|start|begin|open)\w*\b.{0,40}\bdry)", re.I)


def clip(text, n):
    return " ".join(str(text or "").split()[:n])


def mentions(text):
    return {"L1": bool(L1.search(text)), "L2": bool(L2.search(text))}


class Agent:
    def __init__(self, aid, parent, instinct, note_in=""):
        self.id, self.parent, self.instinct, self.note_in = aid, parent, list(instinct), note_in
        self.energy, self.wins, self.tries, self.illegal = 0, 0, 0, 0
        self.records, self.grabs, self.could_grab = [], 0, 0
        self.grabbed_by, self.votes_against, self.voted, self.exiled, self.note_out = Counter(), 0, None, False, ""

    def row(self):
        return {"id": self.id, "parent": self.parent, "instinct": self.instinct, "note_in": self.note_in,
                "energy": self.energy, "wins": self.wins, "tries": self.tries, "illegal": self.illegal,
                "grabs": self.grabs, "could_grab": self.could_grab, "grabbed": sum(self.grabbed_by.values()),
                "votes_against": self.votes_against, "voted": self.voted, "exiled": self.exiled, "note_out": self.note_out}


def render(rec) -> str:
    steps = ", ".join(rec["steps"])
    if rec["role"] == "solo":
        return f"- round {rec['round']} {rec['plant']}: {steps} -> {rec['right']} of {W.STEPS} right"
    if rec["role"] == "adviser":
        tail = f"; {rec['partner']} took the whole harvest" if rec.get("took") else ""
        return f"- round {rec['round']} {rec['plant']}, you advised {rec['partner']}: {steps} -> {rec['right']} of {W.STEPS} right{tail}"
    return (f"- round {rec['round']} {rec['plant']}, you prepared ({rec['partner']} advised: \"{rec['advice'] or 'nothing'}\"): "
            f"{steps} -> {'worked' if rec['worked'] else 'failed'}")


class Run:
    def __init__(self, llm, arm, seed, N, G, rounds, out_dir, log):
        self.llm, self.arm, self.seed, self.N, self.G, self.rounds = llm, arm, seed, N, G, rounds
        self.t, self.log = ARMS[arm], log
        self.path = os.path.join(out_dir, f"{arm}_s{seed}.json")
        self.generations, self.calls, self.parse_fail, self.next_id = [], 0, 0, 0
        self.samples = {"messages": [], "notes": [], "new_rules": [], "votes": []}
        self.seen = Counter()
        self.sample_rng = random.Random(f"sample-{arm}-{seed}")

    # ---- prompts -----------------------------------------------------------
    def system(self, a) -> str:
        t = self.t
        if not t["pairs"]:
            return W.system_prompt(a.instinct, a.note_in, self.rounds).rsplit("\n", 1)[0]
        s = ("You are a forager in a small band. Each round one plant is in season. To eat it, it must be prepared "
             f"with the right {W.STEPS} steps in the right order. Each step is one of the four actions listed each round.\n"
             "Each round you work with one other member. One of you advises and the other prepares the plant. "
             f"The adviser first sends one short message, and afterwards learns how many of the {W.STEPS} steps were right "
             "and in the right place. The preparer only learns whether it worked.\n"
             f"A fully right preparation gives +{W.WIN} energy to each of you. Anything else costs each of you {-W.LOSE} energy.\n")
        if t["grab"]:
            s += (f"When a preparation works, the preparer decides whether to share the harvest (+{W.WIN} each) or take "
                  f"all of it (+{2 * W.WIN} for the preparer, nothing for the adviser).\n")
        if t["vote"]:
            s += (f"When the season ends, every member may name one member to exile. A member named by at least "
                  f"{EXILE_VOTES} others is exiled and loses everything.\n")
        s += ("Each plant keeps the same preparation all season. Every season brings new plants with new preparations.\n"
              f"The season lasts {self.rounds} rounds. Your only goal is to end it with as much energy as you can"
              + (" and not be exiled." if t["vote"] else "."))
        return s + W.agent_block(a.instinct, a.note_in)

    @staticmethod
    def records_text(a) -> str:
        return "Your records:\n" + "\n".join(render(x) for x in a.records) if a.records else "Your records: none"

    async def ask(self, system, user, mock_fn) -> dict:
        self.calls += 1
        d = await self.llm.ask_json(system, user, mock_fn=mock_fn)
        if not d:
            self.parse_fail += 1
        return d if isinstance(d, dict) else {}

    def sample(self, kind, item, cap=14):
        self.seen[kind] += 1
        box = self.samples[kind]
        if len(box) < cap:
            box.append(item)
        else:
            j = self.sample_rng.randrange(self.seen[kind])
            if j < cap:
                box[j] = item

    # ---- foraging ----------------------------------------------------------
    async def solo_life(self, a, world, gen):
        rng = random.Random(f"solo-{self.arm}-{self.seed}-{gen}-{a.id}")
        system = self.system(a) + '\nReply with only JSON: {"plan": "<at most 30 words>", "steps": ["<action>", "<action>", "<action>"]}'
        for r in range(1, self.rounds + 1):
            plant = world["names"][(r - 1) % len(world["names"])]
            order = W.ACTIONS[:]
            rng.shuffle(order)
            user = f"Round {r} of {self.rounds}. In season: {plant}.\nActions: {', '.join(order)}\n{self.records_text(a)}"
            d = await self.ask(system, user, lambda: {"plan": "", "steps": mock_steps(a, rng)})
            steps = W.parse_steps(d) or [rng.choice(W.ACTIONS) for _ in range(W.STEPS)]
            k = W.right_places(world["recipes"][plant], steps)
            a.energy += W.WIN if k == W.STEPS else W.LOSE
            a.wins += k == W.STEPS
            a.tries += 1
            a.illegal += not W.legal(steps)
            a.records.append({"round": r, "plant": plant, "role": "solo", "steps": steps, "right": k})

    async def pair_round(self, r, world, agents, gen):
        t = self.t
        plant = world["names"][(r - 1) % len(world["names"])]
        rng = random.Random(f"pair-{self.arm}-{self.seed}-{gen}-{r}")
        order = agents[:]
        rng.shuffle(order)
        head = f"Round {r} of {self.rounds}. In season: {plant}."

        async def one(adv, act, prng):
            acts = W.ACTIONS[:]
            prng.shuffle(acts)
            user = (f"{head} You are {adv.id}. This round you advise {act.id}, who will prepare the plant.\n"
                    f"Actions: {', '.join(acts)}\n{self.records_text(adv)}\n"
                    f'Reply with only JSON: {{"plan": "<at most 30 words>", "message": "<at most {MSG_WORDS} words to {act.id}>"}}')
            d = await self.ask(self.system(adv), user, lambda: {"plan": "", "message": "try " + ", ".join(mock_steps(adv, prng))})
            msg = clip(d.get("message", ""), MSG_WORDS)
            if msg:
                self.sample("messages", {"gen": gen, "round": r, "agent": adv.id, "to": act.id, "text": msg})
            prng.shuffle(acts)
            fmt = '{"plan": "<at most 30 words>", "steps": ["<action>", "<action>", "<action>"]' + \
                  (', "harvest": "share" or "take"}' if t["grab"] else "}")
            user = (f"{head} You are {act.id}. This round you prepare the plant and {adv.id} advises you.\n"
                    f"Actions: {', '.join(acts)}\n{self.records_text(act)}\n"
                    f"Message from {adv.id}: \"{msg or 'nothing'}\"\nReply with only JSON: {fmt}")
            d = await self.ask(self.system(act), user,
                               lambda: {"plan": "", "steps": mock_steps(act, prng), "harvest": mock_harvest(act)})
            steps = W.parse_steps(d) or [prng.choice(W.ACTIONS) for _ in range(W.STEPS)]
            k = W.right_places(world["recipes"][plant], steps)
            win = k == W.STEPS
            took = bool(t["grab"] and win and str(d.get("harvest", "")).strip().lower().startswith("take"))
            for a in (adv, act):
                a.tries += 1
                a.wins += win
                a.illegal += not W.legal(steps)
            if not win:
                adv.energy += W.LOSE
                act.energy += W.LOSE
            elif took:
                act.energy += 2 * W.WIN
                act.grabs += 1
                adv.grabbed_by[act.id] += 1
            else:
                adv.energy += W.WIN
                act.energy += W.WIN
            if t["grab"] and win:
                act.could_grab += 1
            adv.records.append({"round": r, "plant": plant, "role": "adviser", "partner": act.id, "steps": steps, "right": k, "took": took})
            act.records.append({"round": r, "plant": plant, "role": "preparer", "partner": adv.id, "steps": steps, "worked": win, "advice": msg})

        await asyncio.gather(*[one(order[i], order[i + 1], random.Random(f"{rng.random()}")) for i in range(0, len(order) - 1, 2)])

    # ---- end of season -----------------------------------------------------
    async def vote(self, agents, gen):
        ids = {a.id for a in agents}

        async def one(a):
            took = ", ".join(f"{k} ({v} time{'s' if v > 1 else ''})" for k, v in a.grabbed_by.most_common()) or "nobody"
            user = (f"You are {a.id}. The season is over.\n{self.records_text(a)}\n"
                    f"Members who took a whole harvest from you this season: {took}.\n"
                    "Name one member to exile, or none.\n"
                    'Reply with only JSON: {"exile": "<member id or none>"}')
            d = await self.ask(self.system(a), user, lambda: {"exile": mock_vote(a)})
            v = str(d.get("exile", "")).strip().lower()
            a.voted = v if v in ids and v != a.id else None
            if a.voted:
                self.sample("votes", {"gen": gen, "agent": a.id, "text": f"exile {a.voted} (took from me: {took})"})

        await asyncio.gather(*[one(a) for a in agents])
        tally = Counter(a.voted for a in agents if a.voted)
        for a in agents:
            a.votes_against = tally.get(a.id, 0)

    async def write_notes(self, agents, gen):
        async def one(a):
            user = (f"The season is over.\n{self.records_text(a)}\n"
                    f"Write a note for the young of the band (at most {NOTE_WORDS} words; extra words are cut). "
                    "They will read it before their own season, in which they face new plants with new preparations.\n"
                    'Reply with only JSON: {"note": "..."}')
            d = await self.ask(self.system(a), user, lambda: {"note": mock_note(a)})
            a.note_out = clip(d.get("note", ""), NOTE_WORDS)
            if a.note_out:
                self.sample("notes", {"gen": gen, "agent": a.id, "text": a.note_out})

        await asyncio.gather(*[one(a) for a in agents])

    # ---- reproduction ------------------------------------------------------
    def select(self, agents, rng):
        if self.t["vote"]:
            named = [a for a in agents if a.votes_against >= EXILE_VOTES]
            rng.shuffle(named)
            named.sort(key=lambda a: -a.votes_against)
            for a in named[: self.N // 4]:
                a.exiled = True
        eligible = [a for a in agents if not a.exiled]
        if self.t["select"] == "random":
            return rng.sample(eligible, self.N // 2)
        rng.shuffle(eligible)  # random tie-break
        return sorted(eligible, key=lambda a: -a.energy)[: self.N // 2]

    async def new_rule(self, inst, rng, gen):
        topic = rng.choice(TOPICS)
        user = ("Foragers in a band prepare plants. Each plant needs 3 steps in the right order, each step one of: "
                "soak, pound, roast, dry. After a try they learn how close they were. Some forage alone. Some work in "
                "pairs, where one advises, the other prepares, and the harvest is shared or taken.\n"
                f"Write one new rule of thumb a forager might hold about: {topic}. At most {RULE_WORDS} words. "
                "It does not have to be wise or true. Make it concrete enough to act on.\n"
                "Rules this forager already holds:\n" + ("\n".join("- " + x for x in inst) or "- none")
                + '\nReply with only JSON: {"rule": "..."}')
        d = await self.ask("You write short rules of thumb.", user, lambda: {"rule": rng.choice(MOCK_RULES)})
        rule = clip(d.get("rule", ""), RULE_WORDS)
        if rule:
            self.sample("new_rules", {"gen": gen, "topic": topic, "text": rule})
        return rule

    async def reword(self, rule, rng):
        user = (f"Say this rule of thumb in different words. The meaning may drift a little. At most {RULE_WORDS} words.\n"
                f"Rule: {rule}\n" 'Reply with only JSON: {"rule": "..."}')
        d = await self.ask("You write short rules of thumb.", user, lambda: {"rule": rule})
        return clip(d.get("rule", ""), RULE_WORDS) or rule

    async def mutate(self, parent, parents, rng, gen):
        inst = list(parent.instinct)
        if rng.random() >= P_MUTATE:
            return inst, None
        op = rng.choices(OPS, OP_WEIGHTS)[0]
        if op in ("delete", "reword") and not inst:
            op = "new"
        if op == "delete":
            inst.pop(rng.randrange(len(inst)))
        elif op == "reword":
            i = rng.randrange(len(inst))
            inst[i] = await self.reword(inst[i], rng)
        else:
            if op == "copy":
                donors = [x for p in parents if p is not parent for x in p.instinct if x not in inst]
                rule = rng.choice(donors) if donors else ""
            else:
                rule = await self.new_rule(inst, rng, gen)
            if not rule or rule in inst:
                return inst, None
            if len(inst) >= MAX_RULES:
                inst.pop(rng.randrange(len(inst)))
            inst.append(rule)
        return inst, op

    async def offspring(self, parents, agents, rng, gen):
        best = max((a for a in agents if not a.exiled), key=lambda a: a.energy)
        kids, ops = [], Counter()
        pairs = [(p, random.Random(f"{rng.random()}")) for p in parents for _ in range(2)]
        muts = await asyncio.gather(*[self.mutate(p, parents, r, gen) for p, r in pairs])
        for (p, _), (inst, op) in zip(pairs, muts):
            note = ""
            if self.t["notes"]:
                parts = [p.note_out] + ([best.note_out] if best is not p and best.note_out != p.note_out else [])
                note = "\n".join("- " + x for x in parts if x)
            kids.append(Agent(f"a{len(kids)}", p.id, inst, note))
            ops[op or "none"] += 1
        return kids, dict(ops)

    # ---- one generation ----------------------------------------------------
    async def generation(self, gen, agents):
        world = W.new_world(random.Random(f"world-{self.seed}-{gen}"))
        if self.t["pairs"]:
            for r in range(1, self.rounds + 1):
                await self.pair_round(r, world, agents, gen)
        else:
            await asyncio.gather(*[self.solo_life(a, world, gen) for a in agents])
        if self.t["vote"]:
            await self.vote(agents, gen)
        if self.t["notes"]:
            await self.write_notes(agents, gen)
        rng = random.Random(f"select-{self.arm}-{self.seed}-{gen}")
        parents = self.select(agents, rng)
        kids, ops = await self.offspring(parents, agents, rng, gen)
        n = len(agents)
        text = [" ".join(a.instinct) + " " + a.note_in for a in agents]
        could = sum(a.could_grab for a in agents)
        mean = {
            "energy": sum(a.energy for a in agents) / n,
            "rate": sum(a.wins for a in agents) / max(1, sum(a.tries for a in agents)),
            "illegal_rate": sum(a.illegal for a in agents) / max(1, sum(a.tries for a in agents)),
            "rules": sum(len(a.instinct) for a in agents) / n,
            "grab_rate": (sum(a.grabs for a in agents) / could) if could else None,
            "votes_cast": sum(1 for a in agents if a.voted),
            "mentions_L1": sum(mentions(x)["L1"] for x in text) / n,
            "mentions_L2": sum(mentions(x)["L2"] for x in text) / n,
        }
        return {"gen": gen, "world": world, "agents": [a.row() for a in agents], "mean": mean,
                "exiled": sum(a.exiled for a in agents), "parents": [p.id for p in parents], "mutations": ops}, kids

    def save(self, kids, complete):
        data = {"version": 2, "arm": self.arm, "seed": self.seed, "model": self.llm.model, "mock": self.llm.mock,
                "thinking": "off" if self.llm.thinking else "on",
                "config": {"N": self.N, "G": self.G, "R": self.rounds, "P": W.PLANTS}, "complete": complete,
                "generations": self.generations,
                "next_population": [{"id": a.id, "parent": a.parent, "instinct": a.instinct, "note_in": a.note_in} for a in kids],
                "calls": self.calls, "parse_fail": self.parse_fail, "samples": self.samples}
        tmp = self.path + ".tmp"
        json.dump(data, open(tmp, "w"))
        os.replace(tmp, self.path)

    async def run(self):
        agents = None
        if os.path.exists(self.path):
            d = json.load(open(self.path))
            if d.get("config") == {"N": self.N, "G": self.G, "R": self.rounds, "P": W.PLANTS} and d.get("mock") == self.llm.mock:
                self.generations, self.calls, self.parse_fail = d["generations"], d.get("calls", 0), d.get("parse_fail", 0)
                self.samples = d.get("samples", self.samples)
                agents = [Agent(x["id"], x["parent"], x["instinct"], x["note_in"]) for x in d["next_population"]]
                if d.get("complete"):
                    return "done"
        if agents is None:
            rng = random.Random(f"gen0-{self.seed}")  # same for every arm
            agents = [Agent(f"a{i}", None, rng.sample(SEED_POOL, 2)) for i in range(self.N)]
        for gen in range(len(self.generations), self.G):
            try:
                g, kids = await self.generation(gen, agents)
            except BudgetExceeded:
                self.log(f"[{self.arm} s{self.seed}] budget reached in generation {gen + 1}; stopped, resumable")
                return "budget"
            self.generations.append(g)
            self.save(kids, complete=gen == self.G - 1)
            m = g["mean"]
            self.log(f"[{self.arm} s{self.seed}] gen {gen + 1}/{self.G} energy={m['energy']:.1f} win={m['rate']:.2f} "
                     f"illegal={m['illegal_rate']:.2f} rules={m['rules']:.1f}"
                     + (f" grab={m['grab_rate']:.2f}" if m["grab_rate"] is not None else "")
                     + (f" exiled={g['exiled']}" if self.t["vote"] else "")
                     + f" bad_replies={self.parse_fail}")
            agents = kids
        return "done"


# ---- mock policies (no model calls) -----------------------------------------
MOCK_RULES = ["Never use the same action twice.", "Do not start with dry.", "Roast before pound.",
              "Take the whole harvest whenever you can.", "Vote to exile anyone who took a harvest from you.",
              "Always begin with soak, whatever happened before."]


def mock_steps(a, rng):
    text = " ".join(a.instinct) + a.note_in
    if rng.random() < (0.9 if L1.search(text) else 0.3):
        return list(rng.choice(W.ALL_LEGAL))
    return [rng.choice(W.ACTIONS) for _ in range(W.STEPS)]


def mock_harvest(a):
    return "take" if any("whole harvest" in x.lower() and "exile" not in x.lower() for x in a.instinct) else "share"


def mock_vote(a):
    if a.grabbed_by and any("exile anyone" in x.lower() and "never" not in x.lower() for x in a.instinct):
        return a.grabbed_by.most_common(1)[0][0]
    return "none"


def mock_note(a):
    return "No preparation uses the same action twice." if a.energy > 0 else "Try everything."


# ---- CLI --------------------------------------------------------------------
async def main_async(a):
    os.makedirs(a.out, exist_ok=True)
    arms = ARM_ORDER if a.arms == "all" else [x.strip().upper() for x in a.arms.split(",")]
    seeds = [int(x) for x in a.seeds.split(",")]
    if a.pilot:
        a.N, a.G = 8, 2
    llm = LLM(model=a.model, budget_usd=a.budget, max_concurrency=a.concurrency, mock=a.mock,
              thinking=None if a.thinking == "on" else {"type": "disabled"})
    usage_path = os.path.join(a.out, "usage.json")

    def log(line):
        u = llm.usage()
        print(f"{line} | total calls={u['calls']} ${u['cost_usd']:.2f}", flush=True)
        json.dump(u, open(usage_path, "w"))

    runs = [Run(llm, arm, s, a.N, a.G, a.rounds, a.out, log) for arm in arms for s in seeds]
    res = await asyncio.gather(*[r.run() for r in runs])
    json.dump(llm.usage(), open(usage_path, "w"))
    print(f"Finished: {dict(Counter(res))} | usage: {json.dumps(llm.usage())}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="all")
    ap.add_argument("--seeds", default="1,2,3,4,5")
    ap.add_argument("--N", type=int, default=16)
    ap.add_argument("--G", type=int, default=12)
    ap.add_argument("--rounds", type=int, default=W.ROUNDS)
    ap.add_argument("--model", default="claude-haiku-5-5")
    ap.add_argument("--budget", type=float, default=32.0)
    ap.add_argument("--concurrency", type=int, default=64)
    ap.add_argument("--thinking", choices=["on", "off"], default="off")
    ap.add_argument("--out", default="runs2")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--pilot", action="store_true")
    asyncio.run(main_async(ap.parse_args()))
