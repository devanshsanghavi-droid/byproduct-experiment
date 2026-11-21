"""Held-out ability surface: three tasks that are never used for selection.

    python3 -m evo.heldout --model claude-haiku-5-5 --budget 8 --reps 24 \
        --out runs/heldout.json [--mock]

Ability is measured per gene level, then looked up for each population:

    H1 code-breaking   depends on MEM         5 levels x reps
    H3 rule induction  depends on MEM         5 levels x reps
    H2 stranger relay  depends on MEM, TALK   25 cells x reps/2

Capacity limits are enforced in code through evo/genome.py: an agent is only ever
shown the records its MEM allows, and its messages are cut to its TALK words. The
model is never asked to pretend to be limited.

The same hidden code / rule / target is used for rep i at every gene level, so
levels are compared on identical task instances.
"""
import argparse
import asyncio
import json
import os
import random
import re
import statistics

from evo import genome as G
from evo.llm import LLM, BudgetExceeded

H1_TURNS, H1_LEN, H1_SYMBOLS = 8, 3, 4
H2_ROUNDS, H2_PARTS, H2_SLOTS = 4, 4, 3
H3_PROBES, H3_TESTS, H3_LO, H3_HI = 6, 8, 1, 12
PARTNER = {"MEM": 2, "TALK": 2, "LORE": 0, "AGG": 0}
LEVELS = range(G.MAX_LEVEL + 1)


def agent(mem: int, talk: int = 0) -> dict:
    return {"MEM": mem, "TALK": talk, "LORE": 0, "AGG": 0}


def nonsense_words(rng: random.Random, n: int) -> list:
    """Unique made-up words, so the model's prior knowledge has nothing to hold on to."""
    cons, vows = "bdfgklmnprstvz", "aeiou"
    words = set()
    while len(words) < n:
        words.add(rng.choice(cons) + rng.choice(vows) + rng.choice(cons) + rng.choice(vows))
    words = sorted(words)
    rng.shuffle(words)
    return words


def tokens(text) -> list:
    return re.findall(r"[a-z]+", str(text).lower())


class _Stop(Exception):
    """Raised inside a rep once the run is ending (budget reached or fatal error)."""


class Runner:
    def __init__(self, llm: LLM, seed: int, concurrency: int):
        self.llm = llm
        self.seed = seed
        # Bounds how many reps are in flight, so few calls sit queued past the budget check.
        self.gate = asyncio.Semaphore(max(4, concurrency * 2))
        self.budget_hit = False
        self.fatal = None
        self.invalid = {"H1": 0, "H2": 0, "H3": 0}  # replies replaced by a random default

    async def ask(self, system: str, user: str, mock_fn) -> dict:
        if self.budget_hit or self.fatal is not None:
            raise _Stop()
        return await self.llm.ask_json(system, user, mock_fn=mock_fn)

    async def guard(self, coro):
        """Run one rep. Returns None if the run is ending; records the first fatal error."""
        try:
            return await coro
        except BudgetExceeded:
            self.budget_hit = True
        except _Stop:
            pass
        except Exception as e:  # noqa: BLE001 - re-raised by main() after partial save
            if self.fatal is None:
                self.fatal = e
        return None

    def rng(self, *parts) -> random.Random:
        return random.Random("|".join(str(p) for p in (self.seed,) + parts))

    # ------------------------------------------------------------------ H1
    async def h1_rep(self, mem: int, rep: int):
        """Returns (score, solved). Score = (9 - turn solved) / 8, or 0 if unsolved."""
        task = self.rng("H1", rep)
        symbols = nonsense_words(task, H1_SYMBOLS)
        code = [task.choice(symbols) for _ in range(H1_LEN)]
        rng = self.rng("H1", mem, rep)
        genome = agent(mem)
        system = (
            "You are playing a code-guessing game.\n"
            f"A hidden code has {H1_LEN} positions. Each position holds one of these "
            f"{H1_SYMBOLS} symbols: {', '.join(symbols)}. A symbol may appear more than once.\n"
            f"You have {H1_TURNS} turns. Each turn you guess all {H1_LEN} positions. After each "
            "guess you are told, for each position, whether your symbol there was right or wrong.\n"
            "The goal is to find the code in as few turns as you can.\n"
            'Reply with only a JSON object: {"guess": ["<symbol>", "<symbol>", "<symbol>"]}'
        )
        records = []
        async with self.gate:
            for turn in range(1, H1_TURNS + 1):
                vis = G.visible_records(records, genome)
                if vis:
                    lines = "\n".join(
                        f"- turn {r['turn']}: {', '.join(r['guess'])} -> "
                        + ", ".join(f"position {i + 1} {'right' if ok else 'wrong'}"
                                    for i, ok in enumerate(r["fb"]))
                        for r in vis)
                    history = "Earlier guesses on record:\n" + lines
                else:
                    history = "Earlier guesses on record: none."
                user = f"Turn {turn} of {H1_TURNS}.\n{history}"
                data = await self.ask(system, user,
                                      lambda vis=vis: {"guess": h1_mock(vis, symbols, rng)})
                raw = data.get("guess")
                guess = [str(x).strip().lower() for x in raw] if isinstance(raw, list) else []
                if len(guess) != H1_LEN or any(g not in symbols for g in guess):
                    self.invalid["H1"] += 1
                    guess = [guess[i] if i < len(guess) and guess[i] in symbols
                             else rng.choice(symbols) for i in range(H1_LEN)]
                fb = [g == c for g, c in zip(guess, code)]
                records.append({"turn": turn, "guess": guess, "fb": fb})
                if all(fb):
                    return (H1_TURNS + 1 - turn) / H1_TURNS, 1
        return 0.0, 0

    # ------------------------------------------------------------------ H3
    async def h3_rep(self, mem: int, rep: int):
        """Returns accuracy on 8 unseen triples (4 fit the rule, 4 do not)."""
        task = self.rng("H3", rep)
        _, rule = task.choice(RULES)
        seed_example = sample_triple(task, rule, True)
        tests = [sample_triple(task, rule, True) for _ in range(H3_TESTS // 2)]
        tests += [sample_triple(task, rule, False) for _ in range(H3_TESTS - len(tests))]
        task.shuffle(tests)
        truth = [rule(t) for t in tests]
        rng = self.rng("H3", mem, rep)
        genome = agent(mem)
        system = (
            f"There is a hidden rule about triples of whole numbers from {H3_LO} to {H3_HI}. "
            "Some triples fit the rule and others do not.\n"
            f"First you get {H3_PROBES} probes: on each probe you name any triple and are told "
            "whether it fits the rule.\n"
            f"Then you are shown {H3_TESTS} new triples and must say for each one whether it "
            "fits the rule. You score one point for each correct answer."
        )

        def history(vis):
            if not vis:
                return "Records on file: none."
            return "Records on file:\n" + "\n".join(
                f"- {r['triple'][0]}, {r['triple'][1]}, {r['triple'][2]} -> "
                f"{'fits' if r['fits'] else 'does not fit'}" for r in vis)

        records = [{"triple": seed_example, "fits": True}]
        async with self.gate:
            for probe in range(1, H3_PROBES + 1):
                vis = G.visible_records(records, genome)
                user = (f"Probe {probe} of {H3_PROBES}.\n{history(vis)}\n"
                        'Reply with only a JSON object: {"triple": [<n>, <n>, <n>]}')
                data = await self.ask(system, user, lambda: {"triple": random_triple(rng)})
                triple = as_triple(data.get("triple"))
                if triple is None:
                    self.invalid["H3"] += 1
                    triple = random_triple(rng)
                records.append({"triple": triple, "fits": bool(rule(triple))})
            vis = G.visible_records(records, genome)
            listing = "\n".join(f"{i + 1}) {t[0]}, {t[1]}, {t[2]}" for i, t in enumerate(tests))
            user = (f"The probes are over.\n{history(vis)}\n"
                    f"The {H3_TESTS} new triples:\n{listing}\n"
                    'Reply with only a JSON object: {"answers": [<true or false>, ...]} with '
                    f"{H3_TESTS} entries in order, true meaning the triple fits the rule.")
            data = await self.ask(system, user,
                                  lambda vis=vis: {"answers": h3_mock(vis, tests, rng)})
        raw = data.get("answers")
        if not isinstance(raw, list) or not raw:
            self.invalid["H3"] += 1
            raw = [rng.random() < 0.5 for _ in tests]
        answers = [as_bool(x) for x in raw[:H3_TESTS]]
        return sum(a is not None and a == t for a, t in zip(answers, truth)) / H3_TESTS

    # ------------------------------------------------------------------ H2
    def h2_task(self, rep: int):
        task = self.rng("H2", rep)
        words = nonsense_words(task, H2_PARTS + H2_PARTS * H2_SLOTS * 2)
        names, pool = words[:H2_PARTS], words[H2_PARTS:]
        options = {}
        for name in names:
            options[name] = [(pool.pop(), pool.pop()) for _ in range(H2_SLOTS)]
        target = {name: [task.choice(pair) for pair in options[name]] for name in names}
        return {"names": names, "options": options, "target": target}

    @staticmethod
    def h2_sheet(task) -> str:
        lines = "\n".join(
            f"- {name}: " + " ".join(f"({a}|{b})" for a, b in task["options"][name])
            for name in task["names"])
        return (
            f"A target has {H2_PARTS} parts: {', '.join(task['names'])}. Each part is "
            f"{H2_SLOTS} words. In each word position there are two possible words, "
            f"shown as (x|y):\n{lines}\n")

    async def h2_send(self, task, genome, rng) -> list:
        """The sender's messages as delivered (already cut to its word limit)."""
        limit = G.talk_words(genome)
        if limit <= 0:
            return [""] * H2_ROUNDS
        system = (
            f"You and a partner are playing a relay game over {H2_ROUNDS} rounds.\n"
            + self.h2_sheet(task) +
            "You can see the target. Your partner cannot. Each round you send your partner "
            f"one message. Your messages are cut off after {limit} words.\n"
            "After the last round your partner must state the whole target. The pair scores "
            "one point for each part stated exactly right.\n"
            'Reply with only a JSON object: {"message": "<your message>"}'
        )
        shown = "\n".join(f"- {n}: {' '.join(task['target'][n])}" for n in task["names"])
        records = []
        for rnd in range(1, H2_ROUNDS + 1):
            vis = G.visible_records(records, genome)
            if vis:
                sent = "Your earlier messages on record, as delivered:\n" + "\n".join(
                    f"- round {r['round']}: {r['text'] or '(empty)'}" for r in vis)
            else:
                sent = "Your earlier messages on record: none."
            user = f"Round {rnd} of {H2_ROUNDS}.\nThe target:\n{shown}\n{sent}"
            data = await self.ask(
                system, user,
                lambda vis=vis, rnd=rnd: {"message": h2_mock_send(task, vis, rnd, limit)})
            if "message" not in data:
                self.invalid["H2"] += 1
            records.append({"round": rnd, "text": G.clip_words(data.get("message", ""), limit)})
        return [r["text"] for r in records]

    async def h2_receive(self, task, messages, genome, rng) -> float:
        """Fraction of parts the receiver states exactly right."""
        system = (
            f"You and a partner are playing a relay game over {H2_ROUNDS} rounds.\n"
            + self.h2_sheet(task) +
            "Your partner can see the target. You cannot. Each round your partner sends you "
            "one message.\n"
            "After the last round you must state the whole target. The pair scores one point "
            "for each part stated exactly right.\n"
            "Reply with only a JSON object: {"
            + ", ".join(f'"{n}": "<{H2_SLOTS} words>"' for n in task["names"]) + "}"
        )
        earlier = [{"round": i + 1, "text": m} for i, m in enumerate(messages[:-1])]
        vis = G.visible_records(earlier, genome)
        if vis:
            on_record = "Earlier messages on record:\n" + "\n".join(
                f"- round {r['round']}: {r['text'] or '(empty)'}" for r in vis)
        else:
            on_record = "Earlier messages on record: none."
        user = (f"All {H2_ROUNDS} rounds are done.\n"
                f"Message just received (round {H2_ROUNDS}): {messages[-1] or '(empty)'}\n"
                f"{on_record}\nState the target.")
        seen = [r["text"] for r in vis] + [messages[-1]]
        data = await self.ask(system, user, lambda: h2_mock_receive(task, seen, rng))
        if not data:
            self.invalid["H2"] += 1
            data = h2_mock_receive(task, [], rng)  # uniform random guess
        lowered = {str(k).strip().lower(): v for k, v in data.items()}
        right = 0
        for name in task["names"]:
            value = lowered.get(name, "")
            if isinstance(value, list):
                value = " ".join(str(x) for x in value)
            right += tokens(value) == task["target"][name]
        return right / H2_PARTS

    async def h2_partner_sends(self, rep: int):
        task = self.h2_task(rep)
        async with self.gate:
            return await self.h2_send(task, PARTNER, self.rng("H2", "partner-send", rep))

    async def h2_as_receiver(self, mem: int, rep: int, partner_task):
        """Tested agent receives from the fixed partner. Its TALK plays no part here."""
        messages = await partner_task
        if messages is None:
            raise _Stop()
        task = self.h2_task(rep)
        async with self.gate:
            return await self.h2_receive(task, messages, agent(mem),
                                         self.rng("H2", "recv", mem, rep))

    async def h2_as_sender(self, mem: int, talk: int, rep: int):
        """Tested agent sends to the fixed partner."""
        task = self.h2_task(rep)
        rng = self.rng("H2", "send", mem, talk, rep)
        async with self.gate:
            messages = await self.h2_send(task, agent(mem, talk), rng)
            return await self.h2_receive(task, messages, PARTNER, rng)


# ---------------------------------------------------------------------- H3 rules
def _rules():
    r = [
        ("strictly ascending", lambda t: t[0] < t[1] < t[2]),
        ("strictly descending", lambda t: t[0] > t[1] > t[2]),
        ("all even", lambda t: all(x % 2 == 0 for x in t)),
        ("all odd", lambda t: all(x % 2 == 1 for x in t)),
        ("middle is the largest", lambda t: t[1] > t[0] and t[1] > t[2]),
        ("middle is the smallest", lambda t: t[1] < t[0] and t[1] < t[2]),
        ("contains a multiple of 3", lambda t: any(x % 3 == 0 for x in t)),
        ("contains a repeated number", lambda t: len(set(t)) < 3),
        ("all different", lambda t: len(set(t)) == 3),
        ("first is less than last", lambda t: t[0] < t[2]),
        ("first is greater than last", lambda t: t[0] > t[2]),
        ("largest minus smallest is at most 4", lambda t: max(t) - min(t) <= 4),
        ("largest minus smallest is at least 7", lambda t: max(t) - min(t) >= 7),
        ("every number is at least 5", lambda t: min(t) >= 5),
        ("every number is at most 8", lambda t: max(t) <= 8),
        ("first is even", lambda t: t[0] % 2 == 0),
        ("last is odd", lambda t: t[2] % 2 == 1),
    ]
    for k in (16, 19, 22):
        r.append((f"sum is above {k}", lambda t, k=k: sum(t) > k))
    for k in (17, 20, 23):
        r.append((f"sum is below {k}", lambda t, k=k: sum(t) < k))
    return r


RULES = _rules()


def random_triple(rng):
    return [rng.randint(H3_LO, H3_HI) for _ in range(3)]


def sample_triple(rng, rule, want: bool):
    while True:
        t = random_triple(rng)
        if bool(rule(t)) == want:
            return t


def as_triple(raw):
    try:
        t = [int(x) for x in raw]
    except (TypeError, ValueError):
        return None
    return t if len(t) == 3 and all(H3_LO <= x <= H3_HI for x in t) else None


def as_bool(x):
    if isinstance(x, bool):
        return x
    s = str(x).strip().lower()
    if s in ("true", "yes", "fits", "1"):
        return True
    if s in ("false", "no", "0"):
        return False
    return None


# ---------------------------------------------------------------------- mock policies
# Each mock uses only what the prompt for that call shows, so mock scores rise with
# MEM and TALK for the same reason real scores can.
def h1_mock(vis, symbols, rng):
    guess = []
    for pos in range(H1_LEN):
        right = [r["guess"][pos] for r in vis if r["fb"][pos]]
        wrong = {r["guess"][pos] for r in vis if not r["fb"][pos]}
        if right:
            guess.append(right[-1])
        else:
            guess.append(rng.choice([s for s in symbols if s not in wrong] or symbols))
    return guess


def h3_mock(vis, tests, rng):
    alive = [f for _, f in RULES if all(bool(f(r["triple"])) == r["fits"] for r in vis)]
    alive = alive or [f for _, f in RULES]
    out = []
    for t in tests:
        yes = sum(bool(f(t)) for f in alive)
        out.append(rng.random() < 0.5 if yes * 2 == len(alive) else yes * 2 > len(alive))
    return out


def h2_mock_send(task, vis, rnd, limit):
    per_msg = limit // H2_SLOTS
    seen = set()
    for r in vis:
        seen.update(tokens(r["text"]))
    names = task["names"]
    start = ((rnd - 1) * per_msg) % H2_PARTS
    order = [names[(start + i) % H2_PARTS] for i in range(H2_PARTS)]
    unsent = [n for n in order if not all(w in seen for w in task["target"][n])]
    chosen = (unsent or order)[:per_msg]
    return " ".join(w for n in chosen for w in task["target"][n])


def h2_mock_receive(task, seen_messages, rng):
    heard = {}
    for text in seen_messages:          # later messages override earlier ones
        for w in tokens(text):
            heard[w] = True
    answer = {}
    for name in task["names"]:
        words = []
        for a, b in task["options"][name]:
            if a in heard and b not in heard:
                words.append(a)
            elif b in heard and a not in heard:
                words.append(b)
            else:
                words.append(rng.choice((a, b)))
        answer[name] = " ".join(words)
    return answer


# ---------------------------------------------------------------------- aggregation
def cell(scores, **extra) -> dict:
    scores = [s for s in scores if s is not None]
    if not scores:
        return {"mean": None, "sd": None, "n": 0, **extra}
    sd = statistics.stdev(scores) if len(scores) > 1 else 0.0
    return {"mean": round(statistics.fmean(scores), 4), "sd": round(sd, 4),
            "n": len(scores), **extra}


def mean_or_none(xs):
    xs = [x for x in xs if x is not None]
    return round(statistics.fmean(xs), 4) if xs else None


async def run(args) -> dict:
    llm = LLM(model=args.model, budget_usd=args.budget, max_concurrency=args.concurrency,
              mock=args.mock)
    rn = Runner(llm, args.seed, args.concurrency)
    reps, reps2 = args.reps, max(1, args.reps // 2)
    spawn = lambda coro: asyncio.ensure_future(rn.guard(coro))  # noqa: E731

    h1 = {m: [spawn(rn.h1_rep(m, r)) for r in range(reps)] for m in LEVELS}
    h3 = {m: [spawn(rn.h3_rep(m, r)) for r in range(reps)] for m in LEVELS}
    partner = [spawn(rn.h2_partner_sends(r)) for r in range(reps2)]
    recv = {m: [spawn(rn.h2_as_receiver(m, r, partner[r])) for r in range(reps2)]
            for m in LEVELS}
    send = {}
    for t in LEVELS:
        # With TALK 0 the sender makes no call, so its MEM is irrelevant: share one run.
        shared = [spawn(rn.h2_as_sender(0, 0, r)) for r in range(reps2)] if t == 0 else None
        for m in LEVELS:
            send[m, t] = shared or [spawn(rn.h2_as_sender(m, t, r)) for r in range(reps2)]

    pending = [x for group in (h1, h3, recv) for xs in group.values() for x in xs]
    pending += partner + [x for xs in send.values() for x in xs]
    await asyncio.gather(*pending)

    out = {"model": args.model, "mock": args.mock, "reps": reps, "seed": args.seed,
           "complete": not rn.budget_hit and rn.fatal is None,
           "tasks": {"H1": {"turns": H1_TURNS, "positions": H1_LEN, "symbols": H1_SYMBOLS,
                            "score": "(turns + 1 - turn solved) / turns, 0 if unsolved"},
                     "H2": {"rounds": H2_ROUNDS, "parts": H2_PARTS, "words_per_part": H2_SLOTS,
                            "partner": PARTNER, "reps_per_cell": reps2,
                            "score": "mean of sender-role and receiver-role fraction of parts right",
                            "chance": round(0.5 ** H2_SLOTS, 4)},
                     "H3": {"probes": H3_PROBES, "test_triples": H3_TESTS, "rules": len(RULES),
                            "score": "accuracy", "chance": 0.5}},
           "H1": {}, "H3": {}, "H2": {}}
    for m in LEVELS:
        res = [x.result() for x in h1[m]]
        out["H1"][str(m)] = cell([r[0] if r else None for r in res],
                                 solved_rate=mean_or_none([r[1] if r else None for r in res]))
        out["H3"][str(m)] = cell([x.result() for x in h3[m]])
        for t in LEVELS:
            s = [x.result() for x in send[m, t]]
            v = [x.result() for x in recv[m]]
            both = [(a + b) / 2 if a is not None and b is not None else None
                    for a, b in zip(s, v)]
            out["H2"][f"{m},{t}"] = cell(both, sender_role=mean_or_none(s),
                                         receiver_role=mean_or_none(v))
    out["invalid_replies"] = dict(rn.invalid)
    out["usage"] = llm.usage()
    out["_fatal"] = rn.fatal
    return out


def save(out: dict, path: str):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(out, f, indent=1)
    os.replace(tmp, path)


def main():
    p = argparse.ArgumentParser(description="Measure the held-out ability surface.")
    p.add_argument("--model", default="claude-haiku-5-5")
    p.add_argument("--budget", type=float, default=8.0)
    p.add_argument("--reps", type=int, default=24)
    p.add_argument("--out", default="runs/heldout.json")
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--mock", action="store_true")
    args = p.parse_args()

    out = asyncio.run(run(args))
    fatal = out.pop("_fatal")
    save(out, args.out)

    fmt = lambda c: "  -  " if c["mean"] is None else f"{c['mean']:.2f}"  # noqa: E731
    print("H1 by MEM:", "  ".join(fmt(out["H1"][str(m)]) for m in LEVELS))
    print("H3 by MEM:", "  ".join(fmt(out["H3"][str(m)]) for m in LEVELS))
    print("H2 rows MEM 0-4, columns TALK 0-4:")
    for m in LEVELS:
        print("   ", "  ".join(fmt(out["H2"][f"{m},{t}"]) for t in LEVELS))
    print("invalid replies:", out["invalid_replies"], "| usage:", out["usage"])
    print(f"saved {args.out}" + ("" if out["complete"] else "  (PARTIAL: run ended early)"))
    if fatal is not None:
        raise fatal


if __name__ == "__main__":
    main()
