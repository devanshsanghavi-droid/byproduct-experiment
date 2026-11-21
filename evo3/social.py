"""Version 4 simulation: social arms with changing weights. Design and rules: SPEC4.md.

A population of N agents shares one adapter. Each agent carries a repertoire of task families
(inherited, not in the weights). A life is `rounds` one-question tasks drawn from the repertoire;
the j-th attempt at one family pays max(1, 5 - j) if right and -1 otherwise. Parents are the top
`parents` by energy. The adapter is then trained on parents' task replies that were themselves
correct (SOLO-W: all parent replies), from the last `window` generations.

Energy is the only thing selection sees. Reserved and held families never appear here.

The same code runs the null model: pass a TableLLM, which answers from a fixed score table and
never learns.
"""
import math
import random
import time
from collections import Counter

from evo3 import breadth as B
from evo3 import wide

ARMS = {
    "DRIFT":   dict(pair=False, grab=False, vote=False, swap=False, select=False, whole=False),
    "SOLO":    dict(pair=False, grab=False, vote=False, swap=False, select=True, whole=False),
    "SOLO-W":  dict(pair=False, grab=False, vote=False, swap=False, select=True, whole=True),
    "COOP":    dict(pair=True, grab=False, vote=False, swap=False, select=True, whole=False),
    "ALPHA":   dict(pair=True, grab=True, vote=False, swap=False, select=True, whole=False),
    "DOMEST":  dict(pair=True, grab=True, vote=True, swap=False, select=True, whole=False),
    "CULTURE": dict(pair=False, grab=False, vote=False, swap=True, select=True, whole=False),
    "MEDLEY":  dict(pair=True, grab=True, vote=True, swap=True, select=True, whole=False),
}
CONTROL = {"SOLO": "DRIFT", "SOLO-W": "SOLO", "COOP": "SOLO", "ALPHA": "COOP", "DOMEST": "ALPHA",
           "CULTURE": "SOLO", "MEDLEY": "SOLO"}
THEORIES = {"COOP": "Theory 2, cooperation", "DOMEST": "Theory 3, self-domestication", "CULTURE": "Theory 4, cultural brain"}
DEFAULTS = {"G": 8, "N": 32, "parents": 8, "rounds": 24, "start": 2, "cap": 12, "gain": 0.3, "loss": 0.3,
            "swap": 0.5, "iters": 150, "lr": 2e-4, "batch_size": 8, "temp": 0.7, "window": 3,
            "vote_min": 4, "vote_bar": 2, "test_at": [4, 8], "founder": 1}
TRAINABLE = sorted(set(B.POOL) - set(B.reserved()))

KEEP_SYS = "You forage with a partner. Your own energy decides whether you raise the next generation."
VOTE_SYS = "You live in a group of foragers. The group can bar a forager from raising the next generation."


class TableLLM:
    """Null model: right with the probability in `table`, fixed keep and vote rates, no learning."""

    def __init__(self, table, seed, keep_rate=0.5, vote_rate=0.5):
        self.table, self.rng, self.keep_rate, self.vote_rate, self.temp = table, random.Random(seed), keep_rate, vote_rate, 0.0


def effective(counts) -> float:
    """Effective number of families: exponential of the Shannon entropy of the shares."""
    tot = sum(counts.values())
    if tot <= 0:
        return 0.0
    return math.exp(-sum(v / tot * math.log(v / tot) for v in counts.values() if v > 0))


def name_of(i) -> str:
    return f"F{i + 1:02d}"


def fresh(arm, seed, P) -> dict:
    r0 = random.Random(seed * 7919 + 1)        # generation 1 repertoires: the same in every arm of a seed
    if P.get("founder"):                       # founder population: every agent starts with the same families
        first = sorted(r0.sample(TRAINABLE, P["start"]))
        reps = [list(first) for _ in range(P["N"])]
    else:
        reps = [sorted(r0.sample(TRAINABLE, P["start"])) for _ in range(P["N"])]
    return {"out": {"arm": arm, "seed": seed, "config": dict(P), "generations": [], "tests": {}, "complete": False},
            "reps": reps, "copied": [set() for _ in range(P["N"])], "buffer": [], "draws": {}, "late": {},
            "rng": random.Random(1000 * seed + 17).getstate()}


# ---- model calls ---------------------------------------------------------------------------------
async def answer(llm, jobs, seed, g) -> list:
    """One task reply per job (unit, round, family, member). Both members of a pair get the same item."""
    if getattr(llm, "table", None) is not None:
        r, t = llm.rng, llm.table
        return [{"right": r.random() < t[fam], "valid": True, "ex": None} for _, _, fam, _ in jobs]
    items, mrng = {}, random.Random(seed * 31 + g)

    async def one(u, k, fam, m):
        it = items.get((u, k))
        if it is None:
            it = items[(u, k)] = wide.items_for(fam, random.Random(f"v4-{seed}-{g}-{u}-{k}"), 1)[0]
        log = []
        d = await llm.ask_json(wide.system_for(fam, it["kind"]), it["user"],
                               mock_fn=lambda: {"answer": wide.random_answer(it, mrng)}, max_tokens=32, log=log)
        right, valid = wide.check(it, d)
        return {"right": bool(right), "valid": bool(valid), "ex": log[0]}

    return await llm.run([one(*j) for j in jobs])


async def keep_or_split(llm, harvests, seed, g) -> list:
    """One decision per pair holder: (keep all, valid, example). `harvests` is a list of (pair index, energy)."""
    if getattr(llm, "table", None) is not None:
        return [(llm.rng.random() < llm.keep_rate, True, None) for _ in harvests]
    mrng = random.Random(seed * 37 + g)

    async def one(u, e):
        opts = ['{"take": "all"}', '{"take": "half"}']
        if random.Random(f"v4k-{seed}-{g}-{u}").random() < 0.5:     # option order varies, so position cannot decide
            opts.reverse()
        user = (f"You and your partner gathered {e} energy together. You are holding all of it. "
                f"You may keep all of it or give your partner half.\nReply with only JSON: {opts[0]} or {opts[1]}")
        log = []
        d = await llm.ask_json(KEEP_SYS, user, mock_fn=lambda: {"take": mrng.choice(["all", "half"])}, max_tokens=16, log=log)
        text = str(d.get("take", d.get("_raw", ""))).lower()
        has_all, has_half = "all" in text, "half" in text
        return (has_all and not has_half, has_all != has_half, log[0])

    return await llm.run([one(u, e) for u, e in harvests])


async def vote(llm, keepers, n, seed, g) -> list:
    """One vote per agent: (index of the keeper named or None, example or None)."""
    if not keepers:
        return [(None, None)] * n
    if getattr(llm, "table", None) is not None:
        out = []
        for i in range(n):
            pool = [k for k in keepers if k != i]
            out.append((llm.rng.choice(pool) if pool and llm.rng.random() < llm.vote_rate else None, None))
        return out
    mrng = random.Random(seed * 41 + g)

    async def one(i):
        pool = [k for k in keepers if k != i]
        if not pool:
            return (None, None)
        random.Random(f"v4v-{seed}-{g}-{i}").shuffle(pool)
        names = [name_of(k) for k in pool]
        user = (f"These foragers kept a whole shared harvest for themselves this season: {', '.join(names)}.\n"
                'Name one of them to bar, or name nobody.\nReply with only JSON: {"bar": "<name>"} or {"bar": "none"}')
        log = []
        d = await llm.ask_json(VOTE_SYS, user, mock_fn=lambda: {"bar": mrng.choice(names + ["none"])}, max_tokens=16, log=log)
        text = str(d.get("bar", d.get("_raw", "")))
        found = [k for k in pool if name_of(k) in text]
        return (found[0] if len(found) == 1 else None, log[0])

    return await llm.run([one(i) for i in range(n)])


# ---- one generation ------------------------------------------------------------------------------
async def generation(llm, state, g) -> dict:
    out = state["out"]
    arm, seed, P = out["arm"], out["seed"], out["config"]
    F, N, reps, copied = ARMS[arm], P["N"], state["reps"], state["copied"]
    table_mode = getattr(llm, "table", None) is not None
    rng = random.Random()
    rng.setstate(state["rng"])
    t0 = time.time()
    llm.temp = P["temp"]

    # who forages with whom, and which family each round poses
    if F["pair"]:
        order = list(range(N))
        random.Random(seed * 100003 + g).shuffle(order)      # same pairing in every pair arm of a seed
        units = [(order[i], order[i + 1]) for i in range(0, N, 2)]
    else:
        units = [(i,) for i in range(N)]
    plans, jobs = [], []
    for u, members in enumerate(units):
        fams = sorted({f for m in members for f in reps[m]})
        random.Random(seed * 1000003 + g * 1009 + u).shuffle(fams)
        plans.append([fams[k % len(fams)] for k in range(P["rounds"])])
        jobs += [(u, k, fam, m) for k, fam in enumerate(plans[-1]) for m in members]
    replies = await answer(llm, jobs, seed, g)

    # pay: the j-th attempt at a family in one life pays max(1, 5 - j); a pair is paid if either is right
    harvest, idx = [], 0
    for u, members in enumerate(units):
        seen, e = {}, 0
        for fam in plans[u]:
            j = seen[fam] = seen.get(fam, 0) + 1
            rs = replies[idx: idx + len(members)]
            idx += len(members)
            e += max(1, 5 - j) if any(x["right"] for x in rs) else -1
        harvest.append(e)
    energy, social, keepers, keep_valid = [0.0] * N, [], [], []
    if not F["pair"]:
        for u, (i,) in enumerate(units):
            energy[i] = harvest[u]
    else:
        holders = [members[random.Random(seed * 100019 + g * 1013 + u).randrange(2)] for u, members in enumerate(units)]
        # the holder is asked only when there is something to keep; a zero or negative harvest is shared
        asked = [u for u in range(len(units)) if harvest[u] > 0] if F["grab"] else []
        decisions = dict(zip(asked, await keep_or_split(llm, [(u, harvest[u]) for u in asked], seed, g))) if asked else {}
        for u, members in enumerate(units):
            keep, valid, ex = decisions.get(u, (False, True, None))
            h = holders[u]
            other = members[0] if members[1] == h else members[1]
            if keep:
                energy[h], energy[other] = harvest[u], 0.0
                keepers.append(h)
            else:
                energy[h] = energy[other] = harvest[u] / 2
            if u in decisions:
                keep_valid.append(valid)
                social.append((h, ex))
    barred, votes_cast = [], 0
    if F["vote"]:
        ballots = await vote(llm, keepers, N, seed, g)
        tally = Counter(k for k, _ in ballots if k is not None)
        votes_cast = sum(tally.values())
        ranked = sorted(tally, key=lambda k: (-tally[k], rng.random()))
        barred = [k for k in ranked if tally[k] >= P["vote_min"]][:P["vote_bar"]]
        social += [(i, ex) for i, (_, ex) in enumerate(ballots) if ex is not None]

    # parents
    if F["select"]:
        chosen = sorted((i for i in range(N) if i not in barred), key=lambda i: (-energy[i], rng.random()))[:P["parents"]]
    else:
        chosen = rng.sample(range(N), P["parents"])
    cset = set(chosen)

    # what the adapter is trained on: parents' replies that were themselves right (SOLO-W: all of them)
    added = [(fam, r["ex"]) for (_, _, fam, m), r in zip(jobs, replies) if m in cset and (F["whole"] or r["right"])]
    added += [(None, ex) for i, ex in social if i in cset]
    state["buffer"] = (state["buffer"] + [(g, added)])[-P["window"]:]
    entries = [(gg, fam, ex) for gg, part in state["buffer"] for fam, ex in part]
    losses = []
    if entries:
        w = P["iters"] * P["batch_size"] / len(entries)      # draws cycle through the buffer, so each entry gets this many
        for gg, fam, _ in entries:
            if fam is not None:
                state["draws"][fam] = state["draws"].get(fam, 0.0) + w
                if gg >= P["G"] - 1:
                    state["late"][fam] = state["late"].get(fam, 0.0) + w
        if not table_mode:
            losses = llm.train([ex for _, _, ex in entries], P["iters"], P["lr"], P["batch_size"], seed=seed * 100 + g)

    task = [r for r in replies]
    gen = {"g": g, "energy": round(sum(energy) / N, 3),
           "parent_energy": round(sum(energy[i] for i in chosen) / len(chosen), 3),
           "acc": round(sum(r["right"] for r in task) / len(task), 4),
           "invalid": round(1 - sum(r["valid"] for r in task) / len(task), 4),
           "rep_size": round(sum(len(r) for r in reps) / N, 3),
           "distinct": len({f for r in reps for f in r}),
           "practice_eff": round(effective(Counter(fam for _, _, fam, _ in jobs)), 3),
           "parent_fams": len({f for i in chosen for f in reps[i]}),
           "added": sum(1 for fam, _ in added if fam is not None),
           "added_fams": dict(Counter(fam for fam, _ in added if fam is not None)),
           "examples": len(entries),
           "keep_rate": (round(len(keepers) / len(keep_valid), 4) if keep_valid else None) if F["grab"] else None,
           "keep_asked": len(keep_valid) if F["grab"] else None,
           "keep_invalid": round(1 - sum(keep_valid) / len(keep_valid), 4) if keep_valid else None,
           "votes": votes_cast, "barred": len(barred),
           "copied_parents": sum(1 for i in chosen if copied[i] & set(reps[i])),
           "loss": losses, "seconds": round(time.time() - t0, 1)}

    # children: copy the parent's repertoire, then gain, lose and (culture arms) swap
    best = sorted(chosen, key=lambda i: -energy[i])
    new_reps, new_copied = [], []
    for p in chosen:
        for _ in range(N // len(chosen)):
            rep, cp = list(reps[p]), set(copied[p])
            if rng.random() < P["gain"] and len(rep) < P["cap"]:
                rep.append(rng.choice([f for f in TRAINABLE if f not in rep]))
            if rng.random() < P["loss"] and len(rep) > 1:
                drop = rng.choice(rep)
                rep.remove(drop)
                cp.discard(drop)
            if F["swap"] and rng.random() < P["swap"]:
                model = next((m for m in best if m != p), None)
                cand = [f for f in reps[model] if f not in rep] if model is not None else []
                if cand:
                    new, old = rng.choice(cand), rng.choice(rep)
                    rep.remove(old)
                    cp.discard(old)
                    rep.append(new)
                    cp.add(new)
            new_reps.append(sorted(rep))
            new_copied.append(cp & set(rep))
    state["reps"], state["copied"] = new_reps, new_copied
    state["rng"] = rng.getstate()
    return gen


def scores_of(b) -> dict:
    """Battery scores plus the in-world skill: mean over the 49 trainable families."""
    s = dict(b["scores"])
    s["inworld"] = round(sum(s["pool_" + f] for f in TRAINABLE) / len(TRAINABLE), 4)
    return s


async def run_sim(llm, state, on_gen=None, tests=True, stop=None) -> dict:
    """Advance `state` (from fresh() or a checkpoint) to the end. Calls on_gen(state) after each generation
    and after each test, so a restarted worker repeats at most one generation or one test."""
    out = state["out"]
    P = out["config"]
    table_mode = getattr(llm, "table", None) is not None
    g = len(out["generations"])
    while True:
        if g >= 1 and tests and not table_mode and g in P["test_at"] and str(g) not in out["tests"]:
            b = await B.battery(llm, [], None, far=False, public=False, near=B.reserved())
            out["tests"][str(g)] = scores_of(b)
            if g == P["G"]:
                out["items"] = b["items"]
            if on_gen:
                on_gen(state)
        if g >= P["G"]:
            break
        if stop and stop():
            out["stopped"] = "budget"
            return out
        g += 1
        out["generations"].append(await generation(llm, state, g))
        out["breadth"] = round(effective(state["draws"]), 4)
        out["late_breadth"] = round(effective(state["late"]), 4)
        if on_gen:
            on_gen(state)
    out["draws"] = {k: round(v, 2) for k, v in sorted(state["draws"].items())}
    out["complete"] = True
    if not table_mode:
        out["usage"] = llm.usage()
    if on_gen:
        on_gen(state)
    return out
