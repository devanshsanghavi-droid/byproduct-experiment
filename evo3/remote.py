"""Code that runs inside the Modal worker. One call = one task (speed, headroom, ceiling, sim).

Spend guard: every task records its running cost in the shared board every 30 seconds. If the
total across all tasks reaches CAP, the worker stops itself. The cost is an upper estimate
(8 cores plus the full memory limit for every second), not Modal's own bill.
"""
import asyncio
import gc
import json
import os
import pickle
import random
import resource
import threading
import time

P_CPU, P_MEM = 0.0000131, 0.00000222   # USD per core-second, per GiB-second (modal.com/pricing, 2026-11-12)
CORES, MEM_REQ, MEM_LIM = 8, 6144, 12288
RATE = CORES * P_CPU + MEM_LIM / 1024 * P_MEM
P_L4 = 0.000306                         # USD per second, billed at the A10 rate (L4 is 0.000222) so the ledger is an upper bound
GPU_CORES, GPU_MEM = 4, 16384
RATE_GPU = P_L4 + GPU_CORES * P_CPU + GPU_MEM / 1024 * P_MEM
ON_GPU = False                          # set by run() for the worker this process is on
BASE_SPENT = 7.02                       # ledger total before the breadth experiment (first $10 limit)
RERUN_BASE = 9.68                       # ledger total before the near-transfer rerun
STRONG_BASE = 12.26                     # ledger total before the two strengthening experiments
S4_BASE = 20.15                         # ledger total before version 4 (social arms)
CAP = 31.15                             # own ledger stops here: $11.00 for version 4 (user approved an $11 limit, 2026-11-16)
GPU_MEM_BIG = 32768                     # the 3B model needs more host memory while it loads
RATE_GPU_BIG = P_L4 + GPU_CORES * P_CPU + GPU_MEM_BIG / 1024 * P_MEM
START_ALLOWANCE = 20                    # seconds added per task for worker start-up
VERSION = 16                            # shown in status, to confirm which code a worker ran


def spent(board) -> float:
    return sum(v["usd"] for k, v in board.items() if str(k).startswith("cost:"))


BBH_TASKS = ["boolean_expressions", "navigate", "object_counting", "tracking_shuffled_objects_three_objects",
             "logical_deduction_three_objects", "date_understanding", "web_of_lies", "temporal_sequences",
             "reasoning_about_colored_objects", "penguins_in_a_table", "multistep_arithmetic_two"]


def make_llm(model, seed=0):
    from evo3.torch_llm import TorchLLM
    llm = TorchLLM(model=model, threads=GPU_CORES if ON_GPU else CORES, device="cuda" if ON_GPU else "cpu", seed=seed)
    if "3B" in model:
        llm.token_budget = 6000        # smaller test batches only: 12 GB of weights leave less room on a 24 GB card
        llm.micro = 4                  # training step in two groups of 4, gradients summed (SPEC3.md, deviation for B)
    return llm


async def k_speed(name, p, tick, vol):
    from evo3 import world3 as W
    out = {"rate_usd_per_hour": round((RATE_GPU if ON_GPU else RATE) * 3600, 4), "models": []}
    for model in p["models"]:
        t = time.time()
        llm = make_llm(model)
        row = {"model": model, "load_s": round(time.time() - t, 1), "threads": llm.config["threads"]}
        llm.temp = 0.7
        logs = [[] for _ in range(p.get("lives", 16))]
        w = W.new_world(random.Random(1))
        t = time.time()
        res = await llm.run([W.life(llm.proxy(logs[i]), w, i) for i in range(len(logs))])
        dt = time.time() - t
        ex = [e for l in logs for e in l]
        row.update(lives_s=round(dt, 1), calls=len(ex), calls_per_s=round(len(ex) / dt, 2),
                   tokens_per_s=round((llm.prompt_tokens + llm.gen_tokens) / dt, 1),
                   prompt_tokens_per_call=round(llm.prompt_tokens / len(ex), 1),
                   gen_tokens_per_call=round(llm.gen_tokens / len(ex), 1),
                   energy=round(sum(r["energy"] for r in res) / len(res), 2),
                   invalid=round(sum(r["invalid"] for r in res) / len(ex), 3))
        t = time.time()
        losses = llm.train(ex, p.get("iters", 8), batch_size=8)
        row.update(train_s_per_iter=round((time.time() - t) / p.get("iters", 8), 2), losses=losses, sample=ex[:2])
        out["models"].append(row)
        tick({"note": f"{model} timed", "rows": [{k: v for k, v in r.items() if k != "sample"} for r in out["models"]]})
        del llm
        gc.collect()
    return out


async def k_headroom(name, p, tick, vol):
    from evo3.calibrate import calibrate
    return await calibrate(make_llm(p["model"]), n=p.get("n", 24), temps=p.get("temps", (0.0, 0.7)), tick=tick,
                           battery=p.get("battery", True))


async def k_ceiling(name, p, tick, vol):
    """Train directly on correct play, then test. Reference only: this breaks the thesis rule on purpose.
    Saves after every stage so a preempted worker resumes instead of starting over."""
    from evo3 import tests3, world3 as W
    llm = make_llm(p["model"])
    cfg, sizes = p.get("world") or W.DEFAULT, p.get("test_sizes")
    tcfg = p.get("test_world") or cfg      # the tests may use a wider world than the one trained on
    d = f"/runs/{name}"
    os.makedirs(d, exist_ok=True)
    logs = []
    for i in range(p.get("lives", 120)):
        await W.life(None, W.new_world(random.Random(300000 + i), cfg), 300000 + i, oracle=True, log=logs)
    if os.path.exists(f"{d}/ceil.pkl"):
        out = pickle.load(open(f"{d}/ceil.pkl", "rb"))
        if os.path.exists(f"{d}/adapter.pt"):
            llm.load_adapter(f"{d}/adapter.pt")
    else:
        out = {"model": llm.config, "world": cfg, "test_world": tcfg, "examples": len(logs), "stages": [], "trained": 0}

    def save():
        llm.save_adapter(f"{d}/adapter.pt")
        pickle.dump(out, open(f"{d}/ceil.pkl.tmp", "wb"))
        os.replace(f"{d}/ceil.pkl.tmp", f"{d}/ceil.pkl")
        vol.commit()

    if "samples" not in out:
        raw = [[] for _ in range(4)]
        llm.temp = 0.7
        await llm.run([W.life(llm.proxy(raw[i]), W.new_world(random.Random(77 + i), cfg), 77 + i) for i in range(4)])
        out["samples"] = [{"user": e["user"][-160:], "out": e["out"]} for l in raw for e in l[:9]]
        if p.get("before", True):
            b = await tests3.battery(llm, tcfg, sizes)
            out["stages"].append({"iters": 0, "scores": b["scores"], "items": b["items"],
                                  "replies": await tests3.transcripts(llm)})
            tick({"note": "before-training tests done", "scores": b["scores"]})
        save()
    for upto in p.get("stages", [150, 400]):
        if any(st["iters"] == upto for st in out["stages"]):
            continue
        losses = []
        while out["trained"] < upto:     # train in chunks of 50 steps, saving after each
            n = min(50, upto - out["trained"])
            losses += llm.train(logs, n, lr=p.get("lr", 1e-4), batch_size=p.get("batch_size", 8), seed=out["trained"])
            out["trained"] += n
            save()
            tick({"note": f"trained {out['trained']} of {upto} steps, loss {losses[-1]}"})
        b = await tests3.battery(llm, tcfg, sizes)
        out["stages"].append({"iters": upto, "loss": losses, "scores": b["scores"], "items": b["items"],
                              "replies": await tests3.transcripts(llm)})
        save()
        tick({"note": f"tested after {upto} steps", "scores": b["scores"]})
    out["usage"] = llm.usage()
    return out


async def k_sim(name, p, tick, vol):
    from evo3 import sim3, world3 as W
    llm = make_llm(p["model"])
    d = f"/runs/{name}"
    os.makedirs(d, exist_ok=True)
    P = dict(sim3.DEFAULTS, **p.get("P", {}))
    if os.path.exists(f"{d}/state.pkl"):
        state = pickle.load(open(f"{d}/state.pkl", "rb"))
        if os.path.exists(f"{d}/adapter.pt"):
            llm.load_adapter(f"{d}/adapter.pt")
    else:
        state = sim3.fresh(p["arm"], p["seed"], p.get("world") or W.DEFAULT, P)

    def on_gen(st):
        llm.save_adapter(f"{d}/adapter.pt")
        pickle.dump(st, open(f"{d}/state.pkl.tmp", "wb"))
        os.replace(f"{d}/state.pkl.tmp", f"{d}/state.pkl")
        vol.commit()
        gens = st["out"]["generations"]
        tick({"arm": p["arm"], "seed": p["seed"], "gen": len(gens), "G": P["G"],
              "energy": [g["energy"] for g in gens], "plants": [g["plants"] for g in gens],
              "tracks": [g["tracks"] for g in gens], "trade": [g["trade"] for g in gens],
              "invalid": [g["invalid"] for g in gens], "max_energy": gens[-1]["max_energy"] if gens else None,
              "note": "testing" if len(gens) == P["G"] and not st["out"]["complete"] else ""})

    out = await sim3.run_sim(llm, state, on_gen=on_gen, tests=p.get("tests", True), test_sizes=p.get("test_sizes"))
    out["model"] = llm.config
    return out


async def k_social(name, p, tick, vol):
    """Version 4: one arm, one seed (SPEC4.md). Saves after every generation and test, so a preempted worker resumes."""
    from evo3 import social as S
    llm = make_llm(p["model"], seed=p["seed"])
    d = f"/runs/{name}"
    os.makedirs(d, exist_ok=True)
    P = dict(S.DEFAULTS, **p.get("P", {}))
    if os.path.exists(f"{d}/state.pkl"):
        state = pickle.load(open(f"{d}/state.pkl", "rb"))
        if os.path.exists(f"{d}/adapter.pt"):
            llm.load_adapter(f"{d}/adapter.pt")
    else:
        state = S.fresh(p["arm"], p["seed"], P)

    def on_gen(st):
        llm.save_adapter(f"{d}/adapter.pt")
        pickle.dump(st, open(f"{d}/state.pkl.tmp", "wb"))
        os.replace(f"{d}/state.pkl.tmp", f"{d}/state.pkl")
        vol.commit()
        o = st["out"]
        gens, tests = o["generations"], o["tests"]
        sc = tests[str(max(int(k) for k in tests))] if tests else {}
        pending = len(gens) in P["test_at"] and str(len(gens)) not in tests
        tick({"arm": p["arm"], "seed": p["seed"], "gen": len(gens), "G": P["G"],
              "energy": [x["energy"] for x in gens], "acc": [x["acc"] for x in gens], "invalid": [x["invalid"] for x in gens],
              "rep": [x["rep_size"] for x in gens], "pfams": [x["parent_fams"] for x in gens],
              "keep": [x["keep_rate"] for x in gens], "barred": [x["barred"] for x in gens],
              "breadth": o.get("breadth"), "inworld": sc.get("inworld"), "near": sc.get("near"), "held": sc.get("held"),
              "tested_at": max((int(k) for k in tests), default=None), "note": "testing" if pending else ""})

    out = await S.run_sim(llm, state, on_gen=on_gen, tests=p.get("tests", True))
    out["model"] = llm.config
    return out


async def k_battery(name, p, tick, vol):
    """The test battery for the unmodified model, or for a saved adapter."""
    from evo3 import tests3, world3 as W
    llm = make_llm(p["model"])
    if p.get("adapter"):
        vol.reload()
        llm.load_adapter(p["adapter"])
    b = await tests3.battery(llm, p.get("world") or W.DEFAULT, p.get("test_sizes"))
    tick({"scores": b["scores"]})
    return {"model": llm.config, "adapter": p.get("adapter"), "scores": b["scores"], "items": b["items"],
            "usage": llm.usage()}


async def k_breadth(name, p, tick, vol):
    """Train on correct play in K one-question families with a fixed budget, then run the breadth battery.
    No families means the untrained baseline. Saves every 200 steps so a preempted worker resumes."""
    from evo3 import breadth as B
    fams, seed = p.get("families") or [], p.get("seed", 0)
    llm = make_llm(p["model"], seed=seed)
    steps = p.get("steps", 1200) if fams else 0
    logs = B.training_set(fams, p.get("examples", 9600), 500000 + seed) if fams else []
    d = f"/runs/{name}"
    os.makedirs(d, exist_ok=True)
    if os.path.exists(f"{d}/state.pkl"):
        out = pickle.load(open(f"{d}/state.pkl", "rb"))
        if os.path.exists(f"{d}/adapter.pt"):
            llm.load_adapter(f"{d}/adapter.pt")
    else:
        out = {"model": llm.config, "families": fams, "k": len(fams), "seed": seed, "examples": len(logs),
               "steps": steps, "trained": 0, "loss": []}
    while out["trained"] < steps:
        n = min(200, steps - out["trained"])
        out["loss"] += llm.train(logs, n, lr=p.get("lr", 2e-4), batch_size=8, seed=seed * 100000 + out["trained"])
        out["trained"] += n
        llm.save_adapter(f"{d}/adapter.pt")
        pickle.dump(out, open(f"{d}/state.pkl.tmp", "wb"))
        os.replace(f"{d}/state.pkl.tmp", f"{d}/state.pkl")
        vol.commit()
        tick({"k": len(fams), "note": f"trained {out['trained']} of {steps} steps, loss {out['loss'][-1]}"})
    tick({"k": len(fams), "note": "testing"})
    b = await B.battery(llm, fams, p.get("sizes"), far=p.get("far", True), public=p.get("public", True), near=p.get("near"))
    out.update(scores=b["scores"], items=b["items"], usage=llm.usage())
    out["micro"], out["micro_splits"] = getattr(llm, "micro", None), getattr(llm, "micro_splits", 0)
    sc = b["scores"]
    tick({"k": len(fams), "note": "tested", "held": sc.get("held"), "seen": sc.get("seen"), "unseen": sc.get("unseen"), "near": sc.get("near"),
          "bbh": sc.get("bbh"), "general": sc.get("general"), "retention": sc.get("retention")})
    return out


async def k_bbhdump(name, p, tick, vol):
    """Raw replies on the public test, for the untrained model or a saved adapter. Used to check the scoring."""
    from evo3 import breadth as B
    llm = make_llm(p["model"])
    if p.get("adapter"):
        vol.reload()
        llm.load_adapter(p["adapter"])
    llm.temp = 0.0

    async def one(task, text, target):
        d = await llm.ask_json(B.BBH_SYSTEM, text, max_tokens=24)
        return {"task": task, "target": target, "reply": d.get("answer", d.get("_raw", "")), "input": text}

    rows = await llm.run([one(*it) for it in B.bbh_items(p.get("n", 40))])
    return {"adapter": p.get("adapter"), "rows": rows}


KINDS = {"social": k_social, "bbhdump": k_bbhdump, "breadth": k_breadth, "battery": k_battery, "speed": k_speed, "headroom": k_headroom, "ceiling": k_ceiling, "sim": k_sim}


def run(name, kind, params, board, vol, gpu=False, big=False):
    global ON_GPU
    ON_GPU = gpu
    rate = RATE_GPU_BIG if big else (RATE_GPU if gpu else RATE)
    t0 = time.time()
    key, last = f"cost:{name}:{int(t0)}", {}

    def cost():
        sec = time.time() - t0 + START_ALLOWANCE
        return round(sec), round(sec * rate, 4)

    def tick(extra=None, state="running"):
        sec, usd = cost()
        board[key] = {"seconds": sec, "usd": usd}
        if extra:
            last.update(extra)
        rss = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1048576, 2)
        board[f"status:{name}"] = dict(last, name=name, kind=kind, state=state, updated=time.time(), usd=usd,
                                       seconds=sec, peak_gb=rss, ver=VERSION, gpu=gpu)

    if spent(board) >= CAP:
        tick(state="refused: budget reached")
        return {"refused": "budget"}
    # Modal restarts a task whose worker died or was preempted. Allow 3 starts; every start is on the ledger.
    starts = sum(1 for k in board.keys() if str(k).startswith(f"cost:{name}:"))
    if starts >= (8 if kind in ("sim", "social") else 3):   # a sim resumes from its checkpoint, so restarts are cheap
        board[f"status:{name}"] = dict(board.get(f"status:{name}", {}), name=name, kind=kind, updated=time.time(),
                                       state="failed", error="worker died; restart refused to protect the budget")
        return {"refused": "restart"}
    tick({"note": "loading model"})
    alive = True

    def watchdog():
        while alive:
            time.sleep(30)
            if not alive:
                return
            try:
                tick()
                if spent(board) >= CAP:
                    tick(state="stopped: budget reached")
                    os._exit(0)
            except Exception:
                pass

    threading.Thread(target=watchdog, daemon=True).start()
    try:
        out = asyncio.run(KINDS[kind](name, params, tick, vol))
        alive = False
        out["name"], out["kind"] = name, kind
        out["seconds"], out["usd"] = cost()
        os.makedirs("/runs/results", exist_ok=True)
        json.dump(out, open(f"/runs/results/{name}.json", "w"))
        vol.commit()
        tick(state="done")
        return out
    except BaseException as e:
        alive = False
        tick({"error": repr(e)[:400]}, state="failed")
        raise
