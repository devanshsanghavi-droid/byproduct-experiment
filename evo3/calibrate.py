"""Check 1 (headroom): where does the unmodified model sit between chance and correct play?

For each family and difficulty: win rate of a random policy, of the model, and of correct play.
Then the full test battery for the model and for a random policy.
"""
import copy
import random

from evo3 import tests3
from evo3 import world3 as W
from evo3.local import LocalLLM

GRID = {
    "plants": [{"steps": 1, "actions": 4, "n": 2, "tries": 5}, {"steps": 2, "actions": 3, "n": 2, "tries": 5},
               {"steps": 2, "actions": 4, "n": 2, "tries": 5}],
    "tracks": [{"ring": 6, "n": 2, "rounds": 4}, {"ring": 8, "n": 2, "rounds": 4}],
    "trade": [{"offers": 2, "bundle": 1, "rounds": 8}, {"offers": 3, "bundle": 1, "rounds": 8}],
}


async def rate(llm, fam, cfg, n, oracle=False):
    worlds = [W.new_world(random.Random(500 + i), cfg) for i in range(n)]
    res = await llm.run([W.FAMILIES[fam](llm, w, random.Random(i), oracle) for i, w in enumerate(worlds)])
    tot = sum(r["rounds"] for r in res)
    return round(sum(r["wins"] for r in res) / tot, 3), round(sum(r["invalid"] for r in res) / tot, 3)


async def calibrate(llm, n=24, temps=(0.0, 0.7), tick=None, battery=True) -> dict:
    mock = LocalLLM(mock=True)
    out = {"model": llm.config, "families": [], "battery": {}}
    for fam, levels in GRID.items():
        for lv in levels:
            cfg = copy.deepcopy(W.DEFAULT)
            cfg[fam] = lv
            row = {"family": fam, "level": lv, "random": (await rate(mock, fam, cfg, 300))[0],
                   "correct": (await rate(mock, fam, cfg, 300, oracle=True))[0]}
            for temp in temps:
                llm.temp = temp
                row[f"model_t{temp}"], row[f"invalid_t{temp}"] = await rate(llm, fam, cfg, n)
            out["families"].append(row)
            if tick:
                tick({"note": f"{fam} level {len([r for r in out['families'] if r['family'] == fam])} done"})
    if battery:
        out["battery"]["random"] = (await tests3.battery(mock))["scores"]
        b = await tests3.battery(llm)
        out["battery"]["model"], out["items"] = b["scores"], b["items"]
    out["usage"] = llm.usage()
    return out
