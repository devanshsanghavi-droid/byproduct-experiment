"""Start tasks on the deployed Modal app, or fetch results.

    python -m evo3.submit '[{"name": "speed", "kind": "speed", "params": {...}}]'
    python -m evo3.submit --fetch        # status, spend, and results saved to runs3/modal/
"""
import json
import os
import sys

import modal

from evo3.remote import CAP, spent

board = modal.Dict.from_name("evo3-board", create_if_missing=True)
if sys.argv[1] == "--fetch":
    os.makedirs("runs3/modal", exist_ok=True)
    vol = modal.Volume.from_name("evo3-runs")
    for k, v in sorted(board.items()):
        if str(k).startswith("status:"):
            print(f"{v['name']:<22} {v['state']:<10} ${v['usd']:<7} {v['seconds']:>6}s  "
                  + " ".join(f"{a}={v[a]}" for a in ("gen", "G", "note", "error") if v.get(a) not in (None, "")))
            if v["state"] == "done" and not os.path.exists(f"runs3/modal/{v['name']}.json"):
                with open(f"runs3/modal/{v['name']}.json", "wb") as f:
                    for chunk in vol.read_file(f"results/{v['name']}.json"):
                        f.write(chunk)
    json.dump({k: v for k, v in board.items() if str(k).startswith("cost:")}, open("runs3/modal/ledger.json", "w"))
    print(f"ledger total ${spent(board):.4f} of cap ${CAP:.2f} (own upper estimate)")
else:
    fns = {False: modal.Function.from_name("evo3", "task"), True: modal.Function.from_name("evo3", "task_gpu"),
           "big": modal.Function.from_name("evo3", "task_gpu_big")}
    arg = sys.argv[1]
    for spec in (json.load(open(arg)) if arg.endswith(".json") else json.loads(arg)):
        call = fns["big" if spec.get("big") else bool(spec.get("gpu"))].spawn(spec["name"], spec["kind"], spec.get("params", {}))
        print(spec["name"], call.object_id)
