"""Live tracker: a tiny local web page that shows run progress, refreshing every 3 seconds.

    python3 -m evo2.live_server --runs runs2 --port 8767

Reads the run files the simulation rewrites after every generation. Read-only.
"""
import argparse
import glob
import json
import os
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ARM_ORDER = ["DRIFT", "SOLO", "COOP", "ALPHA", "DOMEST", "CULTURE", "MEDLEY"]
ARM_LABEL = {
    "DRIFT": "Theory 1: chance alone (noise floor)",
    "SOLO": "Baseline: solo foraging",
    "COOP": "Theory 2: cooperation (adviser and preparer)",
    "ALPHA": "Control for 3: grabbing, no vote",
    "DOMEST": "Theory 3: self-domestication",
    "CULTURE": "Theory 4: cultural brain (notes)",
    "MEDLEY": "Theories 2 + 3 + 4",
}
GENES = ["energy", "rate", "illegal_rate", "rules", "grab_rate", "mentions_L1", "mentions_L2"]
THESIS = ("All three surviving theories say intelligence was a byproduct. Agents are scored only "
          "on staying alive and being copied; ability is measured separately afterward. Version 2: prompt rules and notes evolve.")


def status(runs_dir: str) -> dict:
    runs = []
    for path in sorted(glob.glob(os.path.join(runs_dir, "*_s*.json"))):
        try:
            d = json.load(open(path))
        except (json.JSONDecodeError, OSError):
            continue  # file is mid-write; next refresh will catch it
        gens = d.get("generations", [])
        runs.append({
            "arm": d.get("arm"), "seed": d.get("seed"),
            "done": len(gens), "total": d.get("config", {}).get("G", 0),
            "complete": bool(d.get("complete")),
            "cost": 0, "calls": d.get("calls", 0),
            "means": [g.get("mean", {}) for g in gens],
            "exiled": sum(g.get("exiled", 0) for g in gens),
            "parse_fail": d.get("parse_fail", 0),
            "updated": os.path.getmtime(path),
        })
    arms = []
    for arm in ARM_ORDER:
        rs = [r for r in runs if r["arm"] == arm]
        if not rs:
            continue
        depth = max(r["done"] for r in rs)
        traj = []
        for g in range(depth):
            have = [r["means"][g] for r in rs if g < len(r["means"])]
            row = {"n": len(have)}
            for k in GENES:
                vals = [m[k] for m in have if m.get(k) is not None]
                row[k] = sum(vals) / len(vals) if vals else None
            traj.append(row)
        arms.append({"arm": arm, "label": ARM_LABEL[arm], "traj": traj,
                     "runs": sorted([{k: r[k] for k in ("seed", "done", "total", "complete")} for r in rs],
                                    key=lambda r: r["seed"]),
                     "exiled": sum(r["exiled"] for r in rs)})
    log = []
    log_path = os.path.join(runs_dir, "sim.log")
    if os.path.exists(log_path):
        with open(log_path, "rb") as f:
            f.seek(0, os.SEEK_END)
            f.seek(max(0, f.tell() - 6000))
            log = f.read().decode("utf-8", "replace").splitlines()[-14:]
    heldout = None
    cost = 0.0
    hp = os.path.join(runs_dir, "measure.json")
    if os.path.exists(hp):
        try:
            h = json.load(open(hp))
            heldout = {"complete": bool(h.get("complete")), "done": len(h.get("populations", {})),
                       "cost": h.get("usage", {}).get("cost_usd", 0)}
            cost += heldout["cost"]
        except (json.JSONDecodeError, OSError):
            pass
    up = os.path.join(runs_dir, "usage.json")
    if os.path.exists(up):
        try:
            cost += json.load(open(up)).get("cost_usd", 0)
        except (json.JSONDecodeError, OSError):
            pass
    done = sum(r["done"] for r in runs)
    total = sum(r["total"] for r in runs)
    return {"thesis": THESIS, "now": time.time(), "runs_dir": runs_dir,
            "gens_done": done, "gens_total": total,
            "runs_complete": sum(r["complete"] for r in runs), "runs_total": len(runs),
            "cost": round(cost, 3), "calls": sum(r["calls"] for r in runs),
            "parse_fail": sum(r["parse_fail"] for r in runs),
            "last_update": max([r["updated"] for r in runs], default=None),
            "arms": arms, "log": log, "heldout": heldout}


PAGE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Live Run Tracker</title>
<style>
:root{--bg:#f7f6f2;--card:#fff;--ink:#1c1b19;--mute:#6b6862;--line:#dedbd3;--acc:#2f6f5e;--warn:#b4532a;--bar:#e9e6de}
@media (prefers-color-scheme:dark){:root{--bg:#161614;--card:#1f1f1c;--ink:#ecebe6;--mute:#9b978f;--line:#33322e;--acc:#6fbfa8;--warn:#e08a5e;--bar:#2b2a26}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 ui-sans-serif,system-ui,sans-serif}
main{max-width:1000px;margin:0 auto;padding:20px 16px 48px}
h1{font-size:19px;line-height:1.3;margin:0 0 4px;font-weight:650}
.sub{color:var(--mute);margin:0 0 16px}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:10px;margin-bottom:14px}
.kpi{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:10px 12px}
.kpi b{display:block;font-size:20px;font-variant-numeric:tabular-nums}.kpi span{color:var(--mute);font-size:12px}
.big{height:10px;background:var(--bar);border-radius:5px;overflow:hidden;margin-bottom:18px}.big i{display:block;height:100%;background:var(--acc);transition:width .6s}
.wrap{overflow-x:auto;background:var(--card);border:1px solid var(--line);border-radius:8px;margin-bottom:16px}
table{border-collapse:collapse;width:100%;min-width:980px}th,td{padding:7px 10px;text-align:right;border-bottom:1px solid var(--line);font-variant-numeric:tabular-nums;white-space:nowrap}
th{font-size:11px;text-transform:uppercase;letter-spacing:.04em;color:var(--mute);font-weight:600}
td:first-child,th:first-child{text-align:left;white-space:normal}tr:last-child td{border-bottom:0}
.arm{font-weight:650}.lab{color:var(--mute);font-size:12px}
.seeds{display:flex;gap:3px;justify-content:flex-end}.seed{width:34px;height:8px;background:var(--bar);border-radius:4px;overflow:hidden}.seed i{display:block;height:100%;background:var(--acc)}
.up{color:var(--acc)}.down{color:var(--warn)}.mute{color:var(--mute)}
pre{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:10px 12px;overflow-x:auto;font-size:11.5px;margin:0;color:var(--mute)}
h2{font-size:13px;text-transform:uppercase;letter-spacing:.05em;color:var(--mute);margin:18px 0 8px}
.dot{display:inline-block;width:8px;height:8px;border-radius:50%;background:var(--acc);margin-right:6px;animation:p 1.6s infinite}.stale .dot{background:var(--warn);animation:none}
@keyframes p{50%{opacity:.25}}
</style></head><body><main>
<h1 id="thesis"></h1><p class="sub" id="sub"></p>
<div class="kpis" id="kpis"></div><div class="big"><i id="bigbar" style="width:0"></i></div>
<h2>Latest generation, with change since the first generation (average over seeds)</h2>
<div class="wrap"><table><thead><tr><th>Arm</th><th>Seeds</th><th>Gen</th><th>Energy</th><th>Win rate</th><th>Law-breaking tries</th><th>Rules held</th><th>Grab rate</th><th>Law 1 in text</th><th>Law 2 in text</th><th>Exiled</th></tr></thead><tbody id="rows"></tbody></table></div>
<h2>Latest log lines</h2><pre id="log">waiting for the run to start</pre>
</main><script>
const f=(x,d=2)=>x==null?"-":x.toFixed(d);
function cell(traj,k,d=2){const a=traj[0]?.[k],b=traj[traj.length-1]?.[k];if(b==null)return"<td>-</td>";
 const dl=b-(a??b),c=Math.abs(dl)<0.02?"mute":"up";return `<td>${f(b,d)} <span class="${c}">${dl>=0?"+":""}${f(dl,d)}</span></td>`}
async function tick(){let s;try{s=await (await fetch("status.json",{cache:"no-store"})).json()}catch(e){return}
 document.getElementById("thesis").textContent=s.thesis;
 const age=s.last_update?Math.round(s.now-s.last_update):null,stale=age==null||age>180;
 const fin=s.runs_total>0&&s.runs_complete===s.runs_total;
 document.getElementById("sub").className="sub"+(stale&&!fin?" stale":"");
 document.getElementById("sub").innerHTML=fin?"All runs finished.":`<span class="dot"></span>`+(age==null?"No run files yet.":`Last update ${age}s ago. Folder: ${s.runs_dir}`);
 const pct=s.gens_total?100*s.gens_done/s.gens_total:0;
 document.getElementById("bigbar").style.width=pct+"%";
 document.getElementById("kpis").innerHTML=[[f(pct,0)+"%","generations done ("+s.gens_done+" of "+s.gens_total+")"],[s.runs_complete+" / "+s.runs_total,"runs finished"],["$"+f(s.cost),"spent so far"],[s.calls.toLocaleString(),"model calls"],[s.parse_fail,"unusable replies"],[s.heldout?(s.heldout.complete?"done":s.heldout.done+" groups"):"not started","transfer tests"]].map(([b,t])=>`<div class="kpi"><b>${b}</b><span>${t}</span></div>`).join("");
 document.getElementById("rows").innerHTML=s.arms.map(a=>{const t=a.traj;
  const seeds=a.runs.map(r=>`<div class="seed" title="seed ${r.seed}: ${r.done}/${r.total}"><i style="width:${r.total?100*r.done/r.total:0}%"></i></div>`).join("");
  return `<tr><td><span class="arm">${a.arm}</span><br><span class="lab">${a.label}</span></td><td><div class="seeds">${seeds}</div></td><td>${t.length}</td>${cell(t,"energy",1)}${cell(t,"rate")}${cell(t,"illegal_rate")}${cell(t,"rules",1)}${cell(t,"grab_rate")}${cell(t,"mentions_L1")}${cell(t,"mentions_L2")}<td>${a.exiled}</td></tr>`}).join("");
 if(s.log.length)document.getElementById("log").textContent=s.log.join("\n");}
tick();setInterval(tick,3000);
</script></body></html>"""


def make_handler(runs_dir):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            if self.path.startswith("/status.json"):
                body, ctype = json.dumps(status(runs_dir)).encode(), "application/json"
            else:
                body, ctype = PAGE.encode(), "text/html; charset=utf-8"
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
    return H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs2")
    ap.add_argument("--port", type=int, default=8767)
    args = ap.parse_args()
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    runs_dir = args.runs if os.path.isabs(args.runs) else os.path.join(here, args.runs)
    print(f"Live tracker on http://localhost:{args.port} reading {runs_dir}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(runs_dir)).serve_forever()


if __name__ == "__main__":
    main()
