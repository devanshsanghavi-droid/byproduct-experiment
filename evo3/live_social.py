"""Live tracker for the version 4 runs on Modal: a tiny local page that refreshes every 5 seconds.

    python3 -m evo3.live_social --port 8769

Reads the shared status board on Modal. Read-only; does no model work on this machine.
"""
import argparse
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import modal

from evo3.remote import CAP, RATE_GPU, S4_BASE

THESIS = ("Version 4: social arms with changing weights. Each row is one population of 32 agents sharing one adapter. "
          "Question for this run: do social rules widen what a population practises, beyond what the rules do by themselves? "
          "Declared before any run: score lift on unseen tasks is not testable at this scale.")
board = modal.Dict.from_name("evo3-board", create_if_missing=True)
cache = {"t": 0, "data": {}}


def status() -> dict:
    if time.time() - cache["t"] < 4:
        return cache["data"]
    tasks, total = [], 0.0
    for k, v in board.items():
        if str(k).startswith("cost:"):
            total += v["usd"]
        elif str(k).startswith("status:") and str(v.get("name", "")).startswith("s4-"):
            v = dict(v)
            v["age"] = round(time.time() - v.pop("updated", time.time()))
            tasks.append(v)
    tasks.sort(key=lambda v: (v["state"] != "running", "smoke" not in v["name"], v.get("seed", 0), v["name"]))
    cache.update(t=time.time(), data={"tasks": tasks, "spent": round(max(0.0, total - S4_BASE), 3), "limit": round(CAP - S4_BASE, 2),
                                      "done": sum(1 for v in tasks if v["state"] == "done"), "n": len(tasks),
                                      "rate": round(RATE_GPU * 3600, 2), "thesis": THESIS})
    return cache["data"]


PAGE = """<!doctype html><meta charset="utf-8"><title>Byproduct Experiment 4 live</title>
<style>
body{font:14px/1.45 -apple-system,system-ui,sans-serif;margin:0;padding:20px 16px;background:#f7f6f2;color:#1d1c1a}
@media(prefers-color-scheme:dark){body{background:#161614;color:#e9e7e1}th{color:#9a978f!important}td,th{border-color:#33312d!important}.bar{background:#33312d!important}}
h1{font-size:16px;margin:0 0 6px;max-width:980px;font-weight:600}p{margin:4px 0;max-width:980px}
table{border-collapse:collapse;margin-top:14px;width:100%;max-width:1280px}
td,th{padding:6px 8px;border-bottom:1px solid #ddd9cf;text-align:left;vertical-align:top;font-variant-numeric:tabular-nums}
th{font-size:12px;color:#6b6860;font-weight:600}.bar{height:8px;background:#ddd9cf;border-radius:4px;max-width:420px;margin:6px 0}
.bar div{height:8px;border-radius:4px;background:#2f7d5b}.running{color:#2f7d5b;font-weight:600}.failed{color:#b3402a;font-weight:600}
small{color:#85827a}.wrap{overflow-x:auto}
</style>
<h1 id="thesis"></h1><p id="money"></p><div class="bar"><div id="fill" style="width:0"></div></div>
<p><small id="when"></small></p><div class="wrap"><table id="t"></table></div>
<script>
function ends(a,dp){ if(!a||!a.length) return ""; a=a.filter(function(x){return x!=null;}); if(!a.length) return ""; var f=function(x){return (+x).toFixed(dp);}; return a.length>1 ? f(a[0])+" → "+f(a[a.length-1]) : f(a[0]); }
function f3(x){ return x==null?"":(+x).toFixed(3); }
async function tick(){
  try{
    const d=await (await fetch("/status")).json();
    thesis.textContent=d.thesis;
    money.textContent="Spend on version 4: $"+d.spent.toFixed(2)+" (own upper estimate) of the $"+d.limit.toFixed(2)+" limit, which is also the auto-stop. Finished "+d.done+" of "+d.n+" runs started so far. One GPU worker costs $"+d.rate+" per hour.";
    fill.style.width=Math.min(100,100*d.spent/d.limit)+"%";
    let h="<tr><th>Run</th><th>State</th><th>Generation</th><th>Share right, first → latest</th><th>Mean energy</th><th>Families per agent</th><th>Families among parents</th><th>Breadth so far</th><th>Keep-all rate</th><th>Barred by vote, latest</th><th>Practised skill (untrained 0.510)</th><th>Similar tasks (untrained 0.543)</th><th>Cost</th><th>Last update</th></tr>";
    for(const v of d.tasks){
      const cls=v.state=="running"?"running":(v.state=="failed"?"failed":"");
      const b=v.barred&&v.barred.length?v.barred[v.barred.length-1]:"";
      h+="<tr><td>"+v.name+"</td><td class='"+cls+"'>"+v.state+"</td><td>"+(v.gen==null?"":v.gen+" of "+v.G)+(v.note?"<br><small>"+v.note+"</small>":"")+(v.error?"<br><small>"+v.error+"</small>":"")+"</td><td>"+ends(v.acc,2)+"</td><td>"+ends(v.energy,1)+"</td><td>"+ends(v.rep,1)+"</td><td>"+ends(v.pfams,0)+"</td><td>"+(v.breadth==null?"":(+v.breadth).toFixed(1))+"</td><td>"+ends(v.keep,2)+"</td><td>"+b+"</td><td>"+f3(v.inworld)+(v.tested_at?"<br><small>after generation "+v.tested_at+"</small>":"")+"</td><td>"+f3(v.near)+"</td><td>$"+(+v.usd).toFixed(2)+"<br><small>"+Math.round(v.seconds/60)+" min</small></td><td>"+v.age+"s ago</td></tr>";
    }
    t.innerHTML=h; when.textContent="Refreshed "+new Date().toLocaleTimeString()+". Runs happen on Modal, not on this Mac.";
  }catch(e){when.textContent="Could not reach the tracker: "+e}
}
tick(); setInterval(tick,5000);
</script>"""


class H(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.startswith("/status"):
            body, ctype = json.dumps(status()).encode(), "application/json"
        else:
            body, ctype = PAGE.encode(), "text/html; charset=utf-8"
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8769)
    ThreadingHTTPServer(("127.0.0.1", ap.parse_args().port), H).serve_forever()
