"""Live tracker for the Modal runs: a tiny local page that refreshes every 5 seconds.

    python3 -m evo3.live_modal --port 8768

Reads the shared status board on Modal. Read-only; does no model work on this machine.
"""
import argparse
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import modal

from evo3.remote import CAP, RATE_GPU, RATE_GPU_BIG, STRONG_BASE

THESIS = ("Two strengthening experiments. A (n2-): the confirmed test repeated with a second, separate set of 16 reserved "
          "families. B (m3-): the confirmed test repeated on a model twice the size. Rule fixed in advance for each: "
          "does the reserved-family score rise with the number of families trained on (K)?")
board = modal.Dict.from_name("evo3-board", create_if_missing=True)
cache = {"t": 0, "data": {}}


def status() -> dict:
    if time.time() - cache["t"] < 4:
        return cache["data"]
    tasks, total = [], 0.0
    for k, v in board.items():
        if str(k).startswith("cost:"):
            total += v["usd"]
        elif str(k).startswith("status:"):
            v = dict(v)
            v["age"] = round(time.time() - v.pop("updated", time.time()))
            tasks.append(v)
    tasks = [v for v in tasks if str(v.get("name", "")).startswith(("n2-", "m3-"))] or [v for v in tasks if str(v.get("name", "")).startswith("nr-")]
    tasks.sort(key=lambda v: (v["state"] != "running", v["name"][:2], v.get("k", 0), v["name"]))
    done = sum(1 for v in tasks if v["state"] == "done")
    cache.update(t=time.time(), data={"tasks": tasks, "spent": round(max(0.0, total - STRONG_BASE), 3),
                                      "cap": round(CAP - STRONG_BASE, 2), "limit": 10.0, "done": done, "n": len(tasks),
                                      "rate": round(RATE_GPU * 3600, 2), "rate_big": round(RATE_GPU_BIG * 3600, 2), "thesis": THESIS})
    return cache["data"]


PAGE = """<!doctype html><meta charset="utf-8"><title>Byproduct Experiment 3 live</title>
<style>
body{font:14px/1.45 -apple-system,system-ui,sans-serif;margin:0;padding:20px 16px;background:#f7f6f2;color:#1d1c1a}
@media(prefers-color-scheme:dark){body{background:#161614;color:#e9e7e1}th{color:#9a978f!important}td,th{border-color:#33312d!important}.bar{background:#33312d!important}}
h1{font-size:17px;margin:0 0 6px;max-width:900px;font-weight:600}p{margin:4px 0;max-width:900px}
table{border-collapse:collapse;margin-top:14px;width:100%;max-width:1100px}
td,th{padding:6px 8px;border-bottom:1px solid #ddd9cf;text-align:left;vertical-align:top;font-variant-numeric:tabular-nums}
th{font-size:12px;color:#6b6860;font-weight:600}.bar{height:8px;background:#ddd9cf;border-radius:4px;max-width:420px;margin:6px 0}
.bar div{height:8px;border-radius:4px;background:#2f7d5b}.running{color:#2f7d5b;font-weight:600}.failed{color:#b3402a;font-weight:600}
small{color:#85827a}.wrap{overflow-x:auto}
</style>
<h1 id="thesis"></h1><p id="money"></p><div class="bar"><div id="fill" style="width:0"></div></div>
<p><small id="when"></small></p><div class="wrap"><table id="t"></table></div>
<script>
function arrows(a){ if(a==null) return ""; return a.map(function(x){ return (+x).toFixed(Math.abs(x)<1?2:1); }).join(" → "); }
function last(a){ return a&&a.length ? (+a[a.length-1]).toFixed(2) : ""; }
async function tick(){
  try{
    const d=await (await fetch("/status")).json();
    thesis.textContent=d.thesis;
    money.textContent="Spend on these two experiments: $"+d.spent.toFixed(2)+" (own upper estimate) of under $"+d.limit.toFixed(2)+" approved. Auto-stop at $"+d.cap.toFixed(2)+". Finished "+d.done+" of "+d.n+" tasks started so far. One GPU worker costs $"+d.rate+" per hour ($"+d.rate_big+" for the larger model).";
    fill.style.width=Math.min(100,100*d.spent/d.limit)+"%";
    let h="<tr><th>Task</th><th>State</th><th>Families trained on (K)</th><th>Progress</th><th>Reserved families (the test)</th><th>Held-out families (new skills)</th><th>Trained families</th><th>Cost</th><th>Last update</th></tr>";
    for(const v of d.tasks){
      const cls=v.state=="running"?"running":(v.state=="failed"?"failed":"");
      const f3=function(x){ return x==null?"":(+x).toFixed(3); };
      h+="<tr><td>"+v.name+"<br><small>"+v.kind+"</small></td><td class='"+cls+"'>"+v.state+"</td><td>"+(v.k==null?"":v.k)+"</td><td>"+(v.note||"")+(v.error?(" "+v.error):"")+"</td><td>"+f3(v.near)+"</td><td>"+f3(v.held)+"</td><td>"+f3(v.seen)+"</td><td>$"+(+v.usd).toFixed(2)+"<br><small>"+Math.round(v.seconds/60)+" min</small></td><td>"+v.age+"s ago</td></tr>";
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
    ap.add_argument("--port", type=int, default=8768)
    ThreadingHTTPServer(("127.0.0.1", ap.parse_args().port), H).serve_forever()
