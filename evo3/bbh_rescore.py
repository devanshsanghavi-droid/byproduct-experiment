"""Check of the public-test scoring, added after seeing the data.

The main runs scored a multiple-choice reply as right only if it named the option letter. A model trained to answer
with a word may reply with the option's text instead. This rescoring also accepts a reply equal to the text of exactly
one option. Reads runs3/modal/bbh-dump-*.json -> runs3/bbh_rescore.json.

    PYTHONPATH=. python3 -m evo3.bbh_rescore
"""
import collections
import glob
import json
import re

from evo3.breadth import bbh_right


def norm(x):
    return re.sub(r"[^a-z0-9 ]", "", str(x).lower()).strip()


def options(text):
    return {m.group(1): norm(m.group(2)) for m in re.finditer(r"^\(([A-R])\)\s*(.+)$", text, flags=re.M)}


def kind(row):
    """What the reply looks like on a multiple-choice item."""
    g, opts = str(row["reply"]).strip(), options(row["input"])
    if re.fullmatch(r"\W*\(?[A-R]\)?\W*", g):
        return "letter"
    if norm(g) in opts.values():
        return "option text"
    return "other"


def lenient(row):
    if bbh_right(row["target"], row["reply"]):
        return True
    m = re.fullmatch(r"\(([A-R])\)", row["target"].strip())
    if not m:
        return False
    opts, g = options(row["input"]), norm(row["reply"])
    return bool(g) and [k for k, v in opts.items() if v == g] == [m.group(1)]


if __name__ == "__main__":
    out = {}
    for f in sorted(glob.glob("runs3/modal/bbh-dump-*.json")):
        d = json.load(open(f))
        rows, name = d["rows"], d["name"].replace("bbh-dump-", "")
        mc = [r for r in rows if re.fullmatch(r"\([A-R]\)", r["target"].strip())]
        kinds = collections.Counter(kind(r) for r in mc)
        strict = sum(bbh_right(r["target"], r["reply"]) for r in rows) / len(rows)
        len_ = sum(lenient(r) for r in rows) / len(rows)
        by = {}
        for r in rows:
            by.setdefault(r["task"], []).append(lenient(r))
        out[name] = {"strict": round(strict, 4), "lenient": round(len_, 4), "n": len(rows), "multiple_choice": len(mc),
                     "reply_kinds": {k: round(v / len(mc), 3) for k, v in kinds.items()},
                     "lenient_by_task": {t: round(sum(v) / len(v), 3) for t, v in by.items()},
                     "samples": [{"task": r["task"], "target": r["target"], "reply": str(r["reply"])[:60]} for r in mc[::37][:8]]}
        print(f"{name:<8} strict {strict:.3f}  lenient {len_:.3f}  multiple-choice replies: {dict(kinds)}")
        for s in out[name]["samples"][:5]:
            print(f"      {s['task'][:28]:<28} target {s['target']}  reply {s['reply']!r}")
    json.dump(out, open("runs3/bbh_rescore.json", "w"), indent=1)
