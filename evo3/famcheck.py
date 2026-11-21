"""Checker for one-question task families.

    PYTHONPATH=. python3 -m evo3.famcheck evo3/fam_a.py [more files]

A family file defines FAMILIES = {name: (generator, role_sentence)}. A generator takes a random.Random and
returns wide.q(user, answer, kind, ...). Exit code 1 if any family has a problem.
"""
import collections
import importlib.util
import json
import random
import re
import sys

from evo3 import wide

N = 400


def load(path):
    spec = importlib.util.spec_from_file_location("fam_" + re.sub(r"\W", "_", path), path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.FAMILIES


def check_family(name, gen, role):
    bad = []
    if not re.fullmatch(r"[a-z][a-z0-9]{2,11}", name):
        bad.append("name must be 3-12 lowercase letters or digits")
    if not (isinstance(role, str) and role.startswith("You ") and role.endswith(".") and len(role) <= 120):
        bad.append("role must be one sentence starting with 'You ', ending with '.', at most 120 characters")
    try:
        items = [gen(random.Random(1000 + i)) for i in range(N)]
        again = gen(random.Random(1000))
    except Exception as e:  # noqa: BLE001
        return [f"generator raised {e!r}"], {}
    if again != items[0]:
        bad.append("not deterministic: same seed gave a different item (use only the rng passed in)")
    rng, hits = random.Random(7), 0
    for it in items:
        k, a, u = it.get("kind"), it.get("answer"), it.get("user")
        if k not in wide.FMT:
            bad.append(f"unknown kind {k!r}")
            break
        if not (isinstance(u, str) and 20 <= len(u) <= 400):
            bad.append(f"question must be 20-400 characters, got {len(str(u))}")
            break
        if k == "int" and not (isinstance(a, int) and not isinstance(a, bool) and isinstance(it.get("lo"), int)
                               and isinstance(it.get("hi"), int) and it["lo"] <= a <= it["hi"] and it["lo"] >= 0):
            bad.append(f"int answer needs lo <= answer <= hi with lo >= 0 (answer {a!r}, lo {it.get('lo')!r}, hi {it.get('hi')!r})")
            break
        if k == "yesno" and a not in ("yes", "no"):
            bad.append(f"yesno answer must be 'yes' or 'no', got {a!r}")
            break
        if k == "word":
            o = it.get("options")
            if not (isinstance(o, list) and 2 <= len(o) <= 8 and len(set(o)) == len(o) and a in o
                    and all(isinstance(x, str) and re.fullmatch(r"[a-z]+", x) for x in o)):
                bad.append(f"word answer needs options: 2-8 distinct lowercase single words that include the answer (got {o!r}, answer {a!r})")
                break
        if k == "list" and not (isinstance(a, list) and 2 <= len(a) <= 6
                                and all(isinstance(x, str) and re.fullmatch(r"[a-z0-9]+", x) for x in a)):
            bad.append(f"list answer must be 2-6 lowercase tokens (letters or digits, no spaces), got {a!r}")
            break
        right, valid = wide.check(it, json.loads(json.dumps({"answer": a})))
        if not (right and valid):
            bad.append(f"the correct answer is not accepted by wide.check: {a!r} for {u!r}")
            break
        hits += wide.check(it, {"answer": wide.random_answer(it, rng)})[0]
    stats = {}
    if not bad:
        users = len({it["user"] for it in items})
        top = collections.Counter(json.dumps(it["answer"]) for it in items).most_common(1)[0][1] / N
        kinds = collections.Counter(it["kind"] for it in items)
        stats = {"kind": "/".join(sorted(kinds)), "distinct": users, "top_answer": round(top, 2), "chance": round(hits / N, 2),
                 "chars": round(sum(len(it["user"]) for it in items) / N)}
        if users < 80:
            bad.append(f"only {users} distinct questions in {N} draws; need at least 80")
        if top > (0.65 if "yesno" in kinds else 0.5):
            bad.append(f"one answer covers {top:.0%} of items; balance the answers")
        if hits / N > 0.55:
            bad.append(f"a random reply is right {hits / N:.0%} of the time; need at most 55%")
    return bad, stats


def main(paths):
    seen, failed = {}, 0
    for path in paths:
        fams = load(path)
        print(f"{path}: {len(fams)} families")
        for name, (gen, role) in fams.items():
            bad, st = check_family(name, gen, role)
            if name in seen or name in wide.ALL:
                bad.append(f"name already used ({seen.get(name, 'evo3/wide.py')})")
            seen[name] = path
            failed += bool(bad)
            ex = gen(random.Random(3))
            print(f"  {'FAIL' if bad else 'ok  '} {name:<12} {st}")
            print(f"       e.g. {ex['user']}  ->  {ex['answer']}")
            for b in bad:
                print(f"       PROBLEM: {b}")
    print(f"{len(seen)} families checked, {failed} with problems")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
