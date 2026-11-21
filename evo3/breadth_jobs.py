"""Write the job files for the breadth experiment.

    PYTHONPATH=. python3 -m evo3.breadth_jobs
    -> runs3/job_breadth_smoke.json   one tiny job, to prove the code path on the GPU
    -> runs3/job_breadth.json         1 untrained baseline + 3 runs at each of K = 2, 4, 8, 16, 32, 64
"""
import json

from evo3 import breadth as B

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


def job(name, fams, seed, **extra):
    return {"name": name, "kind": "breadth", "gpu": True,
            "params": dict({"model": MODEL, "families": fams, "seed": seed, "steps": 1200, "examples": 9600, "lr": 2e-4}, **extra)}


if __name__ == "__main__":
    smoke = [job("br-smoke", B.subset(2, 9), 99, steps=20, examples=160,
                 sizes={"pool": 1, "held": 2, "bbh": 2, "code": 2, "rule": 2, "order": 2, "retention": 4})]
    jobs = [job("br-k00-base", [], 0)]
    for k in B.LEVELS:
        for d in range(1, B.DRAWS + 1):
            jobs.append(job(f"br-k{k:02d}-r{d}", B.subset(k, d), 100 * k + d))
    # near-transfer rerun (SPEC3.md): 16 reserved pool families, 4 runs at each of K = 2..32, no public or far tests
    near, lean = B.reserved(), {"far": False, "public": False}
    rerun = [job("nr-k00-base", [], 0, near=near, **lean)]
    for k in (2, 4, 8, 16, 32):
        for d in range(1, 5):
            rerun.append(job(f"nr-k{k:02d}-r{d}", B.subset2(k, d), 7000 + 100 * k + d, near=near, **lean))
    json.dump([job("nr-smoke", B.subset2(2, 9), 98, steps=20, examples=160, near=near, sizes={"pool": 1, "held": 2, "near": 2}, **lean)],
              open("runs3/job_near_smoke.json", "w"), indent=1)
    json.dump(rerun, open("runs3/job_near.json", "w"), indent=1)
    print(f"rerun: {len(rerun)} jobs, reserved {near}")
    # strengthening experiments (SPEC3.md, 2026-11-15): A = second reserved set at 1.5B, B = reserved set 1 at 3B
    near2, big = B.reserved2(), "Qwen/Qwen2.5-3B-Instruct"
    a = [job("n2-k00-base", [], 0, near=near2, **lean)]
    b = [dict(job("m3-k00-base", [], 0, near=near, model=big, **lean), big=True)]
    for k in (2, 4, 8, 16, 32):
        for d in range(1, 5):
            a.append(job(f"n2-k{k:02d}-r{d}", B.subset3(k, d), 8000 + 100 * k + d, near=near2, **lean))
            b.append(dict(job(f"m3-k{k:02d}-r{d}", B.subset2(k, d), 7000 + 100 * k + d, near=near, model=big, **lean), big=True))
    assert not set(near2) & set(near) and all(not set(j["params"]["families"]) & set(near2) for j in a)
    assert all(not set(j["params"]["families"]) & set(near) for j in b)
    half = {"pool": 4, "held": 16, "near": 16}
    json.dump([dict(job("m3-smoke", B.subset2(2, 9), 98, steps=20, examples=160, near=near, model=big, sizes=half, **lean), big=True)],
              open("runs3/job_m3_smoke.json", "w"), indent=1)
    json.dump(a, open("runs3/job_n2.json", "w"), indent=1)
    json.dump(b, open("runs3/job_m3.json", "w"), indent=1)
    json.dump([j for j in b if not j["name"].endswith("r4")], open("runs3/job_m3_three.json", "w"), indent=1)
    print(f"strengthening: A {len(a)} jobs, reserved {near2}; B {len(b)} jobs on {big}")
    json.dump(smoke, open("runs3/job_breadth_smoke.json", "w"), indent=1)
    json.dump(jobs, open("runs3/job_breadth.json", "w"), indent=1)
    print(f"pool {len(B.POOL)} families, held-out {len(B.HELD)} families, {len(jobs)} jobs")
    for j in jobs:
        print(f"  {j['name']:<12} K={len(j['params']['families']):<3} seed={j['params']['seed']}")
