"""Job lists for version 4 (SPEC4.md).

    PYTHONPATH=. python3 -m evo3.social_jobs   -> runs3/job_s4_smoke.json, runs3/job_s4.json
"""
import json

from evo3 import social as S

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


def job(name, arm, seed, **P):
    return {"name": name, "kind": "social", "gpu": True, "params": {"model": MODEL, "arm": arm, "seed": seed, "P": P}}


if __name__ == "__main__":
    smoke = [job(f"s4-smoke-{arm}", arm, 91, G=4, test_at=[4]) for arm in ("SOLO", "MEDLEY")]
    full = [job(f"s4-{arm}-s{seed}", arm, seed) for seed in range(1, 6) for arm in S.ARMS]
    json.dump(smoke, open("runs3/job_s4_smoke.json", "w"), indent=1)
    json.dump(full, open("runs3/job_s4.json", "w"), indent=1)
    print(len(smoke), "smoke jobs,", len(full), "full jobs")
