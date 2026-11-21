"""Modal app for version 3 (CPU only).

    modal deploy -m evo3.modal_app          # once, and after every code change
    python -m evo3.submit '[{"name": "speed", "kind": "speed", "params": {...}}]'
"""
import modal

from evo3.remote import BBH_TASKS, CORES, GPU_CORES, GPU_MEM, GPU_MEM_BIG, MEM_LIM, MEM_REQ

MODELS = ["Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct"]


def fetch():
    from huggingface_hub import snapshot_download
    for m in MODELS:
        snapshot_download(m, allow_patterns=["*.json", "*.safetensors", "*.txt"])


def fetch_big():
    from huggingface_hub import snapshot_download
    snapshot_download("Qwen/Qwen2.5-3B-Instruct", allow_patterns=["*.json", "*.safetensors", "*.txt"])


def fetch_bbh():
    """Public test set (BIG-Bench Hard), from the benchmark authors' repository. Data only; parsed as JSON."""
    import json
    import os
    import urllib.request
    os.makedirs("/data/bbh", exist_ok=True)
    for t in BBH_TASKS:
        raw = urllib.request.urlopen(
            f"https://raw.githubusercontent.com/suzgunmirac/BIG-Bench-Hard/main/bbh/{t}.json", timeout=60).read()
        ex = json.loads(raw)["examples"]
        assert len(ex) >= 100 and all(isinstance(e.get("input"), str) and isinstance(e.get("target"), str) for e in ex), t
        open(f"/data/bbh/{t}.json", "wb").write(raw)


image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("torch", index_url="https://download.pytorch.org/whl/cpu")
         .pip_install("transformers", "peft", "accelerate", "safetensors", "huggingface_hub")
         .env({"HF_HOME": "/hf", "TOKENIZERS_PARALLELISM": "false", "HF_HUB_DISABLE_TELEMETRY": "1"})
         .run_function(fetch)
         .env({"HF_HUB_OFFLINE": "1"})
         .add_local_python_source("evo", "evo2", "evo3"))

gpu_image = (modal.Image.debian_slim(python_version="3.11")
             .pip_install("torch")   # default wheel includes CUDA
             .pip_install("transformers", "peft", "accelerate", "safetensors", "huggingface_hub")
             .env({"HF_HOME": "/hf", "TOKENIZERS_PARALLELISM": "false", "HF_HUB_DISABLE_TELEMETRY": "1"})
             .run_function(fetch)
             .env({"HF_HUB_OFFLINE": "1"})
             .run_function(fetch_bbh)
             .add_local_python_source("evo", "evo2", "evo3"))

big_image = (modal.Image.debian_slim(python_version="3.11")
             .pip_install("torch")
             .pip_install("transformers", "peft", "accelerate", "safetensors", "huggingface_hub")
             .env({"HF_HOME": "/hf", "TOKENIZERS_PARALLELISM": "false", "HF_HUB_DISABLE_TELEMETRY": "1"})
             .run_function(fetch)
             .run_function(fetch_big)
             .env({"HF_HUB_OFFLINE": "1"})
             .add_local_python_source("evo", "evo2", "evo3"))

app = modal.App("evo3")
vol = modal.Volume.from_name("evo3-runs", create_if_missing=True)
board = modal.Dict.from_name("evo3-board", create_if_missing=True)


@app.function(image=image, cpu=(CORES, CORES), memory=(MEM_REQ, MEM_LIM), timeout=5 * 3600,
              volumes={"/runs": vol}, max_containers=6)
def task(name: str, kind: str, params: dict):
    from evo3 import remote
    return remote.run(name, kind, params, board, vol)


@app.function(image=gpu_image, gpu=["L4", "A10"], cpu=(GPU_CORES, GPU_CORES), memory=(GPU_MEM, GPU_MEM), timeout=3 * 3600,
              volumes={"/runs": vol}, max_containers=8)
def task_gpu(name: str, kind: str, params: dict):
    from evo3 import remote
    return remote.run(name, kind, params, board, vol, gpu=True)


@app.function(image=big_image, gpu=["L4", "A10"], cpu=(GPU_CORES, GPU_CORES), memory=(GPU_MEM_BIG, GPU_MEM_BIG), timeout=3 * 3600,
              volumes={"/runs": vol}, max_containers=8)
def task_gpu_big(name: str, kind: str, params: dict):
    from evo3 import remote
    return remote.run(name, kind, params, board, vol, gpu=True, big=True)
