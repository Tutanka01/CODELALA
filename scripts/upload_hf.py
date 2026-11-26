"""Publish CodeLALA-v1 (GGUF + full HF safetensors) cleanly to Hugging Face.

Reads the token silently from HF_TOKEN or a local .env file (NEVER prints the token).
Uses a proxy-safe httpx client (disables stale keep-alive reuse) + automatic retries.
Quantizes Q4_K_M and Q8_0 alongside existing Q6_K, creates both repositories under the
authenticated user, and uploads GGUFs first followed by full BF16 weights.
"""
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

import httpx
from huggingface_hub import HfApi, set_client_factory


def make_proxy_safe_client() -> httpx.Client:
    return httpx.Client(
        limits=httpx.Limits(max_keepalive_connections=0, keepalive_expiry=0.0, max_connections=8),
        timeout=httpx.Timeout(connect=30.0, read=600.0, write=600.0, pool=30.0),
        follow_redirects=True,
    )


def with_retry(fn, label: str, attempts: int = 5):
    for i in range(1, attempts + 1):
        try:
            return fn()
        except Exception as e:
            print(f"  [retry {i}/{attempts}] {label} failed: {type(e).__name__}: {str(e)[:220]}", flush=True)
            if i == attempts:
                raise
            time.sleep(min(30, 5 * i))


def load_token() -> str:
    env_tok = os.environ.get("HF_TOKEN", "")
    if env_tok:
        return env_tok.strip()
    env_path = Path(os.environ.get("HF_TOKEN_FILE", ".env"))
    if env_path.exists():
        m = re.search(r"hf_[A-Za-z0-9]+", env_path.read_text())
        if m:
            return m.group(0)
    raise RuntimeError("No HF_TOKEN found in environment or .env")


def main():
    set_client_factory(make_proxy_safe_client)
    root = Path(os.environ.get("CODELALA_DIR", Path(__file__).resolve().parent.parent))
    tok = load_token()
    api = HfApi(token=tok)
    user = with_retry(lambda: api.whoami()["name"], "whoami")
    print(f"Authenticated as HF user: {user}", flush=True)

    gguf_repo = f"{user}/CodeLALA-v1-7A1B-GGUF"
    hf_repo = f"{user}/CodeLALA-v1-7A1B"
    banner = root / "hf_cards/codelala-banner.jpg"

    # 1. Prepare GGUF release directory with Q6_K, Q4_K_M, Q8_0
    qbin = root / "src/llama.cpp-olala/build-cuda/bin/llama-quantize"
    bf16 = root / "models/gguf/olala-v4-py-dpo-bf16.gguf"
    q6 = root / "models/gguf/olala-v4-py-dpo-q6_k.gguf"
    stage_gguf = root / "release/CodeLALA-v1-7A1B-GGUF"
    stage_gguf.mkdir(parents=True, exist_ok=True)

    dst_q6 = stage_gguf / "CodeLALA-v1-7A1B-Q6_K.gguf"
    dst_q4 = stage_gguf / "CodeLALA-v1-7A1B-Q4_K_M.gguf"
    dst_q8 = stage_gguf / "CodeLALA-v1-7A1B-Q8_0.gguf"

    if not dst_q6.exists():
        shutil.copy2(q6, dst_q6)
    for dst, qtype in [(dst_q4, "Q4_K_M"), (dst_q8, "Q8_0")]:
        if not dst.exists():
            print(f"Quantizing {qtype} -> {dst.name} ...", flush=True)
            subprocess.run([str(qbin), str(bf16), str(dst), qtype], check=True)

    for p in sorted(stage_gguf.iterdir()):
        print(f"  GGUF staged: {p.name} ({p.stat().st_size / 1e9:.2f} GB)", flush=True)

    def get_remote_sizes(repo_id: str) -> dict[str, int]:
        try:
            info = with_retry(lambda: api.model_info(repo_id=repo_id, files_metadata=True), f"model_info({repo_id})")
            return {s.rfilename: (s.size or 0) for s in (info.siblings or [])}
        except Exception:
            return {}

    # 2. Create & upload GGUF model files FIRST (NEVER touch README.md)
    print(f"\n=== Creating/updating GGUF repo (models only): {gguf_repo} ===", flush=True)
    with_retry(
        lambda: api.create_repo(repo_id=gguf_repo, repo_type="model", exist_ok=True, private=False),
        f"create_repo({gguf_repo})",
    )
    remote_gguf = get_remote_sizes(gguf_repo)
    t0 = time.time()
    gguf_files = ["CodeLALA-v1-7A1B-Q6_K.gguf", "CodeLALA-v1-7A1B-Q4_K_M.gguf", "CodeLALA-v1-7A1B-Q8_0.gguf"]
    for fname in gguf_files:
        fp = stage_gguf / fname
        if not fp.exists():
            continue
        fsize = fp.stat().st_size
        if remote_gguf.get(fname) == fsize:
            print(f"  Skipping {fname} (already uploaded, {fsize / 1e6:.1f} MB)", flush=True)
            continue
        ft0 = time.time()
        print(f"Uploading {fname} ({fsize / 1e6:.1f} MB) to {gguf_repo} ...", flush=True)
        with_retry(
            lambda fp=fp, fname=fname: api.upload_file(
                path_or_fileobj=str(fp),
                path_in_repo=fname,
                repo_id=gguf_repo,
                repo_type="model",
                commit_message=f"Upload {fname} for CodeLALA-v1",
            ),
            f"upload({gguf_repo}/{fname})",
        )
        print(f"  Uploaded {fname} in {time.time() - ft0:.1f}s", flush=True)
    print(f"GGUF_REPO_DONE: https://huggingface.co/{gguf_repo} ({time.time() - t0:.1f}s)", flush=True)

    # 3. Upload full HF safetensors repo (runs/v4_py_dpo/merged) — NEVER touch README.md
    merged = root / "runs/v4_py_dpo/merged"
    print(f"\n=== Creating/updating HF safetensors repo (models only): {hf_repo} ===", flush=True)
    with_retry(
        lambda: api.create_repo(repo_id=hf_repo, repo_type="model", exist_ok=True, private=False),
        f"create_repo({hf_repo})",
    )
    remote_hf = get_remote_sizes(hf_repo)
    t1 = time.time()
    # Upload model/config/tokenizer files one-by-one (skip README.md and already-uploaded files)
    skip_names = {"README.md", "MERGE_INFO.txt", "modeling_olala.py.orig"}
    all_files = sorted(
        [p for p in merged.iterdir() if p.is_file() and p.name not in skip_names and not p.name.endswith(".orig")],
        key=lambda p: p.stat().st_size,
    )
    for fp in all_files:
        fsize = fp.stat().st_size
        if remote_hf.get(fp.name) == fsize:
            print(f"  Skipping {fp.name} (already uploaded, {fsize / 1e6:.1f} MB)", flush=True)
            continue
        ft0 = time.time()
        print(f"Uploading {fp.name} ({fsize / 1e6:.1f} MB) to {hf_repo} ...", flush=True)
        with_retry(
            lambda fp=fp: api.upload_file(
                path_or_fileobj=str(fp),
                path_in_repo=fp.name,
                repo_id=hf_repo,
                repo_type="model",
                commit_message=f"Upload {fp.name} for CodeLALA-v1",
            ),
            f"upload({hf_repo}/{fp.name})",
        )
        print(f"  Uploaded {fp.name} in {time.time() - ft0:.1f}s", flush=True)
    print(f"HF_REPO_DONE: https://huggingface.co/{hf_repo} ({time.time() - t1:.1f}s)", flush=True)
    print("ALL_HF_UPLOADS_COMPLETE!", flush=True)


if __name__ == "__main__":
    main()
