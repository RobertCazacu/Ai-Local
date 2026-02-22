import argparse
import json
import os
from datetime import datetime

import numpy as np
import pandas as pd

from store import ensure_store, load_all_shards


def build_centroids(store_dir: str) -> str:
    ensure_store(store_dir)
    meta, emb = load_all_shards(store_dir)
    if meta.empty:
        raise RuntimeError("Store gol")
    label_col = "CategoryID" if "CategoryID" in meta.columns else "Categorie"
    groups = {}
    for i, cid in enumerate(meta[label_col].astype(str).tolist()):
        groups.setdefault(cid, []).append(emb[i])

    cids = sorted(groups.keys())
    cent = np.vstack([np.mean(np.vstack(groups[c]), axis=0) for c in cids]).astype(np.float32)
    counts = np.array([len(groups[c]) for c in cids], dtype=np.int32)
    out = os.path.join(store_dir, "centroids.npz")
    np.savez(out, category_ids=np.array(cids), centroids=cent, counts=counts)
    return out


def save_model_version(store_dir: str, artifact_path: str) -> str:
    models = os.path.join(store_dir, "models")
    os.makedirs(models, exist_ok=True)
    existing = sorted([x for x in os.listdir(models) if x.startswith("model_v")])
    vid = len(existing) + 1
    version = f"model_v{vid:03d}"
    dst = os.path.join(models, f"{version}.npz")
    with open(artifact_path, "rb") as src, open(dst, "wb") as out:
        out.write(src.read())
    meta = {"version": version, "created_at": datetime.utcnow().isoformat(), "artifact": dst}
    with open(os.path.join(models, "current_model.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return dst


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--store", default="data_store")
    p.add_argument("--mode", choices=["full", "centroids_only"], default="full")
    args = p.parse_args()

    art = build_centroids(args.store)
    versioned = save_model_version(args.store, art)
    print(f"OK retrain {args.mode}: {versioned}")


if __name__ == "__main__":
    main()
