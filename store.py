import json
import os
import sqlite3
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from embeddings import build_clean_text, content_hash, embed_texts_batched



PERSISTENT_LABEL_COLUMNS = [
    "SKU",
    "Nume",
    "Brand",
    "Descriere",
    "clean_text",
    "content_hash",
    "predicted_category_id",
    "predicted_category_text",
    "topk_candidates",
    "confidence",
    "margin",
    "needs_review",
    "final_category_id",
    "final_category_text",
    "source",
    "created_at",
    "updated_at",
]

def now_iso() -> str:
    return datetime.utcnow().isoformat()

def _is_blank_csv(path: str) -> bool:
    """
    True dacă fișierul nu există, are 0 bytes sau conține doar whitespace/newlines.
    """
    if not os.path.exists(path):
        return True
    if os.path.getsize(path) == 0:
        return True
    try:
        with open(path, "r", encoding="utf-8") as f:
            chunk = f.read(4096)
        return chunk.strip() == ""
    except Exception:
        # dacă nu îl putem citi, nu riscăm să-l suprascriem
        return False


def _write_header_csv(path: str, columns: List[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pd.DataFrame(columns=columns).to_csv(path, index=False)

def ensure_store(store_dir: str) -> None:
    os.makedirs(store_dir, exist_ok=True)
    os.makedirs(os.path.join(store_dir, "shards"), exist_ok=True)
    os.makedirs(os.path.join(store_dir, "logs"), exist_ok=True)
    os.makedirs(os.path.join(store_dir, "exports"), exist_ok=True)

    manifest_path = os.path.join(store_dir, "manifest.json")
    if not os.path.exists(manifest_path):
        manifest = {
            "store_version": 1,
            "embedding_model": "",
            "embedding_dim": 0,
            "created_at": now_iso(),
            "updated_at": now_iso(),
            "shards": [],
        }
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)

    conn = sqlite3.connect(os.path.join(store_dir, "hash_index.sqlite"))
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS embeddings_cache (
          content_hash TEXT PRIMARY KEY,
          shard_id INTEGER NOT NULL,
          row_idx INTEGER NOT NULL,
          sku TEXT,
          created_at TEXT NOT NULL
        )
        """
    )
    cur.execute("CREATE INDEX IF NOT EXISTS idx_sku ON embeddings_cache(sku)")
    conn.commit()
    conn.close()

    for fname in ["corrections_gold.csv", "pseudo_labels.csv", "review_queue.csv"]:
        path = os.path.join(store_dir, fname)
    if _is_blank_csv(path):
        _write_header_csv(path, PERSISTENT_LABEL_COLUMNS)


def _load_manifest(store_dir: str) -> Dict:
    with open(os.path.join(store_dir, "manifest.json"), "r", encoding="utf-8") as f:
        return json.load(f)


def _save_manifest(store_dir: str, manifest: Dict) -> None:
    manifest["updated_at"] = now_iso()
    with open(os.path.join(store_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


def _next_shard_id(manifest: Dict) -> int:
    if not manifest["shards"]:
        return 1
    return max(s["id"] for s in manifest["shards"]) + 1


def init_store_from_legacy(store_dir: str, legacy_meta: str, legacy_emb: str, text_cols: List[str], embedding_model: str) -> Dict:
    ensure_store(store_dir)
    manifest = _load_manifest(store_dir)
    if manifest["shards"]:
        raise RuntimeError("Store deja inițializat.")

    meta = pd.read_csv(legacy_meta, dtype=str, keep_default_na=False)
    emb = np.load(legacy_emb)
    shard_id = 1
    meta_path = os.path.join(store_dir, "shards", f"meta_{shard_id:05d}.csv")
    emb_path = os.path.join(store_dir, "shards", f"emb_{shard_id:05d}.npy")

    clean_texts = []
    hashes = []
    for _, row in meta.iterrows():
        txt = build_clean_text(row.to_dict(), text_cols)
        clean_texts.append(txt)
        hashes.append(content_hash(txt))
    meta = meta.copy()
    meta["clean_text"] = clean_texts
    meta["content_hash"] = hashes
    meta.to_csv(meta_path, index=False)
    np.save(emb_path, emb.astype(np.float16))

    conn = sqlite3.connect(os.path.join(store_dir, "hash_index.sqlite"))
    cur = conn.cursor()
    for i, h in enumerate(hashes):
        sku = meta.iloc[i].get("SKU", "") if "SKU" in meta.columns else ""
        cur.execute(
            "INSERT OR IGNORE INTO embeddings_cache(content_hash, shard_id, row_idx, sku, created_at) VALUES(?,?,?,?,?)",
            (h, shard_id, i, sku, now_iso()),
        )
    conn.commit()
    conn.close()

    manifest["embedding_model"] = embedding_model
    manifest["embedding_dim"] = int(emb.shape[1]) if emb.ndim == 2 and emb.shape[0] else 0
    manifest["shards"].append({"id": shard_id, "meta_path": meta_path, "emb_path": emb_path, "rows": int(len(meta)), "created_at": now_iso()})
    _save_manifest(store_dir, manifest)
    return {"rows": len(meta), "shard_id": shard_id}


def ingest_file_incremental(
    df: pd.DataFrame,
    store_dir: str,
    text_cols: List[str],
    embedding_model: str,
    ollama_url: str,
    workers: int = 4,
    sku_col: str = "SKU",
    progress_cb=None,
) -> Dict:
    ensure_store(store_dir)
    manifest = _load_manifest(store_dir)
    shard_id = _next_shard_id(manifest)

    records = []
    for _, row in df.iterrows():
        rec = row.to_dict()
        clean = build_clean_text(rec, text_cols)
        rec["clean_text"] = clean
        rec["content_hash"] = content_hash(clean)
        records.append(rec)
    work_df = pd.DataFrame(records)

    conn = sqlite3.connect(os.path.join(store_dir, "hash_index.sqlite"))
    cur = conn.cursor()
    exists = set()
    for h in work_df["content_hash"].tolist():
        cur.execute("SELECT content_hash FROM embeddings_cache WHERE content_hash=?", (h,))
        if cur.fetchone():
            exists.add(h)

    new_df = work_df[~work_df["content_hash"].isin(exists)].reset_index(drop=True)
    dup_count = len(work_df) - len(new_df)

    if len(new_df) == 0:
        conn.close()
        return {"total": len(work_df), "new": 0, "duplicates": dup_count, "errors": 0, "shard_id": None}

    texts = new_df["clean_text"].astype(str).tolist()
    emb = embed_texts_batched(texts, embedding_model, ollama_url, workers=workers, progress_cb=progress_cb)
    emb = emb.astype(np.float16)

    meta_path = os.path.join(store_dir, "shards", f"meta_{shard_id:05d}.csv")
    emb_path = os.path.join(store_dir, "shards", f"emb_{shard_id:05d}.npy")
    new_df.to_csv(meta_path, index=False)
    np.save(emb_path, emb)

    sku_values = new_df[sku_col].astype(str).tolist() if sku_col in new_df.columns else [""] * len(new_df)
    for i, (h, sku) in enumerate(zip(new_df["content_hash"].tolist(), sku_values)):
        cur.execute(
            "INSERT OR IGNORE INTO embeddings_cache(content_hash, shard_id, row_idx, sku, created_at) VALUES(?,?,?,?,?)",
            (h, shard_id, i, sku, now_iso()),
        )
    conn.commit()
    conn.close()

    manifest["embedding_model"] = embedding_model or manifest.get("embedding_model", "")
    manifest["embedding_dim"] = int(emb.shape[1]) if emb.ndim == 2 else manifest.get("embedding_dim", 0)
    manifest["shards"].append({"id": shard_id, "meta_path": meta_path, "emb_path": emb_path, "rows": int(len(new_df)), "created_at": now_iso()})
    _save_manifest(store_dir, manifest)
    return {"total": len(work_df), "new": len(new_df), "duplicates": dup_count, "errors": 0, "shard_id": shard_id}


def load_all_shards(store_dir: str) -> Tuple[pd.DataFrame, np.ndarray]:
    ensure_store(store_dir)
    manifest = _load_manifest(store_dir)
    metas, embs = [], []
    for shard in manifest["shards"]:
        if os.path.exists(shard["meta_path"]) and os.path.exists(shard["emb_path"]):
            metas.append(pd.read_csv(shard["meta_path"], dtype=str, keep_default_na=False))
            embs.append(np.load(shard["emb_path"]).astype(np.float32))
    if not metas:
        return pd.DataFrame(), np.zeros((0, 1), dtype=np.float32)
    return pd.concat(metas, ignore_index=True), np.vstack(embs)
