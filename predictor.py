import json
import os
from datetime import datetime
from typing import Dict, List

import numpy as np
import pandas as pd

from embeddings import build_clean_text, content_hash, embed_texts_batched
from store import ensure_store, load_all_shards


def _softmax(x: np.ndarray) -> np.ndarray:
    z = x - np.max(x)
    e = np.exp(z)
    return e / (np.sum(e) + 1e-12)


def _normalize(mat: np.ndarray) -> np.ndarray:
    if mat.size == 0:
        return mat
    n = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12
    return mat / n


def _append_csv(path: str, df: pd.DataFrame) -> None:
    if df.empty:
        return
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        df.to_csv(path, index=False)
    else:
        old = pd.read_csv(path, dtype=str, keep_default_na=False)
        merged = pd.concat([old, df], ignore_index=True)
        if "content_hash" in merged.columns:
            merged = merged.drop_duplicates(subset=["content_hash"], keep="last")
        merged.to_csv(path, index=False)


def count_warm_categories(store_dir: str) -> Dict[str, int]:
    gold_path = os.path.join(store_dir, "corrections_gold.csv")
    if not os.path.exists(gold_path) or os.path.getsize(gold_path) == 0:
        return {}
    df = pd.read_csv(gold_path, dtype=str, keep_default_na=False)
    if "final_category_id" not in df.columns:
        return {}
    return df["final_category_id"].value_counts().to_dict()


def run_predict(
    input_df: pd.DataFrame,
    store_dir: str,
    text_cols: List[str],
    embedding_model: str,
    ollama_url: str,
    id_to_text: Dict[str, str],
    auto_accept_conf: float = 0.93,
    min_margin: float = 0.05,
    min_gold_per_cat: int = 10,
    topk: int = 5,
    workers: int = 4,
) -> pd.DataFrame:
    ensure_store(store_dir)
    base_meta, base_emb = load_all_shards(store_dir)
    if base_meta.empty:
        raise RuntimeError("Store gol. Fă ingest înainte de predict.")

    label_col = "CategoryID" if "CategoryID" in base_meta.columns else "Categorie"
    labels = base_meta[label_col].astype(str).tolist()

    rows = []
    for _, r in input_df.iterrows():
        d = r.to_dict()
        clean = build_clean_text(d, text_cols)
        d["clean_text"] = clean
        d["content_hash"] = content_hash(clean)
        rows.append(d)
    qdf = pd.DataFrame(rows)

    q_emb = embed_texts_batched(qdf["clean_text"].tolist(), embedding_model, ollama_url, workers=workers)
    qn = _normalize(q_emb)
    bn = _normalize(base_emb.astype(np.float32))
    sims = qn @ bn.T

    warm_counts = count_warm_categories(store_dir)
    out_rows = []
    for i in range(len(qdf)):
        row = qdf.iloc[i].to_dict()
        s = sims[i]
        k = min(topk, len(labels))
        idx = np.argpartition(-s, k - 1)[:k]
        idx = idx[np.argsort(-s[idx])]

        cat_scores: Dict[str, float] = {}
        top_items = []
        for j in idx:
            cid = str(labels[int(j)])
            sc = float(s[int(j)])
            cat_scores[cid] = max(cat_scores.get(cid, -1e9), sc)
            top_items.append({"category_id": cid, "score": sc, "category_text": id_to_text.get(cid, cid)})

        ranked = sorted(cat_scores.items(), key=lambda x: x[1], reverse=True)
        top1_id, top1_score = ranked[0]
        top2_score = ranked[1][1] if len(ranked) > 1 else top1_score - 1e-9
        margin = float(top1_score - top2_score)
        probs = _softmax(np.array([x[1] for x in ranked[:topk]], dtype=np.float32))
        conf = float(probs[0])

        warm_ok = warm_counts.get(top1_id, 0) >= int(min_gold_per_cat)
        validator_ok = top1_id in id_to_text
        auto = conf >= auto_accept_conf and margin >= min_margin and warm_ok and validator_ok

        row.update(
            {
                "predicted_category_id": top1_id,
                "predicted_category_text": id_to_text.get(top1_id, top1_id),
                "topk_candidates": json.dumps(top_items, ensure_ascii=False),
                "confidence": conf,
                "margin": margin,
                "needs_review": not auto,
                "source": "auto" if auto else "review",
                "created_at": datetime.utcnow().isoformat(),
            }
        )
        out_rows.append(row)

    out = pd.DataFrame(out_rows)

    pseudo = out[~out["needs_review"]].copy()
    pseudo["final_category_id"] = pseudo["predicted_category_id"]
    pseudo["final_category_text"] = pseudo["predicted_category_text"]

    review = out[out["needs_review"]].copy()

    _append_csv(os.path.join(store_dir, "pseudo_labels.csv"), pseudo)
    _append_csv(os.path.join(store_dir, "review_queue.csv"), review)
    return out
