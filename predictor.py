import json
import os
from datetime import datetime
from typing import Dict, List
from pandas.errors import EmptyDataError
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

    # dacă nu există sau e 0 bytes -> scrie direct (cu header)
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        df.to_csv(path, index=False)
        return

    try:
        old = pd.read_csv(path, dtype=str, keep_default_na=False)
    except EmptyDataError:
        # fișier invalid (blank). îl refacem corect.
        df.to_csv(path, index=False)
        return

    merged = pd.concat([old, df], ignore_index=True)
    if "content_hash" in merged.columns:
        merged = merged.drop_duplicates(subset=["content_hash"], keep="last")
    merged.to_csv(path, index=False)


def count_warm_categories(store_dir: str) -> Dict[str, int]:
    gold_path = os.path.join(store_dir, "corrections_gold.csv")
    if not os.path.exists(gold_path) or os.path.getsize(gold_path) == 0:
        return {}

    try:
        df = pd.read_csv(gold_path, dtype=str, keep_default_na=False)
    except EmptyDataError:
        # fișierul are bytes dar nu are coloane (blank / doar newline)
        return {}

    if df.empty:
        return {}

    if "final_category_id" not in df.columns:
        return {}

    return df["final_category_id"].astype(str).value_counts().to_dict()

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
    progress_cb=None,

    
) -> pd.DataFrame:
    def report(pct: float, msg: str) -> None:
        if progress_cb is None:
            return
        try:
            progress_cb(float(pct), str(msg))
        except Exception:
            return

    report(0.02, "Pregătesc predict...")

    ensure_store(store_dir)
    report(0.08, "Încarc store-ul (shards + embeddings)...")

    base_meta, base_emb = load_all_shards(store_dir)
    if base_meta.empty:
        raise RuntimeError("Store gol. Fă ingest înainte de predict.")

    label_col = "CategoryID" if "CategoryID" in base_meta.columns else "Categorie"
    labels = base_meta[label_col].astype(str).tolist()

    # --- Preprocesare (clean_text + hash) ---
    n = len(input_df)
    report(0.15, f"Preprocesez {n} produse (clean_text + content_hash)...")

    rows = []
    for i, (_, r) in enumerate(input_df.iterrows()):
        d = r.to_dict()
        clean = build_clean_text(d, text_cols)
        d["clean_text"] = clean
        d["content_hash"] = content_hash(clean)
        rows.append(d)

        if i % 200 == 0 or i == n - 1:
            # 15% -> 30%
            report(0.15 + 0.15 * (i + 1) / max(n, 1), f"Preprocesez: {i+1}/{n}")

    qdf = pd.DataFrame(rows)

    # --- Embeddings ---
    report(0.30, f"Generez embeddings pentru {len(qdf)} produse...")

    def embed_progress_cb(*args):
        # acceptă mai multe forme (done,total) sau (pct,msg)
        try:
            if len(args) >= 2 and isinstance(args[0], (int, float)) and isinstance(args[1], (int, float)):
                done = float(args[0])
                total = float(args[1])
                pct = 0.30 + 0.45 * (done / max(total, 1.0))  # 30% -> 75%
                report(pct, f"Embeddings: {int(done)}/{int(total)}")
            elif len(args) >= 1 and isinstance(args[0], (int, float)):
                pct0 = float(args[0])
                pct = 0.30 + 0.45 * max(0.0, min(1.0, pct0))
                report(pct, "Generez embeddings...")
        except Exception:
            pass

    # dacă embed_texts_batched suportă progress_cb, îl folosim; dacă nu, merge oricum
    try:
        q_emb = embed_texts_batched(
            qdf["clean_text"].tolist(),
            embedding_model,
            ollama_url,
            workers=workers,
            progress_cb=embed_progress_cb,
        )
    except TypeError:
        q_emb = embed_texts_batched(
            qdf["clean_text"].tolist(),
            embedding_model,
            ollama_url,
            workers=workers,
        )

    report(0.76, "Calculez similarități și scoruri (topK)...")

    qn = _normalize(q_emb)
    bn = _normalize(base_emb.astype(np.float32))
    sims = qn @ bn.T

    warm_counts = count_warm_categories(store_dir)

    # --- Scorare / topK ---
    m = len(qdf)
    out_rows = []
    for i in range(m):
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

        if i % 200 == 0 or i == m - 1:
            # 76% -> 97%
            report(0.76 + 0.21 * (i + 1) / max(m, 1), f"Scorare: {i+1}/{m}")

    out = pd.DataFrame(out_rows)

    report(0.98, "Salvez pseudo_labels și review_queue...")

    pseudo = out[~out["needs_review"]].copy()
    pseudo["final_category_id"] = pseudo["predicted_category_id"]
    pseudo["final_category_text"] = pseudo["predicted_category_text"]

    review = out[out["needs_review"]].copy()

    _append_csv(os.path.join(store_dir, "pseudo_labels.csv"), pseudo)
    _append_csv(os.path.join(store_dir, "review_queue.csv"), review)

    report(1.0, "Predict finalizat.")
    return out