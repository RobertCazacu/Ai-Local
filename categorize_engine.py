import argparse
import hashlib
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

OLLAMA_URL = "http://localhost:11434"


# -----------------------------
# Helpers
# -----------------------------
def check_ollama() -> None:
    r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=10)
    r.raise_for_status()


def clean_text(s: str, max_len: int) -> str:
    s = str(s)
    # scoate HTML
    s = re.sub(r"<[^>]+>", " ", s)
    # normalizează spații
    s = re.sub(r"\s+", " ", s).strip()
    # taie
    if len(s) > max_len:
        s = s[:max_len]
    return s


def ollama_embed_one(text: str, model: str, timeout: int = 180, retries: int = 3) -> np.ndarray:
    payload = {"model": model, "prompt": text}

    last_err: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            r = requests.post(f"{OLLAMA_URL}/api/embeddings", json=payload, timeout=timeout)
            if r.status_code >= 500:
                raise RuntimeError(f"Ollama 5xx: {r.status_code} {r.text[:200]}")
            r.raise_for_status()
            return np.array(r.json()["embedding"], dtype=np.float32)
        except Exception as e:
            last_err = e
            time.sleep(1.0 * attempt)

    raise last_err  # type: ignore


def embed_texts(
    texts: List[str],
    model: str,
    workers: int = 2,
    progress_cb: Optional[Callable[[int, int, str], None]] = None,
    phase: str = "embeddings",
    row_info: Optional[List[str]] = None,
    cache_dir: Optional[str] = None,
) -> np.ndarray:
    total = len(texts)
    vecs: List[Optional[np.ndarray]] = [None] * total
    done = 0

    def cache_path_for(i: int) -> Optional[str]:
        if not cache_dir:
            return None
        key = hashlib.sha1(f"{model}\n{texts[i]}".encode("utf-8")).hexdigest()
        return os.path.join(cache_dir, f"{key}.npy")

    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        for i in range(total):
            cpath = cache_path_for(i)
            if not cpath:
                continue
            if os.path.exists(cpath):
                vecs[i] = np.load(cpath).astype(np.float32)
                done += 1

    if progress_cb:
        progress_cb(done, total, phase)

    missing_idxs = [i for i, v in enumerate(vecs) if v is None]

    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        futures = {ex.submit(ollama_embed_one, texts[i], model): i for i in missing_idxs}
        for fut in as_completed(futures):
            i = futures[fut]
            try:
                emb = fut.result()
                vecs[i] = emb

                cpath = cache_path_for(i)
                if cpath:
                    tmp_path = f"{cpath}.tmp"
                    with open(tmp_path, "wb") as tf:
                        np.save(tf, emb)
                    os.replace(tmp_path, cpath)
            except Exception as e:
                info = row_info[i] if row_info and i < len(row_info) else f"index={i}"
                preview = texts[i][:200].replace("\n", " ")
                raise RuntimeError(f"Embedding a picat la {info}. Preview: {preview}") from e

            done += 1
            if progress_cb:
                progress_cb(done, total, phase)

    if any(v is None for v in vecs):
        raise RuntimeError("Nu toate embedding-urile au fost calculate.")

    return np.vstack([v for v in vecs if v is not None])


def normalize(mat: np.ndarray) -> np.ndarray:
    if mat.size == 0:
        raise ValueError("Nu pot normaliza o matrice goală.")
    n = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12
    return mat / n


def build_text(row: pd.Series, text_cols: List[str]) -> str:
    parts: List[str] = []
    for c in text_cols:
        v = row.get(c, "")
        if pd.isna(v):
            v = ""
        v = str(v).strip()
        if not v:
            continue

        # descrierea e cea mai “periculoasă” -> o curățăm și o limităm
        if c.lower() in ["descriere", "description", "desc"]:
            v = clean_text(v, max_len=800)
        else:
            v = clean_text(v, max_len=200)

        if v:
            parts.append(f"{c}: {v}")

    return " | ".join(parts)


def detect_id_col(df: pd.DataFrame) -> Optional[str]:
    for cand in ["SKU", "sku", "Id", "ID", "id", "Cod", "cod", "ProductID", "product_id"]:
        if cand in df.columns:
            return cand
    return None


# -----------------------------
# Build (train)
# -----------------------------
def build_index(
    labeled_path: str,
    out_dir: str,
    text_cols: List[str],
    label_col: str,
    embed_model: str = "nomic-embed-text",
    workers: int = 2,
    progress_cb: Optional[Callable[[int, int, str], None]] = None,
) -> None:
    check_ollama()
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_excel(labeled_path).reset_index(drop=True)
    df["__excel_row__"] = df.index + 2  # header=1, data starts at row 2

    missing = [c for c in text_cols + [label_col] if c not in df.columns]
    if missing:
        raise ValueError(f"Lipsesc coloane în labeled.xlsx: {missing}")

    id_col = detect_id_col(df)

    # păstrăm doar rândurile cu label valid
    df = df.dropna(subset=[label_col]).copy()
    df[label_col] = df[label_col].astype(str).str.strip()
    df = df[df[label_col] != ""].reset_index(drop=True)

    if df.empty:
        raise ValueError("Nu există rânduri valide cu etichetă după filtrare.")

    # construim text + row_info (aliniat 1:1 cu texts)
    df["__text__"] = df.apply(lambda r: build_text(r, text_cols), axis=1)

    if id_col:
        row_info = [
            f"ExcelRow={r} ({id_col}={v})"
            for r, v in zip(df["__excel_row__"].tolist(), df[id_col].astype(str).tolist())
        ]
    else:
        row_info = [f"ExcelRow={r}" for r in df["__excel_row__"].tolist()]

    emb = embed_texts(
        df["__text__"].tolist(),
        embed_model,
        workers=workers,
        progress_cb=progress_cb,
        phase="build_embeddings",
        row_info=row_info,
        cache_dir=os.path.join(out_dir, "build_cache"),
    )
    emb = normalize(emb)

    np.save(os.path.join(out_dir, "embeddings.npy"), emb)
    meta = df.drop(columns=["__text__"]).reset_index(drop=True)
    meta.to_csv(os.path.join(out_dir, "meta.csv"), index=False, encoding="utf-8")

    cfg = {
        "text_cols": text_cols,
        "label_col": label_col,
        "embed_model": embed_model,
        "dim": int(emb.shape[1]),
        "rows": int(emb.shape[0]),
    }
    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)


# -----------------------------
# Predict
# -----------------------------
def predict(
    input_path: str,
    out_dir: str,
    output_path: str,
    k: int = 15,
    min_conf: float = 0.55,
    workers: int = 2,
    progress_cb: Optional[Callable[[int, int, str], None]] = None,
) -> None:
    check_ollama()

    cfg_path = os.path.join(out_dir, "config.json")
    emb_path = os.path.join(out_dir, "embeddings.npy")
    meta_path = os.path.join(out_dir, "meta.csv")

    if not (os.path.exists(cfg_path) and os.path.exists(emb_path) and os.path.exists(meta_path)):
        raise RuntimeError("Nu găsesc indexul. Rulează întâi build.")

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    text_cols = cfg["text_cols"]
    label_col = cfg["label_col"]
    embed_model = cfg["embed_model"]

    base_emb = np.load(emb_path).astype(np.float32)  # (N, D) normalizat
    meta = pd.read_csv(meta_path, dtype=str, keep_default_na=False)
    labels = meta[label_col].astype(str).tolist()

    df = pd.read_excel(input_path).reset_index(drop=True)
    df["__excel_row__"] = df.index + 2
    id_col = detect_id_col(df)

    missing = [c for c in text_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Lipsesc coloane în new.xlsx: {missing}")

    df["__text__"] = df.apply(lambda r: build_text(r, text_cols), axis=1)

    if id_col:
        row_info = [
            f"ExcelRow={r} ({id_col}={v})"
            for r, v in zip(df["__excel_row__"].tolist(), df[id_col].astype(str).tolist())
        ]
    else:
        row_info = [f"ExcelRow={r}" for r in df["__excel_row__"].tolist()]

    q_emb = embed_texts(
    df["__text__"].tolist(),
    embed_model,
    workers=workers,
    progress_cb=progress_cb,
    phase="predict_embeddings",
    row_info=row_info,
    cache_dir=os.path.join(out_dir, "predict_cache"),
    )
    q_emb = normalize(q_emb)

    sims_all = q_emb @ base_emb.T  # (Q, N) cosine similarity

    pred_labels = []
    pred_conf = []
    top_matches = []

    N = base_emb.shape[0]
    if N == 0:
        raise RuntimeError("Indexul este gol. Rulează din nou build cu date valide.")

    k_eff = min(k, N)
    if k_eff < 1:
        raise ValueError("Parametrul k trebuie să fie cel puțin 1.")

    for i in range(sims_all.shape[0]):
        sims = sims_all[i]
        idxs = np.argpartition(-sims, k_eff - 1)[:k_eff]
        idxs = idxs[np.argsort(-sims[idxs])]
        top_sims = sims[idxs]

        score_by_cat: Dict[str, float] = {}
        pairs: List[Tuple[str, float]] = []

        for idx, sim in zip(idxs, top_sims):
            cat = labels[int(idx)]
            score_by_cat[cat] = score_by_cat.get(cat, 0.0) + float(sim)
            pairs.append((cat, float(sim)))

        sorted_cats = sorted(score_by_cat.items(), key=lambda x: x[1], reverse=True)
        best_cat, best_score = sorted_cats[0]
        denom = sum(v for _, v in sorted_cats[: min(len(sorted_cats), 10)]) + 1e-12
        conf = float(best_score / denom)

        pred_labels.append(best_cat)
        pred_conf.append(conf)

        pairs_sorted = sorted(pairs, key=lambda x: x[1], reverse=True)[:5]
        top_matches.append(" | ".join([f"{c}:{sim:.3f}" for c, sim in pairs_sorted]))

    out = df.drop(columns=["__text__"]).copy()
    out["Categorie_propusa"] = pred_labels
    out["Scor_incredere"] = pred_conf
    out["Top_matchuri"] = top_matches
    out["Necesita_verificare"] = out["Scor_incredere"] < float(min_conf)
    out.to_excel(output_path, index=False)


# -----------------------------
# CLI
# -----------------------------
def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build")
    b.add_argument("--labeled", required=True)
    b.add_argument("--out_dir", required=True)
    b.add_argument("--text_cols", required=True)
    b.add_argument("--label_col", required=True)
    b.add_argument("--embed_model", default="nomic-embed-text")
    b.add_argument("--workers", type=int, default=2)

    pr = sub.add_parser("predict")
    pr.add_argument("--input", required=True)
    pr.add_argument("--out_dir", required=True)
    pr.add_argument("--output", required=True)
    pr.add_argument("--k", type=int, default=15)
    pr.add_argument("--min_conf", type=float, default=0.55)
    pr.add_argument("--workers", type=int, default=2)

    args = p.parse_args()

    if args.cmd == "build":
        cols = [c.strip() for c in args.text_cols.split(",") if c.strip()]
        build_index(args.labeled, args.out_dir, cols, args.label_col, args.embed_model, args.workers)
        print("OK: build.")

    if args.cmd == "predict":
        predict(args.input, args.out_dir, args.output, args.k, args.min_conf, args.workers)
        print("OK: predict.")


if __name__ == "__main__":
    main()
