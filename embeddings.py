import hashlib
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, List, Optional

import numpy as np
import requests


def normalize_text(s: str, max_len: int = 4000) -> str:
    s = str(s)
    s = re.sub(r"<[^>]+>", " ", s)
    s = s.lower().replace("_", " ").replace("-", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s[:max_len]


def build_clean_text(row: dict, text_cols: List[str]) -> str:
    return " | ".join(normalize_text(row.get(c, "")) for c in text_cols)


def content_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def embed_one(text: str, model: str, ollama_url: str, timeout: int = 180, retries: int = 3) -> np.ndarray:
    payload = {"model": model, "prompt": text}
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            r = requests.post(f"{ollama_url}/api/embeddings", json=payload, timeout=timeout)
            r.raise_for_status()
            return np.array(r.json()["embedding"], dtype=np.float32)
        except Exception as e:
            last_err = e
            time.sleep(attempt)
    raise RuntimeError("Embedding failed") from last_err


def embed_texts_batched(
    texts: List[str],
    model: str,
    ollama_url: str,
    workers: int = 4,
    progress_cb: Optional[Callable[[int, int], None]] = None,
) -> np.ndarray:
    total = len(texts)
    out: List[Optional[np.ndarray]] = [None] * total
    done = 0
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        futures = {ex.submit(embed_one, texts[i], model, ollama_url): i for i in range(total)}
        for fut in as_completed(futures):
            i = futures[fut]
            out[i] = fut.result()
            done += 1
            if progress_cb:
                progress_cb(done, total)
    return np.vstack([x for x in out if x is not None])
