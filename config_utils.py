import json
import os
from copy import deepcopy
from typing import Any, Dict

DEFAULT_CONFIG: Dict[str, Any] = {
    "out_dir": "model_trendyol",
    "embed_model": "nomic-embed-text",
    "workers": 4,
    "k": 5,
    "min_conf": 0.55,
    "OLLAMA_URL": "http://localhost:11434",
    "embedding_model": "nomic-embed-text",
    "store_dir": "data_store",
    "text_cols": ["Nume", "Brand", "Descriere"],
    "label_col": "CategoryID",
    "AUTO_ACCEPT_CONF": 0.93,
    "MIN_MARGIN": 0.05,
    "MIN_GOLD_PER_CAT": 10,
    "topK": 5,
    "ui_port": 8501,
}


def load_config(path: str = "config.json") -> Dict[str, Any]:
    cfg = deepcopy(DEFAULT_CONFIG)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            existing = json.load(f)
        cfg.update(existing)
    return cfg


def save_config(cfg: Dict[str, Any], path: str = "config.json") -> None:
    merged = deepcopy(DEFAULT_CONFIG)
    merged.update(cfg)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
