import json
import os
from typing import Dict, Tuple

import pandas as pd

from embeddings import normalize_text


def load_catalog(path: str) -> pd.DataFrame:
    if path.lower().endswith(".xlsx"):
        return pd.read_excel(path)
    return pd.read_csv(path)


def build_catalog_mappings(df: pd.DataFrame, id_col: str = "CategoryID", text_col: str = "Categoria Text", path_col: str = "CategoryPath") -> Tuple[Dict[str, str], Dict[str, str], pd.DataFrame]:
    data = df.copy()
    data[id_col] = data[id_col].astype(str)
    data[text_col] = data[text_col].astype(str)

    dupes = data[data.duplicated(subset=[text_col], keep=False)].copy()
    if path_col in data.columns:
        data[text_col] = data.apply(lambda r: str(r[path_col]) if data[text_col].duplicated(keep=False).loc[r.name] else str(r[text_col]), axis=1)

    id_to_text = dict(zip(data[id_col], data[text_col]))
    text_to_id = {normalize_text(v): k for k, v in id_to_text.items()}
    return id_to_text, text_to_id, dupes


def load_overrides(store_dir: str) -> Dict[str, str]:
    path = os.path.join(store_dir, "catalog_overrides.json")
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_overrides(store_dir: str, overrides: Dict[str, str]) -> None:
    path = os.path.join(store_dir, "catalog_overrides.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(overrides, f, ensure_ascii=False, indent=2)
