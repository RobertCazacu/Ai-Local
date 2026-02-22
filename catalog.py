import json
import os
import re
from datetime import datetime
from typing import Dict, Optional, Tuple

import pandas as pd

from embeddings import normalize_text


ID_CANDIDATES = ["CategoryID", "category_id", "categoryId", "id"]
NAME_CANDIDATES = ["CategoryName", "category_name", "name"]
TEXT_CANDIDATES = ["CategoryText", "category_text", "Categoria Text", "category", "Categorie"]


def parse_category_from_filename(filename: str) -> Optional[Tuple[str, str]]:
    base = os.path.basename(str(filename)).strip()
    m = re.match(r"^\s*(\d+)\s+(.+?)\s*\.(xlsx|xls)\s*$", base, flags=re.IGNORECASE)
    if not m:
        return None
    cid = m.group(1).strip()
    name = re.sub(r"\s+", " ", m.group(2)).strip()
    if not cid or not name:
        return None
    return cid, name


def load_catalog(path: str) -> pd.DataFrame:
    if path.lower().endswith(".xlsx") or path.lower().endswith(".xls"):
        return pd.read_excel(path)
    return pd.read_csv(path)


def _pick_col(df: pd.DataFrame, candidates):
    existing = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in existing:
            return existing[cand.lower()]
    return None


def build_catalog_mappings(df: pd.DataFrame, id_col: str = "CategoryID", text_col: str = "Categoria Text", path_col: str = "CategoryPath") -> Tuple[Dict[str, str], Dict[str, str], pd.DataFrame]:
    data = df.copy()

    id_detected = _pick_col(data, [id_col] + ID_CANDIDATES)
    name_detected = _pick_col(data, NAME_CANDIDATES)
    text_detected = _pick_col(data, [text_col] + TEXT_CANDIDATES)

    if not id_detected:
        raise ValueError(f"Nu găsesc coloană ID. Disponibile: {list(data.columns)}")

    if not text_detected and not name_detected:
        raise ValueError(f"Nu găsesc coloană text/nume categorie. Disponibile: {list(data.columns)}")

    data["CategoryID"] = data[id_detected].astype(str).str.strip()

    if text_detected:
        data["CategoryText"] = data[text_detected].astype(str).str.strip()
    else:
        data["CategoryText"] = data[name_detected].astype(str).str.strip()  # type: ignore[index]

    if name_detected:
        data["CategoryName"] = data[name_detected].astype(str).str.strip()
    else:
        data["CategoryName"] = data["CategoryText"]

    dupes = data[data.duplicated(subset=["CategoryName"], keep=False)].copy()
    if path_col in data.columns:
        dup_mask = data["CategoryName"].str.lower().str.strip().duplicated(keep=False)
        data.loc[dup_mask, "CategoryText"] = data.loc[dup_mask, path_col].astype(str).str.strip()

    data = data[data["CategoryID"] != ""].copy()
    id_to_text = dict(zip(data["CategoryID"], data["CategoryText"]))
    text_to_id = {normalize_text(v): k for k, v in id_to_text.items()}
    return id_to_text, text_to_id, dupes


def build_catalog_from_folder(folder_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not os.path.isdir(folder_path):
        raise ValueError(f"Folder invalid/inexistent: {folder_path}")

    rows = []
    for fname in sorted(os.listdir(folder_path)):
        full = os.path.join(folder_path, fname)
        if not os.path.isfile(full):
            continue
        if not (fname.lower().endswith(".xlsx") or fname.lower().endswith(".xls")):
            continue
        parsed = parse_category_from_filename(fname)
        if not parsed:
            continue
        cid, name = parsed
        rows.append(
            {
                "CategoryID": str(cid),
                "CategoryName": str(name),
                "CategoryText": f"{cid} {name}".strip(),
                "FileName": fname,
                "FilePath": full,
            }
        )

    df_catalog = pd.DataFrame(rows)
    if df_catalog.empty:
        return df_catalog, pd.DataFrame(columns=["CategoryNameNorm", "count"])

    name_norm = df_catalog["CategoryName"].astype(str).str.lower().str.strip()
    counts = name_norm.value_counts().reset_index()
    counts.columns = ["CategoryNameNorm", "count"]
    df_dupes = counts[counts["count"] > 1].copy()
    return df_catalog, df_dupes


def save_catalog_outputs(df_catalog, df_dupes, out_dir="data_store/catalog", versioned=True) -> Dict[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    cat_name = f"catalog_normalized_{ts}.xlsx" if versioned else "catalog_normalized.xlsx"
    dup_name = f"duplicates_{ts}.xlsx" if versioned else "duplicates.xlsx"

    catalog_path = os.path.join(out_dir, cat_name)
    id_to_text_path = os.path.join(out_dir, "id_to_text.json")
    text_to_id_path = os.path.join(out_dir, "text_to_id.json")
    dup_path = os.path.join(out_dir, dup_name)

    df_catalog.to_excel(catalog_path, index=False)

    id_to_text = {
        str(r["CategoryID"]): str(r.get("CategoryText", "")).strip()
        for _, r in df_catalog.iterrows()
        if str(r.get("CategoryID", "")).strip()
    }
    text_to_id = {normalize_text(v): k for k, v in id_to_text.items()}

    with open(id_to_text_path, "w", encoding="utf-8") as f:
        json.dump(id_to_text, f, ensure_ascii=False, indent=2)
    with open(text_to_id_path, "w", encoding="utf-8") as f:
        json.dump(text_to_id, f, ensure_ascii=False, indent=2)

    out = {
        "catalog_path": catalog_path,
        "id_to_text_path": id_to_text_path,
        "text_to_id_path": text_to_id_path,
    }
    if df_dupes is not None and not df_dupes.empty:
        df_dupes.to_excel(dup_path, index=False)
        out["duplicates_path"] = dup_path
    return out


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
