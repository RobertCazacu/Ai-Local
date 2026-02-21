import os
import re
import argparse
import pandas as pd
from rapidfuzz import process, fuzz

def parse_filename(file_name: str):
    base = os.path.splitext(os.path.basename(file_name))[0].strip()
    # format: "369 Other Accessories" / "371 Cufflinks"
    m = re.match(r"^\s*(\d+)\s*(.*)\s*$", base)
    if not m:
        return None, base
    cat_id = m.group(1)
    cat_name = m.group(2).strip()
    return cat_id, cat_name

def norm(s: str) -> str:
    s = str(s).strip().lower()
    s = s.replace("_", " ").replace("-", " ")
    s = re.sub(r"\s+", " ", s)
    return s

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cat_dir", required=True, help="Folder cu fisierele XLSX Trendyol Categories")
    ap.add_argument("--labeled", default="", help="Optional: labeled_trendyol.xlsx (training) pentru mapping")
    ap.add_argument("--category_col", default="Categorie", help="Coloana categorie (text) din labeled_trendyol.xlsx")
    ap.add_argument("--out_catalog", default="trendyol_categories_catalog.xlsx")
    ap.add_argument("--out_mapping", default="trendyol_text_to_id_mapping.xlsx")
    ap.add_argument("--out_labeled_id", default="labeled_trendyol_id.xlsx")
    ap.add_argument("--min_fuzzy_score", type=int, default=90)
    args = ap.parse_args()

    # 1) Catalog din numele fisierelor
    rows = []
    for f in os.listdir(args.cat_dir):
        if not f.lower().endswith(".xlsx"):
            continue
        full = os.path.join(args.cat_dir, f)
        cat_id, cat_name = parse_filename(f)
        if not cat_id:
            continue
        rows.append({
            "category_id": str(cat_id),
            "category_name": cat_name,
            "category_name_norm": norm(cat_name),
            "file": full,
        })

    catalog = pd.DataFrame(rows)
    if catalog.empty:
        raise RuntimeError("Nu am gasit fisiere .xlsx in cat_dir sau nu se potriveste formatul 'ID Nume.xlsx'.")

    # detectam duplicate de nume (posibil ambiguu)
    dup = catalog.groupby("category_name_norm")["category_id"].nunique()
    dup = dup[dup > 1].reset_index().rename(columns={"category_id": "ids_count"})
    catalog.to_excel(args.out_catalog, index=False)
    print(f"OK: catalog salvat -> {args.out_catalog} (rows={len(catalog)})")
    if not dup.empty:
        dup.to_excel("trendyol_duplicate_names.xlsx", index=False)
        print("ATENTIE: exista nume duplicate. Am salvat lista in trendyol_duplicate_names.xlsx")

    # 2) Mapping (optional) din labeled_trendyol.xlsx
    if args.labeled:
        df = pd.read_excel(args.labeled)
        if args.category_col not in df.columns:
            raise ValueError(f"Nu exista coloana {args.category_col} in {args.labeled}")

        uniq = sorted(set(df[args.category_col].astype(str).str.strip().tolist()))
        uniq = [u for u in uniq if u]

        name_to_ids = catalog.groupby("category_name_norm")["category_id"].apply(list).to_dict()
        catalog_norm_list = catalog["category_name_norm"].tolist()

        map_rows = []
        for t in uniq:
            t_norm = norm(t)

            # exact match pe norm
            if t_norm in name_to_ids:
                ids = name_to_ids[t_norm]
                if len(ids) == 1:
                    map_rows.append({
                        "categorie_text": t,
                        "categorie_text_norm": t_norm,
                        "category_id": ids[0],
                        "match_type": "exact",
                        "score": 100,
                        "notes": ""
                    })
                else:
                    # ambiguu: acelasi nume -> mai multe ID-uri
                    map_rows.append({
                        "categorie_text": t,
                        "categorie_text_norm": t_norm,
                        "category_id": "",
                        "match_type": "ambiguous_exact",
                        "score": 100,
                        "notes": f"Mai multe ID-uri pentru acelasi nume: {ids[:10]}..."
                    })
                continue

            # fuzzy match
            best = process.extractOne(t_norm, catalog_norm_list, scorer=fuzz.token_sort_ratio)
            if best:
                best_norm, score, idx = best
                best_id = catalog.iloc[idx]["category_id"]
                match_type = "fuzzy_ok" if score >= args.min_fuzzy_score else "fuzzy_low"
                map_rows.append({
                    "categorie_text": t,
                    "categorie_text_norm": t_norm,
                    "category_id": best_id if score >= args.min_fuzzy_score else "",
                    "match_type": match_type,
                    "score": int(score),
                    "notes": catalog.iloc[idx]["category_name"]
                })
            else:
                map_rows.append({
                    "categorie_text": t,
                    "categorie_text_norm": t_norm,
                    "category_id": "",
                    "match_type": "none",
                    "score": 0,
                    "notes": ""
                })

        mapping = pd.DataFrame(map_rows)
        mapping.to_excel(args.out_mapping, index=False)
        print(f"OK: mapping salvat -> {args.out_mapping}")

        # 3) Aplica mapping in labeled si scoate labeled_trendyol_id.xlsx
        df["_cat_norm"] = df[args.category_col].astype(str).str.strip().map(norm)
        mapping_small = mapping[["categorie_text_norm", "category_id"]].rename(columns={"categorie_text_norm": "_cat_norm"})
        merged = df.merge(mapping_small, on="_cat_norm", how="left")

        # category_id devine label-ul bun
        merged.rename(columns={"category_id": "CategoryID"}, inplace=True)
        merged.drop(columns=["_cat_norm"], inplace=True)

        merged.to_excel(args.out_labeled_id, index=False)
        print(f"OK: labeled cu ID salvat -> {args.out_labeled_id}")

if __name__ == "__main__":
    main()