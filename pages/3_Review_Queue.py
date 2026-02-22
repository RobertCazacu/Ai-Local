import os
import pandas as pd
import streamlit as st

from catalog import build_catalog_mappings, load_catalog, load_overrides
from config_utils import load_config

st.title("Review Queue (Corectare)")
cfg = load_config()
store_dir = cfg["store_dir"]
review_path = os.path.join(store_dir, "review_queue.csv")
gold_path = os.path.join(store_dir, "corrections_gold.csv")

if not os.path.exists(review_path) or os.path.getsize(review_path) == 0:
    st.info("Review queue este goală.")
    st.stop()

qdf = pd.read_csv(review_path, dtype=str, keep_default_na=False)
st.caption("Salvarea corecției mută produsul în corrections_gold.csv și îl scoate din review queue.")
st.dataframe(qdf.head(200), use_container_width=True)

catalog_path = st.text_input("Catalog path", value="trendyol_categories_catalog.xlsx")
cat = load_catalog(catalog_path)
id_to_text, _, _ = build_catalog_mappings(cat)
id_to_text.update(load_overrides(store_dir))

idx = st.number_input("Row index", min_value=0, max_value=max(0, len(qdf) - 1), value=0, help="Selectează produsul de corectat")
selected = qdf.iloc[int(idx)].to_dict()
st.write({k: selected.get(k, "") for k in ["SKU", "Nume", "Brand", "Descriere", "predicted_category_text", "confidence"]})

cid = st.selectbox("Selectează categoria corectă", options=sorted(id_to_text.keys()), format_func=lambda x: f"{x} - {id_to_text[x]}", help="Caută și selectează categoria corectă.")

if st.button("Save correction (Gold label)", help="Salvează corecția, scoate produsul din coadă și întărește sistemul pentru viitor."):
    row = selected.copy()
    row["final_category_id"] = cid
    row["final_category_text"] = id_to_text[cid]
    row["source"] = "human"
    gdf = pd.DataFrame([row])

    if os.path.exists(gold_path) and os.path.getsize(gold_path) > 0:
        old = pd.read_csv(gold_path, dtype=str, keep_default_na=False)
        gdf = pd.concat([old, gdf], ignore_index=True)
        if "content_hash" in gdf.columns:
            gdf = gdf.drop_duplicates(subset=["content_hash"], keep="last")
    gdf.to_csv(gold_path, index=False)

    qdf2 = qdf.drop(index=int(idx)).reset_index(drop=True)
    qdf2.to_csv(review_path, index=False)
    st.success("Corecție salvată.")
