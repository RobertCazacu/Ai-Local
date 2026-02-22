import pandas as pd
import streamlit as st

from catalog import build_catalog_mappings, load_catalog, load_overrides
from config_utils import load_config
from jobs import job_context
from predictor import run_predict

st.title("Mapare (Predict)")
cfg = load_config()
store_dir = cfg["store_dir"]

st.caption("Generează predicții; cazurile foarte sigure merg în pseudo_labels, restul în review queue.")
file = st.file_uploader("Upload produse pentru mapare", type=["xlsx", "csv"], help="Poți urca un nou fișier pentru predict.")
catalog_path = st.text_input("Catalog path", value=cfg.get("active_catalog_path", "trendyol_categories_catalog.xlsx"), help="Fișierul catalog folosit pentru maparea CategoryID -> Categoria Text.")

if file and st.button("Run Predict", help="Rulează predicția cu gating: auto-accept pentru cazurile sigure, review pentru cele neclare."):
    df = pd.read_excel(file) if file.name.lower().endswith("xlsx") else pd.read_csv(file)
    cat = load_catalog(catalog_path)
    id_to_text, _, _ = build_catalog_mappings(cat)
    ov = load_overrides(store_dir)
    id_to_text.update(ov)

    with job_context(store_dir, "predict") as (_, log):
        out = run_predict(
            df,
            store_dir,
            cfg["text_cols"],
            cfg["embedding_model"],
            cfg["OLLAMA_URL"],
            id_to_text,
            auto_accept_conf=float(cfg["AUTO_ACCEPT_CONF"]),
            min_margin=float(cfg["MIN_MARGIN"]),
            min_gold_per_cat=int(cfg["MIN_GOLD_PER_CAT"]),
            topk=int(cfg.get("topK", 5)),
            workers=int(cfg["workers"]),
        )
        log(f"rows={len(out)}")

    st.success(f"Predict finalizat: {len(out)} rânduri")
    q = st.slider("Min confidence", 0.0, 1.0, 0.0, help="Filtru minim de încredere")
    only_review = st.checkbox("show only needs_review", value=False)
    show = out[out["confidence"] >= q]
    if only_review:
        show = show[show["needs_review"]]
    show = show.sort_values("confidence", ascending=True)
    st.dataframe(show, use_container_width=True)
