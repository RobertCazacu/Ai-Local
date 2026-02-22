import os
import pandas as pd
import streamlit as st

from catalog import build_catalog_mappings, load_catalog, load_overrides
from config_utils import load_config
from jobs import job_context
from predictor import run_predict


st.title("Mapare (Predict)")
cfg = load_config()
store_dir = cfg["store_dir"]

st.caption(
    "Generează predicții; cazurile foarte sigure merg în pseudo_labels, restul în review queue. "
    "După ce rulezi o dată, rezultatul rămâne în pagină (poți filtra fără să dispară)."
)

# --- UI inputs ---
file = st.file_uploader(
    "Upload produse pentru mapare",
    type=["xlsx", "csv"],
    help="Urcă fișierul cu produsele pe care vrei să le mapezi.",
)

catalog_path = st.text_input(
    "Catalog path",
    value=cfg.get("active_catalog_path", cfg.get("catalog_path", "trendyol_categories_catalog.xlsx")),
    help="Catalogul folosit pentru mapping CategoryID -> Categoria Text (string).",
)

# --- Session state keys ---
OUT_KEY = "predict_out_df"
STATUS_KEY = "predict_status"
LAST_FILE_KEY = "predict_last_file_name"

if STATUS_KEY not in st.session_state:
    st.session_state[STATUS_KEY] = ""

# --- Run predict ---
run_clicked = st.button(
    "Run Predict",
    help="Rulează predicția cu gating: auto-accept pentru cazurile sigure, review pentru cele neclare.",
    disabled=(file is None),
)

if run_clicked:
    if file is None:
        st.warning("Încarcă un fișier înainte.")
        st.stop()

    # Citește fișier
    st.session_state[STATUS_KEY] = "Citesc fișierul..."
    if file.name.lower().endswith(".xlsx"):
        df = pd.read_excel(file)
    else:
        df = pd.read_csv(file)

    # Încarcă catalog + overrides
    st.session_state[STATUS_KEY] = "Încarc catalogul..."
    catalog_path_abs = os.path.abspath(str(catalog_path).strip())
    cat = load_catalog(catalog_path_abs)
    id_to_text, _, _ = build_catalog_mappings(cat)
    id_to_text.update(load_overrides(store_dir))

    # Rulează predict cu spinner
    st.session_state[STATUS_KEY] = "Rulez predicția (poate dura în funcție de nr. produse)..."
    with job_context(store_dir, "predict") as (_, log):
        with st.spinner("Rulez predicția..."):
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

    # Salvează rezultatul în session_state ca să nu dispară la filtre
    st.session_state[OUT_KEY] = out.copy()
    st.session_state[LAST_FILE_KEY] = file.name
    st.session_state[STATUS_KEY] = f"Predict finalizat: {len(out)} rânduri."
    st.success(st.session_state[STATUS_KEY])

# --- Show status (persistent) ---
if st.session_state.get(STATUS_KEY):
    st.info(st.session_state[STATUS_KEY])

# --- Filtering UI (works even after reruns) ---
out_df = st.session_state.get(OUT_KEY)
if out_df is None or len(out_df) == 0:
    st.caption("Nu există rezultate încă. Încarcă un fișier și apasă Run Predict.")
    st.stop()

st.subheader("Rezultate + filtre")
st.caption(f"Ultimul fișier procesat: {st.session_state.get(LAST_FILE_KEY, '-')}")
min_conf = st.slider("Min confidence", 0.0, 1.0, 0.0, 0.01, help="Afișează doar rândurile cu confidence >= prag.")
only_review = st.checkbox("Show only needs_review", value=False, help="Arată doar produsele care au intrat în review_queue.")
sort_asc = st.checkbox("Sort by confidence asc", value=True, help="Util ca să vezi întâi cazurile cele mai incerte.")

show = out_df.copy()
if "confidence" in show.columns:
    show = show[pd.to_numeric(show["confidence"], errors="coerce").fillna(0.0) >= float(min_conf)]
if only_review and "needs_review" in show.columns:
    show = show[show["needs_review"] == True]  # noqa: E712

if sort_asc and "confidence" in show.columns:
    show = show.sort_values("confidence", ascending=True)

st.dataframe(show, use_container_width=True)

# --- Quick counters (optional, helpful) ---
c1, c2, c3 = st.columns(3)
if "needs_review" in out_df.columns:
    c1.metric("Total", len(out_df))
    c2.metric("Needs review", int(out_df["needs_review"].astype(bool).sum()))
    c3.metric("Auto accepted", int((~out_df["needs_review"].astype(bool)).sum()))