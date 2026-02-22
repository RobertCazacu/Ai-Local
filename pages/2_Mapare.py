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
    "După ce rulezi o dată, rezultatul rămâne în pagină și poți filtra fără să dispară."
)

# --------- inputs ----------
file = st.file_uploader(
    "Upload produse pentru mapare",
    type=["xlsx", "csv"],
    help="Urcă fișierul cu produsele pe care vrei să le mapezi.",
)

catalog_path = st.text_input(
    "Catalog path",
    value=cfg.get("active_catalog_path", cfg.get("catalog_path", "trendyol_categories_catalog.xlsx")),
    help="Catalog folosit pentru mapping CategoryID -> Categoria Text.",
)

# --------- session state keys ----------
OUT_KEY = "predict_out_df"
STATUS_KEY = "predict_status"
LAST_FILE_KEY = "predict_last_file"
if STATUS_KEY not in st.session_state:
    st.session_state[STATUS_KEY] = ""

run_clicked = st.button(
    "Run Predict",
    help="Rulează predicția cu progres: auto-accept pentru cazurile sigure, review pentru cele neclare.",
    disabled=(file is None),
)

# --------- RUN (totul aici, ca să existe df) ----------
if run_clicked:
    if file is None:
        st.warning("Încarcă un fișier înainte.")
        st.stop()

    # UI progress widgets (există doar în acest run)
    progress = st.progress(0.0, text="Pornesc predict...")
    status_box = st.empty()

    def ui_progress_cb(pct: float, msg: str):
        pct = max(0.0, min(1.0, float(pct)))
        progress.progress(pct, text=msg)
        status_box.info(msg)

    # 1) citire fișier
    ui_progress_cb(0.01, "Citesc fișierul...")
    if file.name.lower().endswith(".xlsx"):
        df = pd.read_excel(file)
    else:
        df = pd.read_csv(file)

    # 2) catalog + overrides
    ui_progress_cb(0.05, "Încarc catalogul...")
    catalog_path = str(catalog_path or "").strip()
    catalog_path_abs = os.path.abspath(catalog_path)

    if not os.path.exists(catalog_path_abs):
        progress.empty()
        status_box.empty()
        st.error(f"Nu găsesc catalogul la: {catalog_path_abs}")
        st.stop()

    cat = load_catalog(catalog_path_abs)
    id_to_text, _, _ = build_catalog_mappings(cat)
    id_to_text.update(load_overrides(store_dir))

    # 3) predict
    ui_progress_cb(0.08, "Rulez predicția...")

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
            progress_cb=ui_progress_cb,  # <-- progress real
        )
        log(f"rows={len(out)}")

    ui_progress_cb(1.0, "Predict finalizat.")
    status_box.success(f"Predict finalizat: {len(out)} rânduri.")

    # salvează rezultatul ca să nu dispară la filtre
    st.session_state[OUT_KEY] = out.copy()
    st.session_state[LAST_FILE_KEY] = file.name
    st.session_state[STATUS_KEY] = f"Predict finalizat: {len(out)} rânduri."

# --------- persistent status ----------
if st.session_state.get(STATUS_KEY):
    st.info(st.session_state[STATUS_KEY])

# --------- show results + filters (persistente) ----------
out_df = st.session_state.get(OUT_KEY)
if out_df is None or len(out_df) == 0:
    st.caption("Nu există rezultate încă. Încarcă un fișier și apasă Run Predict.")
    st.stop()

st.subheader("Rezultate + filtre")
st.caption(f"Ultimul fișier procesat: {st.session_state.get(LAST_FILE_KEY, '-')}")
min_conf = st.slider("Min confidence", 0.0, 1.0, 0.0, 0.01)
only_review = st.checkbox("Show only needs_review", value=False)
sort_asc = st.checkbox("Sort by confidence asc", value=True)

show = out_df.copy()
if "confidence" in show.columns:
    show["_conf_num"] = pd.to_numeric(show["confidence"], errors="coerce").fillna(0.0)
    show = show[show["_conf_num"] >= float(min_conf)]
    show = show.drop(columns=["_conf_num"])

if only_review and "needs_review" in show.columns:
    show = show[show["needs_review"] == True]  # noqa: E712

if sort_asc and "confidence" in show.columns:
    show = show.sort_values("confidence", ascending=True)

st.dataframe(show, use_container_width=True)