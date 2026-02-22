import os
import pandas as pd
import streamlit as st

from config_utils import load_config
from jobs import job_context
from store import ensure_store, ingest_file_incremental, init_store_from_legacy

st.title("Insert new Products(Upload & Add to Build)")
cfg = load_config()
store_dir = cfg["store_dir"]
ensure_store(store_dir)

st.caption("Adaugă incremental doar produse noi. Nu suprascrie build-ul existent.")
file = st.file_uploader("Upload xlsx/csv", type=["xlsx", "csv"], help="Încarcă feed-ul de produse pentru ingest incremental.")

if file:
    df = pd.read_excel(file) if file.name.lower().endswith("xlsx") else pd.read_csv(file)
    st.dataframe(df.head(20), use_container_width=True)
    cols = st.multiselect("Text columns", options=list(df.columns), default=[c for c in cfg["text_cols"] if c in df.columns], help="Coloane folosite pentru normalizare, hash și embeddings.")

    st.info("Add to Build adaugă doar produse noi pe baza content_hash; duplicatele sunt ignorate.")
    if st.button("Add to Build (incremental)"):
        with job_context(store_dir, "ingest") as (_, log):
            prog = st.progress(0)
            status = st.empty()

            def cb(done, total):
                prog.progress(done / max(1, total))
                status.write(f"Processed {done}/{total}")

            rep = ingest_file_incremental(df, store_dir, cols, cfg["embedding_model"], cfg["OLLAMA_URL"], workers=int(cfg["workers"]), progress_cb=cb)
            log(str(rep))
            st.success(rep)

st.warning("Initialize from legacy se rulează o singură dată, când migrezi meta.csv + embeddings.npy în store-ul incremental.")
if st.button("Initialize store from legacy", help="Importă build-ul vechi ca shard_00001; nu folosi dacă store-ul e deja populat."):
    try:
        rep = init_store_from_legacy(store_dir, os.path.join(cfg["out_dir"], "meta.csv"), os.path.join(cfg["out_dir"], "embeddings.npy"), cfg["text_cols"], cfg["embedding_model"])
        st.success(rep)
    except Exception as e:
        st.error(str(e))
