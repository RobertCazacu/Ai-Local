import json
import os

import pandas as pd
import streamlit as st

from config_utils import load_config

st.set_page_config(page_title="AI Local Incremental", layout="wide")
cfg = load_config()
store_dir = cfg["store_dir"]

st.title("AI Categorii Local - Workflow complet")
st.caption("Dashboard pentru ingest incremental, mapare, review, export și retrain.")

manifest_path = os.path.join(store_dir, "manifest.json")
if os.path.exists(manifest_path):
    manifest = json.load(open(manifest_path, encoding="utf-8"))
    shards = manifest.get("shards", [])
else:
    shards = []

gold_path = os.path.join(store_dir, "corrections_gold.csv")
pseudo_path = os.path.join(store_dir, "pseudo_labels.csv")
review_path = os.path.join(store_dir, "review_queue.csv")
jobs_path = os.path.join(store_dir, "jobs.jsonl")

def rows(path: str) -> int:
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return 0
    try:
        return len(pd.read_csv(path, dtype=str, keep_default_na=False))
    except Exception:
        return 0

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total shards", len(shards))
col2.metric("Gold labels", rows(gold_path))
col3.metric("Pseudo labels", rows(pseudo_path))
col4.metric("Review queue", rows(review_path))
col5.metric("Jobs", rows(jobs_path))

st.info("Gold = corecții umane (adevăr de referință). Pseudo = auto-acceptate cu încredere mare. Review = cazuri neclare pentru validare umană.")

if os.path.exists(jobs_path):
    with open(jobs_path, "r", encoding="utf-8") as f:
        lines = [x.strip() for x in f if x.strip()]
    if lines:
        st.subheader("Ultimele 5 evenimente job")
        st.dataframe(pd.DataFrame([json.loads(l) for l in lines[-5:]]), use_container_width=True)

st.success("Folosește meniul din stânga (pages) pentru workflow complet.")
