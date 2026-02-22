import json
import os

import pandas as pd
import streamlit as st

from config_utils import load_config

st.title("Jobs & Logs")
cfg = load_config()
store_dir = cfg["store_dir"]
jobs_path = os.path.join(store_dir, "jobs.jsonl")

if os.path.exists(jobs_path):
    jobs = [json.loads(x) for x in open(jobs_path, encoding="utf-8") if x.strip()]
    st.dataframe(pd.DataFrame(jobs).tail(100), use_container_width=True)
else:
    st.info("Nu există joburi încă.")

log_path = st.text_input("Log file path", value="", help="Inserează calea completă din tabel pentru a vedea logul.")
if log_path and os.path.exists(log_path):
    st.code(open(log_path, encoding="utf-8").read()[-8000:])

exports = os.path.join(store_dir, "exports")
st.text_input("Exports folder", value=exports, help="Poți copia path-ul pentru acces rapid.")
st.caption("Open exports folder: copiază path-ul și deschide-l în File Explorer.")
