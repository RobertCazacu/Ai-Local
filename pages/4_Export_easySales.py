import os
from datetime import datetime

import pandas as pd
import streamlit as st

from config_utils import load_config

st.title("Export easySales")
cfg = load_config()
store_dir = cfg["store_dir"]
exports_dir = os.path.join(store_dir, "exports")
os.makedirs(exports_dir, exist_ok=True)

mode = st.selectbox("Ce exportezi", ["auto-accepted only", "corrected only", "all"], help="Alege sursa pentru export.")
include_debug = st.checkbox("Include coloane debug", value=False, help="Adaugă confidence, margin și category_id în fișier.")

if st.button("Generate easySales export", help="Exportă Categoria Text (string) compatibil easySales."):
    pseudo = pd.read_csv(os.path.join(store_dir, "pseudo_labels.csv"), dtype=str, keep_default_na=False) if os.path.exists(os.path.join(store_dir, "pseudo_labels.csv")) and os.path.getsize(os.path.join(store_dir, "pseudo_labels.csv")) > 0 else pd.DataFrame()
    gold = pd.read_csv(os.path.join(store_dir, "corrections_gold.csv"), dtype=str, keep_default_na=False) if os.path.exists(os.path.join(store_dir, "corrections_gold.csv")) and os.path.getsize(os.path.join(store_dir, "corrections_gold.csv")) > 0 else pd.DataFrame()

    if mode == "auto-accepted only":
        data = pseudo.copy()
    elif mode == "corrected only":
        data = gold.copy()
    else:
        data = pd.concat([pseudo, gold], ignore_index=True)

    if data.empty:
        st.warning("Nu există date pentru export.")
    else:
        out = pd.DataFrame()
        out["SKU"] = data.get("SKU", "")
        out["Categoria"] = data.get("final_category_text", data.get("predicted_category_text", ""))
        if include_debug:
            out["category_id"] = data.get("final_category_id", data.get("predicted_category_id", ""))
            out["confidence"] = data.get("confidence", "")
            out["margin"] = data.get("margin", "")

        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        csv_path = os.path.join(exports_dir, f"easySales_export_{ts}.csv")
        xlsx_path = os.path.join(exports_dir, f"easySales_export_{ts}.xlsx")
        out.to_csv(csv_path, index=False)
        out.to_excel(xlsx_path, index=False)
        st.success(f"Export creat: {csv_path}")
        st.download_button("Download CSV", data=open(csv_path, "rb").read(), file_name=os.path.basename(csv_path))
