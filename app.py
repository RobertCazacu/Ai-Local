import os
import pandas as pd
import streamlit as st

from categorize_engine import build_index, predict

st.set_page_config(page_title="AI Categorii - Local", layout="wide")
st.title("AI local: Build și Predict for categories Trendyol")

with st.sidebar:
    out_dir = st.text_input("Folder index (out_dir)", value="model_trendyol")
    embed_model = st.text_input("Model embeddings (Ollama)", value="nomic-embed-text")
    workers = st.number_input("Workers", min_value=1, max_value=16, value=4)
    k = st.number_input("k (vecini)", min_value=3, max_value=50, value=15)
    min_conf = st.number_input("Prag încredere (min_conf)", min_value=0.1, max_value=0.95, value=0.55)

tab1, tab2 = st.tabs(["1) Build (învățare)", "2) Predict (mapare)"])

# -------------------------
# TAB 1: BUILD
# -------------------------
with tab1:
    st.subheader("Încarcă labeled_trendyol.xlsx și construiește indexul")

    labeled_file = st.file_uploader("labeled_trendyol.xlsx", type=["xlsx"], key="labeled")

    if labeled_file is None:
        st.info("Încarcă fișierul labeled_trendyol.xlsx ca să poți rula Build.")
    else:
        df = pd.read_excel(labeled_file)
        st.write("Coloane detectate:", list(df.columns))
        st.dataframe(df.head(20), use_container_width=True)

        label_col_default = "Categorie" if "Categorie" in df.columns else df.columns[0]
        label_col = st.selectbox(
            "Coloana categorie (label_col)",
            options=list(df.columns),
            index=list(df.columns).index(label_col_default),
        )

        default_text_cols = [c for c in ["Nume", "Brand", "Descriere"] if c in df.columns]
        if not default_text_cols:
            default_text_cols = [df.columns[0]]

        text_cols = st.multiselect(
            "Coloane folosite pentru text (text_cols)",
            options=list(df.columns),
            default=default_text_cols,
        )

        if st.button("Build", key="build_btn"):
            if not text_cols:
                st.error("Alege măcar o coloană pentru text_cols.")
            else:
                os.makedirs("tmp_uploads", exist_ok=True)

                labeled_path = os.path.join("tmp_uploads", "labeled_trendyol.xlsx")
                with open(labeled_path, "wb") as f:
                    f.write(labeled_file.getbuffer())

                progress = st.progress(0)
                status = st.empty()

                def cb(done, total, phase):
                    if total <= 0:
                        return
                    progress.progress(min(1.0, done / total))
                    status.write(f"{phase}: {done}/{total}")

                try:
                    build_index(
                        labeled_path=labeled_path,
                        out_dir=out_dir,
                        text_cols=text_cols,
                        label_col=label_col,
                        embed_model=embed_model,
                        workers=int(workers),
                        progress_cb=cb,  # trebuie să existe în categorize_engine.py
                    )
                    status.write("build: index finalizat")
                    progress.progress(1.0)
                    st.success(f"Index construit în: {out_dir}")
                except Exception as e:
                    st.exception(e)

# -------------------------
# TAB 2: PREDICT
# -------------------------
with tab2:
    st.subheader("Încarcă new.xlsx și descarcă rezultat_mapare.xlsx")

    new_file = st.file_uploader("new.xlsx", type=["xlsx"], key="new")

    if new_file is None:
        st.info("Încarcă fișierul new.xlsx ca să poți rula Predict.")
    else:
        df2 = pd.read_excel(new_file)
        st.write("Coloane detectate:", list(df2.columns))
        st.dataframe(df2.head(20), use_container_width=True)

        if st.button("Predict", key="predict_btn"):
            os.makedirs("tmp_uploads", exist_ok=True)

            input_path = os.path.join("tmp_uploads", "new.xlsx")
            output_path = os.path.join("tmp_uploads", "rezultat_mapare.xlsx")

            with open(input_path, "wb") as f:
                f.write(new_file.getbuffer())

            progress = st.progress(0)
            status = st.empty()

            def cb(done, total, phase):
                if total <= 0:
                    return
                progress.progress(min(1.0, done / total))
                status.write(f"{phase}: {done}/{total}")

            try:
                predict(
                    input_path=input_path,
                    out_dir=out_dir,
                    output_path=output_path,
                    k=int(k),
                    min_conf=float(min_conf),
                    workers=int(workers),
                    progress_cb=cb,  # trebuie să existe în categorize_engine.py
                )
                status.write("predict: finalizat")
                progress.progress(1.0)

                with open(output_path, "rb") as f:
                    data = f.read()

                st.download_button(
                    label="Descarcă rezultat_mapare.xlsx",
                    data=data,
                    file_name="rezultat_mapare.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
            except Exception as e:
                st.exception(e)