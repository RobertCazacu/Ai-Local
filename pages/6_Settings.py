import streamlit as st

from config_utils import DEFAULT_CONFIG, load_config, save_config

st.title("Settings")
cfg = load_config()

st.caption("Modificările se aplică la rulările viitoare.")
ollama = st.text_input("OLLAMA_URL", value=str(cfg["OLLAMA_URL"]), help="URL Ollama pentru embeddings.")
model = st.text_input("embedding_model", value=str(cfg["embedding_model"]), help="Model embeddings folosit pentru ingest/predict.")
text_cols = st.text_input("text columns (comma-separated)", value=",".join(cfg["text_cols"]), help="Coloane text implicite pentru hash + embeddings.")
auto = st.number_input("AUTO_ACCEPT_CONF", 0.0, 1.0, float(cfg["AUTO_ACCEPT_CONF"]), help="Prag pentru auto-accept.")
margin = st.number_input("MIN_MARGIN", 0.0, 1.0, float(cfg["MIN_MARGIN"]), help="Diferență minimă top1-top2.")
min_gold = st.number_input("MIN_GOLD_PER_CAT", 0, 1000, int(cfg["MIN_GOLD_PER_CAT"]), help="Nr minim gold labels pentru categorie warm.")
workers = st.number_input("workers", 1, 32, int(cfg["workers"]))
port = st.number_input("ui_port", 1, 65535, int(cfg.get("ui_port", 8501)))

if st.button("Save settings", help="Salvează config-ul extins, compatibil cu setările existente."):
    cfg.update({
        "OLLAMA_URL": ollama,
        "embedding_model": model,
        "text_cols": [x.strip() for x in text_cols.split(",") if x.strip()],
        "AUTO_ACCEPT_CONF": float(auto),
        "MIN_MARGIN": float(margin),
        "MIN_GOLD_PER_CAT": int(min_gold),
        "workers": int(workers),
        "ui_port": int(port),
    })
    save_config(cfg)
    st.success("Settings salvate.")

if st.button("Reset to defaults", help="Resetează la valorile implicite."):
    save_config(DEFAULT_CONFIG)
    st.warning("Reset aplicat.")
