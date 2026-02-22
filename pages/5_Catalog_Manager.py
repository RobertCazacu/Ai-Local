import streamlit as st

from catalog import build_catalog_mappings, load_catalog, load_overrides, save_overrides
from config_utils import load_config

st.title("Catalog Manager")
cfg = load_config()
store_dir = cfg["store_dir"]
path = st.text_input("Catalog file", value="trendyol_categories_catalog.xlsx", help="Încarcă catalogul pentru mapping ID -> text export.")

cat = load_catalog(path)
id_to_text, _, dupes = build_catalog_mappings(cat)
overrides = load_overrides(store_dir)
id_to_text.update(overrides)

st.metric("Total categorii", len(id_to_text))
st.subheader("Duplicate names")
st.dataframe(dupes.head(200), use_container_width=True)

cid = st.selectbox("CategoryID pentru override", options=sorted(id_to_text.keys()))
text = st.text_input("Export text override", value=id_to_text[cid], help="Override manual pentru ambiguități de denumire.")
if st.button("Save overrides", help="Overrides au prioritate la export."):
    overrides[cid] = text
    save_overrides(store_dir, overrides)
    st.success("Override salvat.")

st.subheader("ID -> Text (sample)")
st.write(dict(list(id_to_text.items())[:20]))
