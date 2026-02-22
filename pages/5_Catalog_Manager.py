import streamlit as st

from catalog import (
    build_catalog_from_folder,
    build_catalog_mappings,
    load_catalog,
    load_overrides,
    save_catalog_outputs,
    save_overrides,
)
from config_utils import load_config, save_config

st.title("Catalog Manager")
cfg = load_config()
store_dir = cfg["store_dir"]
default_catalog = cfg.get("active_catalog_path", "trendyol_categories_catalog.xlsx")

st.subheader("Catalog activ")
path = st.text_input(
    "Catalog file",
    value=default_catalog,
    help="Încarcă catalogul pentru mapping ID -> text export. Acceptă și coloane alternative (ex: category_id/category_name).",
)

try:
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
except Exception as e:
    st.error(f"Nu am putut încărca catalogul activ: {e}")

with st.expander("Build catalog from folder (Trendyol Categories)", expanded=True):
    st.caption(
        "Scanează un folder cu fișiere numite ca: ‘369 Other Accessories.xlsx’. "
        "Extrage automat CategoryID și CategoryName din numele fișierului."
    )
    folder_path = st.text_input(
        "Folder path",
        value="",
        help="Ex: C:\\Users\\...\\Trendyol Categories. Funcționează dacă UI rulează pe același PC unde există folderul.",
    )

    fmt = st.selectbox(
        "CategoryText format",
        ["ID + Name (recomandat)", "Name only (risc duplicate)"],
        help="Name only poate produce duplicate de text între categorii diferite; recomandat este ID + Name.",
    )

    if st.button(
        "Build & Save Catalog",
        help="Scanează folderul, extrage ID+Nume din numele fișierelor și salvează catalogul în data_store/catalog/. Nu modifică fișierele din folder.",
    ):
        try:
            df_catalog, df_dupes = build_catalog_from_folder(folder_path)
            if df_catalog.empty:
                st.warning("Nu am găsit fișiere valide în formatul '<ID> <Nume>.xlsx'.")
            else:
                if fmt == "Name only (risc duplicate)":
                    df_catalog["CategoryText"] = df_catalog["CategoryName"].astype(str)

                paths = save_catalog_outputs(df_catalog, df_dupes, out_dir=f"{store_dir}/catalog", versioned=True)

                cfg["active_catalog_path"] = paths["catalog_path"]
                save_config(cfg)

                st.success(f"Catalog construit cu succes. Total categorii: {len(df_catalog)}")
                st.dataframe(df_catalog.head(50), use_container_width=True)

                if not df_dupes.empty:
                    st.warning(
                        f"Au fost detectate duplicate de nume: {len(df_dupes)}. "
                        "Recomandat: folosește formatul CategoryText = 'ID + Name'."
                    )
                    st.dataframe(df_dupes, use_container_width=True)

                st.info("Fișiere salvate:")
                for k, v in paths.items():
                    st.write(f"- {k}: {v}")
                st.info(f"Catalog activ actualizat în config: {cfg['active_catalog_path']}")
        except Exception as e:
            st.error(f"Build catalog failed: {e}")
