import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")

from embeddings import content_hash, normalize_text
from catalog import build_catalog_mappings


def test_normalize_text_deterministic():
    a = normalize_text("  <b>Produs-ABC</b>   _X  ")
    b = normalize_text("produs abc x")
    assert a == b


def test_content_hash_consistent():
    t = normalize_text("Produs test")
    assert content_hash(t) == content_hash(t)


def test_catalog_mapping_duplicate_prefers_path_when_present():
    df = pd.DataFrame(
        {
            "CategoryID": ["1", "2"],
            "Categoria Text": ["Rochii", "Rochii"],
            "CategoryPath": ["Femei > Rochii", "Copii > Rochii"],
        }
    )
    id_to_text, _, dupes = build_catalog_mappings(df)
    assert len(dupes) == 2
    assert id_to_text["1"] in ["Femei > Rochii", "Rochii"]

def test_dedup_second_ingest_zero_new(tmp_path, monkeypatch):
    import store

    def fake_embed(texts, model, ollama_url, workers=4, progress_cb=None):
        return np.ones((len(texts), 4), dtype=np.float32)

    monkeypatch.setattr(store, "embed_texts_batched", fake_embed)

    df = pd.DataFrame({"Nume": ["a", "b"], "Brand": ["x", "y"], "Descriere": ["d1", "d2"], "CategoryID": ["1", "2"]})
    sdir = str(tmp_path / "data_store")
    r1 = store.ingest_file_incremental(df, sdir, ["Nume", "Brand", "Descriere"], "m", "http://localhost:11434")
    r2 = store.ingest_file_incremental(df, sdir, ["Nume", "Brand", "Descriere"], "m", "http://localhost:11434")
    assert r1["new"] == 2
    assert r2["new"] == 0
