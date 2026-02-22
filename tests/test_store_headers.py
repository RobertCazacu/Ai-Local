import pytest

pd = pytest.importorskip('pandas')

from store import PERSISTENT_LABEL_COLUMNS, ensure_store


def test_persistent_csv_files_have_headers(tmp_path):
    sdir = tmp_path / 'data_store'
    ensure_store(str(sdir))

    for fname in ['corrections_gold.csv', 'pseudo_labels.csv', 'review_queue.csv']:
        df = pd.read_csv(sdir / fname, dtype=str, keep_default_na=False)
        assert list(df.columns) == PERSISTENT_LABEL_COLUMNS
