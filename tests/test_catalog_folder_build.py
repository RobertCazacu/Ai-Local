import pytest

pd = pytest.importorskip("pandas")

from catalog import build_catalog_from_folder, parse_category_from_filename


def test_parse_category_from_filename():
    assert parse_category_from_filename("369 Other Accessories.xlsx") == ("369", "Other Accessories")
    assert parse_category_from_filename(" 371 Cufflinks .xlsx") == ("371", "Cufflinks")
    assert parse_category_from_filename("no_id.xlsx") is None


def test_build_catalog_from_folder(tmp_path):
    (tmp_path / "369 Other Accessories.xlsx").write_text("x")
    (tmp_path / "371 Cufflinks.xlsx").write_text("x")
    (tmp_path / "372 Cufflinks.xlsx").write_text("x")
    (tmp_path / "bad_name.xlsx").write_text("x")

    catalog_df, dupes_df = build_catalog_from_folder(str(tmp_path))
    assert len(catalog_df) == 3
    for col in ["CategoryID", "CategoryName", "CategoryText", "FileName", "FilePath"]:
        assert col in catalog_df.columns
    assert not dupes_df.empty
    assert (dupes_df["CategoryNameNorm"] == "cufflinks").any()
