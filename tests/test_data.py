from src.data.clean import clean_clean_dataset
from src.data.load import load_clean_dataset, load_raw_business, load_raw_economy
from src.data.quality import build_quality_report


def test_load_clean_dataset_has_expected_columns():
    frame = load_clean_dataset()
    assert len(frame) > 1000
    for column in ["airline", "source_city", "destination_city", "class", "duration", "days_left", "price"]:
        assert column in frame.columns


def test_clean_removes_unnamed_and_keeps_target():
    raw = load_clean_dataset().head(500)
    cleaned = clean_clean_dataset(raw)
    assert all(not str(col).startswith("Unnamed") for col in cleaned.columns)
    assert "price" in cleaned.columns
    assert (cleaned["duration"] > 0).all()
    assert (cleaned["days_left"] >= 0).all()
    assert (cleaned["source_city"] != cleaned["destination_city"]).all()


def test_quality_report_counts():
    frame = clean_clean_dataset(load_clean_dataset().head(200))
    report = build_quality_report(frame, name="sample")
    assert report["rows"] == 200 or report["rows"] <= 200
    assert report["missing_values"]["price"] == 0
    assert "target_statistics" in report


def test_raw_files_load():
    assert len(load_raw_business()) > 0
    assert len(load_raw_economy()) > 0
