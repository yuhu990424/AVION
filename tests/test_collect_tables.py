from avion.evaluation.collect_tables import collect_metric_rows, infer_run_fields, summarize_metric_rows, write_csv, write_markdown
from avion.utils.io import write_json


def test_collect_tables(tmp_path):
    run = tmp_path / "run1"
    write_json({"accuracy": 12.5}, run / "metrics.json")
    rows = collect_metric_rows(tmp_path)
    assert rows[0]["accuracy"] == 12.5
    write_csv(rows, tmp_path / "tables" / "metrics.csv")
    write_markdown(rows, tmp_path / "tables" / "metrics.md")
    assert (tmp_path / "tables" / "metrics.csv").exists()
    assert (tmp_path / "tables" / "metrics.md").exists()


def test_infer_and_summarize_fewshot_rows(tmp_path):
    run1 = tmp_path / "aid" / "shots_16" / "AVION_GeoRSCLIP" / "tag" / "seed1"
    run2 = tmp_path / "aid" / "shots_16" / "AVION_GeoRSCLIP" / "tag" / "seed2"
    write_json({"accuracy": 80.0}, run1 / "metrics.json")
    write_json({"accuracy": 84.0}, run2 / "metrics.json")

    fields = infer_run_fields(run1)
    assert fields["protocol"] == "few_shot"
    assert fields["dataset"] == "aid"
    assert fields["shots"] == 16
    assert fields["seed"] == 1

    rows = collect_metric_rows(tmp_path, protocol="few_shot")
    summary = summarize_metric_rows(rows)
    assert len(summary) == 1
    assert summary[0]["accuracy_mean"] == 82.0
    assert summary[0]["num_runs"] == 2


def test_infer_retrieval_rows(tmp_path):
    run = tmp_path / "retrieval" / "rsitmd" / "AVION_GeoRSCLIP" / "tag" / "seed3"
    write_json({"mR": 55.0}, run / "metrics.json")
    rows = collect_metric_rows(tmp_path, protocol="retrieval")
    assert rows[0]["protocol"] == "retrieval"
    assert rows[0]["dataset"] == "rsitmd"
    assert rows[0]["seed"] == 3
