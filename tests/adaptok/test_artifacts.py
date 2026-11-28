import csv

import pytest

from baselines.adaptok.artifacts import exclusive_writer
from common.artifacts import write_csv


def test_comparison_csv_keeps_metric_columns_missing_from_cp(tmp_path):
    path = tmp_path / "comparison.csv"
    write_csv(path, [{"model": "CP", "psnr": 15.5}, {"model": "Official", "psnr": 25., "lpips-vgg": 0.2}])
    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["lpips-vgg"] == ""
    assert rows[1]["lpips-vgg"] == "0.2"


def test_result_directory_has_one_writer_and_can_be_reopened(tmp_path):
    with exclusive_writer(tmp_path):
        with pytest.raises(BlockingIOError):
            with exclusive_writer(tmp_path):
                pytest.fail("second writer entered")
    with exclusive_writer(tmp_path):
        assert (tmp_path / "experiment.lock").exists()
