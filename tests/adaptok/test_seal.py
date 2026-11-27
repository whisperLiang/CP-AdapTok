import json

import pytest

from baselines.adaptok.video_reader import manifest_sources
from common.data import dataset_path, require_unsealed_path
from common.paths import ROOT


def test_adaptok_rejects_the_sealed_manifest():
    manifest = ROOT / "data/manifests/cpat_test.json"
    record = json.loads(manifest.read_text())["records"][0]
    with pytest.raises(PermissionError, match="sealed"):
        manifest_sources(manifest, [record["content_id"]])


def test_sealed_directory_and_frame_cannot_be_selected(monkeypatch, tmp_path):
    monkeypatch.setenv("CP_ADAPTOK_DATA_ROOT", str(tmp_path))
    manifest = ROOT / "data/manifests/cpat_test.json"
    record = json.loads(manifest.read_text())["records"][0]
    directory = dataset_path(record["path"])
    frame = directory / record["frame_hashes"][0]["file"]
    for candidate in (directory.parent, directory, frame):
        with pytest.raises(PermissionError, match="sealed"):
            require_unsealed_path(candidate)
