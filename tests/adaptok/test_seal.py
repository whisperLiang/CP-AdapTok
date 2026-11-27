import json

import pytest

from baselines.adaptok.video_reader import manifest_sources, source_metadata
from common.data import dataset_path
from common.paths import ROOT


def test_sealed_manifest_and_direct_path_are_rejected():
    manifest_path = ROOT / "data/manifests/cpat_test.json"
    test_record = json.loads(manifest_path.read_text())["records"][0]
    with pytest.raises(PermissionError, match="sealed"):
        manifest_sources(manifest_path, [test_record["content_id"]])
    with pytest.raises(PermissionError, match="sealed"):
        source_metadata(dataset_path(test_record["path"]))
