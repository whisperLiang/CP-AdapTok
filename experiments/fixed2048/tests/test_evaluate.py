import json

import pytest

from experiments.fixed2048.checkpoint import sha256
from experiments.fixed2048.evaluate import check_test_seal, stage_times


def test_test_split_is_closed_until_identities_and_config_are_sealed(tmp_path):
    checkpoint = tmp_path / 'model.pt'
    checkpoint.write_bytes(b'trained')
    manifest = tmp_path / 'cpat_test.json'
    manifest.write_text('{}')
    config = {'manifest_dir':str(tmp_path), 'final_test':{'frames':64}}
    with pytest.raises(ValueError, match='seal'):
        check_test_seal(config, checkpoint, None)
    seal = tmp_path / 'seal.json'
    seal.write_text(json.dumps({'checkpoint_sha256s':[sha256(checkpoint)], 'test_config':config['final_test'],
                                'test_manifest_sha256':sha256(manifest)}))
    check_test_seal(config, checkpoint, seal)
    checkpoint.write_bytes(b'changed weights')
    with pytest.raises(ValueError, match='identity'):
        check_test_seal(config, checkpoint, seal)


def test_r0_latency_does_not_count_unrelated_paired_tokenization():
    from types import SimpleNamespace
    encoded = SimpleNamespace(timings={'encoder_ms':12, 'tokenizer_ms':7})
    output = SimpleNamespace(timings={'dcvc_decoder_ms':9})
    assert stage_times(encoded, output, 'R0')['end_to_end_ms'] == 21
    assert stage_times(encoded, output, 'R1')['end_to_end_ms'] == 28
