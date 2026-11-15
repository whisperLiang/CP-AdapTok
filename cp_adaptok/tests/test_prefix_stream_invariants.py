import copy
from dataclasses import replace

import pytest
import torch

from cp_adaptok.models.causal_progressive_codec import CausalProgressiveCodec
from cp_adaptok.streaming.bitstream import StreamError
from cp_adaptok.streaming.receiver import CodecDecoder
from cp_adaptok.streaming.sender import CodecEncoder


def test_progressive_real_rans_prefixes_commit_only_base_and_have_explicit_model_id(codec):
    torch.manual_seed(37)
    model = CausalProgressiveCodec(replace(codec.config, token_aggregation="shared_progressive_v2")).eval()
    sender, dense, cached, base_only = CodecEncoder(model), CodecDecoder(model), CodecDecoder(model), CodecDecoder(model)
    for _ in range(3):
        rgb = torch.rand(1, 3, 16, 24)
        encoded = sender.encode_frame(rgb, budgets=[8, 16, 24, 32])
        for receiver in (dense, cached, base_only):
            receiver.start_frame(encoded.header_bytes)
        dense.append_packets(encoded.packets)
        base_only.append_packets(encoded.prefix_packets(8))
        base_only.decode_base()
        for index, packet in enumerate(encoded.packets):
            cached.append_packets([packet])
            k = (index + 1) * 8
            a = cached.decode_prefix(k, use_cache=True)
            b = dense.decode_prefix(k)
            torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-5)
            torch.testing.assert_close(cached.decode_prefix(k, use_cache=True), a, atol=0, rtol=0)
        fingerprints = {receiver.get_state().fingerprint() for receiver in (sender, dense, cached, base_only)}
        assert len(fingerprints) == 1
        assert tuple(cached.stream.token_ids) == encoded.token_ids
    assert sender.base_commits == dense.base_commits == cached.base_commits == base_only.base_commits == 3
    old = CodecDecoder(copy.deepcopy(codec))
    with pytest.raises(StreamError, match="model_id"):
        old.start_frame(encoded.header_bytes)
