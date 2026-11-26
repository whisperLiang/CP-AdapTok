from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from hla_tok.uf_v3.deploy import JointUnified, inference_digest


def native(key):
    parameter = torch.nn.Parameter(torch.ones(3))
    calls = []
    return SimpleNamespace(
        named_parameters=lambda: [(key, parameter)],
        update=lambda skip_thres: calls.append(("update", skip_thres)),
        proxy=SimpleNamespace(set_param=lambda params, skip: calls.append(("proxy", params, skip))),
        add_cdf_to_state_dict=lambda state: state | {"cdf": "updated"},
        state_dict=lambda: {key: parameter.detach().clone()},
        gaussian_encoder=SimpleNamespace(skip_thres=0),
    ), parameter, calls


def test_deployment_takes_the_checkpoint_tensor_and_refreshes_the_proxy(monkeypatch):
    new = torch.arange(3.)
    intra_key, inter_key = "enc.enc_2.6.weight", "encoder.down.weight"
    overlays = {"native_i": {intra_key: new}, "native_p": {inter_key: new + 5}}
    assert inference_digest(overlays) == inference_digest(dict(reversed(list(overlays.items()))))
    assert inference_digest(overlays) != inference_digest(overlays | {"native_p": {inter_key: new}})
    i, i_weight, i_calls = native(intra_key)
    p, p_weight, p_calls = native(inter_key)
    endpoint = SimpleNamespace(intra=i, _video=lambda: p, stream=SimpleNamespace(synchronize=lambda: None))
    mock = JointUnified.__new__(JointUnified)
    mock.adjustments = mock.lite = None
    mock.overlays = overlays
    monkeypatch.setattr(torch.cuda, "stream", lambda _: nullcontext())
    mock._install(endpoint)
    for key, layer, weight, calls in (("native_i", intra_key, i_weight, i_calls),
                                       ("native_p", inter_key, p_weight, p_calls)):
        assert torch.equal(weight, overlays[key][layer])
        assert calls[0] == ("update", 0) and calls[1][0] == "proxy"
        assert calls[1][1]["cdf"] == "updated" and torch.equal(calls[1][1][layer], weight)
    mock.overlays = overlays | {"native_i": {}}
    with pytest.raises(ValueError, match="incomplete or unknown"):
        mock._install(endpoint)
    mock.overlays = overlays | {"native_i": {intra_key: torch.tensor([float("nan"), 0., 0.])}}
    with pytest.raises(ValueError, match="invalid parameter tensor"):
        mock._install(endpoint)
