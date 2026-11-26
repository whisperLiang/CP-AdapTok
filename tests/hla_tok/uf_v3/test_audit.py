import torch

from scripts.hla_tok.uf_v3.audit import tensor_delta


def test_tensor_audit_distinguishes_changes_from_declared_trainability():
    before = torch.tensor([1., 2., 3.])
    same = tensor_delta(before, before.clone())
    updated = tensor_delta(before, torch.tensor([1., 2.5, 3.]))
    assert same["changed"] == 0 and same["before"] == same["after"]
    assert updated["changed"] == 1 and updated["l2"] == 0.5
    assert updated["before"] != updated["after"]
