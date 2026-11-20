import torch

from aip_vtok.joint.transform import LatentTransform


def test_initial_permutation_and_learned_inverse():
    orders = torch.stack((torch.arange(127, -1, -1), torch.arange(128)))
    transform = LatentTransform(orders)
    x = torch.randn(2, 128, 7, 9)
    assert torch.equal(transform.project(x, transform.matrix(True)), x[:, orders[0]])
    assert torch.equal(transform.project(x, transform.matrix(False)), x)
    with torch.no_grad():
        transform.skew.normal_(std=0.01)
    for intra in (True, False):
        matrix = transform.matrix(intra)
        assert torch.allclose(matrix.T @ matrix, torch.eye(128), atol=2e-6, rtol=0)
        assert torch.allclose(transform.project(transform.project(x, matrix), matrix.T), x, atol=5e-6)
