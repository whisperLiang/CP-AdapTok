import pytest
import torch

from hla_tok import HLATok
from hla_tok.progressive.geometry import NATIVE_SHAPE
from hla_tok.streaming.continuous import (
    FULL_RANKS,
    MAX_K,
    REGIONS,
    TAIL_REGIONS,
    TOKEN_DIM,
    ContinuousReceiver,
    Order,
    RankOrder,
    encode_continuous,
    truncate,
)
from scripts.hla_tok.continuous.inputs import V3, open_inputs
from scripts.hla_tok.v2.protocol import setup

pytestmark = pytest.mark.filterwarnings("ignore:Arguments other than a weight enum")


@pytest.fixture(scope="module")
def sequence():
    return torch.arange(MAX_K, dtype=torch.float32).view(1, -1, 1).expand(1, MAX_K, TOKEN_DIM)


def test_rank_major_vector_mapping_and_inverse(sequence):
    reverse = RankOrder.reversed()
    reordered = reverse.apply(sequence)
    assert reordered.shape == (1, 16320, 32)
    assert torch.equal(reordered[:, :TAIL_REGIONS], sequence[:, -TAIL_REGIONS:])
    assert torch.equal(reordered[0, TAIL_REGIONS:TAIL_REGIONS + REGIONS, 0],
                       torch.arange(119 * 135, 120 * 135))
    assert torch.equal(reverse.restore_full(reordered), sequence)


@pytest.mark.parametrize("count", [1, 2, 134, 135, 136, 510, MAX_K - 1, MAX_K])
def test_prefix_budget_is_exact(sequence, count):
    assert torch.equal(truncate(sequence, count), sequence[:, :count])
    assert truncate(sequence, count).shape == (1, count, TOKEN_DIM)


@pytest.mark.parametrize("count", [True, 0, -1, 1.5, MAX_K + 1])
def test_illegal_budget_is_rejected(sequence, count):
    with pytest.raises(ValueError, match="budget"):
        truncate(sequence, count)


def test_order_must_preserve_each_rank_exactly_once():
    with pytest.raises(ValueError, match="Rank"):
        RankOrder(Order.O0, tuple(range(FULL_RANKS, -1, -1)))
    with pytest.raises(ValueError, match="Rank"):
        RankOrder(Order.O2, (0,) * FULL_RANKS + (FULL_RANKS,))


def test_receiver_uses_only_the_received_prefix():
    setup()
    model = HLATok("R2").eval()
    receiver = ContinuousReceiver.from_model(model)
    base = torch.zeros(1, MAX_K, TOKEN_DIM)
    altered = base.clone()
    altered[:, 136:] = 20
    order = RankOrder.native()
    with torch.inference_mode():
        left = receiver.recover(truncate(base, 136), order)
        right = receiver.recover(truncate(altered, 136), order)
        assert torch.equal(left, right)
        assert not torch.equal(left, receiver.recover(truncate(altered, 137), order))
    assert {name for name, _ in receiver.named_children()} == {"local_decoder", "rgb_decoder"}


@pytest.mark.gpu
@pytest.mark.parametrize("precision", ["fp32", "bf16"])
def test_c0_published_model_and_max_budget_equivalence(precision):
    if not torch.cuda.is_available() or not V3.exists():
        pytest.skip("published V3 checkpoint and CUDA are required")
    inputs = open_inputs(V3)
    assert inputs.checkpoint["step"] == 10000
    assert inputs.checkpoint["budget_updates"] == {"16320": 10000}
    from hla_tok.evaluation.benchmark import transfer

    video = next(iter(inputs.cohort))
    source = transfer(inputs.reader.read("validation", video, 0)[None])
    model = inputs.model
    receiver = ContinuousReceiver.from_model(model)
    with torch.inference_mode(), torch.autocast("cuda", enabled=precision == "bf16", dtype=torch.bfloat16):
        native = model.encode_training(source, MAX_K, quantize=False)
        values = encode_continuous(model, source)
        reference = model.decode_training(native, MAX_K)
        assert torch.equal(values, native.values)
        assert reference.shape[-2:] == NATIVE_SHAPE
        for order in (RankOrder.native("cuda"), RankOrder.reversed("cuda")):
            assert torch.equal(receiver.decode(order.apply(values), order), reference)


@pytest.mark.parametrize("count", (1, 120, 121, 254, 255, 256, 16319))
def test_reordered_groups_use_original_rank_and_region(sequence, count):
    order = RankOrder.reversed()
    prefix = truncate(order.apply(sequence), count)
    groups = list(order.recovery_groups(prefix))
    assert sum(values.shape[1] for values, _ in groups) == REGIONS
    assert sum(values.shape[1] * values.shape[2] for values, _ in groups) == count
    start = 0
    for values, ranks in groups:
        width = values.shape[1]
        expected = torch.arange(start, start + width)[:, None] + ranks * REGIONS
        assert torch.equal(values[0, :, :, 0], expected)
        start += width
