"""Behavior mutations exercised in disposable copies, with real regression assertions."""

# ruff: noqa: E402
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from hla_tok.evaluation.io import write_json
from scripts.hla_tok.mutation_check import mutate
from scripts.hla_tok.v2.protocol import OUTPUT, ROOT

ARCH = "test_v2_architecture.py::"
CASES = (
    ("drop_spatial_bias", "models/local.py", "bias = self.spatial_bias[ranks]", "bias = None",
     ARCH + "test_production_anchored_query_path_uses_bias"),
    ("scramble_anchors", "models/anchors.py", "first = grid[selected]", "first = grid[selected].flip(0)",
     ARCH + "test_anchor_grid_uses_deterministic_order_and_two_balanced_rounds"),
    ("bidirectional_rank", "models/rank_causal.py", "return keys[None, :] <= queries[:, None]",
     "return keys[None, :] >= 0 * queries[:, None]",
     ARCH + "test_rank_mask_uses_absolute_indices_and_cache_carries_history_and_gradients"),
    ("ignore_history", "models/rank_causal.py", "mixed = F.scaled_dot_product_attention(q, past_k, past_v, attn_mask=allowed)",
     "past_k = torch.cat((torch.zeros_like(past_k[..., :start, :]), k), dim=-2)\n"
     "            past_v = torch.cat((torch.zeros_like(past_v[..., :start, :]), v), dim=-2)\n"
     "            mixed = F.scaled_dot_product_attention(q, past_k, past_v, attn_mask=allowed)",
     ARCH + "test_rank_mask_uses_absolute_indices_and_cache_carries_history_and_gradients"),
    ("inspect_suffix_storage", "models/codec.py", "values = self.quantizer.lookup(ids)",
     "values = self.quantizer.lookup(ids.as_strided((ids.shape[0], ids.shape[1] + 1), ids.stride()))",
     ARCH + "test_receiver_cannot_inspect_unreceived_backing_storage"),
    ("read_encoder_in_receiver", "models/codec.py", "return self._decode(values, geometry)",
     "return self._decode(values, geometry) + next(self.encoder.parameters()).sum()",
     ARCH + "test_direct_projection_stops_at_two_spatial_passes_and_decodes_independently"),
    ("vq_lookup_bypass", "models/codec.py", "values = self.quantizer.lookup(ids)",
     "values = torch.zeros(*ids.shape, 32, device=ids.device)",
     ARCH + "test_direct_projection_stops_at_two_spatial_passes_and_decodes_independently"),
    ("illegal_test_alias", "evaluation/data.py", 'raise PermissionError("manifest entry aliases sealed Test content")',
     "return path.read_bytes()", "test_data_boundary.py::test_sealed_test_alias_is_rejected_before_decode"),
)


def main():
    target = OUTPUT / "mutations"
    target.mkdir(exist_ok=False)
    results = [mutate(ROOT, target, case) for case in CASES]
    write_json(target / "summary.json", results)
    if not all(row["killed"] for row in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
