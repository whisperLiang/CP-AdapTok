"""E0 compares the public upstream interface and the actual experiment adapter on a real tile."""

import torch

from .metrics import psnr
from .tiling import TileLayout, extract_batches
from .video_reader import read_input, temporal_windows


@torch.inference_mode()
def check_equivalence(adapter, source):
    count = min(source.get("source_frames", adapter.frame_num), adapter.frame_num)
    clip, metadata = read_input(source["path"], count, fps=source["fps"], raw_yuv=source.get("raw_yuv"), metadata=source)
    clip = next(temporal_windows(clip, adapter.frame_num)).frames
    _, tiles = next(extract_batches(clip.to(adapter.device), TileLayout(1080, 1920, adapter.input_size, adapter.input_size), 4))
    tile = tiles[:1]
    before = adapter.state_digest()
    native = adapter.model.encode_eval(tile)
    reference = adapter.model.decode_eval(native["encoded"], latent_mask=native["latent_mask"]).float().clamp(0, 1)
    full = adapter.encode(tile)
    actual = adapter.decode(full, adapter.total_tokens)
    repeated = adapter.encode(tile)
    torch.testing.assert_close(native["encoded"], full["encoded"], atol=0, rtol=0)
    torch.testing.assert_close(reference, actual, atol=0, rtol=0)
    torch.testing.assert_close(full["encoded"], repeated["encoded"], atol=0, rtol=0)
    budget_checks = {}
    for budget in [adapter.total_tokens // 4, adapter.total_tokens // 2, adapter.total_tokens]:
        mask = adapter.budget_mask(budget, 1)
        native_recon = adapter.model.decode_eval(full["encoded"], latent_mask=mask).float().clamp(0, 1)
        torch.testing.assert_close(native_recon, adapter.decode(full, budget), atol=0, rtol=0)
        counts = mask.reshape(1, adapter.groups, -1).sum(-1).tolist()[0]
        assert counts == [budget // adapter.groups] * adapter.groups
        budget_checks[str(budget)] = {"group_counts": counts, "retained_tokens": int(mask.sum()), "official_output_max_error": 0.0}
    after = adapter.state_digest()
    assert before == after
    batch = adapter.decode(adapter.encode(tiles), adapter.total_tokens)[:1]
    mse = (tile - actual).square().mean().item()
    return {"passed": True, "input": metadata, "tile_coordinate": [0, 0], "input_shape": list(tile.shape),
            "output_shape": list(actual.shape), "output_min": actual.min().item(), "output_max": actual.max().item(),
            "official_max_abs_error": (reference - actual).abs().max().item(), "vq_deterministic": True,
            "state_sha256_before": before, "state_sha256_after": after, "weights_unchanged": before == after,
            "group_masks": budget_checks, "batch1_vs_batch4_max_abs_error": (actual - batch).abs().max().item(),
            "tile_mse": mse, "tile_psnr_db": psnr(mse)}
