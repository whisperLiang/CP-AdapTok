"""Run the released AdapTok weights on the same audited Validation pixels."""

from pathlib import Path

import torch

from baselines.adaptok.model_adapter import OfficialAdapter
from baselines.adaptok.reconstruction import reconstruct_window
from baselines.adaptok.tiling import TileLayout
from baselines.adaptok.video_reader import temporal_windows
from common.artifacts import write_csv, write_json
from common.inputs import inputs, source_clips
from experiments.pht_tok.study import Quality, summarize_quality

BUDGETS = (512, 1024, 2048)


def assess_clip(clip, adapter, quality):
    layout = TileLayout(1080, 1920, adapter.input_size, adapter.input_size)
    output = []
    for window in temporal_windows(clip.rgb, adapter.frame_num):
        frames, timing = reconstruct_window(window.frames, adapter, layout, BUDGETS)
        for budget, reconstructed in frames.items():
            original = window.frames[:window.valid_frames]
            scores = quality.score_rgb(original, reconstructed[:window.valid_frames])
            output.append({"source": clip.clip.video, "start": window.offset, "fps": clip.fps,
                           "unit_valid": window.valid_frames, "physical_bytes": None,
                           "budget": budget, "frame_metrics": scores, "timings": timing,
                           "tile_size": adapter.input_size, "tile_count": len(layout.coordinates),
                           "spatial_padding": list(layout.padded_shape), "resize": False})
    return output


@torch.inference_mode()
def adaptok(checkpoint: Path, output: Path):
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    reader, plans = inputs()
    clips = source_clips(reader, plans["validation"])
    if reader.counts != {"train": 0, "validation": 174, "test": 0}:
        raise ValueError("AdapTok control must read the frozen Validation subset")
    model = OfficialAdapter(str(checkpoint), device="cuda")
    quality = Quality()
    rows = []
    for clip in clips:
        rows.extend(assess_clip(clip, model, quality))
        print("AdapTok", clip.clip.video, flush=True)
    metrics = []
    for budget in BUDGETS:
        result = summarize_quality([row for row in rows if row["budget"] == budget])
        metrics.append({"method": "original_adaptok_quality_only", "tokens_per_tile_window": budget,
                        "mbps": None, "bpp": None, "bytes": None, **result})
    write_csv(output / "adaptok_quality.csv", metrics)
    write_json(output / "adaptok_validation.json", {"reads": reader.counts, "checkpoint": model.verify_unchanged(),
        "temporal_window": model.frame_num, "tile_stride": model.input_size,
        "wire": "The upstream tokenizer exposes token budgets without an entropy-coded transmission format.",
        "windows": rows})
