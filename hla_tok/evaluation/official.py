"""Read-only original AdapTok baseline with real independent Fixed13 payloads."""

import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import torch

from experiments.original_adaptok_1080p.tiling import Stitcher, TileLayout, extract_batches
from hla_tok.evaluation.data import AuditedFrames
from hla_tok.evaluation.io import timed_cuda, write_csv
from hla_tok.evaluation.metrics import FullFrameQuality, sha256_file, summarize_quality
from hla_tok.streaming.fixed13 import pack_tokens_13bit, payload_ledger, unpack_tokens_13bit

GROUP_BUDGETS = (96, 128, 256, 384, 480, 512)
CLIP_LENGTH = 16


class OfficialBaseline:
    def __init__(self, source: Path, checkpoint: Path, manifest: dict):
        head = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
        status = subprocess.check_output(
            ["git", "-C", str(source), "status", "--porcelain", "--untracked-files=no"], text=True
        )
        if head != manifest["official_git_commit"] or status.strip():
            raise ValueError("official AdapTok source differs from the pinned clean checkout")
        for name, field in (("config.json", "config_sha256"), ("model.safetensors", "model_safetensors_sha256")):
            if sha256_file(checkpoint / name) != manifest[field]:
                raise ValueError(f"official {name} differs from pinned provenance")
        sys.path.insert(0, str(source))
        from models.adaptok import AdapTok

        self.model = AdapTok.from_pretrained(str(checkpoint)).eval().requires_grad_(False).cuda()
        self.model.set_vq_eval_deterministic(True)
        self.groups = self.model.mask_generator.tot_groups
        self.ranks = self.model.mask_generator.total_toks
        recorded = self.model.frame_num, self.model.input_size, self.model.codebook_size, self.groups, self.ranks
        if recorded != (16, 128, 8192, 4, 512):
            raise ValueError("unexpected official checkpoint geometry")

    def encode(self, tiles):
        encoded = self.model.encode_eval(tiles, with_latent_mask=False)
        ids = encoded["bottleneck_rep"].reshape(len(tiles), self.groups, self.ranks)
        return ids, encoded["encoded"]

    def decode(self, ids, budget):
        counts = torch.full((len(ids), self.groups), budget, device=ids.device, dtype=torch.int64)
        mask = self.model.mask_generator._left_masking_by_group_adap(counts, device=ids.device).bool()
        # Missing positions are determined by public geometry. Their placeholder values are masked.
        stored = torch.nn.functional.pad(ids, (0, self.ranks - budget)).flatten(1)
        values = self.model.bottleneck.decode(stored)
        return self.model.decode_eval(values, latent_mask=mask).float().clamp(0, 1)

    def parity(self, tiles, budget):
        ids, projected = self.encode(tiles)
        counts = torch.full((len(tiles), self.groups), budget, device=ids.device, dtype=torch.int64)
        mask = self.model.mask_generator._left_masking_by_group_adap(counts, device=ids.device).bool()
        reference = self.model.decode_eval(projected, latent_mask=mask).float().clamp(0, 1)
        independent = self.decode(ids[..., :budget], budget)
        difference = (reference - independent).abs()
        return {"rgb_mean_abs": difference.mean().item(), "rgb_max_abs": difference.max().item(),
                "equal": torch.equal(reference, independent)}


def wire_indices(ids, k):
    cpu = ids[..., :k].contiguous().cpu()
    payload = pack_tokens_13bit(cpu.flatten().tolist())
    recovered = torch.tensor(unpack_tokens_13bit(payload, cpu.numel())).view(cpu.shape)
    if not torch.equal(cpu, recovered):
        raise RuntimeError("integer indices did not round-trip")
    return payload, recovered.to(ids.device)


def source_windows(reader: AuditedFrames, cohort: dict[str, int]):
    for video, count in cohort.items():
        for start in range(0, count, CLIP_LENGTH):
            frames = [reader.read("validation", video, i) for i in range(start, min(count, start + CLIP_LENGTH))]
            clip = torch.stack(frames + [frames[-1]] * (CLIP_LENGTH - len(frames)))
            yield video, start, len(frames), clip


def reconstruct(clip, baseline, layout, budgets, tile_batch):
    clip = clip.cuda().float().div_(255)
    enc_wall, enc_cuda = 0.0, 0.0
    writers = {k: Stitcher(layout, CLIP_LENGTH, 3, clip.device) for k in budgets}
    timings = {k: defaultdict(float) for k in budgets}
    for coordinates, tiles in extract_batches(clip, layout, tile_batch):
        (ids, _), wall, cuda = timed_cuda(lambda: baseline.encode(tiles))
        enc_wall += wall
        enc_cuda += cuda
        for k, writer in writers.items():
            (payload, received), wall, _ = timed_cuda(lambda: wire_indices(ids, k))
            item = timings[k]
            item["wire_ms"] += wall
            item["bytes"] += len(payload)
            item["tokens"] += received.numel()
            decoded, wall, cuda = timed_cuda(lambda: baseline.decode(received, k))
            item["decode_ms"] += wall
            item["decode_cuda_ms"] += cuda
            _, wall, _ = timed_cuda(lambda: writer.add(coordinates, decoded))
            item["stitch_ms"] += wall
    return clip, {k: w.finish() for k, w in writers.items()}, {
        k: dict(timings[k]) | {"encode_ms": enc_wall, "encode_cuda_ms": enc_cuda} for k in writers
    }


def official_window(clip, baseline, layout, budgets, tile_batch):
    torch.cuda.reset_peak_memory_stats()
    (target, pictures, details), wall, _ = timed_cuda(lambda: reconstruct(clip, baseline, layout, budgets, tile_batch))
    peak = torch.cuda.max_memory_allocated() / 2**20
    return target, pictures, details, peak, wall


def summarize_point(frame_rows, windows, k):
    rows = [row for row in frame_rows if row["tokens_per_tile_block"] == k]
    points = [row for row in windows if row["tokens_per_tile_block"] == k]
    result = summarize_quality(rows)
    token_count = int(sum(p["tokens"] for p in points))
    byte_count = int(sum(p["bytes"] for p in points))
    result |= {"tokens_per_tile_block": k} | payload_ledger(byte_count, token_count, len(rows), 1080, 1920)
    for field in ("encode_ms", "decode_ms", "encode_cuda_ms", "decode_cuda_ms", "wire_ms", "stitch_ms"):
        result[field] = sum(p[field] for p in points)
    result["neural_fps"] = 1000 * len(rows) / (result["encode_ms"] + result["decode_ms"])
    result["peak_allocated_mib_all_budgets"] = max(p["peak_allocated_mib"] for p in points)
    return result


@torch.inference_mode()
def quality_run(reader, cohort, baseline, metrics: FullFrameQuality, stride, precision, tile_batch, output: Path,
                budgets=GROUP_BUDGETS):
    layout = TileLayout(1080, 1920, 128, stride)
    frame_rows, windows = [], []
    for video, start, valid, clip in source_windows(reader, cohort):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=precision == "bf16", cache_enabled=True):
            target, pictures, details, peak, wall = official_window(clip, baseline, layout, budgets, tile_batch)
        tags = {"video": video, "stride": stride, "precision": precision, "tile_batch": tile_batch}
        for k in budgets:
            condition = tags | {"tokens_per_tile_block": k}
            windows.append(condition | details[k] | {"start": start, "valid_frames": valid, "padded_frames": 16 - valid,
                                                     "peak_allocated_mib": peak, "all_budgets_wall_ms": wall})
            for index in range(valid):
                row = metrics(target[index:index+1], pictures[k][index:index+1])
                frame_rows.append(condition | {"frame": start + index} | row)
        print(f"Official stride {stride} {precision} {video} frame {start} + {valid}, {wall / 1000:.2f}s codec",
              flush=True)
        write_csv(output / "per_frame.csv", frame_rows)
        write_csv(output / "windows.csv", windows)
    return [summarize_point(frame_rows, windows, k) | {
        "method": "Original AdapTok", "stride": stride, "precision": precision, "tile_batch": tile_batch
    } for k in budgets]


@torch.inference_mode()
def precision_diagnostics(clip, baseline, output, write_json):
    tiles = next(extract_batches(clip.cuda().float() / 255, TileLayout(1080, 1920), 1))[1]
    sample = {}
    for precision in ("fp32", "bf16"):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=precision == "bf16"):
            parity = baseline.parity(tiles, 256)
            ids, _ = baseline.encode(tiles)
            reference = baseline.decode(ids[..., :256], 256)
            pairs = []
            for end in (4, 8, 12):
                past_only = torch.cat((tiles[:, :, :end], tiles[:, :, end-1:end].expand(-1, -1, 16-end, -1, -1)), dim=2)
                (prefix, _), enc_ms, _ = timed_cuda(lambda: baseline.encode(past_only))
                reconstruction, dec_ms, _ = timed_cuda(lambda: baseline.decode(prefix[..., :256], 256))
                pairs.append({
                    "frames_available": end, "integer_ids_equal": torch.equal(ids[:, :end//4], prefix[:, :end//4]),
                    "rgb_max_abs": (reference[:, :, :end] - reconstruction[:, :, :end]).abs().max().item(),
                    "tile_encode_ms": enc_ms, "tile_decode_ms": dec_ms,
                })
            sample[precision] = {"integer_decode_vs_upstream": parity, "past_only_recompute": pairs}
    write_json(output / "precision_diagnostics.json", sample)


@torch.inference_mode()
def tile_batch_diagnostics(clip, baseline, output):
    rows = []
    layout = TileLayout(1080, 1920)
    for precision in ("bf16", "fp32"):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=precision == "bf16"):
            for batch in (4, 8, 16):
                (_, _, details), elapsed, cuda = timed_cuda(lambda: reconstruct(clip, baseline, layout, (256,), batch))
                rows.append({"precision": precision, "batch": batch, "window_frames": 16,
                             "window_wall_ms": elapsed, "window_cuda_ms": cuda, "codec_fps": 16000 / elapsed}
                            | details[256])
                print(f"Official stride128 {precision} tile batch {batch}, {16000 / elapsed:.2f} codec FPS", flush=True)
                write_csv(output / "tile_batch.csv", rows)
