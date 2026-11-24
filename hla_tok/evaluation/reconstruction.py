"""Native Validation with discrete round trips and an explicit payload rate ledger."""

from collections import defaultdict

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

from hla_tok import Geometry
from hla_tok.evaluation.benchmark import transfer
from hla_tok.evaluation.io import write_csv, write_json
from hla_tok.evaluation.metrics import summarize_quality
from hla_tok.progressive.geometry import MAX_QUERIES, NATIVE_REGIONS, NATIVE_SHAPE
from hla_tok.streaming.fixed13 import pack_tokens_13bit, payload_ledger, unpack_tokens_13bit


class CodeUsage:
    def __init__(self, size):
        self.size = size
        self.rank = torch.zeros(MAX_QUERIES // 32, size, dtype=torch.long)
        self.region = torch.zeros(NATIVE_REGIONS, size, dtype=torch.long)
        self.cosine, self.repeat_count, self.pairs = 0.0, 0, 0

    @staticmethod
    def _update(table, ids, category):
        keys = category * table.shape[1] + ids
        table.view(-1).index_add_(0, keys, torch.ones_like(keys))

    def update(self, ids, representations=None):
        ids = ids.flatten().cpu()
        positions = torch.arange(len(ids))
        self._update(self.rank, ids, positions // (NATIVE_REGIONS * 32))
        self._update(self.region, ids, positions % NATIVE_REGIONS)
        if len(ids) > NATIVE_REGIONS:
            self.repeat_count += ids[:-NATIVE_REGIONS].eq(ids[NATIVE_REGIONS:]).sum().item()
            self.pairs += len(ids) - NATIVE_REGIONS
            if representations is not None:
                z = representations.flatten(0, 1).float()
                self.cosine += F.cosine_similarity(z[:-NATIVE_REGIONS], z[NATIVE_REGIONS:]).sum().item()

    def _summaries(self, counts):
        rows = []
        for category, hits in enumerate(counts):
            active = int(hits.count_nonzero())
            if active:
                p = hits[hits > 0].double() / hits.sum()
                entropy = -(p * p.log2()).sum().item()
                top = hits.topk(min(8, active))
                rows.append({"category": category, "active_ratio": active / self.size,
                             "dead_ratio": 1 - active / self.size, "entropy_bits": entropy,
                             "perplexity": 2**entropy, "token_count": int(hits.sum()),
                             "top_ids": top.indices.tolist(), "top_counts": top.values.tolist()})
        return rows

    def report(self):
        return {
            "overall": self._summaries(self.rank.sum(0, keepdim=True))[0],
            "by_rank_groups_of_32": self._summaries(self.rank),
            "by_region": self._summaries(self.region),
            "adjacent_local_ranks": self.pairs,
            "same_code_fraction": self.repeat_count / max(self.pairs, 1),
            "adjacent_query_cosine_mean": self.cosine / max(self.pairs, 1),
        }


def save_native(output, label, image):
    output.mkdir(parents=True, exist_ok=True)
    pixels = image[0].float().clamp(0, 1).mul(255).round().byte().permute(1, 2, 0).cpu().numpy()
    Image.fromarray(pixels).save(output / f"{label}.png")
    Image.fromarray(pixels[432:648, 768:1152]).save(output / f"{label}_crop.png")


@torch.inference_mode()
def validate(model, reader, cohort, quality, budgets, output, *, quantize=True, subset=False, bf16=True):
    geometry = Geometry(*NATIVE_SHAPE)
    rows = defaultdict(list)
    usage = {k: CodeUsage(model.quantizer.embedding.num_embeddings) for k in budgets} if quantize else {}
    model.eval()
    for video, count in cohort.items():
        indices = (0, count // 2, count - 1) if subset else range(count)
        for index in indices:
            source = transfer(reader.read("validation", video, index)[None])
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
                if quantize:
                    encoded = model.encode_training(source, max(budgets))
                else:
                    encoded = model.encode_training(source, max(budgets), quantize=False)
            for k in budgets:
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
                    if quantize:
                        ids = encoded.ids[:, :k]
                        payload = pack_tokens_13bit(ids[0].tolist())
                        received = torch.tensor([unpack_tokens_13bit(payload, k)], dtype=torch.long, device="cuda")
                        if not torch.equal(ids, received):
                            raise RuntimeError("validation receiver did not recover the integer prefix")
                        reconstruction = model.decode_prefix(received, geometry)
                        usage[k].update(received, encoded.continuous[:, :k])
                    else:
                        reconstruction = model.decode_training(encoded, k)
                rate = payload_ledger(len(payload), k, 1, *NATIVE_SHAPE) if quantize else {}
                rows[k].append({"video": video, "frame": index, "budget": k} | rate | quality(source, reconstruction))
                if not subset and index == 32 and video.endswith("car-roundabout"):
                    save_native(output / "samples", f"K{k}", reconstruction)
                    save_native(output / "samples", "source", source)
    report = []
    for k in budgets:
        summary = {"budget": k, "quantized": quantize, "matched_174": not subset} | summarize_quality(rows[k])
        if quantize:
            num_bytes = sum(r["total_bytes"] for r in rows[k])
            summary |= payload_ledger(num_bytes, k * len(rows[k]), len(rows[k]), *NATIVE_SHAPE)
        report.append(summary)
    write_csv(output / "validation_metrics.csv", report)
    write_csv(output / "per_frame.csv", [r for k in budgets for r in rows[k]])
    if quantize:
        write_json(output / "codebook.json", {k: stats.report() for k, stats in usage.items()})
        if not subset:
            np.savez_compressed(output / "codebook_distributions.npz", **{
                f"K{k}_{name}": table.numpy() for k, stats in usage.items()
                for name, table in (("rank", stats.rank), ("region", stats.region))
            })
    return report
