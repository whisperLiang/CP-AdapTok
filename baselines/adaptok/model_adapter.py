"""Strict upstream loading and native group masking. No scorer, IPAL, or CP codec."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import torch

from common.integrity import sha256_file as file_sha256

ROOT = Path(__file__).resolve().parents[2]
REPO_ID = "yeahhhh326/AdapTok"
DEFAULT_REVISION = "c514b22615a140c06f4c9792e93cbef9e5c79f32"


def validate_budget(budget, groups, tokens_per_group):
    if type(budget) is not int or not 0 < budget <= groups * tokens_per_group or budget % groups:
        raise ValueError("budget must be positive, divisible by temporal groups and <= all native tokens")
    return budget // groups


def verified_checkpoint(local_dir=None, revision=DEFAULT_REVISION):
    lock = json.loads((ROOT / "configs/baselines/adaptok_checkpoint.json").read_bytes())
    if revision != lock["revision"]:
        raise ValueError("only the frozen official AdapTok revision is supported")
    if local_dir is None:
        from huggingface_hub import snapshot_download

        local_dir = snapshot_download(REPO_ID, revision=revision, allow_patterns=list(lock["files"]))
    directory = Path(local_dir).resolve()
    hashes = {name: file_sha256(directory / name) for name in lock["files"]}
    if hashes != lock["files"]:
        raise ValueError("checkpoint differs from the frozen official Hugging Face revision")
    return directory, {"repo_id": REPO_ID, "revision": revision, "config_sha256": hashes["config.json"],
                       "model_safetensors_sha256": hashes["model.safetensors"], "official_hash_verified": True}



class OfficialAdapter:
    def __init__(self, checkpoint=None, revision=DEFAULT_REVISION, device="cuda:0"):
        directory, self.manifest = verified_checkpoint(checkpoint, revision)
        upstream = ROOT / "third_party/AdapTok"
        status = subprocess.check_output(["git", "-C", str(upstream), "status", "--porcelain", "--untracked-files=no"], text=True)
        commit = subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip()
        lock = json.loads((ROOT / "configs/baselines/adaptok_checkpoint.json").read_bytes())
        if status.strip() or commit != lock["source_commit"]:
            raise ValueError("official AdapTok must use the clean pinned source")
        sys.path.insert(0, str(upstream))
        from models.adaptok import AdapTok

        # Upstream's from_pretrained calls load_state_dict with strict=True by default.
        self.model = AdapTok.from_pretrained(str(directory))
        self.model.eval()
        self.model.set_vq_eval_deterministic(True)
        self.model.requires_grad_(False)
        self.device = torch.device(device)
        self.model.to(self.device)
        self.input_size, self.frame_num = self.model.input_size, self.model.frame_num
        self.total_tokens = self.model.bottleneck_token_num
        self.groups = self.model.mask_generator.tot_groups
        self.tokens_per_group = self.model.mask_generator.total_toks
        if self.groups * self.tokens_per_group != self.total_tokens:
            raise ValueError("native group geometry does not cover all latent tokens")
        self.manifest.update({"directory": str(directory), "config": json.loads((directory / "config.json").read_text()),
                              "official_git_commit": commit,
                              "input_size": self.input_size, "frame_num": self.frame_num,
                              "codebook_size": self.model.codebook_size, "latent_tokens": self.total_tokens,
                              "temporal_groups": self.groups, "tokens_per_group": self.tokens_per_group,
                              "parameters": sum(p.numel() for p in self.model.parameters()),
                              "dtype": str(self.model.dtype), "strict_loading": True,
                              "scorer_used": False, "ipal_used": False, "cross_window_state": False})
        self.initial_digest = self.state_digest()
        self.manifest["state_sha256_before"] = self.initial_digest

    def state_digest(self):
        digest = hashlib.sha256()
        for name, value in self.model.state_dict().items():
            digest.update(name.encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()

    def verify_unchanged(self):
        digest = self.state_digest()
        if digest != self.initial_digest:
            raise RuntimeError("official model state changed during inference")
        self.manifest.update(state_sha256_after=digest, weights_unchanged=True)
        return self.manifest

    def budget_mask(self, budget, batch_size):
        count = validate_budget(budget, self.groups, self.tokens_per_group)
        native_counts = torch.full((batch_size, self.groups), count, dtype=torch.int64, device=self.device)
        return self.model.mask_generator._left_masking_by_group_adap(native_counts, device=self.device).bool()

    @torch.inference_mode()
    def encode(self, tiles):
        if tuple(tiles.shape[1:]) != (3, self.frame_num, self.input_size, self.input_size):
            raise ValueError("tile input must match the unmodified checkpoint geometry")
        return self.model.encode_eval(tiles, with_latent_mask=False)

    @torch.inference_mode()
    def decode(self, encoded, budget):
        mask = self.budget_mask(budget, len(encoded["encoded"]))
        reconstructed = self.model.decode_eval(encoded["encoded"], latent_mask=mask)
        return reconstructed.float().clamp(0, 1)
