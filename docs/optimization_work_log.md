# CP-AdapTok optimization action log

Baseline: 911c9cc5a206f90b529ff65299350248a04b4b12. Work began 2026-11-15 (Asia/Shanghai).

- Read the complete user task and ub-mode skill; inspected the requested codec, training, data, runtime, streaming and evaluation paths. Working tree was clean. No user edits to preserve.
- Confirmed H100 PCIe 80 GB, PyTorch 2.9.1/cu130, official weights and patched native latent API. Another user's GPU process exists; do not assume exclusive device use.
- Saved checkpoint hashes, source/config/environment manifest under `results/optimization_20261115/baseline_manifest.json`. Original checkpoints and experiment directories remain intact.
- Started unchanged baseline pytest. Added failing behavioral tests for diagnostics, stochastic/continuous VQ and real-sample initialization before implementation.
- Named crapper/dryer/mutator tools are unavailable; use repository radon/coverage/AST and isolated mutation scripts and disclose their limits.
- Baseline: 94/94 existing tests passed. Real 256-pixel clip: I uses 2 codes, P uses 1; P pre-VQ token cosine ~0.999. The precise diagnostic is recorded instead of repeating an unconditional "every frame uses one" claim.
- Implemented one shared quantizer with full-codebook cosine sampling, absolute-step temperature schedules, deterministic inference, diagnostic-only continuous mode, reproducible spherical K-means and opt-in window-based code recovery with optimizer-row reset. No uniform-entropy penalty added.
- Restricted the default exact native teacher to I frames; legacy mismatched I/P targets require an explicit diagnostic flag. Same-context P teachers are rejected until numerically verified. Preserved the correct native decoder latent API.
- Added per-code histograms, token variance/rank/cosine, nearest distances, training/deterministic assignment statistics, prefix latent deltas, losses and per-module gradients. Checkpoints retain VQ window state and RNG.
- Continuous real-frame 100-update run reduced image and teacher errors; it did not establish useful prefix gains. Began equal-100-update VQ ablations.
- Added rectangular native crops, source-content split identities and image-sequence input. Verified 256/512/540p/720p/1080p padding and crop behavior. Existing source files remain untouched.
- DAVIS official train-only full-resolution subset is being imported with verified HTTP ranges and ZIP CRCs. The initial full download was too slow; its partial file is retained. No validation/test sequence is silently reassigned to training.
- New red tests reproduced missing features before edits. A mean-reduction assertion exposed floating accumulation sensitivity; replaced it with a stricter per-pixel equality assertion. Updated suite: 109 passed before the final continuous-overfit test addition.
