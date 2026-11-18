# AdapTok-HR Unified RGB–Perceptual

Run from the repository root with the same native Python environment and licensed Train/Validation sources as Fine Detail v2. This branch starts at `1851ecf`; the historical experiment directories remain independent. `ub-mode` action log is in `work_log.md`.

The one production trainer supports `reconstruction.mode: legacy|unified`. The unified objective requires `l1: 1`, a distinct calibrated `perceptual` coefficient and `vq`; nonzero MSE, temporal, chroma, monotonic, high-frequency, suffix and GAN terms are rejected. `network: alex|vgg` selects the frozen pretrained LPIPS trunk. LPIPS averages a 480×270 global view with a native 256×256 view; both decodes share each frame's independently drawn crop coordinates. Training views do not replace full-resolution Validation metrics.

`budget_mode: paired`, `paired_probability: 0.5` samples adjacent pairs uniformly. Every frame consumes the same three RNG draws regardless of pairing probability, preserving main histories and latent auxiliary proposals for U3 (`0.75`). For this matching scheme main-budget endpoint probabilities are 1/6 and interior probabilities 1/3. Auxiliary decodes never advance the reference. Sampling, budget and crop RNGs are independently checkpointed; no missing-RNG compatibility path is supplied.

`encoder_fusion: cross_scale_v1` enables S128→S64→S32 top-down 1×1 projections, bilinear fusion and depthwise residual refinement before the original 16D projections. `fine_input: local_residual|direct` independently controls the existing architecture prior. Default serialization and wire framing retain historical identities. New architecture parameters are initialized without shifting shared tensor initialization.

Use the original repository as the read-only source of each seed's D3 1300 checkpoint, its JSON metadata and full optimizer log. Change `--parent-root` when that checkout resides elsewhere.

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
.venv-native/bin/python -m experiments.adaptok_hr.unified.calibrate --parent-root /home/whisperliang/CP-AdapTok
.venv-native/bin/python -m experiments.adaptok_hr.unified.run pilot --parent-root /home/whisperliang/CP-AdapTok --seeds 1 2
.venv-native/bin/python -m experiments.adaptok_hr.unified.evaluate_pilot --names S1_U0 S1_U1 S1_U2 S1_U3 S2_U0 S2_U1 S2_U2 S2_U3 --steps 1600 --frames 64
.venv-native/bin/python -m experiments.adaptok_hr.unified.benchmark --names S1_U0 S1_U1 S1_U2 S1_U3 S2_U0 S2_U1 S2_U2 S2_U3 --steps 1600
.venv-native/bin/python -m experiments.adaptok_hr.unified.decision A --steps 1600 --frames 64
```

Each A arm makes exactly 300 additional Adam updates. U0 retains the original loss and full-resolution Alex perception but shares the new budget schedule. Therefore it isolates the loss change, not the budget scheduler change. D3 fine-tuning says nothing about from-scratch convergence.

The study command serializes training, native evaluation and idle-GPU benchmarks, then executes only permitted B milestones. It first uses the unchanged-width N0 architecture and subsequently compares the two fusion configurations. All three groups use two independent seeds, identical data/crop/main/auxiliary schedules and optimizers. B validates both 300 and 1000 as requested. At 300, relative quality and collapse are diagnostics; nonfinite gradients stop early. The all-budget rate-distortion/Fine-information and stability gate at 1000 controls 3000/5000. A failed gate archives the negative result and stops longer training.

```bash
.venv-native/bin/python -m experiments.adaptok_hr.unified.study
```

To reproduce one from-scratch phase explicitly, substitute the actual `winner` from `results/adaptok_hr_unified/decision_A.json` below. No parent model or optimizer is loaded for B.

```bash
.venv-native/bin/python -m experiments.adaptok_hr.unified.run scratch --seeds 1 2 --arms N0 N1 N2 --loss-arm U1 --steps 300
.venv-native/bin/python -m experiments.adaptok_hr.unified.evaluate --names S1_N0 S1_N1 S1_N2 S2_N0 S2_N1 S2_N2 --steps 300 --frames 64
.venv-native/bin/python -m experiments.adaptok_hr.unified.decision B --steps 300 --frames 64
```

Native evaluation requests 64 contiguous frames per source. The fixed sources actually supply 50/64/60 frames, totaling 174. Every budget has a separate 64-frame car-roundabout loop; no short source is repeated. The I/P/late-GOP tables use actual source frames. Both the original incremental Fine [1024:2048] and entire transmitted Fine [645:2048] are perturbed; interventions never commit state. Raw packets, original input hashes and all RGB metrics are retained.

```bash
.venv-native/bin/python -m coverage run --branch --source=adaptok_hr --data-file=results/adaptok_hr_unified/.coverage -m pytest -q -o junit_family=legacy adaptok_hr/tests cp_adaptok/tests/test_regression.py --junitxml=results/adaptok_hr_unified/tests_full.xml
.venv-native/bin/python -m coverage json --data-file=results/adaptok_hr_unified/.coverage -o results/adaptok_hr_unified/coverage.json
.venv-native/bin/python -m experiments.adaptok_hr.quality audit --coverage results/adaptok_hr_unified/coverage.json --output results/adaptok_hr_unified/quality_audit.json
.venv-native/bin/python -m experiments.adaptok_hr.unified.mutations
.venv-native/bin/python -m experiments.adaptok_hr.unified.verify
.venv-native/bin/python -m experiments.adaptok_hr.unified.figures --qa-scripts /home/whisperliang/.codex/skills/nature-figure/scripts
.venv-native/bin/python -m experiments.adaptok_hr.unified.report
.venv-native/bin/python -m experiments.adaptok_hr.unified.archive
.venv-native/bin/python -m experiments.adaptok_hr.unified.archive --verify-only
```

All new run output is under `results/adaptok_hr_unified`; all new complete model/optimizer checkpoints are under `checkpoints/adaptok_hr_unified`. The tracked `evidence` package supplies actual raw records, checkpoint metadata, hashes, native samples and figures. Full weight files remain in the independent local checkpoint root. Each native validation directory saves 1920×1080 original and decoded PNGs, including I and P frames. These images compare actual ID prefixes on the common real K2048 history; low-budget P images are not independent-loop examples. Rate-distortion tables use independent loops. PNG is an 8-bit display export; metrics use float32 reconstructions. Charts export exact point data and editable PDF/SVG. Source validation, PDF text and collision audits accompany the figures.

Before any B update, the milestone rule was corrected to complete both requested early validation milestones. `protocol_A_snapshot.json` preserves the exact protocol hash used by the already-running A evaluation; only B continuation policy changed, not A selection or native metric definitions. Existing A evidence can be reproduced with its recorded source revision and frozen protocol. Start fresh result roots when changing a protocol; cached evaluations reject identity mismatches.

`evaluate_pilot` only selects the immutable A protocol and delegates to the same native evaluator. It preserves strict reuse of the completed A measurements after the B-only milestone amendment; it contains no second metric implementation or compatibility fallback. B uses `evaluate` with the current B protocol.
