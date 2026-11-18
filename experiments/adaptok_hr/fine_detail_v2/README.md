# AdapTok-HR Fine detail v2

The four arms change two factors in the shared-codebook, active-prefix A1 model. R00 is the control, R10 removes the local S32 projection mean, R01 supervises the transmitted suffix against the coarse reconstruction residual, and R11 combines both. The native setup uses one H100 and the existing Train and Validation manifests. Run from the repository root.

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
.venv-native/bin/python -m experiments.adaptok_hr.fine_detail_v2.run --seed 1 --steps 300
.venv-native/bin/python -m experiments.adaptok_hr.fine_detail_v2.evaluate --names S1_R00 S1_R10 S1_R01 S1_R11 --steps 300 --frames 8
.venv-native/bin/python -m experiments.adaptok_hr.fine_detail_v2.probe --names S1_R00 S1_R10 S1_R01 S1_R11
.venv-native/bin/python -m experiments.adaptok_hr.fine_detail_v2.study confirm
```

The last command verifies initialization, the data and budget sequence and each optimizer step, then applies the ranking in `protocol.json`. It resumes the screen winner and R00 to 1000. A positive primary information test triggers the independently initialized, matched second seed. Training holds an exclusive per-run lock. An interrupted run reuses a committed optimizer checkpoint, and an interrupted evaluation retains its previous directory. The evaluated packets and causal reference chain are checked again by the delivery audit.

The high budget adds slots 1024–2047 (zero-based). The preceding 379 Fine slots already belong to K1024. The supervisor averages the BT.709 luma reconstruction residual in 8×8 subcells of each 32×32 tile, centers over valid pixels, and compares its 16 values with the actual discrete embedding in transmission order. Missing bottom pixels have zero weight. Only the main decode advances training history.

Each independent budget loop owns separate sender and receiver model instances. The same-context comparison replays the actual K2048 history and truncates its packets. Diagnostic replacement uses the next video in the preregistered source order, at the same frame index and position, on the common length of all three sources. Perturbations are decoded in the original context and never committed. The full 174-frame pool comprises 50, 64 and 60 frames.

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 .venv-native/bin/python -m coverage run --branch --source=adaptok_hr --data-file=results/adaptok_hr_fine_detail_v2/.coverage.full -m pytest -o junit_family=legacy -q adaptok_hr/tests cp_adaptok/tests/test_regression.py --junitxml=results/adaptok_hr_fine_detail_v2/tests.xml
.venv-native/bin/python -m coverage json --data-file=results/adaptok_hr_fine_detail_v2/.coverage.full -o results/adaptok_hr_fine_detail_v2/coverage.json
.venv-native/bin/python -m experiments.adaptok_hr.quality audit --coverage results/adaptok_hr_fine_detail_v2/coverage.json --output results/adaptok_hr_fine_detail_v2/quality_audit.json
.venv-native/bin/python -m experiments.adaptok_hr.fine_detail_v2.quality_checks
.venv-native/bin/python -m experiments.adaptok_hr.fine_detail_v2.verify
.venv-native/bin/python -m experiments.adaptok_hr.fine_detail_v2.figures --qa-scripts /home/whisperliang/.codex/skills/nature-figure/scripts
```

Detailed measurements are in `results/adaptok_hr_fine_detail_v2/`; local optimizer checkpoints are in `checkpoints/adaptok_hr_fine_detail_v2/`. The tracked evidence directory contains compressed raw records, packets, selected native images, and per-file hashes. GPU timings for training or quality evaluation do not serve as speed benchmarks.
