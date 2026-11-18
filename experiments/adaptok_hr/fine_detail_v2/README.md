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

Both measured seeds passed suffix information with R11. To reproduce the conditional phase for this study, run the frozen probes and the matched decoder continuations. D0, D1, D2 and D3 use the winning representation with the original decoder, active residual branch, high-frequency loss, and both changes respectively. C0 continues the original control for an equal number of updates. Seed two repeats the selected decoder, D0 and C0. The driver reads the completed confirmation decision before training.

```bash
.venv-native/bin/python -m experiments.adaptok_hr.fine_detail_v2.probe --names S1_R00 S1_R11 --steps 1000
.venv-native/bin/python -m experiments.adaptok_hr.fine_detail_v2.probe --names S2_R00 S2_R11 --steps 0 10 25 100 300 1000
.venv-native/bin/python -m experiments.adaptok_hr.fine_detail_v2.decoder_stage
```

Run standalone timings on the idle H100 after the quality evaluations. Each measured checkpoint has four budget timings. The 1000-step representation and the selected 1300-step decoder also have three repetitions at K2048.

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 .venv-native/bin/python -m coverage run --branch --source=adaptok_hr --data-file=results/adaptok_hr_fine_detail_v2/.coverage.full -m pytest -o junit_family=legacy -q adaptok_hr/tests cp_adaptok/tests/test_regression.py --junitxml=results/adaptok_hr_fine_detail_v2/tests.xml
.venv-native/bin/python -m coverage json --data-file=results/adaptok_hr_fine_detail_v2/.coverage.full -o results/adaptok_hr_fine_detail_v2/coverage.json
.venv-native/bin/python -m experiments.adaptok_hr.quality audit --coverage results/adaptok_hr_fine_detail_v2/coverage.json --output results/adaptok_hr_fine_detail_v2/quality_audit.json
.venv-native/bin/python -m experiments.adaptok_hr.fine_detail_v2.quality_checks
.venv-native/bin/python -m experiments.adaptok_hr.fine_detail_v2.benchmark --all-validated
.venv-native/bin/python -m experiments.adaptok_hr.fine_detail_v2.verify
.venv-native/bin/python -m experiments.adaptok_hr.fine_detail_v2.figures --qa-scripts /home/whisperliang/.codex/skills/nature-figure/scripts
```

Detailed measurements are in `results/adaptok_hr_fine_detail_v2/`. Local optimizer checkpoints are in `checkpoints/adaptok_hr_fine_detail_v2/`. The tracked evidence directory contains compressed raw records, packets, selected native images, and per-file hashes. The standalone stream measurements include the sender's local decoder, independent receiver and 13-bit framing.

The actual 300-step selection and schedules are readable in [selection.json](evidence/selection.json) and [match_S1_300.json](evidence/match_S1_300.json). The historical A1 control reproduces every training-step MSE and every final model tensor. Recreate the portable evidence and Markdown after completing the applicable gate and benchmarks.

```bash
.venv-native/bin/python -m experiments.adaptok_hr.fine_detail_v2.archive
.venv-native/bin/python -m experiments.adaptok_hr.fine_detail_v2.report
.venv-native/bin/python -m experiments.adaptok_hr.fine_detail_v2.archive --verify-only
```

The compressed evidence uses repository-relative member names beneath the literal run name. Checkpoint metadata contains the full config and digests. The weight files themselves are local. The archive verification command streams every member and checks its SHA256 without extracting anything.
