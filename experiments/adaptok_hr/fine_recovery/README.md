# Fine Token recovery

Start at `cc87bd4` on `experiment/adaptok-hr-fine-recovery`. All commands run from the repository root with the existing `.venv-native` runtime and local native 1080p manifest. No data downloads are performed. The historical branch, old results and checkpoints stay unchanged.

Set `CUBLAS_WORKSPACE_CONFIG=:4096:8` for every GPU command. The single H100 runs all training sequentially. Quality evaluation may overlap training; those latency fields are **not** used for throughput claims. Standalone benchmarks require an otherwise idle GPU and no metric networks in that process.

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
.venv-native/bin/python -m experiments.adaptok_hr.fine_recovery.diagnose
.venv-native/bin/python -m experiments.adaptok_hr.fine_recovery.gradient_probe
.venv-native/bin/python -m experiments.adaptok_hr.fine_recovery.run train --steps 300
.venv-native/bin/python -m experiments.adaptok_hr.fine_recovery.run match --steps 300
.venv-native/bin/python -m experiments.adaptok_hr.fine_recovery.run evaluate --steps 300 --frames 8
.venv-native/bin/python -m experiments.adaptok_hr.fine_recovery.report screen
```

Continue **only the two names in `results/adaptok_hr_fine_recovery/selection_300.json`**:

```python
import json
from pathlib import Path
from experiments.adaptok_hr.fine_recovery.run import train, evaluate, verify_matching

names = json.loads(Path("results/adaptok_hr_fine_recovery/selection_300.json").read_text())["selected"]
for name in names:
    train(name, 1000)  # optimizer, sampler, budget RNG restored from step 300
verify_matching(names, 1000)
for name in names:
    evaluate(name, 1000, 64)
```

For each measured checkpoint (replace A0/300 with the actual name/step):

```bash
.venv-native/bin/python -m experiments.adaptok_hr.fine_recovery.benchmark \
  --checkpoint checkpoints/adaptok_hr_fine_recovery/A0/step_000300.pt \
  --output results/adaptok_hr_fine_recovery/A0/benchmark_000300.json
.venv-native/bin/python -m experiments.adaptok_hr.fine_recovery.report decide
.venv-native/bin/python -m experiments.adaptok_hr.fine_recovery.mechanism
.venv-native/bin/python -m experiments.adaptok_hr.fine_recovery.quality
.venv-native/bin/python -m pytest adaptok_hr/tests cp_adaptok/tests/test_regression.py -q
```

The preregistered `protocol.json` controls selection and conditional later stages. D1/D2/D3 are not allowed until the transmitted Fine IDs demonstrably carry information. If that still fails at 1000 updates, preserve negative results and stop expansion. A positive monotonicity loss or increased code count alone is insufficient.

Plotting follows the established Python workflow. Pass the installed nature-figure skill's `scripts` directory to `figures --qa-scripts ...`; run its source, PDF glyph and collision audits on the actual exported figures. Then run `report report` to generate the Markdown report and archive raw JSON/JSONL/XML plus checkpoint identities. No figure uses invented data or independent-seed error bars.

Frozen continuous bypasses and Fine interventions are explicitly `diagnostic_only`; they never enter standard RD curves. The fixed 300-step screen uses 3×8 Validation frames. The 1000-step verification uses the same three preselected Validation sources up to 64 frames each (174 actual frames); it is not all 21 Validation sources. The benchmark separately uses 64 contiguous car-roundabout frames, both independent endpoints, GOP32 and resets at index32. No old Final Test source participates in model selection.
