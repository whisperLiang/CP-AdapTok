# HLA-Tok V2 复现

在 `experiment/hla-tok-v2` 的 worktree 中运行。`LEGACY` 指向原 HLA-Tok worktree 并包含原检查点，`JOINT` 指向封存的 Joint V1 worktree，`ORIGINAL` 指向冻结的原工作区。`OFFICIAL_PYTHON` 和 `OFFICIAL_CHECKPOINT` 指向官方推理环境及已核验权重目录。

## D0 与严格前缀

```bash
OUT=results/hla_tok/v2/diag0
python scripts/hla_tok/run_phase1.py --stage continuous \
  --evaluate "$LEGACY/checkpoints/hla_tok/continuous/step_00500.pt" --output "$OUT/replay_continuous"
python scripts/hla_tok/run_phase1.py \
  --evaluate "$LEGACY/checkpoints/hla_tok/fixed/step_04500.pt" --output "$OUT/replay_vq"
python scripts/hla_tok/run_phase1.py --bypass-vq \
  --evaluate "$LEGACY/checkpoints/hla_tok/fixed/step_04500.pt" --output "$OUT/replay_bypass"

python scripts/hla_tok/v2/diagnose.py --legacy-root "$LEGACY"
python scripts/hla_tok/v2/oracle.py --legacy-root "$LEGACY"
python scripts/hla_tok/run_phase1.py \
  --evaluate "$LEGACY/checkpoints/hla_tok/fixed/step_04500.pt" --output "$OUT/replay_vq_post_refactor"
python -m pytest -q -m gpu tests/hla_tok
python scripts/hla_tok/v2/mutations.py
```

诊断子集和 Train 小样本的源文件哈希及裁块范围见 [冻结帧清单](../../../configs/hla_tok/v2/diagnostic_frames.json)。诊断脚本在 Validation 子集还核验 RGB24 哈希。稠密旁路使用 128D/32D 桥接，两组冻结 Encoder，并调整相同结构的 RGB Decoder。编码字节仅在真实查表和 Fixed13 轮次计量。

## 匹配训练

```bash
python scripts/hla_tok/v2/train.py --stage rgb
python scripts/hla_tok/v2/train.py --stage overfit
python scripts/hla_tok/v2/gates.py overfit
python scripts/hla_tok/v2/train.py --stage pilot
python scripts/hla_tok/v2/gates.py pilot
python scripts/hla_tok/v2/pilot_interventions.py

for arm in R0 R1 R2 R3; do
  python scripts/hla_tok/v2/benchmark.py --arm "$arm"
done

python scripts/hla_tok/v2/extend_pilot.py --candidate R2
python scripts/hla_tok/v2/decisions.py D2
python scripts/hla_tok/v2/pilot_interventions.py --arms R0 R2 \
  --output results/hla_tok/v2/continuous_pilot/seed1/extension_R2/interventions
```

单样本诊断以 3 dB 的同样本增益判定可恢复性，约相当于 MSE 减半。正式连续筛选要求同时满足 2 dB PSNR、0.10 LPIPS-Alex 和正 DISTS 改善，共同样本、像素、更新和初始化哈希均要匹配。补训命令会检查首轮指标和最后一段曲线后才恢复配对训练。Seed 1 达标后才为一个锚定候选执行配对 Seed 2，后续阶段从上一阶段选定检查点开始。本次复现至 D2 停止。

## 图与同帧视频

```bash
python scripts/hla_tok/v2/samples.py pilots
python scripts/hla_tok/v2/samples.py extension
python scripts/hla_tok/v2/samples.py legacy --legacy-root "$LEGACY"
"$OFFICIAL_PYTHON" scripts/hla_tok/v2/samples.py official \
  --legacy-root "$LEGACY" --official-checkpoint "$OFFICIAL_CHECKPOINT"
python scripts/hla_tok/v2/samples.py joint --joint-root "$JOINT"
python scripts/hla_tok/v2/figures.py --alignment-tools "$FIGURE_QA_SCRIPTS"
python scripts/hla_tok/v2/acceptance.py --legacy-root "$LEGACY" --original "$ORIGINAL" --joint "$JOINT"
```

选用冻结 Validation 的 car-roundabout 32–47 帧，全部原生 1920×1080。原版使用 stride128 FP32、480 Token/Tile/4 帧，Joint V1 由档案字节 CLOSED 重放。R0 Fixed13 视频来自核验过的旧实验，PNG 来自本次重放。原生 PNG 和 FFV1 视频的表示类型和检查点见各自清单。图的源数据是阶段 CSV，学习曲线固定每次评估 9 帧，同预算时间测量使用真实 64 帧、两次重复。

复现实验使用独立输出目录或另一 worktree，保留各次运行的正式结果。
