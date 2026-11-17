# Fixed-2048 原生 1080p 重建专项

所有命令在仓库根目录运行。依赖沿用 `pyproject.toml` 的 `eval,dev` extras，使用 CUDA PyTorch 环境。
实际执行环境：H100 PCIe 80GB，PyTorch 2.9.1+cu130；本机解释器 `/home/whisperliang/CP-AdapTok/.venv-native/bin/python`。

本目录复用原始五项重建组件，完整模型先严格核对现有 Checkpoint。实验对象不持有 Scorer、Entropy Model 或传输模块；未调用这些模块的前向，不产生码流或 bpp。

- R0：原始 Encoder `y` 直接到 Backbone Decoder。
- R0-spatial：`y` 固定池化 17×30 后双线性恢复。
- R1：显式 `project_features → project_out`，eval 同样绕过 Codebook。
- R2：原 VQ 的 deterministic nearest + 原 STE/VQ Loss。
- raw_192：独立训练控制，仅用于与相同训练路线的 R1 比较，不能解释为精确的16D不可逆信息损失。
- R3：仅迁移 Decoder Grid，二维 bilinear 插值 query，逐参数 ledger 后严格加载。

`paired` 用同一历史 Original128 上下文比较全部候选，提交发生在比较完成后。
`paired_full2048` 用 R1 的完整重建历史比较 R1/R2；两者当前帧 Encoder Latent 和 Tokenizer 输出完全共享。
`full2048` 每条路径独立闭环，`original128` 的显示图像固定2048但状态只由独立VQ128路径提交，`i_only` 每帧空状态。
训练只允许 `full2048`，没有多预算循环或 Token Mask。

## 复现顺序

```bash
python -m pytest experiments/fixed2048/tests cp_adaptok/tests -q
python -m experiments.fixed2048.evaluate --output results/fixed2048/baseline --sensitivity
python -m experiments.fixed2048.backbone_control
python -m experiments.fixed2048.pilot
python -m experiments.fixed2048.train --output results/fixed2048/continuous --checkpoint-dir checkpoints/fixed2048/continuous/main --stop-after 100
python -m experiments.fixed2048.train --output results/fixed2048/continuous --checkpoint-dir checkpoints/fixed2048/continuous/main --resume checkpoints/fixed2048/continuous/main/latest.pt
python -m experiments.fixed2048.evaluate --checkpoint checkpoints/fixed2048/continuous/main/best_validation.pt --paths R1 R2 --references paired_full2048 i_only full2048 original128 --output results/fixed2048/continuous/evaluation --sensitivity
python -m experiments.fixed2048.calibrate --source-checkpoint checkpoints/fixed2048/continuous/main/best_validation.pt
python -m experiments.fixed2048.train --path R2 --source-checkpoint checkpoints/fixed2048/vq/calibrated.pt --output results/fixed2048/quantized/training --checkpoint-dir checkpoints/fixed2048/vq/main
python -m experiments.fixed2048.evaluate --checkpoint checkpoints/fixed2048/vq/main/best_validation.pt --paths R1 R2 --references paired_full2048 i_only full2048 original128 --output results/fixed2048/quantized/evaluation --sensitivity
python -m experiments.fixed2048.train --decoder-grid 34 60 --output results/fixed2048/grid_ablation/decoder34 --checkpoint-dir checkpoints/fixed2048/grid34/main
python -m experiments.fixed2048.evaluate --checkpoint checkpoints/fixed2048/grid34/main/best_validation.pt --paths R1 --references i_only full2048 --output results/fixed2048/grid_ablation/decoder34/evaluation
```

仅在 R1/R2 通过验证且 Grid 改善不明显时，再独立运行 Encoder Grid 控制；保持 Decoder 为17×30：

```bash
python -m experiments.fixed2048.train --encoder-kv-grid 34 60 --output results/fixed2048/grid_ablation/encoder34 --checkpoint-dir checkpoints/fixed2048/encoder34/main
python -m experiments.fixed2048.train --path raw_192 --output results/fixed2048/raw192 --checkpoint-dir checkpoints/fixed2048/raw192/main
```

所有三项 Pilot 均从 B1 S4 独立初始化。正式 R1 / R3 / Encoder Grid / raw192 从同一 B1 初始化、相同训练窗口种子和300更新。R2 为 Train-only K-means 校准后的量化感知微调，继承 R1 验证选定权重。模型以 Validation 的全局 MSE 选择，Step0 也参与选模。

每个输出目录只允许一个写入者，已存在结果拒绝覆盖；训练必须显式 `--resume`。配置、优化器、固定 LR Scheduler、采样/全局 Python/NumPy/Torch/CUDA RNG 和全部历史一并保存。Resume 截断尚未提交的日志；下个更新从已完成步骤+1开始。

## 最终测试

先准备 `final_plan.json`，给出每个模型的 checkpoint、paths、references 与验证选模依据，再执行：

```bash
python -m experiments.fixed2048.seal --plan results/fixed2048/final_plan.json
python -m experiments.fixed2048.evaluate --checkpoint checkpoints/fixed2048/continuous/main/best_validation.pt --split test --frames 64 --paths R1 R2 --references full2048 --seal results/fixed2048/final_test/seal.json --output results/fixed2048/final_test/R1
```

测试固定为 cows/goat/parkour 各64帧。未封存权重、修改参考模式、改变帧数、视频或 Manifest 均拒绝读取测试像素。封存后不再训练或选模。其余模型使用 plan 中声明的路径/参考模式分别评估。`report.py` 从真实原始数据生成中文报告和表 A–D。

## 指标与接口注意

主结果保留逐帧 MSE、PSNR、LPIPS-Alex、DISTS、MS-SSIM、I/P Latent MSE / Relative RMSE、Token 方差及真实码本使用。区分逐帧 PSNR 均值、视频级均值和全局 MSE 对应 PSNR。速度测量使用真实1080p，`profile.py` 的显存不含感知指标网络；evaluation 的显存包括这些网络。所有结果 `real_bitstream: false, bpp: null`。

官方 I-frame prior 在原始 y 上先后保留两组潜变量尺度，Decoder 的连续输入为 `y*q_scale_y_enc*q_scale_y_dec`。本任务字面 R0 省略了这两组因子；原始R0 I质量差不能证明DCVC失效。`backbone_control.py` 单独保留连续尺度，完全不经过 rounding、prior 或任何码流；不会修改历史 Adapter 或将其冒充严格上界。
