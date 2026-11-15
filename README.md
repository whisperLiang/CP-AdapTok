# CP-AdapTok

时间因果、学习式有序 1D VQ Token 与可追加 rANS 视频流研究原型。主模型显式复用固定版本 DCVC-UF LD / DMCI 的 PyTorch 模块，保留 AdapTok 的 learned queries、VQ、前缀训练和质量分配思想。

**当前仍是研究验证原型。已实际完成 VQ/Teacher 修复、短程多视频及1080p延长训练；确定性码本使用恢复，但有效渐进画质与30/60 FPS目标仍未达成。** 同一模型32帧实际码流在逐位一致的整数指纹融合后由16.66提升至18.38 FPS；该模型质量验收失败。完整证据、失败结论与恢复命令见 [优化验收报告](docs/optimization_report.md)。原100步实验和权重仍保留，详见 [原生骨干历史报告](docs/pretrained_native_report.md) 与 [官方权重记录](docs/pretrained_weights.md)。旧 smoke checkpoint 的骨干仍为随机初始化。

在本机使用：

```bash
cd /home/whisperliang/CP-AdapTok
source .venv-native/bin/activate
python -m pytest cp_adaptok/tests -q
python -m cp_adaptok.cli.eval --config configs/pretrained_s1.yaml \
  --checkpoint checkpoints/pretrained_s1/step100.pt --max-frames 8 --output results/pretrained_s1_test
python -m cp_adaptok.cli.eval --config configs/pretrained_s1.yaml --baseline native \
  --max-frames 8 --output results/pretrained_native_baseline
python -m scripts.benchmark_native
```

`step100.pt` 保留本轮100步结果，`latest.pt` 已验证恢复至101步。该短程 checkpoint 状态为 `trained`，表示新增参数经过更新；其 RD 验收仍为 false。当前应先解决码本塌缩，再扩展训练量。

运行已有结构训练 checkpoint 的真实 1080p 字节流评估：

```bash
python -m cp_adaptok.cli.eval --config configs/s1_tokenizer.yaml \
  --budgets 128 256 512 1024 2048 \
  --checkpoint checkpoints/s3_smoke/step100.pt --allow-untrained
python -m cp_adaptok.cli.benchmark --resolution 1920 1080 --measure actual-bitstream \
  --checkpoint checkpoints/s3_smoke/step100.pt --allow-untrained
python -m cp_adaptok.cli.stream_demo --mode localhost --progressive \
  --checkpoint checkpoints/s3_smoke/step100.pt --allow-untrained
```

在线预算已接入真实 TCP，按当前帧带宽、实际未发送字节和源播放截止时间选择前缀；基础显示 ACK 之后才追加增强。运行8帧带宽变化、超时、尾部舍弃与慢 ACK 场景：

```bash
python scripts/run_streaming_experiments.py \
  --checkpoint checkpoints/s3_smoke/step100.pt --allow-untrained \
  --output results/streaming_verified
python -m cp_adaptok.cli.stream_demo --mode localhost --progressive --config configs/streaming.yaml \
  --checkpoint checkpoints/s3_smoke/step100.pt --allow-untrained \
  --allocation rate-only --bandwidth-kbps 32 128 512 2048 128 512 2048 512 \
  --playback-delay-ms 1200 --no-save-frames
```

`--allocation` 支持 `fixed`（需 `--fixed-k`）、`rate-only` 和 `scorer`；`--enhancement-cap 512` 模拟未发送增强尾部的舍弃，`--ack-delay-ms` 设置接收端确认延迟。`--no-save-frames` 以重建 Tensor 就绪作为显示完成，省去 PNG 写盘。带宽整形在应用层进行，未模拟 TCP 随机丢包或物理链路头；冷启动和实际迟到均保留。详见 [实测结果与复现口径](docs/streaming_report.md)。

结构训练和恢复：

```bash
python -m cp_adaptok.cli.train --config configs/s1_tokenizer.yaml --max-steps 100 --smoke --allow-untrained
python -m cp_adaptok.cli.train --config configs/s1_tokenizer.yaml --checkpoint checkpoints/s1_smoke/latest.pt \
  --resume --max-steps 102 --smoke --allow-untrained
python -m cp_adaptok.cli.train --config configs/s3_entropy_scorer.yaml --checkpoint checkpoints/s1_smoke/step100.pt \
  --max-steps 100 --smoke --allow-untrained
```

阶段定义见 `configs/s0_baseline.yaml` 至 `s4_finetune.yaml`。本轮使用 `configs/pretrained_s1.yaml`，独立保存官方骨干实验并从64帧训练区间随机取连续窗口。迁移机器需修改数据位置或运行数据准备脚本；后续阶段须相应配置来源 checkpoint 和足够长的训练区间。这些短片不足以支持数据集级结论。

新环境初始化：

原生码流与 teacher 使用独立环境，构建脚本不会修改系统 CUDA 或上游 submodule：

```bash
bash scripts/setup_native.sh
source .venv-native/bin/activate
python scripts/prepare_pretrained_data.py
```

只运行纯 PyTorch 路径时也可安装基础环境：

```bash
git submodule update --init --recursive
python -m venv .venv
source .venv/bin/activate
pip install -e '.[eval,dev]'
```

本机原有 `.venv` 通过 `.pth` 只读复用 InfoTok 的 PyTorch2.6/cu124；新 `.venv-native` 为 Python3.12/PyTorch2.9.1/cu130，两个环境独立。构建副本应用最小 latent 导出补丁，以取得原生解码器实际使用的量化 tensor；不再通过近似重算 prior 生成 teacher。

文档：

- [架构及接口形状](docs/architecture.md)
- [训练与恢复](docs/training.md)
- [官方骨干、原生码流与100步实测](docs/pretrained_native_report.md)
- [评估口径](docs/evaluation.md)
- [二进制格式](docs/bitstream_spec.md)
- [在线 TCP 场景与实测结果](docs/streaming_report.md)
- [真实实施与实验报告](docs/implementation_report.md)
- [ub-mode 操作日志](docs/work_log.md)
- [本轮优化验收报告](docs/optimization_report.md)
- [量化与教师策略说明](docs/quantization_training.md)
- [本轮 ub-mode 操作日志](docs/optimization_work_log.md)

`third_party` 是固定提交的 Git submodule，清单见 `third_party/manifest.json`。上游源码保持原样；固定空间 Tokenizer 只通过 `configs/ablation_spatial.yaml` 显式启用为实验对照。
