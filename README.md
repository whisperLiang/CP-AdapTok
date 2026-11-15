# CP-AdapTok

时间因果、学习式有序 1D VQ Token 与可追加 rANS 视频流研究原型。主模型显式复用固定版本 DCVC-UF LD / DMCI 的 PyTorch 模块，保留 AdapTok 的 learned queries、VQ、前缀训练和质量分配思想。

**当前是工程验证原型。缺少官方预训练权重和原生 CUDA 扩展，尚未完成正式训练、有效 RD 对照或 30/60 FPS 目标。** `--allow-untrained` 和 `--smoke` 的输出会记录随机骨干状态，不能作为训练效果。

在本机使用：

```bash
cd /home/whisperliang/CP-AdapTok
source .venv/bin/activate
python -m pytest cp_adaptok/tests -q
python scripts/run_acceptance.py
```

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

正式流程依次使用 `configs/s0_baseline.yaml` 至 `s4_finetune.yaml`。安装官方模型到配置中的路径，使用匹配 PyTorch/CUDA 构建上游扩展，再移除 smoke/untrained 开关。数据默认指向本机已有的合法视频；迁移机器时修改 `configs/base.yaml`。这些短片只适合工程检查，不足以支持研究结论。

新环境初始化：

```bash
git submodule update --init --recursive
python -m venv .venv
source .venv/bin/activate
pip install -e '.[eval,dev]'
```

本机 `.venv` 通过 `.pth` 只读复用 InfoTok 已安装的 PyTorch 2.6/cu124，新增测试工具安装在独立环境中，没有修改其环境。原生 UF CUDA Proxy 上游建议 PyTorch 2.9.1/CUDA 13；本原型的纯 PyTorch路径已在 2.6 验证，不能据此声称 Proxy ABI 可用。

文档：

- [架构及接口形状](docs/architecture.md)
- [训练与恢复](docs/training.md)
- [评估口径](docs/evaluation.md)
- [二进制格式](docs/bitstream_spec.md)
- [在线 TCP 场景与实测结果](docs/streaming_report.md)
- [真实实施与实验报告](docs/implementation_report.md)
- [ub-mode 操作日志](docs/work_log.md)

`third_party` 是固定提交的 Git submodule，清单见 `third_party/manifest.json`。上游源码保持原样；固定空间 Tokenizer 只通过 `configs/ablation_spatial.yaml` 显式启用为实验对照。
