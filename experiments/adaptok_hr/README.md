# AdapTok-HR 运行与复现

本目录使用独立 `adaptok_hr` 包。原 `experiments/original_adaptok_1080p`、CP生产路径、子模块和封存结果不改。
默认配置见 `configs/adaptok_hr/default.yaml`，设计见 `docs/adaptok_hr_design.md`。单H100运行，下面各GPU命令顺序执行。

## 环境与数据

本机复用 `.venv-native`：Python3.12、PyTorch2.9.1+cu130、OpenCV、PyYAML、LPIPS、DISTS、pytest、coverage、radon、ruff。
重新安装项目可用 `uv pip install --python .venv-native/bin/python -e '.[eval,dev]'`；CUDA版Torch按实际环境安装。
指标AlexNet/VGG16权重需要合法已有缓存；缺少指标依赖/权重时正式评估显式失败，不用随机权重替代。

```bash
.venv-native/bin/python -m adaptok_hr.data.manifest --cp-manifests data/manifests --output data/adaptok_hr/index.json
OMP_NUM_THREADS=4 .venv-native/bin/python -m pytest adaptok_hr/tests -q
```

输入来自既有 `cpat_train/validation/test.json`。索引自动保留原生至少1920×1080源，更高分辨率只裁剪。支持本地已授权外部源：

`content_id` 必须标识原视频源；同源裁剪/片段沿用相同ID，不能通过为每段重新编号绕过按源划分检查。

```json
{"license":"用户确认的许可标识", "usage_conditions":"允许本项目研究训练的使用条件", "records":[{"path":"/absolute/video", "content_id":"dataset:source", "source":"出处", "official_split":"train", "role":"train", "resolution":[1920,1080], "original_frames":100, "fps":30, "resized":false, "sha256":"真实源哈希", "frame_hashes":[]}]}
```

用 `--external /absolute/sources.json` 导入。图像序列应保存每帧真实哈希；可用原有 `cp_adaptok.large_scale_data.audit_source` 生成源信息。源ID/路径/哈希不得跨split；不自动下载数据。索引包含完整source与split身份。
原始YUV先用封存官方实验的显式颜色范围/矩阵读取器生成经核验RGB来源，再生成独立manifest；禁止猜测YUV色彩参数。默认路径与CP相同OpenCV BGR→RGB、float32/255。

## 工程验证、拟合与联合训练

```bash
OMP_NUM_THREADS=4 .venv-native/bin/python -m experiments.adaptok_hr.pilot phase0 --output results/adaptok_hr/phase0
OMP_NUM_THREADS=4 .venv-native/bin/python -m adaptok_hr.training.train --output results/adaptok_hr/fit --steps 100 --overfit DAVIS2017:bear --overfit-frames 4
OMP_NUM_THREADS=4 .venv-native/bin/python -m adaptok_hr.training.train --output results/adaptok_hr/fit --steps 300 --resume checkpoints/adaptok_hr/fit/step_000100.pt --overfit DAVIS2017:bear --overfit-frames 4
OMP_NUM_THREADS=4 .venv-native/bin/python -m experiments.adaptok_hr.pilot fit-diagnostic --output results/adaptok_hr/fit
OMP_NUM_THREADS=4 .venv-native/bin/python -m adaptok_hr.training.train --output results/adaptok_hr/joint --steps 300
```

完成300步后先用下一节完整Validation，再依据记录决定是否恢复到1000：

```bash
OMP_NUM_THREADS=4 .venv-native/bin/python -m adaptok_hr.training.train --output results/adaptok_hr/joint --steps 1000 --resume checkpoints/adaptok_hr/joint/step_000300.pt
```

`--steps` 是累计已完成优化器步数目标；不是追加步数。检查点为唯一提交边界，崩溃后日志回滚到该边界，未提交计算可能重做但不重复计账。
训练输出目录有单写者锁；不得同时向同名checkpoint目录运行另一实验。模型/optimizer/scheduler/RNG/sampler/manifest/config齐全恢复。不同配置、源身份和最终Test封存后的run拒绝继续训练。
训练T从4到16到32的配置阈值为1000/5000已完成更新；TBPTT2或4单独控制图长，不能把工程32帧探测说成已完成32帧正式训练。

## 完整质量与独立性能

```bash
OMP_NUM_THREADS=4 .venv-native/bin/python -m adaptok_hr.evaluation.evaluate --checkpoint checkpoints/adaptok_hr/joint/step_000300.pt --split validation --ids DAVIS2017:blackswan DAVIS2017:car-roundabout DAVIS2017:dog --frames 64 --output results/adaptok_hr/validation_300
OMP_NUM_THREADS=4 .venv-native/bin/python -m adaptok_hr.evaluation.evaluate --checkpoint checkpoints/adaptok_hr/joint/step_001000.pt --split validation --ids DAVIS2017:blackswan DAVIS2017:car-roundabout DAVIS2017:dog --frames 64 --output results/adaptok_hr/validation_1000
OMP_NUM_THREADS=4 .venv-native/bin/python -m adaptok_hr.evaluation.benchmark --checkpoint checkpoints/adaptok_hr/joint/step_001000.pt --warmup 5 --repeats 20 --output results/adaptok_hr/benchmark.json
OMP_NUM_THREADS=4 .venv-native/bin/python -m experiments.adaptok_hr.sequence_benchmark --checkpoint checkpoints/adaptok_hr/joint/step_001000.pt --frames 64 --warmup 5 --output results/adaptok_hr/sequence_benchmark.json
```

每预算独立闭环，输出真实 `.bin`、逐帧JSONL、聚合JSON和原生样图。blackswan仅50帧、dog60帧，不补帧；car-roundabout为连续64帧。正式长序列统计应再检查独立Test中64帧源。
质量过程含指标模型的显存与时间；部署速度以独立benchmark为准。联合速度包括发送端本地解码及接收端解码；纯Encoder FPS单独列出。
`benchmark`测固定I/首P状态；`sequence_benchmark`预热5对I/P后，逐预算实际处理连续64帧、完整推进双方状态，单遍记录各帧同步计时，不加载感知网络。读取、H2D、落盘、指标及端点相等断言均不计入编解码延迟；该口径没有流水并行。

完成Validation选定冻结Checkpoint后，才能运行：

```bash
OMP_NUM_THREADS=4 .venv-native/bin/python -m adaptok_hr.evaluation.evaluate --checkpoint checkpoints/adaptok_hr/joint/step_001000.pt --split historical --frames 64 --output results/adaptok_hr/historical
OMP_NUM_THREADS=4 .venv-native/bin/python -m adaptok_hr.evaluation.evaluate --checkpoint checkpoints/adaptok_hr/joint/step_001000.pt --split test --frames 64 --seal-run results/adaptok_hr/joint --output results/adaptok_hr/final_test
```

HR Test是从原CP已索引的Test源中划出的本轮未读像素集合，不等于全新独立数据集；历史cows/goat/parkour独立标注。实际最终checkpoint以报告与seal.json为准，不能根据Test换模型。

## 消融与Scorer

复制完整默认配置并记录独立输出目录，保持训练步数/seed/数据split相同：

| 条件 | 配置修改 |
| --- | --- |
| Conv-only | `model.attention_blocks: 0`, `model.temporal: false` |
| Conv+local | `model.attention_blocks: 1`, `model.temporal: false` |
| Conv+local+temporal | 默认配置 |
| Fixed K | `training.budget_mode: fixed`, `training.fixed_budget: 2048` |
| Nested | 默认配置 |

未运行消融不产生结果。Tokenizer未稳定时Scorer只验证工程接口：

```bash
.venv-native/bin/python -m adaptok_hr.training.train_scorer --checkpoint checkpoints/adaptok_hr/joint/step_001000.pt --readiness results/adaptok_hr/decision.json --steps 1000 --output results/adaptok_hr/scorer
```

该入口要求真实决策中 `scorer_ready=true` 且 `checkpoint_sha256` 匹配当前Tokenizer，默认不绕过。冻结Tokenizer后同一P帧状态下各预算实际ID重建MSE作标签；只更新Scorer。在线使用 `Codec.encode_adaptive_frame()`，只读当前编码与重建历史，不读未来，不包含离线ILP。

## Python流式接口

```python
from adaptok_hr.codec.codec import Codec
from adaptok_hr.training.checkpoint import load_model
from adaptok_hr.runtime import seed_everything

seed_everything(20261117)  # 收发端使用相同数值设置
model, metadata = load_model("checkpoints/adaptok_hr/joint/step_001000.pt")
codec = Codec(model.eval(), amp_bf16=True)
packet, sender_state = codec.encode_frame(rgb_bchw, None, budget=512, stream_id="camera-1")
reconstructed, receiver_state = codec.decode_frame(packet, None)
# 下一帧双方分别传回各自的state；不能传原始历史。
```

码流v1固定13-bit，无熵压缩；完整统计142-byte帧头、payload与实际bpp。前缀截断用 `truncate_packet(packet,K)` 重写头部/CRC；降低历史预算后后续P帧必须基于新实际重建重新编码。
空状态可在合法I帧同步；P帧缺失、顺序/链/模型/码本不一致和CRC错误显式报错。Codec创建后模型参数发生更新将拒绝继续使用该身份。

```bash
.venv-native/bin/python -m adaptok_hr.evaluation.report
```

生成的报告及 `experiments/adaptok_hr/evidence` 来自实际JSON/JSONL；大Checkpoint和源数据留在本机忽略目录，以SHA256核对。

## 决策和验收证据

完整1000步Validation之后、最终Test之前执行 `python -m experiments.adaptok_hr.decision`，生成停止/继续决策及逐层实际码流诊断。该命令记录本轮负结果，不能替代未来新架构的就绪判据。

```bash
OMP_NUM_THREADS=4 .venv-native/bin/python -m experiments.adaptok_hr.numeric_mode_check
OMP_NUM_THREADS=4 .venv-native/bin/python -m experiments.adaptok_hr.quality mutate --output results/adaptok_hr/mutations.json
.venv-native/bin/python -m experiments.adaptok_hr.quality audit --output results/adaptok_hr/quality_audit.json
OMP_NUM_THREADS=4 .venv-native/bin/python -m experiments.adaptok_hr.acceptance
.venv-native/bin/python -m adaptok_hr.evaluation.report
```

复杂度/重复审计覆盖 `adaptok_hr` 非测试代码，依赖实际测试与命令执行合并生成的 `results/adaptok_hr/coverage.json`；不是全仓库审计。`crapper`、`dryer`、`mutator`未安装，使用radon+coverage、AST相似度及隔离实际变异，结果中保留工具身份和断言失败证据。
GPU逐位恢复需相同软硬件、BF16设置与严格数值模式。早期Pilot完成后才补强严格CUDA确定性；现有权重保留原身份，不承诺从旧种子重新训练能逐位复现早期Pilot。
