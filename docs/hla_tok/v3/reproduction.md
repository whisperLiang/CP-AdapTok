# HLA-Tok V3 复现

在独立的 `experiment/hla-tok-v3` worktree 根目录执行。本分支从
`50bd9cc6f28c6cde0e65f82723af2e7b11e48bd9` 建立；V2 的 D2 FAIL 和不可进入旧 Phase 2 的裁决不变。
需要单张 H100 PCIe、CUDA BF16、PyTorch、LPIPS、DISTS、MS-SSIM、OpenCV、Matplotlib 和 FFmpeg。
使用冻结的原始数据清单、权重与本地路径。旧工作区只读；为既有 CPU 测试初始化原 Gitlink 的
DCVC/AdapTok 子模块，并将既有 DCVC 权重只读链接到本 worktree 的 `checkpoints/`。

## 预检与执行

```bash
V2=/path/to/frozen-hla-tok-v2/CP-AdapTok
RUN() { python -m scripts.hla_tok.v3.run --v2 "$V2" "$@"; }
RUN preflight
RUN baseline-replay
RUN train all
```

`train dense` 只执行 A0/A1。完整顺序为 Seed 1 A→B→C、独立 Seed 2 R2 2000 步→所选 A→B→C、
两个 Seed 的 D。亦可运行 `train continuous --seed 1`、`train quantized --seed 1`、
`train progressive`，但前序实际 PASS 不可绕过。Train 计划从冻结清单预生成；每 500 步保存，
SIGINT/SIGTERM 亦保存最近一个成功更新。用 `RUN resume all` 读取符合身份的最新检查点；
它不会放宽模型、优化器、调度器、RNG、数据游标、样本链、像素与 update 契约。
同一工作区的训练、评估、测速与解封有单写者锁；报告可以读取训练中原子发布的记录。

A0/A1 继承相同 Encoder/RGB Decoder 参数；D1 仅新增两个恒等初始化块。进入 1080p 时保留 AdamW
历史、重启该 leg 的低学习率 warmup/cosine；B 从冻结切到解冻显式重建 optimizer。
Phase A 之前以 16 个 512×512 和 16 个原生 Train 样本校准，目标在像素/感知 RGB 梯度上比对；
仅当超出预注册区间时，感知项乘以 `clip(1/ratio, 0.25, 4.0)`。权重在正式训练前锁定。
B 同样审计适用模块和 latent 梯度，但不再改权重。唯一匹配延长沿用末 leg 的 LR 下限和连续游标。

阶段比较中的 9 帧是趋势；只用 174 帧原生评价判断准入。若 B0 失败且无允许的延长，
`RUN diagnose --seed 1` 对同一批 174 帧重放 A/B0 稠密、B0 Local 特征与 Rank 干预。
本轮未启用 ARD，也不预置 B1 生产路径；若实际进入该分支，须先完成注册的 1000 步匹配
Local Decoder 诊断并经过质量与耗时裁决。未通过时 C/D 不会启动。D 的最高三档如两个相邻
增量皆未超过任一质量噪声容差，则不通过饱和检查。可选单调损失预设关闭（权重 0）。

## 证据与封存

```bash
RUN verify --mutations
RUN evaluate --checkpoint /path/to/step_08000.pt
RUN benchmark --checkpoint /path/to/trained-checkpoint.pt
RUN samples --checkpoint /path/to/trained-checkpoint.pt
RUN report --figures
```

独立接收端只持有 VQ 和合成模块。折算码率必须来自真实 Fixed13 字节，连续/稠密输出没有 bpp。
性能区分同步逐帧 Neural、Fixed13、V2 定义的预取 Offline Pipeline 和不预取的串行输入/输出。
只有最后一种才用于声明 P95 帧截止时间；离线吞吐不是网络传输。FLOPs 是 profiler 能计量的操作，
未计量算子不应被解释为零开销。FFV1 原生样例保留相同 16 帧与同坐标像素裁块。

最终二者 D PASS 和功能、性能门槛满足后，显式运行 `RUN freeze-final`，再运行
`RUN sealed-test`。Test 范围仅由冻结元数据派生：所有未缩放的原生 1920×1080 Test 序列的全部帧；
其他分辨率不做缩放，也不读像素。每一冻结 SHA-256 的 Test 入口为一次性；中断不自动重试，
已完整执行也只返回原始记录。报告只读 Test 结果，不把它用于训练或模型选择。

## 工程验收

```bash
python -m pytest -q cp_adaptok/tests tests -m 'not gpu and not integration' \
  --junitxml=results/hla_tok/v3/cpu_tests.xml
python -m pytest -q cp_adaptok/tests tests -m 'gpu and not integration' \
  --junitxml=results/hla_tok/v3/cuda_tests.xml
ruff check hla_tok scripts/hla_tok tests/hla_tok
mypy --explicit-package-bases --follow-imports=skip --ignore-missing-imports \
  --python-version 3.12 hla_tok scripts/hla_tok
```

正式结果、权重、视频和长日志保留本地；Git 只提交轻量快照和复现代码。图件来自当前实测记录，
未训练预算不画成实测点。`report --figures` 使用 Python；`NATURE_FIGURE_SCRIPTS` 可指定 PDF QA 工具目录。
