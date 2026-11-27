# CP-AdapTok

PHT-Tok 是唯一持续研发的方案。项目面向原生 1920×1080 视频，在一次多帧 Chunk 分析后生成真实的渐进字节前缀，接收端按已收到的层增量解码。研究目标是同时提高重建质量、编解码吞吐和动态网络下的可用画质。

PHT-Tok 沿用官方 DCVC-UF HT-S 的八帧 P-Chunk 分析。发送端和接收端各自提交协商的参考层，后续增强只更新当前显示。连续参考缺失时由独立 I 帧恢复。I/P 调度、码流字节账目、动态 QP 对照和按截止时间的分组调度均由可复现评估覆盖。

| 保留方案 | 用途 |
| --- | --- |
| [PHT-Tok](results/pht_tok/final_report.md) | 多帧联合渐进码流。保留 P1 和 P2 实现、真实字节前缀与独立参考测试 |
| [官方 DCVC-UF LD、HT-S](results/dcvc_uf/calibration.md) | 静态率失真、真实 I/P 码流、H100 性能及动态 QP HT-S 对照 |
| [原版 AdapTok](results/adaptok/acceptance.json) | 官方预训练模型的两种原生 1080p 分块重建。报告 Token 和质量，序列化字节字段为 `null` |
| [HLA-Tok-UF-V2 D1](results/hla_tok_uf_v2/final_verdict.json) | 冻结的单帧 Base 因果基线，使用原报告正式选择的 Seed 2 E0 |

## 开始

初始化两个官方子模块，然后按复现说明安装锁定的环境、准备官方权重和数据清单、构建 CUDA 扩展。新评估使用 Validation 或冻结的 Train Calibration，Test 保持封存。 [安装与资产准备](docs/reproduction.md#准备)

```bash
git submodule update --init
export CP_ADAPTOK_DATA_ROOT="$PWD/data"
export PYTHONPATH="$PWD/.build/pht-tok/pht/site:$PWD"
.venv-native/bin/python -m benchmarks pht-codec \
  --weights "$PWD/checkpoints" --output .build/examples
```

命令使用冻结的 Calibration 样本生成 I/P 码流，增量解码 P1/P2 各四个前缀，并写出真实字节账本。 [单项评估](docs/reproduction.md#单项评估) · [共同 Validation 与回归](docs/reproduction.md#共同-validation-与回归)

## 已验证能力与研究限制

历史 80 个原生 1080p P-Chunk 的 Full 路径等价，连续 16 个 P-Chunk 的独立参考及丢失后 I 帧恢复通过。P1 Full 聚合字节开销最高约 1.99%。原 H100 单卡记录约为编码 434 FPS、解码 282 FPS、串行 171 FPS，均按预加载源、稳态 P-Chunk 折算。

PHT-Tok 的研究准入为 **FAIL**，总体研究裁决为 **PARTIAL_PASS**。归档的 QP 32 第二前缀等 PSNR 码率开销约 30.18%。五条动态轨迹上的掉帧惩罚画质分数均未超过动态 QP HT-S。八帧采集在 30 FPS 下还需约 233 ms 的首尾帧间隔，采集、网络和播放等待均须与核心编解码耗时分别统计。

[架构](docs/architecture.md) · [复现](docs/reproduction.md) · [研究历史](docs/research_history.md) · [本轮清理验收](docs/cleanup_report.md)
