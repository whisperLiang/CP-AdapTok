# HLA-Tok-UF-V3 单卡 H100 验收

**总体裁决 `PARTIAL_PASS`。** 主要路线 **R0**，保留官方单 latent 和 D1 显示路径。
正式 Validation 准入为 **FAIL**。Test 像素读取 **0**。

## 训练真实性、权重与容量

V2 两个 Seed 各有 20,000 次实际更新。P1 属于特征空间训练，P2/P3 使用分离的缓存并训练有限的 P 帧适配参数。H100 重放分别探测码率与失真梯度，核对了官方 FP16 代理安装的折叠权重。

[训练审计](../../../docs/hla_tok/uf_v3/training_audit.md) · [容量配对](../../../docs/hla_tok/uf_v3/capacity_diagnostics.md) · [路线和冻结门槛](diagnostics/capacity.json)

按冻结工程和 RD 门槛排序，当前最优研究对照为 **A2 / V2 E1**。正式部署准入仍为 FAIL。

该端点沿用 V2 `d1-joint-seed2-step20000.pt`，SHA-256 为 `09bb3a578d89ddb9b2e621a9cf2a1c985edaa49538f236046439116c3864c1ec`。 [E1 配表](entropy/static_cdf.json)

V3 实测候选 checkpoint 为 `checkpoints/hla_tok/uf_v3/r0-probe-seed1-step2001.pt`，SHA-256 为 `67fa65db233d20ed9a7f2141ebf938e4e316e16580a95d3a13d1b8c4cb8776aa`。

## 容量诊断与 D0-Fast

| 对照 | Seed 1 PSNR / LPIPS / DISTS | Seed 2 PSNR / LPIPS / DISTS |
|---|---:|---:|
| C0 | 29.626 / 0.40546 / 0.22993 | 29.624 / 0.40526 / 0.23004 |
| C1 | 29.492 / 0.24409 / 0.15607 | 29.399 / 0.24150 / 0.15583 |
| C2 | 29.116 / 0.41665 / 0.23337 | 29.114 / 0.41646 / 0.23344 |
| C3 | 28.953 / 0.25837 / 0.16415 | 28.863 / 0.25693 / 0.16505 |

C0/C1 是连续域机制对照。C2/C3 使用相同的真实整数与 Base 历史。Seed 2 的 C0 在 174/174 帧提高 PSNR。C2 相对 C3 的全局 PSNR 高 0.251 dB，LPIPS 高 0.15953。全有效前缀较 40 档仅增加 0.00104 dB，支持限制来自量化表示和读出折衷。

R0 优先补齐原始 RGB 到骨干的联合优化。R1 Core–Detail 缺少经联合训练的连续容量证据。R2 半步长收益有限且增加实际字节。R3 D0-Fast 实时通过，感知指标落后。R4 已有合法记录剩余的质量增量接近零，重排留作候选。

| 上限 Mbps | D0-Fast FPS | P95 ms |
|---:|---:|---:|
| 5 | 38.19 | 26.65 |
| 10 | 37.30 | 27.26 |
| 20 | 34.73 | 28.99 |
| 40 | 31.20 | 32.98 |

D0-Fast 在 34 帧与 123,477 条前缀边界上与旧 D0 一致。原始 D0 的 5 Mbps 实测为 6.66 FPS、P95 249.39 ms。 [原生计时](diagnostics/validation/d0_fast_timing.csv) · [字节及 RGB](diagnostics/d0_parity.json) · [历史与同实际码率对照](diagnostics/references.json)

## 原生 Validation 率失真与实时性

| 上限 Mbps | 可达帧 | 实际 Mbps | PSNR dB | LPIPS | DISTS | Full-max FPS / P95 ms | 按需 FPS / P95 ms |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 174/174 | 0.4404 | 26.165 | 0.51340 | 0.28248 | — | — |
| 0.5 | 0/174 | — | — | — | — | — | — |
| 1 | 0/174 | — | — | — | — | — | — |
| 1.3947 | 32/174 | — | — | — | — | — | — |
| 1.395 | 32/174 | — | — | — | — | — | — |
| 2 | 174/174 | 1.9927 | 26.651 | 0.34189 | 0.21010 | — | — |
| 3 | 174/174 | 2.9938 | 26.986 | 0.32936 | 0.20264 | — | — |
| 3.594 | 174/174 | 3.5879 | 27.146 | 0.32410 | 0.19927 | — | — |
| 4 | 174/174 | 3.9941 | 27.243 | 0.32106 | 0.19729 | — | — |
| 5 | 174/174 | 4.9948 | 27.464 | 0.31428 | 0.19291 | 40.95 / 25.56 | 49.26 / 20.99 |
| 10 | 174/174 | 9.9942 | 28.235 | 0.29025 | 0.17855 | 39.13 / 27.23 | 43.90 / 23.74 |
| 20 | 174/174 | 19.9880 | 28.824 | 0.27259 | 0.16723 | 35.97 / 29.80 | 39.36 / 26.95 |
| 40 | 174/174 | 31.2154 | 29.046 | 0.26244 | 0.16133 | 34.70 / 30.16 | 35.17 / 30.08 |

0 为完整 Base-only 的实际序列平均码率。低于 Base 的预算记录物理超限和后续 P 帧失步。40 Mbps 的两种模式使用相同生成上限，独立复跑可反映运行波动。官方生产代理为 FP16。原生 FP32 和 BF16 路径为 `NOT_AVAILABLE`。

[Full-max 计时](runtime_trial/validation_full_max/summary.json) · [按需计时](runtime_trial/validation_on_demand/summary.json) · [首次按需计时，结束时有其他 GPU 工作](evaluation/on_demand/summary.json)

## 静态 CDF 实测

Calibration 预注册规则选择 **E1**。固定整数、组序、原生 Base 和 RGB 完全一致。

| E0 前缀上限 | E1 应用字节节省 | E1 增强字节节省 | E1 同上限 LPIPS |
|---|---:|---:|---:|
| 5 | 4.67% | 5.05% | 0.26064 |
| 10 | 7.10% | 7.38% | 0.24064 |
| 20 | 10.31% | 10.51% | 0.22523 |
| 40 | 16.23% | 16.42% | 0.21801 |
| ALL_VALID_RECORDS | 16.87% | 17.06% | 0.21773 |

两套数学口径分别以应用层全包或仅增强段为分母。CDF 为预装资产。逐记录的 mode、头、CRC 与 rANS 终止都计入。V2 当时选 E0 是因为其低档同感知码率更好。

[同符号](entropy/same_symbol.csv) · [同预算](entropy/same_rate.csv) · [时延和选择](entropy/summary.json)

## Base-Lite

| 方法 | I 峰值 B | P 峰值 B | 4,096 B 可达 | 1.3947 Mbps 可达 |
|---|---:|---:|---:|---:|
| L0 | 7593 | 4290 | 0/174 | 32/174 |
| L1 | 8549 | 4872 | 0/174 | 32/174 |

L1 在 I 帧 Hyperprior 之前衰减 latent，逆向补偿 I 恢复的解码步长，并固定 P 分析增益。独立重放的 I 帧 z 字节均值从 2,440 B 升至 3,736 B。原生 y/z 共用 rANS payload。接收条件不完整时记录 Base 超限及之后的失步。Lite 每种设置有独立前缀族标识。

[Base 逐帧与时延](base_lite/summary.json) · [Hyperprior 重放](base_lite/hyperprior.json)

## 同一 E1 后端的 V2 对照

| 上限 Mbps | A2 实际 Mbps | A2 PSNR | A2 LPIPS | A2 DISTS | A2 FPS / P95 ms |
|---:|---:|---:|---:|---:|---:|
| 5 | 4.9952 | 27.374 | 0.30417 | 0.19312 | 41.52 / 24.98 |
| 10 | 9.9943 | 28.116 | 0.28174 | 0.18135 | 39.23 / 26.81 |
| 20 | 19.9718 | 28.692 | 0.26531 | 0.17026 | 35.49 / 29.31 |
| 40 | 29.8092 | 28.863 | 0.25696 | 0.16507 | 32.73 / 32.32 |

在 A2 高档实际 29.8092 Mbps 处，对 V3 实际曲线插值得 29.023 dB、LPIPS 0.26349、DISTS 0.16194。

[A2 原生账本](evaluation/A2/summary.json)

## 联合更新的 Base 超限与恢复

64 步联合探针的累计 2,064 步 checkpoint 在 Calibration 有 2/128 帧超限。最高 19,403 B，两个超限均为 P 帧。第一个 14,757 B Base 已触发官方严格编码入口拒绝。超限权重下的硬预算 RD 与实时性为 `NOT_RUN`。

从累计 2,000 步 checkpoint 重放同一次原生联合更新。累计 2,001 步的 Base 峰值在 Calibration 为 8,558 B，在 Validation 为 7,614 B。通过固定 Base 门槛的研究 checkpoint 为 `r0-probe-seed1-step2001.pt`。这一恢复保留了独立 I/P Encoder 实际更新和原有 HUF1 语义。

[物理逐帧审计与选择](diagnostics/base_recovery.json)

## 联合训练后的最大前缀

相同 48 张 Calibration 上的全有效记录 D1，V2 为 29.041 dB、LPIPS 0.20948，恢复后的 V3 为 29.170 dB、LPIPS 0.20869。 [连续与量化诊断](diagnostics/post_joint/capacity.json)

连续空间精度对照保持冻结 E1 与 Base。D0 在 40 Mbps 上限的半步长相对原步长变化 +0.151 dB 和 -0.00146 LPIPS。此项用于误差归因。 [精度对照](diagnostics/precision.json)

## 按需组包试验

40 Mbps Calibration 的批量工作区复用版 P95 为 29.98 ms，同条件的原后端为 32.07 ms。原生工作区复用保留组序、CDF、CRC 和分段结束状态。 [时延试验](runtime_trial/summary.json)

完整 Validation 的 174 个最大包和全部合法边界哈希匹配。后端最终选择为 **reuse**。 [字节与双模式门控](runtime_trial/selection.json)

## 消融与阶段门槛

| 实验 | 实际状态及证据 |
|---|---|
| A0 V2 D1 | 冻结两个 Seed 原始 checkpoint；已做同一 Base 历史配对 |
| A1 D0-Fast | 已做同符号实验与三轮 Full-max |
| A2 Static | 已为 Seed 2 从 Train 重新拟合并测量 E0/E1/E2 |
| A3 R0 初始模型 | 复用同一单 latent 和 D1；零步物理路径由 A0/A2 覆盖 |
| A4 R0 训练 | 见优化器与独立部署账本 |
| A5 联合排序 | NOT_RUN；全记录配对没有支持采用新排序 |
| Core-Detail | NOT_RUN；R1 条件未满足 |
| L0/L1 | 见 Base-Lite |

- P1 的实际 Pilot 裁决为 `FAIL`。 [逐帧配对](pilot/P1/gate.json)
- P2 的实际 Pilot 裁决为 `FAIL`。 [逐帧配对](pilot/P2/gate.json)
- probe 的实际 Pilot 裁决为 `FAIL`。 [逐帧配对](pilot/probe/gate.json)

20→40 Mbps 多用 11.2274 Mbps，PSNR 增加 0.2224 dB，LPIPS 改善 0.01015。 [逐帧边际变化](evaluation/A4/adjacent_summary.csv)

## 等画质实际码率

LPIPS 0.28978 的实际 Mbps 为 V3 **10.17888**、V2 E0 **8.11728**、V2 E1 **7.79505**、Original AdapTok **1.39471**。Original 的码率采用独立时间窗平均服务口径。

严格等 PSNR 25 dB 为 `OUT_OF_RANGE`。达到至少 25 dB 的最低实测固定逐帧上限端点为 **1.99273 Mbps**。完整 Base-only 的可变帧长平均为 **0.44037 Mbps**。

| 指标 | 目标 | V3 实际 Mbps | Original 实际 Mbps | 共同可比性 |
|---|---:|---:|---:|---|
| PSNR | 25 | — | 3.59350 | OUT_OF_RANGE |
| PSNR | 26 | — | — | OUT_OF_RANGE |
| PSNR | 28 | 8.09025 | — | OUT_OF_RANGE |
| PSNR | 30 | — | — | OUT_OF_RANGE |
| LPIPS | 0.45 | 0.76945 | — | OUT_OF_RANGE |
| LPIPS | 0.3 | 7.54238 | — | OUT_OF_RANGE |
| LPIPS | 0.28978 | 10.17888 | 1.39471 | PASS |

Original 和 V3 的实测 DISTS 区间没有交集。选择共同覆盖的 DISTS 0.18 可比较 V3 与三个原生对照。

| DISTS 0.18 对照 | V3 实际 Mbps | 对照实际 Mbps |
|---|---:|---:|
| V2 D1 | 9.32015 | 11.55086 |
| V2 E1 | 9.32015 | 10.87158 |
| Anchor-Fast | 9.32015 | 14.46177 |

[等质量与同率账本](evaluation/rates.json) · [所有目标表](evaluation/equal_quality.csv)

## 严格前缀、因果参考与部署权重

混合预算复放 189 帧，参考状态失配 0 次。固定上限复放 204 个帧预算组合，独立 Base 与 P DPB 逐位一致。丢失 Base 后拒绝 38 个依赖 P 帧，下一独立 I 帧恢复 2 次。渐进接收复放 19,198 条记录，最终 RGB 与一次接收逐位一致。篡改未来像素保持历史包、整数和参考哈希。使用未来输入的注入被捕获。

实际后端加载的 checkpoint 为 `67fa65db233d20ed9a7f2141ebf938e4e316e16580a95d3a13d1b8c4cb8776aa`。重新安装 FP16 代理后，I/P 帧的原生 y 与字节均相对 V2 对照改变，旧代理注入在 3 个 P 帧被捕获。 [独立接收与 CUDA 张量账本](runtime_trial/reuse_contracts/summary.json)

## 优化器更新

| 阶段 | Seed | 累计 | 本段 | CUDA 事件区间秒 | 墙钟秒 |
|---|---:|---:|---:|---:|---:|
| P1-2 | 1 | 2 | 2 | 4.17 | 17.13 |
| P1-500 | 1 | 500 | 498 | 198.73 | 212.92 |
| P2-2000 | 1 | 2000 | 1500 | 643.63 | 664.09 |
| probe-2001 | 1 | 2001 | 1 | 8.86 | 22.08 |
| probe-2064 | 1 | 2064 | 64 | 42.84 | 55.98 |

Seed 1 的前 2,000 步训练 D1 和量化器并冻结骨干。联合探针首步展开 4 帧原生 1080p，其余 63 步展开 4 帧 192×256 原生空间裁块。64 步训练直接对活跃骨干张量独立求率、失真梯度。P3、P4 和 Seed 2 在阶段门槛处停止前进。

| 张量路径 | 失真梯度 L2 | 码率梯度 L2 |
|---|---:|---:|
| intra_encoder | 0.0082111 | 0.15779 |
| inter_encoder | 0.0054761 | 0.12702 |
| inter_decoder | 0.013929 | 0.21856 |
| context | 0.005067 | 0.1205 |
| inter_reconstruction | 0.020876 | 0 |
| inter_entropy | 0 | 0.0033287 |
| inter_quantizer | 0.00036213 | 0.0071338 |
| inter_d1 | 0.0023872 | 0 |

CUDA 事件区间含设备空闲及主机等待。原生联合更新的实际分辨率、展开与独立梯度见训练账本。缓存仅保留经过核验的 Train 原始 RGB。可微 Base 代价是代理，正式质量和码率由独立真实编解码产生。

## 分项裁决

| 项目 | 状态 |
|---|---|
| TRAINING_AUDIT | `PASS` |
| DEPLOYMENT_WEIGHT | `PASS` |
| CAPACITY_DIAGNOSTICS | `PASS` |
| D0_FAST | `PASS` |
| STATIC_ENTROPY | `PASS` |
| SELECTED_ARCHITECTURE | `PASS` |
| CORE_DETAIL | `NOT_RUN` |
| BASE_LITE | `FAIL` |
| END_TO_END_TRAINING | `PARTIAL_PASS` |
| LOW_RATE_QUALITY | `PASS` |
| HIGH_RATE_QUALITY | `FAIL` |
| EQUAL_PSNR_RATE | `OUT_OF_RANGE` |
| MINIMUM_PSNR_THRESHOLD_RATE | `PASS` |
| EQUAL_PERCEPTUAL_RATE | `FAIL` |
| CAUSALITY | `PASS` |
| STRICT_PREFIX | `PASS` |
| INCREMENTAL | `PASS` |
| BASE_BUDGET | `PASS` |
| FULL_MAX_REALTIME | `PASS` |
| ON_DEMAND_REALTIME | `PASS` |
| TEST | `NOT_RUN` |

含复验共核对 **2065** 次已保存的优化器调用，CUDA 事件区间 898.22 s、分段墙钟合计 972.19 s、在线 Teacher 墙钟 257.19 s。另有首轮报告失败前保存的两次更新。 [独立 AdamW 计数核验](training/checkpoint_ledger.json)

## 回归

CPU 471 项通过，CUDA 路径 84 项通过。源码定向变异捕获 25 个，另有实际 CUDA 旧代理与未来帧注入。生产 Python 模块最高 Crapper 分数为 9.0。 [回归账本](qa/summary.json)

## 样例与复现

[机器裁决](final_verdict.json) · [复现命令](../../../docs/hla_tok/uf_v3/reproduction.md) · [工作记录](../../../docs/hla_tok/uf_v3/work_log.md) · [图表清单](figures/manifest.json) · [图表核验](figures/qa.json) · [原生样例哈希](samples/native/manifest.json) · [历史资产哈希](audit/historical_sha256.json) · [实验哈希](experiment_manifest.json)
