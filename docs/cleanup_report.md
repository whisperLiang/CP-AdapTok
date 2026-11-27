# CP-AdapTok 清理验收

本次清理为 **PARTIAL_PASS**。PHT-Tok 是唯一持续研发主线，原总体研究裁决为 **PARTIAL_PASS**，研究准入为 **FAIL**。本次代码基于 PHT-Tok `54674cc`，清理前主工作区检出 AIP-VTok Sparse RFSQ `606216e`。原版 AdapTok 的历史证据来自 `c935509`，D1 来自 `ffbb717`。清理后的单卡代码验收提交为 `f09b357cdc9aed5d8c985d3356928486f116c0c1`，`pht-tok` 和稳定分支 `main` 均已核对到该快照。验收文档随发布提交交付。

## 四项现行实现

- PHT-Tok 使用同一份官方 HT-S 骨干，保留 P1/P2、一次八帧分析、物理字节前缀、独立参考和动态调度。保留原始 80 块 Full 等价、16 块连续参考及失败后的 I 帧恢复证据。
- 官方 DCVC-UF LD 与 HT-S 使用同一官方源码和权重目录，实跑官方脚本的双 QP、I/P 及真实码流。保留官方、历史对齐与 PHT 直接对照配置和 Calibration。
- 官方 AdapTok 以同一官方 FP32 权重运行无重叠和重叠分块。历史报告 K2048 的 PSNR 为 24.6811 和 25.1504 dB，计算吞吐为 1.558 和 0.970 FPS。真实序列化字节缺失，字段为 `null`。
- D1 使用原报告选定的 Seed 2 E0。原检查点的 55 个张量逐项保持 dtype 与数值一致，移除优化器、游标及随机状态。两种子的最终评估均保留。5/10/20/40 Mbps 前缀、因果 Base、独立参考和 Full-max 均实跑。

## 工程回归和 H100

四项方案共同使用 DAVIS2017 car-roundabout 的前 17 帧 Validation。PHT-Tok、LD、HT-S 和 D1 在前后生成的真实码流逐字节一致。重建输出的尺寸及摘要、PSNR、LPIPS、DISTS 逐项一致。PHT 的 UVG 16 块参考连续性、P1/P2 各四个物理前缀、增量解码、丢失恢复和 Verizon 回放通过。动态 QP HT-S 与 PHT 前缀使用相同的轨迹及截止时间计算。 [网络回放记录](../results/cleanup/validation_network_streaming.json)

下表每项在清理前后均得到相同数值，PSNR 采用逐元素合并口径。完整 21 行包含两种 PHT 四前缀、双 QP、D1 四档以及 AdapTok 两种分块预算。 [共同 Validation 率失真表](../results/cleanup/validation_rate_distortion.csv)

| Validation 路径 | PSNR dB | LPIPS | DISTS | 真实字节 |
| --- | ---: | ---: | ---: | ---: |
| DCVC-UF LD QP 32 | 30.2584 | 0.24448 | 0.13548 | 194,155 |
| DCVC-UF HT-S QP 32 | 27.2003 | 0.30793 | 0.15922 | 134,645 |
| PHT-Tok P1 Full QP 32 | 27.2003 | 0.30793 | 0.15922 | 134,782 |
| D1 40 Mbps | 27.9683 | 0.21431 | 0.15144 | 2,513,695 |
| AdapTok stride 128 K2048 | 24.3648 | 0.15491 | 0.07040 | 缺测 |
| AdapTok stride 96 K2048 | 24.8749 | 0.14508 | 0.06523 | 缺测 |

下表使用受控的 H100 配对测量和已移除退役入口的独立工作目录，覆盖同步核心吞吐、串行编解码、峰值显存和模型加载。端到端显示等待另见网络回放。

| 路径 | 编码 FPS 前 → 后 | 解码 FPS 前 → 后 | 串行 FPS 前 → 后 | 串行 P50 ms 前 → 后 | 串行 P95 ms 前 → 后 | 峰值 GiB 前 → 后 | 加载秒数前 → 后 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PHT-Tok | 239.371 → 230.791 | 128.243 → 123.745 | 83.607 → 80.838 | 100.440 → 104.178 | 108.707 → 109.651 | 8.691 → 8.691 | 9.32 → 10.62 |
| DCVC-UF LD | 370.452 → 371.720 | 425.836 → 428.407 | 198.406 → 199.153 | 4.986 → 4.972 | 5.396 → 5.340 | 4.445 → 4.445 | 2.66 → 2.68 |
| DCVC-UF HT-S | 571.736 → 573.002 | 505.653 → 504.361 | 268.334 → 268.248 | 30.196 → 30.052 | 30.831 → 30.816 | 14.289 → 14.289 | 4.67 → 4.68 |
| D1 40 Mbps | 74.495 → 74.593 | 66.046 → 68.118 | 34.904 → 35.499 | 28.507 → 28.198 | 30.065 → 29.395 | 2.123 → 2.123 | 3.81 → 3.71 |
| AdapTok stride 96 | 1.983 → 1.977 | 1.411 → 1.407 | 0.820 → 0.818 | 19505.323 → 19563.578 | 19671.129 → 19598.999 | 5.118 → 5.118 | 8.48 → 6.64 |
| AdapTok stride 128 | 3.277 → 3.262 | 2.333 → 2.324 | 1.354 → 1.349 | 11819.076 → 11857.056 | 11969.123 → 12025.602 | 4.162 → 4.162 | 8.48 → 6.64 |

PHT 追加了按原报告的官方预热执行的四对 H100 复测。PHT 的时延以八帧 Chunk 计，AdapTok 以完整 16 帧窗口计，其余按对应验收帧或 Chunk 计。AdapTok 每个窗口共享一次编码，解码列计两个预算的实际工作。各方法的 CUDA 同步与预热按原方案执行，详细计时包括加载时间。原 PHT 历史快照约为 434/282/171 FPS。受控复测的绝对吞吐受当时主机负载影响，报告同时保留了首次 H100 复测值和受控配对差值。30 FPS 源的八帧采集首尾等待约 233 ms，实际网络显示延迟另由轨迹与播放模型记录。

未绑定 CPU 时 LD 解码 P95 的差值约 7%。将前后进程绑到同一组 H100 本地 NUMA CPU，按 ABBA 交错执行八对独立测量。解码 P95 中位数为 2.590 → 2.563 ms。受控复测的稳态结果满足 5% 门槛，峰值显存相同或略减。额外的无官方预热双代理探针在旧版与迁移版均出现首次 I 参考漂移。正式代码和参考契约沿用原实现，配对测量使用原报告的预热。 [首次测量](../results/cleanup/h100_regression.json) · [独立目录测量](../results/cleanup/isolated_h100_regression.json) · [PHT 复测](../results/cleanup/pht_retiming.json) · [LD 复测](../results/cleanup/ld_retiming.json)

删除后主工作区有一轮与现场 UPG-Tok 任务共用 H100，PHT 报告 rANS 记录异常，部分基线的计时也波动。设备空闲后，原版和现行 PHT CUDA 套件各通过四例，现行真实码流和网络回放通过。五个入口的稳定输出与清理前完全一致。并发环境的正确性和延迟仍待验证，异常根因尚未确认。 [删除后验收](../results/cleanup/post_deletion_gate.json) · [码流和重建摘要](../results/cleanup/validation_receipts.json) · [共享设备计时记录](../results/cleanup/post_cleanup_equivalence.json)

## 实验资产与分支

移除了 CP-AdapTok B0/B1、原历史熵模型和评分器、AdapTok-HR、AIP-VTok、固定 2048、HLA-Tok 的退役变体和单独 HT-L 实验。PHT 基点到最终代码的精确路径清单含 2,733 个原有跟踪路径。必要的读取、切片、指标、NAL 格式和载荷统计、计时和参考管理从原实验抽取。官方两个 Gitlink 保持锁定。历史负结果见 [研究历史](research_history.md)。

| 指标 | 清理前 | 清理后 | 减少 | 降幅 |
| --- | ---: | ---: | ---: | ---: |
| Python 生产代码行 | 48,223 | 6,556 | 41,667 | 86.40% |
| Python 测试行 | 7,723 | 1,661 | 6,062 | 78.49% |
| Python 文件 | 476 | 91 | 385 | 80.88% |
| 跟踪文件 | 2,795 | 275 | 2,520 | 90.16% |
| 实验配置 | 43 | 9 | 34 | 79.07% |
| Checkpoint 数量 | 811 | 147 | 664 | 81.87% |
| Checkpoint 体积 | 71.253 GiB | 2.419 GiB | 68.834 GiB | 96.60% |
| Results 体积 | 51.857 GiB | 3.374 GiB | 48.484 GiB | 93.49% |
| Logs 体积 | 66.46 MiB | 58.35 MiB | 8.11 MiB | 12.20% |
| 注册 worktree | 19 | 6 | 13 | 68.42% |
| 本地分支 | 28 | 7 | 21 | 75.00% |
| 远端分支 | 18 | 4 | 14 | 77.78% |
| 项目实际分配磁盘 | 167.832 GiB | 47.605 GiB | 120.227 GiB | 71.64% |

Python 行数以原 PHT HEAD 作比较，排除第三方子模块、生成的结果脚本和数据清单。物理大小覆盖主工作区、注册的项目 worktree 和共同 Git 对象，按 inode 去重。Checkpoint 统计覆盖 `checkpoints` 目录的指定权重类型。结果和日志使用逻辑文件大小。采样时间为 2026-11-27T13:16:41+08:00，现场任务会继续写入受保护的工作树。实测项目占用减少 **120.227 GiB**。Git 历史对象占用 0.865 GiB，LFS 为 0.00 MiB。

按计划删除 172 个物理资产路径。退役源码的 463 个自动生成字节码文件经 Git、运行进程和跨工作树链接核验后清除。删除 23 个本地实验分支、15 个远端实验分支、14 个已确认的历史 worktree 和临时清理工作树。 [清理计划](../cleanup_plan.json) · [资产清单](../results/cleanup/cleanup_manifest.json) · [字节码路径回执](../results/cleanup/retired_cache_cleanup.json)

### 已删除的分支与历史工作树

本地分支包括 `experiment/adaptok-hr`, `experiment/adaptok-hr-fine-detail-v2`, `experiment/adaptok-hr-fine-recovery`, `experiment/adaptok-hr-perceptual-recovery`, `experiment/adaptok-hr-unified`, `experiment/aip-vtok`, `experiment/aip-vtok-c1`, `experiment/aip-vtok-c1-followup`, `experiment/aip-vtok-sparse-rfsq`, `experiment/fixed2048-reconstruction`, `experiment/hla-tok`, `experiment/hla-tok-cc`, `experiment/hla-tok-compact`, `experiment/hla-tok-continuous`, `experiment/hla-tok-hts-v2`, `experiment/hla-tok-hts-v2.1`, `experiment/hla-tok-temporal`, `experiment/hla-tok-uf`, `experiment/hla-tok-uf-v2`, `experiment/hla-tok-uf-v3`, `experiment/hla-tok-v2`, `experiment/hla-tok-v3`, `experiment/original-adaptok-1080p`。

远端分支包括 `experiment/adaptok-hr`, `experiment/adaptok-hr-fine-detail-v2`, `experiment/adaptok-hr-fine-recovery`, `experiment/adaptok-hr-unified`, `experiment/aip-vtok`, `experiment/aip-vtok-c1`, `experiment/aip-vtok-c1-followup`, `experiment/aip-vtok-sparse-rfsq`, `experiment/fixed2048-reconstruction`, `experiment/hla-tok`, `experiment/hla-tok-compact`, `experiment/hla-tok-continuous`, `experiment/hla-tok-v2`, `experiment/hla-tok-v3`, `experiment/original-adaptok-1080p`。

历史工作树包括 `adaptok-hr-unified`, `fixed2048-reconstruction`, `hla-tok-cc`, `hla-tok-compact`, `hla-tok-continuous`, `hla-tok-hts-v2-1`, `hla-tok-hts-v2`, `hla-tok-temporal`, `hla-tok-uf-v2`, `hla-tok-uf-v3`, `hla-tok-uf`, `hla-tok-v2`, `hla-tok-v3`, `hla-tok`。

## 测试和原始数据

- 本地 CPU 测试为 `122 passed, 9 deselected in 8.77s`。GitHub Actions 的干净环境构建和 122 个 CPU 测试通过。 [CI 回执](https://github.com/whisperLiang/CP-AdapTok/actions/runs/45939175689)
- CUDA 测试通过 9 例，其中 PHT 4 例、D1 4 例、AdapTok 1 例。AdapTok 全套共 40 例。官方 LD、HT-S 两个 QP 的 I/P 路径与官方 CLI 实测通过。官方、PHT 和 D1 原生扩展各自在锁定源码上从头构建。 [官方 CLI 核验](../results/cleanup/official_cli_parity.json) · [原生构建记录](../results/cleanup/source_builds.json)
- Ruff 与 mypy 均通过。15 个 CLI 的帮助入口正常，59 个第一方模块完成 AST 依赖审计。D1、AdapTok 和 PHT 的推理追踪中，旧 worktree 路径与封存 Test 图片的文件访问计数均为零。 [依赖核验](../results/cleanup/dependency_audit.json) · [实际打开文件核验](../results/cleanup/file_access_audit.json) · [复现说明](reproduction.md)
- 原始 Train 10,209 帧和 Validation 1,323 帧的文件 SHA256 与清单逐一相符。Test 676 帧的清单、路径和文件元数据摘要在清理前后一致。Test 像素读取 **0**。选中的模型及网络轨迹 SHA256 通过。
- ub-mode 的定向 PHT CPU 测试检出 8/8 个变异，Test 封存测试检出 1/1 个定向变异。Dryer 在相似度 0.8 的阈值下未报告重复候选。Crapper 只计入 CPU 覆盖，仍有 102 个函数达到或高于目标阈值 10，包括原有的复杂评估函数。 [代码审计](../results/cleanup/ub_mode_checks.json)

## 保留的阻塞项

`experiment/aip-vtok-joint-v1` 仍有用户未提交的报告。原分支、工作树及五项跨工作树依赖保留。两份官方权重及三个 Joint V1 依赖检查点经原路径可达，其中三个旧检查点仍占用本项目权重目录。

`experiment/upg-tok` 是清理期间发现的用户工作，含未提交内容和近期 H100 任务。它引用的旧 `experiment/pht-tok` 工作树保留，后者的官方权重链接依赖 `experiment/hla-tok-hts` 工作树。Joint V1 依赖的 `experiment/aip-vtok-sparse-decoder` 工作树同样保留。存留分支和具体路径见[资产清单](../results/cleanup/cleanup_manifest.json)。

三个未登记目录因项目归属未证实而保留，具体路径见资产清单。Git 历史仍由原有提交持有。

## 执行记录

1. 枚举 Git 共同目录、根和子模块 Gitlink、分支、远端 HEAD、工作树、未提交路径及磁盘归属，冻结删除计划。
2. 从原报告核对 PHT 研究裁决、AdapTok 官方权重与 D1 两种子最后节点，以正式选定的 Seed 2 生成只含权重的冻结包。
3. 用 H100 冻结相同 Validation 源、实际码流和质量。提取共享依赖、统一前缀测试与基准入口，在独立分支复测。
4. 对 LD 的尾延迟差异进行受控成对复测。通过门禁后先推送并核对新分支，通过逐路径和活动进程检查后执行磁盘删除。
5. 在主工作区重新运行 CPU/CUDA、官方双 QP、四项推理和摘要核验。按远端返回的预期 SHA 核对后删除旧分支，将稳定主分支快进到通过验收的快照。
6. 核对原始分区、必要权重和二进制扩展的 SHA256，检查并移除退役字节码，重跑 CPU 测试及静态检查。

研究继续围绕 PHT-Tok 原生 1080p 多帧渐进传输。优先级由已测瓶颈决定，包括中间前缀的率失真、动态 QP HT-S 的网络对照以及采集和播放时延。 [最终裁决](../results/cleanup/final_verdict.json)
