# HLA-Tok V3 Reconstruction-First 实验记录

快照时间：2026-11-25T01:53:32+08:00。整体：**PARTIAL PASS**。 V2 D2 保持 **FAIL**；Test 像素读取 **0**。

## 阶段与训练

| 阶段 | 状态 | 说明 |
|---|---|---|
| A_seed1 | PARTIAL PASS | 正式训练未完成，不给出质量准入结论 |
| B_seed1 | BLOCKED | 前置阶段未取得 PASS |
| C_seed1 | BLOCKED | 前置阶段未取得 PASS |
| D_seed1 | BLOCKED | 前置阶段未取得 PASS |
| A_seed2 | BLOCKED | 前置阶段未取得 PASS |
| B_seed2 | BLOCKED | 前置阶段未取得 PASS |
| C_seed2 | BLOCKED | 前置阶段未取得 PASS |
| D_seed2 | BLOCKED | 前置阶段未取得 PASS |
- A0 seed1：实际 12,000/12,000 更新；最近可恢复 12,000；有效 Train 像素 10,391,552,000。
- A1 seed1：实际 875/12,000 更新；最近可恢复 500；有效 Train 像素 229,376,000。
- A0 seed1 终点质量 **PASS**。单臂达标不等于 Phase A 配对准入。
- A0 第 12,000 步，174 帧 完整 Validation：31.75656 / 0.10040 / 0.06705。
- A0 第 12,000 步，9 帧 趋势诊断：31.77989 / 0.10619 / 0.06795。
- A1 第 500 步，9 帧 趋势诊断：22.77801 / 0.43491 / 0.27176。
- A0 终点 H100 BF16 稠密前向：237.41 FPS，P95 4.227 ms，峰值分配 1653 MiB。 此处没有 Token、码率或网络流水线。

## 历史复现与准入边界

旧 R2 2000 步检查点在同一 174 帧上逐指标精确复现：19.12730 / 0.68087 / 0.39074。 固定阈值仍为 24.92307 dB / 0.20373 / 0.09173；旧 R2 没有已验收的离散码流。

## 十项研究问题

1. 原生稠密可恢复性：A0 12000 步 31.75656 / 0.10040 / 0.06705；以 Phase A 终点裁决为准。
2. V2 感知质量不足是否解决：端到端稠密臂已有达标终点；这不能证明连续或离散 Token 也已达标。
3. 连续 Token 损失：尚无匹配的 B0 全帧读数。
4. 是否需要 ARD：未启用；只有 B0 的真实失配与成对 Local Decoder 诊断能够触发。
5. VQ 即时与轨迹误差：未测；连续 R2 不是量化控制。
6. 原 Phase 1 bpp 质量门槛：Seed 1 固定离散阶段 BLOCKED。
7. 渐进式新增 Token 的边际收益：Seed 1 渐进阶段 BLOCKED。
8. H100 K16320 性能：未测通过质量的 V3 离散模型。
9. 跨种子一致性：Seed 1 / Seed 2 固定离散 BLOCKED / BLOCKED。
10. 自适应传输准入：未获得。

PSNR 为 Global PSNR；LPIPS 为 LPIPS-Alex。9 帧观察不参与任何晋级。质量不确定性按视频重采样，不把相邻帧当独立样本。码率表排除稠密模型和浮点连续表示；历史 FPS 不用于 V3 实时性裁决。

见 [阶段与测试 JSON](acceptance.json)、[逐检查点指标](reconstruction_metrics.csv)、[架构对照](architecture_comparison.md) 和 [复现说明](../../../docs/hla_tok/v3/reproduction.md)。
