# 训练准入与冻结配置

`configs/hla_tok/hts/protocol.json` 保存训练阶段、预算分布、两个 Seed 的更新次数和验收门槛。原生 HT-S 的实际码流必须先在 Calibration 的低码率重叠区间体现相对 LD 或历史 Base 的质量收益，并通过官方 parity 审计。

Phase 2 的 C1 计划使用 Seed 1、2 各 1,000 次 AdamW 更新。C2 从对应 C1 继续，各 2,000 次。准入使用固定 Base 历史、闭环状态、实际渐进前缀以及 0.5～2 Mbps 的同码率对照。只有容量门槛通过才进入每 Seed 20,000 次的正式训练。

运行状态与实际更新次数以 `results/hla_tok/hts/final_verdict.json` 为准。Phase 0 的原生连续探针使用官方未量化 y，属于容量诊断。该实验计量 RGB PSNR、LPIPS-Alex 和 DISTS 的变化，连续表示栏没有码流长度。
