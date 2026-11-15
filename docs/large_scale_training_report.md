# CP-AdapTok 大规模训练执行报告

Git SHA：`0c3af5b11cfb7e0541258b5e317dbe50c94925aa`。实验：`formal_20261116`。日期和日志时区：Asia/Shanghai。

只报告真实完成的更新；`pilot` 是有限预实验。最终测试完成：`False`；
收敛已验证：`False`；当前仅单训练种子，不能据此宣称统计稳定性。

## 数据与隔离

```json
{
  "train": {
    "independent_videos": 120,
    "frames": 10209,
    "native_1080p_frames": 3827,
    "native_1080p_frame_fraction": 0.3748653149182094,
    "eligible_videos_by_crop": {
      "512x512": 120,
      "720x1280": 120,
      "1080x1920": 54
    }
  },
  "validation": {
    "independent_videos": 21,
    "frames": 1323,
    "native_1080p_frames": 1323,
    "native_1080p_frame_fraction": 1.0,
    "eligible_videos_by_crop": {
      "512x512": 21,
      "720x1280": 21,
      "1080x1920": 21
    }
  },
  "test": {
    "independent_videos": 9,
    "frames": 676,
    "native_1080p_frames": 555,
    "native_1080p_frame_fraction": 0.8210059171597633,
    "eligible_videos_by_crop": {
      "512x512": 9,
      "720x1280": 9,
      "1080x1920": 7
    }
  }
}
```

清单逐帧审计，记录原始分辨率、全部帧数、FPS 及名义标记、来源、官方 split、内容 ID、SHA256 和裁剪支持。
训练只允许官方 Train。历史 MCL-JCV、UVG、Basketball 和此前 DAVIS Val 开发记录保留，禁止训练。
最终测试清单在训练前封存，只有 S4 和新模型 Stage3 完成并固定模型身份后才能执行。
数据补充来源、CRC、许可及内容划分见 `dataset_audit.json` 和本实验 `manifests/`。

## 实际训练

| 阶段/分支 | 状态 | 更新数/目标 | 处理帧数 | 更新墙钟秒 | 不同视频 | 不同窗口 |
|---|---|---|---|---|---|---|
| shared_warmup/shared | complete | 2000/2000 | 16000 | 1321.3 | 120 | 3235 |
| s1/B0 | running | 902/8000 | 7216 | 728.3 | 120 | 1649 |

各阶段状态、Checkpoint、上游版本、配置与模型结构哈希、验证记录见 `experiment_manifest.json`；
所有优化器、Scheduler、四个 RNG 及累计训练状态保存到 `latest.pt`。
原始逐帧训练日志为各阶段 `train.jsonl`，实际窗口和增强为 `samples.jsonl`。

实际 Checkpoint：

- `shared_warmup/shared`：`/home/whisperliang/CP-AdapTok/checkpoints/large_scale/formal_20261116/shared_warmup/shared/latest.pt`
- `s1/B0`：`/home/whisperliang/CP-AdapTok/checkpoints/large_scale/formal_20261116/s1/B0/latest.pt`

各阶段当前最佳的实际独立验证（`—` 表示该层级未测量；Quick 的 bpp 绝不冒充真实 RD）：

| 阶段/分支 | 步骤 | 层级 | Token | PSNR dB | LPIPS | DISTS | 真实 bpp |
|---|---|---|---|---|---|---|---|
| shared_warmup/shared | 2000 | full | 128 | 12.956281 | 0.848154 | 0.703101 | 0.001227 |
| shared_warmup/shared | 2000 | full | 256 | 12.964084 | 0.847358 | 0.702752 | 0.002023 |
| shared_warmup/shared | 2000 | full | 512 | 12.957901 | 0.845546 | 0.703166 | 0.003651 |
| shared_warmup/shared | 2000 | full | 1024 | 12.959371 | 0.845225 | 0.703494 | 0.006873 |
| shared_warmup/shared | 2000 | full | 2048 | 12.961740 | 0.845376 | 0.702604 | 0.013235 |

本报告中的耗时为优化更新墙钟，不含所有数据准备、初始化和验证开销；这些会话耗时另存各阶段 `sessions/`。
按更新记录的 GPU 峰值是进程累计峰值，不能直接当作每分支独立峰值；独立 1080p 探测另行重置过计数器。
CP Decoder 计数包含监督预算与 Base 参考解码，不含 Teacher 内部的原生编解码调用。

## 已完成 Pilot 与验收（不计入正式预算）

`pilot_20261116` 实际完成 200 更新、1600 处理帧、6552 次 Decoder 调用，访问 60 条视频、379 个时间窗口。
更新墙钟 125.574 秒，吞吐 12.742 帧/秒，GPU 峰值分配 5082.12 MiB。
快速验证 K128 PSNR：步骤 100 为 12.8768 dB，步骤 200 为 12.8525 dB；确定性 Perplexity 为 27.524 → 15.346。
最佳检查点为步骤 100。画质仍低，后续 Token 收益尚不稳定；这些值不能证明收敛或渐进压缩优势。

在该最佳模型上，另用训练数据校准熵表，再对两条独立原生 1080p 视频各评估 32 帧：

| Token | PSNR dB | LPIPS | DISTS | 真实 bpp | I PSNR | P PSNR |
|---|---|---|---|---|---|---|
| 128 | 11.429842 | 0.833048 | 0.729176 | 0.00094817 | 12.9351 | 11.3813 |
| 256 | 11.429618 | 0.832472 | 0.729269 | 0.00144718 | 12.9344 | 11.3811 |
| 512 | 11.429792 | 0.832887 | 0.729489 | 0.00248240 | 12.9384 | 11.3811 |
| 1024 | 11.429857 | 0.833028 | 0.729719 | 0.00454065 | 12.9375 | 11.3812 |
| 2048 | 11.429697 | 0.832726 | 0.729685 | 0.00862835 | 12.9332 | 11.3812 |

这是 Pilot 的真实 rANS 诊断，包含 Frame Header、Packet 与 framing，旧 Scorer 尚未重训；
不能当作正式 B0/B1 结果。增强预算未改变后续参考状态，64 帧发送/接收状态及追加缓存检查通过。
正式 Full/Final 评估只汇总原始尺寸恰为 1920×1080 的序列，避免混入 4K bpp 分母。

额外原生 1080p **真实训练**资源探测：`{"real_updates": 1, "native_input": [1080, 1920], "tbptt": 2, "gradient_accumulation": 4, "frames": 16, "wall_s": 5.289510036993306, "peak_allocated_mib": 26267.2626953125, "peak_reserved_mib": 26730.0, "stage_transition_optimizer_step": 1, "included_in_formal_budget": false, "source_checkpoint": "/home/whisperliang/CP-AdapTok/checkpoints/large_scale/pilot_20261116/pilot/shared/best_validation.pt"}`。
该探测从已有权重开始，优化器为新建；若结果中 step 为 1，说明跨阶段未沿用旧动量步数。

CUDA 恢复实测：数据、增强、预算、RNG、Scheduler 完全继续；最大模型权重差异
`2.806633710861206e-05`，检查容差 `{'atol': 5e-05, 'rtol': 0.0001}`。
本机 `adaptive_avg_pool2d_backward_cuda` 不支持逐位确定性，严格探测失败日志保留，
不承诺 CUDA 浮点更新逐位重放；CPU 端到端恢复测试要求逐位一致并通过。

## 固定计划与公平性

- S0：2000 更新，512 crop，4 帧、TBPTT 2、累积 2、AdamW 3e-4；前 200 次连续表示，随后仅用 Train Pre-VQ 特征做一次 K-means。随机 VQ 温度 0.1 → 0.01，Teacher 仅 I 帧且逐步衰减。
- S1：每分支 8000 更新，512 crop，4 帧、TBPTT 2、累积 2、LR 1e-4。
- S2：每分支 12000 更新，原生 720p crop，4 帧、TBPTT 2、累积 2、LR 5e-5。
- S3：每分支 8000 更新，原生 1080p crop，4 帧、TBPTT 2、累积 4、LR 2e-5。
- S4：每分支 4000 更新，原生 1080p，8 → 16 连续帧、TBPTT 2、累积 4、LR 1e-5。
- 全部阶段采用非零起点线性 Warmup + Cosine，最低 LR 为起始的 0.1；S0 Warmup 100，其余正式阶段 200。Pilot 独立使用 200 更新、Warmup 20、20 次连续表示，用于覆盖 VQ 切换，不能充当正式 S0。
- B0 保留原多预算采样；B1 每次包含 128、2048 和随机中间预算，启用原 Nested 单向约束。结构、骨干、共享 S0、数据顺序、增强、基础损失、优化器和计划保持相同。
- 同更新次数不代表同 FLOPs。逐预算监督次数、完整 Decoder 调用数、处理帧数、GPU 时间线耗时、墙钟时间均独立记录。GPU 时间线包含主机造成的间隙，不能冒充内核活跃时间；Decoder 次数也不能冒充实测 FLOPs。
- BF16、Grad Clip 1.0；不再次放大迁移 Query，不再次 K-means，不沿用上一阶段动量。默认 Rate Loss 为 0，避免未校准的率项造成塌缩；LPIPS 由 S1 0.01 到 S2–S4 0.03，DISTS 只评估。尚未执行的强率失真微调不作完成声明。

## 验证、模型选择与真实码率

主要预算 `[128,256,512,1024,2048]`。快速验证使用独立视频和确定性离散 VQ；
完整开发验证与最终测试使用真实 rANS、Frame Header、Packet 和传输 framing，校验发送/接收 Base-only 状态一致。
每次完整评估先做 Train-only 熵校准，单独保存带概率表身份的 `calibrated.pt`，不得恢复表示优化器。
最终模型重新训练 Stage3 熵模型和 Scorer；旧 Scorer 不用于正式性能结论。

预先固定选择公式：所有预算平均 MSE + 0.25×(K128 MSE + K2048 MSE)
+ 0.001×质量倒退率 + 0.001×平均 LPIPS + 0.001×平均 DISTS + 0.0001×平均真实 bpp。
只比较相同层级和相同样本的验证分数，Full 层级建立后优先 Full；快速验证缺失的指标不填假值。
不以 DeltaPSNR 或 Perplexity 单独选模。不根据单个验证点或早期前缀收益停止；自动平台早停关闭。

各预算 PSNR/LPIPS/DISTS、I/P PSNR：`validation_metrics.csv`；真实 RD：`rd_comparison.csv`；
最终 B0/B1 逐帧与视频 Bootstrap：本实验 `comparison.json`（仅在实际完成后生成）。
未产生的数值表示尚未测量，不能解释为零。只有完成封存测试后才讨论同 bpp 插值比较。

## 复现与继续

```bash
.venv-native/bin/python -m scripts.prepare_large_scale_data
.venv-native/bin/python -m scripts.prepare_reds_large_scale --sequences 60
.venv-native/bin/python -m scripts.train_large_scale --phase preflight
.venv-native/bin/python -m scripts.train_large_scale --phase pilot --run-id pilot_20261116
.venv-native/bin/python -m scripts.train_large_scale --run-all --run-id formal_20261116
.venv-native/bin/python -m scripts.train_large_scale --resume --run-all --run-id formal_20261116
.venv-native/bin/python -m scripts.train_large_scale --phase s2 --branch both --run-id formal_20261116
.venv-native/bin/python -m scripts.train_large_scale --phase evaluate --run-id formal_20261116
.venv-native/bin/python -m scripts.report_large_scale --run-id formal_20261116
```

`--resume` 恢复同阶段绝对目标，不追加额外预算。输入清单和配置变化会拒绝恢复；
数据扩充后使用新 run-id，以免改变已有实验的数据定义。
持续过程中的状态以 `state.json` 和 `latest.pt` 为准，汇总报告为生成时快照。

## 验收与研究边界

测试日志：`results/large_scale/logs/pytest.txt`、`resume_acceptance.txt`；ub-mode 行为与决策日志：
`docs/large_scale_work_log.md`。图表保存在本实验 `figures/`，只有实际存在的数据才绘图。
仍需依据充分训练后的独立验证判断绝对画质、后续 Token 增益、I/P 差异、过拟合、VQ 二次塌缩和验证平台。
资源或数据阻塞记录在 `blocked.json` / `interruption.json`；未完成预算保留，不编造收敛或最终 RD。
