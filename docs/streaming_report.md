# 在线 TCP 后续验证

本轮把在线预算选择接入真实 localhost TCP：完整编码一次，先可靠交付并确认基础前缀，再按当前带宽和剩余时间选择增强包。每个增强包发送前复查截止时间；接收端仅使用模型和线路字节。**状态仍为 `untrained_backbone_smoke`；这些结果验证工程行为，不验证 RD 或 Scorer 泛化。**

## 实测条件与结果

- 代码提交 `41842026ba6cf13fa8e117e0db86e2248c5bb313`；每份 JSON 同时记录源码 SHA256、模型 ID、配置与 GPU 快照。
- 同一个 Stage 3 结构 checkpoint，N_max=2048、K_ref=128、原生1920×1080 RGB，H100 PCIe。在线场景为8帧/50 FPS源时间轴，每个 CLI 独立进程。源时钟从发送循环开始，包含首帧 kernel 冷启动和积压；checkpoint 加载及模型构建先完成。
- Fixed/Rate-only/Scorer 截止时间1200 ms，当前逐帧带宽为32/128/512/2048/128/512/2048/512 kbps；固定目标无法及时交付时仍可靠保留基础图。
- 截尾场景为2500 ms、2048 kbps，只舍弃未发送的连续增强尾部；ACK 延迟场景为1200 ms、2048 kbps。除两帧五档 PNG 外，显示指 Tensor 就绪并转到 CPU，不含实体显示器扫描。

| 场景 | 帧数 | 实际发送 Token 前缀 | 双向应用 bytes | 基础 ACK 迟到帧 | 显示迟到次数 |
|---|---:|---|---:|---:|---:|
| 全部五档 PNG | 2 | 2048, 2048 | 7334 | — | — |
| 300 ms 截止时间 | 8 | 128, 128, 128, 128, 128, 128, 128, 128 | 3034 | 8 | 8 |
| 固定目标 2048 | 8 | 128, 2048, 128, 128, 128, 128, 128, 128 | 6126 | 4 | 4 |
| Rate-only | 8 | 1024, 128, 128, 128, 128, 128, 128, 128 | 4510 | 6 | 6 |
| Causal Scorer | 8 | 128, 128, 128, 128, 128, 128, 128, 128 | 3034 | 0 | 0 |
| 2048 目标截尾到512 | 8 | 512, 512, 512, 512, 512, 512, 512, 512 | 8378 | 0 | 0 |
| 100 ms ACK 延迟 | 8 | 2048, 128, 128, 128, 128, 128, 128, 128 | 6126 | 6 | 5 |

七组的发送/接收基础提交次数、参考摘要、原包前缀 SHA256 以及双向应用字节全部核对。截尾场景8帧均从2048截到512，省去96个增强包，后续 P 帧无漂移；300 ms 场景所有基础帧迟到，省去120个增强包，不能宣称达到实时目标。表中策略结果受同一帧历史因果状态约束；每次独立冷启动存在调度波动，不能据此推断策略质量优势。

## 修复并复现的问题

1. 基础显示后按整帧计费会重复扣除基础字节；现在只计未发送字节和62-byte 显示交换控制。
2. 排除已显示前缀会迫使 Scorer 选择预测更差的增强；现在保留零字节成本的基础候选。
3. 固定25 ms解码预留在先前实测中造成6.46 ms增强显示迟到；现在取配置下限与已观测显示 ACK 耗时上界的1.25倍中较大者，包含当前基础及历史确认，不读取未来耗时。旧失败证据在 `results/streaming_followup`，修复后证据在 `results/streaming_verified`。
4. 控制字段必须为规范整数，ACK 摘要必须完整128 bit，NaN/Inf截止时间在 I/O 前拒绝；基础缺失不能正常结束会话。红测试日志和行为变异覆盖这些约束。
5. Python API 的 `8.0` Token 预算此前也能通过范围检查并提交参考；现在所有编码/包切片/显示入口统一要求规范整数，在基础提交前拒绝非法值。

## 字节与时间口径

FrameHeader/Packet bytes 不变。新增 CPDS/CPAK 含 CRC 与 length framing 分别23/39 bytes，每次显示共62 bytes；正常结束/确认共26 bytes。两帧五档 PNG 运行双向应用总字节7334，相比旧字符串 ACK 演示增加了完整校验控制，不能混用旧6882-byte测量。

源释放时间为 `origin + frame_id / source_fps`，不会随处理速度放慢。预算扣除编码、实际基础交付及显示 ACK 已花时间；报告区分真实显示迟到与发送端 ACK 迟到。预留仅为因果估计，不能保证冷启动和突发抖动下的硬实时。

前向 record 按当前带宽做应用层交付整形，ACK 路径未限速，但其字节计入总量并保守计入选择预算。没有模拟 TCP 随机丢包、内核重传、无线拥塞，也没有统计 TCP/IP/link 头。未发送尾部舍弃不等同于物理丢包。

## 复现

```bash
cd /home/whisperliang/CP-AdapTok
source .venv/bin/activate
python scripts/run_streaming_experiments.py --checkpoint checkpoints/s3_smoke/step100.pt --allow-untrained --output results/streaming_verified
python -m cp_adaptok.cli.stream_demo --mode localhost --progressive --config configs/streaming.yaml --checkpoint checkpoints/s3_smoke/step100.pt --allow-untrained --allocation rate-only --bandwidth-kbps 32 128 512 2048 128 512 2048 512 --playback-delay-ms 1200 --no-save-frames
```

逐项命令和退出码见 `results/streaming_verified/commands.json`；逐帧预测、实际质量、deadline、字节、ACK、状态摘要在各场景的 `stream_demo.json`。未训练模型的画质仍不合格；官方权重、原生 CUDA、正式长序列 RD 和30/60 FPS验收仍待完成。
