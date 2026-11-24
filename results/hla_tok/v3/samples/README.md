# 原生 Validation 样例

样例来自冻结的 `DAVIS2017:car-roundabout` 第 32–47 帧，30 FPS。每个目录保留 1920×1080 RGB PNG、
同坐标的 384×216 裁块（左上角 `x=768, y=432`）和 16 帧 FFV1 无损视频。清单绑定检查点与文件摘要，
并核验解码所得的 FFV1 首帧与 PNG 逐字节相等。

| 保存点 | 重建路径 | 样例设备和精度 | 清单 |
|---|---|---|---|
| A0 4,000 | 稠密 | H100 BF16 | [样例](A0_step04000/manifest.json) |
| A0 12,000 | 稠密 | H100 BF16 | [样例](A0_step12000/manifest.json) |
| A1 12,000 | 稠密 | CPU BF16 | [样例](A1_step12000/manifest.json) |
| B0 20,000 | 连续 K16320 | CPU BF16 | [样例](B0_step20000/manifest.json) |
| C0 10,000 | 连续对照 K16320 | CPU BF16 | [样例](C0_step10000/manifest.json) |
| C1 10,000 | 真实 Fixed13 K16320，Phase C FAIL | CPU BF16 | [样例](C1_step10000/manifest.json) |

CPU 样例使用独立的只读工作区和原始评价帧。正式质量裁决使用 H100 上的完整 174 帧结果。
C1 的样例经过整数 ID 序列化、解包和独立接收端。渐进八档样例为 BLOCKED。

```bash
python -m scripts.hla_tok.v3.run samples --checkpoint /path/to/step.pt
```
