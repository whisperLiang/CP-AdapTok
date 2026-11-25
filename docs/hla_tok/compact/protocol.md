# Compact 码流

HLCP v1 使用 60 字节的固定小端包头。一包承载当前帧内一个连续 Token 范围。

| 字段 | 宽度 | 含义 |
| --- | ---: | --- |
| Magic | 4 | ASCII `HLCP` |
| Version | 1 | 1 |
| Profile ID | 1 | `hla_tok.compression.profiles.Profile` 的冻结编号 |
| Order ID | 1 | 0、1、2 对应 O0、O1、O2 |
| Dimension | 1 | 固定 Profile 的传输分量数 |
| Frame ID | 4 | 无符号 32 位当前帧标识 |
| Token offset | 2 | 此范围的起始位置 |
| Token count | 2 | 此范围的 Token 数 |
| Height / width | 2 + 2 | 1080 和 1920 |
| Payload length | 4 | 实际 Payload 字节数 |
| Binding SHA-256 | 32 | Profile、排序、原生 Rank 排列表、紧凑资产与 C0 状态的绑定 |
| CRC32 | 4 | 覆盖上述 56 字节和完整 Payload |

INT4、INT6、INT8 按小端位序保存 `[-8,7]`、`[-32,31]`、`[-128,127]` 的二补数。INT4 每两个分量一字节，INT6 每四个分量占三字节。BF16 直接传输小端位模式。注册的 Profile 均按整字节划分 Token，实际 padding 和逐帧边信息均为 0。通用标量打包器支持分量数不对齐时在每 Token 最后一字节补零。每 Token 占 `ceil(d × b / 8)` 字节，非零补位在解包时拒绝。

秩标识和区域标识由共享的排序表及原生几何确定。原生 Rank 120 只含前 120 个光栅区域。O1 和 O2 重排整个 Rank span，同时保留该真实宽度。静态资产包含 FP32 投影、Lift 和尺度，文件及装载成本在 Profile manifest 记账。当前帧每个新增分段计入一个包头，累计 Payload 正好等于相同预算的单包 Payload。帧内的 Profile、资产和排序绑定固定。

乱序分段保存在接收端。只有起始于零且连续收到的范围可以进入 Lift 和 Recovery。重复或重叠范围立即拒绝。接收端提交时丢弃缺口后的待处理分段，可直接恢复此前已收到的连续部分。适配与纯静态资产使用相同的包格式和独立合成端。

6 个总码流上限按 30 FPS 计算。固定长度给出候选 K，评价逐帧检查实际序列长度、CRC32 和真实字节。K16320 耗尽后保留实际用量。旧连续协议保持 28 字节包头，C1 与 Original 的新 F13W 比较采用 24 字节实验包头。三种协议的字节账本分别列出。
