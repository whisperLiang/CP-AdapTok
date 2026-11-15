# CPAT v1 二进制协议

网络整数 big-endian；rANS 初始状态单独采用 little-endian uint32。模型 B=1，每帧 RGB 尺寸独立声明，底/右 padding 到 64 倍数。没有 Python pickle、连续 latent 或 zlib 压缩 payload；zlib 只用于标准 CRC32。

## FrameHeader：108 bytes

Python struct：`>4sBBBBQqHHHHHHHH16s16s16s16s`，后跟 big-endian CRC32(4 bytes)。

| 字段 | bytes | 含义 |
|---|---:|---|
| magic | 4 | CPAT |
| wire_version/frame_type/qp/color_version | 1×4 | 1，I=0/P=1，QP 0..63，BT.709 centered YCbCr 版本 1 |
| frame_id/reference_id | 8+8 | unsigned 当前帧、signed 参考帧；I ref=-1，P ref=frame_id−1 |
| H/W/Hpad/Wpad | 2×4 | 原图和规范 padding |
| N_max/K_ref/packet_tokens/codebook_size | 2×4 | 固定最大长度及 packet 对齐约束 |
| model/codebook/entropy/reference fingerprints | 16×4 | 模型、码本、整数 CDF 的 SHA256 前128 bit；reference 是版本化整数状态摘要再取 SHA128 |
| CRC32 | 4 | 以上全部字段 |

wire_version 固定 rANS 算法、precision 和 packet 编码方式；codebook/entropy 的内容版本由各自摘要绑定。模型摘要包含结构配置、Attention precision、state_fingerprint=mixed_int64_v1 和全部 state_dict 参数。状态摘要检测意外漂移，不是密码学认证。Header 不含本次“最终目标 K”，追加预算永远不需要重写它。

## Packet：28-byte overhead + payload

Python struct：`>4sQIHHI`，后跟 payload，再跟 CRC32(header+payload)。依次为 CPTK、frame_id(8)、first_token_index(4)、token_count(2)、block_id(2)、payload_len(4)。frame_type/shape/模型版本等共享字段在 FrameHeader，避免逐包重复。packet_tokens 默认128，支持64/256；8/16/32用于确定性小测试。N_max/K_ref 均必须对齐。

第一版每块使用已共享的静态/训练得到的 block-position 概率表，同块并行预测，无 token-serial neural AR。表以 float64 CPU largest-remainder 方法确定性量化到整数 CDF，precision=16，所有符号 freq≥1，CDF末值65536；tie按符号index排序。表随模型 artifact 保存，不按帧发送；entropy_id 检查表相同。

每包独立 rANS state，L=2^23。编码逆序处理符号，按 byte renormalization；payload 为4-byte state加反向 renormalization bytes。解码必须精确消费 payload 并终止于 L，尾部 junk 与截断均拒绝。概率表训练后用真实 packet 长度统计收益，不能用 token_count×log2(codebook_size) 代替。

只接受连续 prefix：first_token_index==已收到数、block_id==已完成包数、count==packet_tokens、frame_id一致。CRC、entropy termination、size bound 均通过后才追加 IDs。单个失败包不修改状态；批量 append 中先前已验证包可以保留，后续失败包不会被拼接。乱序、重复、丢包后越过缺口均报错。

## TCP 应用 framing

每 record 为 `uint32 length + body`，大小上限1 MiB，接收精确循环支持任意 socket 分片和 EOF 检查。body 可以是原样的 FrameHeader/Packet，也可以是下面的会话控制。TCP/IP/链路头未抓包统计。

文件 `.cpat` 保存同样的 length-framed records；最大预算文件的旧前缀 bytes 完全不变。会话当前帧的 base 不完整时，不允许无声推进下一 P 帧；显式 reset 后可从任意新的 I 帧同步。

## Socket session v1 控制记录

会话控制不改 CPAT/CPTK 内容，也不在 FrameHeader 中加入最终 K。接收端通过线路上的请求确定何时显示，无需预先共享发送预算。

| 控制 | struct / body | 含义 | 含 CRC 与 framing 字节 |
|---|---|---|---:|
| CPDS | `>4sBQH` + CRC32 | magic、version=1、frame_id、已收到的显示前缀 K | 23 |
| CPAK | `>4sBQH16s` + CRC32 | 同一帧/K、基础参考状态摘要；显示完成后才发送 | 39 |
| CPEN | `CPEN` + version=1 + CRC32 | 正常结束，当前基础包和显示必须完整 | 13 |
| CPED | `CPED` + version=1 + CRC32 | 正常结束确认 | 13 |

每次显示请求/确认共62 bytes。ACK 的帧号、K、CRC、参考摘要全部验证；重复/回退请求、请求未收到的 Token、未知控制、基础包缺失时的结束均失败。默认 PNG 模式先落盘再 ACK；`--no-save-frames` 表示重建 Tensor 已准备好并转到 CPU，没有实际显示设备的扫描延迟。

在线会话先可靠发送并确认 K_ref。之后只用当前帧预测、当前带宽和绝对播放截止时间，按尚未发送的 packet bytes + 62-byte 控制开销选择增强前缀。保留已显示基础图的增量成本为0；Scorer 可在增强预测更差时保持基础图。每个增强包发送前重新检查剩余时间，基础包即使迟到也不跳过。`--enhancement-cap` 仅截去尚未发送的连续增强尾部，不能冒充 TCP 层随机丢包。

解码/显示预留为 `max(--decode-reserve-ms, 1.25 × 已观测显示 ACK 耗时上界)`，每次基础 ACK 后更新，再选择增强预算。观测从请求发完至合法 ACK 到达，包含接收端解码、CPU 拷贝/可选 PNG、ACK 延迟；没有读取未来耗时。首次冷启动和突发抖动仍可能超时，报告同时保留真实显示时间与 ACK 时间，未将其当作硬实时保证。

带宽 profile 在当前帧才提供一个值，最后一个值保持；应用层按 record bytes 延迟交付，ACK 路径未限速。该实验是可靠 TCP 上的应用层整形与尾部舍弃，未模拟内核拥塞控制、重传或真实无线丢包。源视频时间轴不会随处理速度放慢，基础包阻塞产生的后续积压和迟到会如实记录。
