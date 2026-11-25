# HLT v1 应用码流

HLT 帧使用固定 92 字节小端包头，按共享扫描表承载零个或多个增强 Record 和至少一个基础 Record。合法显示前缀必须消费完整包头、全部基础 Record 和任意连续数目的完整增强 Record。包头和每个 Record 都有独立 CRC32。

| 包头字段 | 字节 | 含义 |
| --- | ---: | --- |
| Magic / version | 3 + 1 | ASCII `HLT` 和版本 1 |
| I/P type | 1 | 0 为 I，1 为 P |
| Profile | 2 | B1/B2/B3 标识，使用完整绑定区分精度、扫描等设置 |
| Frame / reference ID | 4 + 4 | P 引用紧前一帧，I 使用 `0xffffffff` |
| Height / width / tokens / components | 2 + 2 + 2 + 1 | 1080、1920、16320、32 |
| Base boundary | 2 | 静态基础 Record 数量 |
| Profile binding | 32 | SHA-256，涵盖几何、格点、扫描、模式、C0 权重及其他静态资产的身份 |
| Reference digest | 32 | 最近提交基础状态的 SHA-256，I 帧全零 |
| Header CRC32 | 4 | 覆盖前 88 字节 |

Record 固定 15 字节控制开销加实际 Payload。其中 index、scan offset、packed mode/type flags、component count、payload bytes 分别占 2、4、1、2、2 字节。末尾 4 字节 CRC 覆盖本 Record 的控制字段和 Payload。高位标记基础层，低位模式如下。

| Mode | Payload |
| --- | --- |
| RAW4 | 有符号二补数存低半字节在先。末尾未用半字节置零 |
| RAW8 | 有符号 int8 的连续字节 |
| ZERO | 空字节串，范围对应全零残差 |
| BITMAP4 | 最低位为先的 bitmap，随后是所有置位符号的 RAW4 串。未用位与半字节置零 |

发送按串行长度选择最短模式，同长保留 RAW4。所有 I/P 帧均计入同一单帧硬上限。预算不足以发送完整基础层时返回 `BudgetTooSmall`。Record 顺序和完整基础表示由固定 Profile 及当前格点和历史决定。批量模式计算仅为已选择前缀生成 Payload 和 CRC。

接收端核验控制字段、CRC、绑定、依据当前 Profile 确定的 Record 范围和 P 帧参考摘要。乱序 Record 可进入缓存，缺口以前的连续区间才可参与显示。基础缺失会进入等待 I 刷新的状态。整数基础 SHA-256 不携带增强信息。基础状态为不可变小端 int16 字节与支持 mask。显示使用独立分量数组和有效支持 mask。

`header_bytes + record_header_bytes + crc_bytes + base_payload_bytes + enhancement_payload_bytes` 等于实际应用流长度。bitmap 已包含在相应 Payload 内。补位在所属字节内，账本另列 `padding_bits`。当前版本没有运动 Record。网络层传输和实际包间延迟属于独立测量条件。
