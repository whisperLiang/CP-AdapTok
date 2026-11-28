# 后端与码流契约

| 概念 | 实际类型 | 含义 |
|---|---|---|
| NativeUnit | `baselines.dcvc_uf.units.Unit` | 原生帧索引、有效及编码帧数、I 和重置标志 |
| NativeSymbols | `NativeSymbols` | 原生 Z 整数、Y 残差整数、打包 CDF 索引、skip 掩码 |
| SymbolDependency | 模式、原生掩码、`ChunkHeader` | 记录的先后关系和引用前一单元的标识 |
| ProgressiveUnit | `Chunk` | 一次编码得到的完整字节串及记录账本 |
| PrefixIndex | `Chunk.prefix_boundaries` | 从记录实长推导的只读截断位置 |
| BitstreamHeader | `ChunkHeader` | 版本、模型指纹、参考版本、几何、QP、层数模式 |
| DecoderState | `ProgressiveDecodeSession` | 未完成记录、已接收层、参考和显示进度 |
| ReferenceState | `ReferenceKey` 和 `Reference` | 参考身份、几何和完整原生张量 |
| DecodeResult | `DecodeResult` | RGB、可显示层数、参考提交状态和消费字节数 |

## Wire 格式

PHT1 与 PHT2 使用大端头和记录长度。36 字节头覆盖模型指纹、单元 ID、父 ID、参考版本、宽高、QP、有效帧数、模式和固定参考层，并带 CRC32。每个熵记录有 8 字节长度与 CRC 元数据。I 单元包含官方 SPS 与 I NAL。P 单元先提供一个超先验 Z 记录，再按模式携带四个或八个 Y 记录。

| Magic | P 模式 | I 模式 | Y 层数 |
|---|---|---|---:|
| PHT1 | HT-S MASK、PRECISION | INTRA | 4 |
| PHT2 | LD_MASK | LD_INTRA | 8 |
| PHT2 | HT-S MASK_FINE | INTRA，沿用 PHT1 | 8 |

PHT1 记录保留原 CRC 规则。PHT2 记录 CRC 的起始值绑定记录序号，可拒绝交换位置的原始记录。CRC 用于传输损坏检测。接收端顺序读取记录长度，由此推导各层位置。合法预算返回同一次完整串的字节切片。第一个可显示的 P 前缀包含 Z 和第一个完整 Y 记录。

## 收发规则

每个访问单元交给 `PHTTok.decode(payload)`。分段到达时，第一段传 `offset=0`，后续传 `new_unit=False, offset=原始偏移`。重复送达相同偏移和字节会复用已有结果。缺口、冲突重放、模型不符、错误后端、无所需参考以及记录损坏会失败。首次完整显示层到达前，结果的 `frames` 为 `None`。截断的尾记录等待余下字节。提前切换到新的访问单元会结束旧的接收会话。

参考提交发生在固定层完整收到时。正常 LD 和 HT-S 更新保留原生时域记忆与 P 特征。独立 I 或原生记忆刷新按代理的 I 特征状态传递。接收端在后续增强中复用已解出的整数，提交过的参考保持稳定。符号、索引和像素按冻结 FP16 路径逐值校验。

每个实例的编解码游标由同一调用方串行驱动。发送端和接收端各自拥有参考与解码会话。CUDA 后端是官方 FP16 推理路径，CPU 测试涵盖字节容器、rANS、索引、预算和媒体输入。 [配对回归](prefix_validation.json)
