# HUF1 码流与 UF-V2 模型身份

历史 HUF1 Base 头 26 B，后接 4 B CRC32 与官方 Base payload。头含 magic、8 B 模型及 prefix-family tag、帧序号、原生宽高、QP、官方熵并行与真正 I/特征刷新标记、原生载荷长度。Base 码流携带联合原生 y/z rANS；不存在可按字节独立拆出的 Hyperprior 字段。

增强头为 8 B，另加 4 B CRC32。字段依次为 uint32 空间/通道组 ID、uint8 精化层、uint8 CDF 和模式、uint16 payload 长度，均为小端。E0 维持旧 rANS 顺序及符号逃逸。E1 从部署资产加载 I/P、精化层、通道组确定的静态整数表。带 Raw fallback 的 E2 使用模式字节高位，低 7 位仍验证表 ID。合法 Raw4 采用最低位 nibble 优先的偏移 8 编码，非 INT4 符号可使用有符号 Raw8。编码端比较完整段字节，等长选 rANS。尾部 nibble 零填充属于比特对齐。

一次编码产生完整最大的合法访问单元；按最终预算选择现有边界直接切片。记录 CRC、长度、组范围、层依赖和 CDF 索引均核验。Base 只有完整解析后才提交。缺少或损坏的增强可结束显示更新，已提交 Base 可供下一帧。缺失 Base 会令后续 P 进入 DESYNC，直到独立 I 恢复。访问单元由应用传输边界结束，半段保留待追加，`finish()` 验证当前边界，应用层用 `close()` 显式结束访问单元，含半段时拒绝关闭。头不携带最终记录数。

字节接收使用 `feed()` 与 `decode_available()`。已校验记录也可按边界交给 `commit_base()` 与 `apply_enhancement()`，总长度、组内层次和重复记录遵循同一规则。`render()` 读取当时的累计整数画布，执行共享精化读出，并让 Base 参考保持固定。

独立 Base-Standard / Base-Lite 配置使用独立 tag。更换 checkpoint、CDF、排序权重或编码模式也变更 tag。运行时硬上限为 Standard 9,212 B、探索 Lite 4,096 B。统计分别列原生 Base payload、应用头/CRC、增强 payload 及记录头/CRC。未测传输层 UDP/RTP 等字节。
