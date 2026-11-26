# 官方 DCVC-UF 访问单元

Phase 0 保存上游 `write_sps` 和 `write_ip` 生成的完整序列码流。每个固定分辨率序列写入一次 SPS。I/P NAL 按官方格式写出原生 QP、rANS 并行段数、参考重置标志和可变长度原生字节。解码前使用官方解析器读取码流。 [官方 SPS/IP 格式](https://github.com/microsoft/DCVC/blob/cbdae87a5445114cdc7f48816da63ea80bdeac40/src/utils/stream_helper.py)

每条账本记录保存序列文件路径、字节偏移、payload 和原生头字节、源帧编号、有效和补齐帧数。一个单元的真实字节覆盖完整官方流，尾块填充计入总数。原生 y 与 z 由共享 rANS 流编码，独立字节数记为 `NOT_IDENTIFIABLE`。 [官方 HT-S rANS 调用](https://github.com/microsoft/DCVC/blob/cbdae87a5445114cdc7f48816da63ea80bdeac40/src/layers/extensions/inference/dmc_hts_proxy.cpp)

固定 30 FPS 的码率用总字节和实际有效帧数计算。源时长来自同一冻结清单的名义 FPS。严格逐块可行性用每个单元的有效帧数计算可发送字节，独立列出 I 单元超限。
