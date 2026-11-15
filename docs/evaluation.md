# 评估口径

`eval` 的每一档都走 `bytes → CRC/metadata → rANS decode → ID lookup → 同一个 Decoder → 官方 I/P 重建 → RGB 裁剪`。同一帧只 encode_full 一次，所有预算复用同一组前置 packet bytes 和同一个 frozen context。发送端本地基础解码也经过序列化 packet。

质量计算来自完整分辨率重建 RGB；PSNR、MS-SSIM、LPIPS 和 DISTS 分别记录。DISTS 默认安装包错误地在 sys.prefix 下查找 weights，本实现显式读取其分发的包内预训练 alpha/beta，不用随机权重代替。任何可选指标缺失都为 null 并保存原因，不能置零冒充结果。

码率使用实际序列化长度。FrameHeader、PacketHeader、CRC、元数据、rANS payload 和每 record 的 TCP 应用 framing 全部计入 bpp/kbps。原始 codec bytes 与 framing bytes 也单独记录。socket demo 另外统计显示 ACK/control 的实际应用字节。未抓取 TCP/IP/link 层包头，因此不声称已经测得物理线路总码率。

`benchmark --measure actual-bitstream` 包括最大序列编码、所有最大序列 packet 的 entropy encode、发送端基础 entropy decode/重建/提交、Scorer、接收端 entropy decode、冻结 context、基础及每个请求显示前缀的重建。模型初始化、指标和 PNG I/O 不进入这项 pipeline 时间；socket demo 的独立 FPS 则包括显示 PNG、socket 与 ACK 等待。明确区分两种口径。

所有显示点都重跑完整后端，不能把 cache Attention 的更新速度当作完整 decoder 的增量执行。报告实际 reconstruction 次数、attention 更新、后续 latent/LD reconstruction 时间、总 pipeline P50/P95、稳态 P 帧 FPS、峰值显存。重建次数也包括预算列表未显式包含 K_ref 时仍必须进行的基础状态重建。

默认测试 2 帧真实 1080p RGB，benchmark 可取同一原始 17 帧片段中的 8 帧。720p 需提供原生匹配视频；本次额外准备的 720p 是从真实 1080p 原图做中心裁剪，明确不称为独立原生 720p 数据集。可选 4K 因缺数据未测。保持分辨率检查，不能静默 resize 来满足 benchmark 参数。

Fixed-K、Rate-only、Causal Scorer 用当前预测和实际 bytes；Offline Oracle 使用全段真实失真及全局 byte budget，明确非因果。实际失真只用于输出评估标签和 oracle，不会泄漏给在线选择。

native LD、固定空间 control、learned query 必须各自训练并以实际 bpp 做 RD 匹配。`ablation_spatial.yaml` 是显式控制组；没有训练 control 时仅允许带 untrained 标记的工程检查，不能生成等率优势结论。AdapTok 原生的分辨率和时间块与本项目不同，其原始代码作为对照资料保留，未冒充同条件基线。

当前实验不能支持 30/60 FPS 或压缩质量达标结论；所有实际结果与缺失实验见 implementation_report。后续 compile/CUDA Graph/FP32 cache 等优化必须先按同一 checkpoint、同一 bytes 和相同显示点通过数值与质量核对，再比较延迟。
