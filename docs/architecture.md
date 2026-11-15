# 架构、形状与因果边界

## 上游固定接口

| 组件 | 本实现复用的真实接口 | 1080p 对应形状 |
|---|---|---|
| 输入 | RGB float [0,1] → 上游 BT.709 YCbCr − 0.5 → replicate pad 到 64 倍数 | `[1,3,1088,1920]` |
| LD Encoder | `DMC.encoder(x,ctx,q_encoder[qp])` | y `[1,128,68,120]` |
| I Encoder | `DMCI.enc(x,q_scale_enc[qp])` | y `[1,256,68,120]` |
| I 参考 | DMCI 解码的、裁剪前 centered YUV → pixel_unshuffle(8) | `[1,192,136,240]` |
| P 参考 | `DMC.decoder(y_hat,ctx,q_decoder[qp])` | `[1,256,136,240]` |
| context | `feature_adaptor_i` 或 `feature_adaptor_m`，随后 `feature_extractor` | memory/ctx `[1,256,136,240]` |
| LD 重建 | `decoder` → `recon_head` → YCbCr→RGB → 原始尺寸裁剪 | `[1,3,1080,1920]` |

`forward_one_frame` 产生训练用 bit estimates，并修改上游 memory/ref_feature/ctx；它不是实际码流 API。本实现从不以其输出当作真实压缩。原生 `compress/decompress` 使用 `DMCLDProxy/DMCIProxy`，包括自己的 prior、entropy coder 与状态管理，不能插入新的 token bottleneck。新主路径调用其 PyTorch 子模块；Proxy 仅留在独立 native baseline / teacher 路径。

## EncodeOnce 与共享 Decoder

每帧只用 `encode_full(current_rgb, decoded_state)` 计算一次 N_max 个固定顺序 learned queries。Query 没有绑定到二维位置，也没有预算参数。低分辨率 latent 经过轻量投影与约 17×30 池化；局部高频支路、历史 ctx 和空间坐标只用于 K/V。两个 cross-attention 层聚合出统一 1D 特征。

VQ 使用 normalized nearest-neighbor、确定性 argmin、16 维/8192 codebook 默认值、ST 梯度、commitment/codebook loss、usage/perplexity。4096 codebook、32 维、N_max 4096/8192 可配置。发送方返回的连续 embedding/latent 从不写入码流。

接收方只有 ID：查共享码本 → token position embedding → fixed grid queries 的 cross-attention → 轻量卷积与 latent 尺寸恢复 → 对应 I/P 官方解码骨干。所有 K 共享同一套参数；I/P 输出 channel adapter 可不同。没有单独 Base/Enhancement Decoder，也没有像素残差压缩器。

完整 Attention 支持严格 softmax mask。缓存只保存 token-to-grid 的 query、running max、exp sum 与 weighted value；追加 K/V 后重新归一化。默认 float64 累加以控制深层随机骨干对浮点误差的放大，输出回到网络参数精度。float32 是明确的精度消融，需重新验证重建误差。每个显示点仍重新运行卷积和 LD/DMCI 解码器，计数单独报告。

## 单一参考所有权

`CausalState` 保存模型 ID、上一帧编号、基础前缀解码 feature、memory、基础 IDs 摘要。`FrozenFrameContext` 由 `begin_frame` 创建；feature adaptor/extractor 每帧各调用一次。所有增强显示复用该 context，不能调用上游有副作用的 `apply_feature_adaptor`。

编码方先把整帧 IDs 分包为 rANS bytes，再实际解包基础前缀并本地重建。双方只提交 `ids[:K_ref]` 对应 feature，`commit` 对同一 context 只能执行一次。显示 `decode_base` 是幂等接口。P 帧 memory 来自上一基础状态；I 帧清空 memory 并从独立前缀生成参考。

Header 检查 frame_id、reference_id、model/codebook/CDF ID、参考 feature/memory 的 SHA 指纹。增强包丢失后可正常推进；基础包缺失必须可靠恢复或显式 reset + I 帧。重复、乱序、损坏、跨帧包均显式拒绝。当前接收器只保留当前显示帧的 packet/cache，会拒绝已经切到下一帧后才抵达的旧增强包。

`CodecEncoder.select_prefix` 只根据当前 Scorer 预测、实际前缀字节、带宽、排队字节与截止时间选择现有 packet slice。不能修改 IDs 或重排。Offline Oracle 采用整段实际失真的 byte-constrained MILP，明确是非因果对照，不进入主发送器。

Glossary：reference prefix = 唯一影响未来状态的前 K_ref 个 ID；display prefix = 当前已收到并显示的前 K 个 ID；packet = 独立终止的 entropy block；frozen context = 一帧内不可由增强重建更新的时序条件；model ID = 参数与结构配置摘要；entropy ID = 整数 CDF 表摘要。
