# AIP-VTok Joint V1 架构契约

基线为 experiment/aip-vtok-sparse-decoder 的 c5cd06dfc8cb6afd19c6da668d7664eb3f32c2ea。分支 experiment/aip-vtok-joint-v1 持有新参数、码流和实验产物。

| 模块 | 输入与输出 | 参数所有权与梯度 | 信息边界 |
| --- | --- | --- | --- |
| AlignedDCVC 编码与 ChannelAlignment | 原生 RGB [1,3,1080,1920] 与已提交历史 → aligned [1,128,68,120] | 旧资产冻结，原始图像编码与目标 latent 在无梯度区计算 | 发送端可见 |
| I/P 正交变换 | 独立 Cayley 参数与 Train 排列，aligned 128D → transformed 128D | A 冻结，B/C 联合更新。接收端转置变换参与 RGB 梯度 | 两端共享参数，发送端截断至前 48D |
| A2 FSQ | [B,8,68,120,6] → 65,280 个 uint16 可表示 ID | 正缩放可训练，前向用整数重建值，反向用 FP32 STE | 接收端仅持有已通过校验的整数与所属组 |
| 共享扫描 | 模型 score 与公共 slot ID → 一个全帧排列 | A 使用 SP-1 初始化，B 使用硬掩码和软梯度，从 24001 起冻结 | 两端共享同一张扫描表，无帧级排序元数据 |
| 熵模型 | 帧型、组、维 → 共享 digit 概率与整数 CDF | B/C 联合更新。符号码率前向用真实硬 digit，反向用相邻 digit 线性插值 | 两端共享模型，单包独立 rANS |
| 稀疏画布与 SLD | 48D 系数、8 组掩码、Gaussian fill、预算嵌入及 P 因果特征 → 128D transformed | SLD 主体沿用 64D 两尺度结构。系数级硬掩码保持已接收值 | 缺失系数由 SLD 补全 |
| 冻结 DCVC 解码 | 转置变换恢复 aligned 128D → I/P 对应原生 latent → 原生 RGB | 参数冻结，RGB 损失经过解码器回传到当前帧的 SLD、FSQ、变换和 scan 代理 | 接收端由整数与历史计算 |
| LatentToRef | 固定参考前缀恢复的 aligned latent → I 192/P 256 通道参考特征 | 投影器冻结。Reference loss 通过投影器回传。跨帧历史以独立副本提交并截断梯度 | 发送端和接收端分别恢复、校验和提交一次 |

完整 128D 变换可用转置求逆。48D 截断留下 80D 给 SLD 补全。DCVC 的 I 帧先经过有损 256→128 投影。I 帧质量上界和 reference 目标均使用冻结 aligned 路径。P 帧保留独立的归一化和原生解码路径。

## 旧实现边界

- 复用 AlignedDCVC、FrameContext、ReferenceState、LatentToRef、BT.709 损失和原生感知输入视图。Decode 的 checkpoint 路径保持当前帧潜在输入可微。
- 复用 SP-1 的空间序列与组次序、A2 的 digit 布局和混合进制规则。旧 Quantizer.encode 的 no_grad 面向测量。Joint 路径增加显式整数硬前向与可微代理。硬 argsort 仅决定前向，sigmoid 阈值路径提供梯度。
- 复用 SLD 的 MultiscaleBody 和 frame_input。SLD-500 仅作初始化探针和冻结对照。Conv 输入按原有效列迁移，输出映射和共享置信度按 I/P 初始排列映射。
- 旧 codec 会话、C1 paired/prefix helper 和 Sparse-RFSQ rollout 各自绑定了旧码流或冻结恢复机制。联合训练直接使用公共状态类型，独立入口消费 Joint 位流和状态。B2、W1、SLD-500 使用原冻结协议做对照。

## 扫描、量化和因果时序

基础四组继承各帧型已经封存的 A2 各组离散级数与尺度。增强四组使用 8/8/8/5/5/5 和封存 Train 排名的对应尺度。全局 slot ID 为 group × 8160 + y × 120 + x。K510 至少包含基础四组每组的 8×8 空间网格各一项。其余席位按稳定优先级填入。相同 score 按 slot ID。训练保留相同硬前向，代理仅用于反向。

GOP 相位来自原视频帧号。I2048/P510 是参考预算。发送端从自己的字节流解码参考，接收端从已到达的完整包解码参考。编码时冻结当前帧上下文，提交后可继续展示当前帧的更长前缀。较低 I 帧展示预算仍支付 I2048 字节成本。训练窗口使用同一整数编码规则，burn-in 从最近的 I 帧开始，参考在帧间 detach。

FIXED 使用既有 C1 A1/B2 参考。CLOSED 独立逐帧提交各模型的参考。联合编码的参数指纹和扫描指纹与帧头协议绑定。正式测量保存逐帧字节和延迟，并记录已使用的冻结权重、指标权重、配置和数据清单 SHA256。
