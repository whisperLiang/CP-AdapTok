# AdapTok-HR 架构与协议

本实现从 experiment/original-adaptok-1080p 的 c935509 开始，独立训练，不加载旧 CP、DCVC 或官方 AdapTok 权重。已有官方分块对照使用 135/220 Tile；fixed2048 的短窗口改善未泛化到长闭环。因此本设计同时约束空间计算、离散传输和实际重建历史。

## 数据结构与术语

- Frame：RGB float32 [0,1]，B×3×H×W；Sequence：B×3×T×H×W。正式 H=1080，W=1920。底部 reflect pad 到 1088，宽度不变。工程小尺寸必须另行标注。
- Candidate：确定位置的 16D 向量。三层依次 Base=S128(9×15)、Medium=S64(17×30)、Fine=S32(34×60)，合计 2685。
- Budget：实际传输 ID 数 K∈{256,512,1024,2048}，每完整帧。所有预算共享一个 8192×16 码本。
- Prefix：同一编码上下文产生完整序列后截取前 K 个。各层采用归一化二维坐标最远点遍历，固定 tie-break；全序显式保存为 (level,y,x) 表。
- State：stream ID、上一帧序号与类型、GOP32、状态版本、上一帧真实重建 RGB、解码器 S32 特征、逻辑码流链身份。独立发送/接收状态；无原始历史。

## 编解码网络

Encoder 七级 stride2 depthwise separable conv，通道依次 48/64/96/128/192/192/192；低分辨率 residual blocks。S32、S64 各一个 6-head、4×4 window attention，实际切分窗口再调用 SDPA；padding keys 屏蔽，padding queries 裁去。Depthwise 3×3 混合邻窗，无全帧注意力矩阵。通道和 block 数可配置。

Encoder 输入当前 RGB 及接收端可复现的前帧重建，低分辨率 temporal fusion 也仅使用前一解码特征。I 帧清空全部参考。各层 1×1 投影为 16D、归一化，再按位置表 gather。VQ FP32 分块最近邻，训练从第一步就经过实际码本；STE 仅传梯度，前向值来自查表。码本有 embedding loss、commitment loss，无长期连续预训练路径。

Decoder 输入仅为传输的前 K 个 embedding、位置表、已有接收状态。每层 scatter 得到 embedding+显式 validity mask；未发送值完全不存在。低分辨率局部卷积及窗口注意力融合，逐层 bilinear upsample+轻卷积；高分辨率通道收窄，最后 pixel shuffle 输出 RGB 并严格裁至 1080×1920。P 帧可融合历史低分辨率特征及有界 RGB blend，参考仅来自实际解码输出。

首帧及 index%32=0 为 I 帧；第 33 帧（index32）重置。训练主分支按真实采样 K 前向及推进历史。辅助预算共用同一编码上下文，只计算损失，不提交状态。每 TBPTT 段反向后 detach；T=4/16/32 表示逐帧展开总长，不要求把整个图留存，也不表示推理等待整段。

## 真实二进制协议

网络字节序帧头：magic/version/I-P flag/index/GOP/height/width/K/payload length/16-byte stream ID/32-byte codebook SHA256/32-byte model SHA256/32-byte preceding packet-chain digest/header CRC32/payload CRC32。13-bit big-endian code packing，末字节未使用位必须为零。准确尺寸以 `bitstream.HEADER.size` 与实际 bytes 为准。

代码身份基于所有解码权重与模型配置，码本身份单独保存。CRC 覆盖元数据及 payload，拒绝未知版本、非法数量、尺寸、身份、padding bits、缺失/重复/乱序 P 帧及错误流状态。空接收端只能从合法 I 帧重新同步。链指纹证明逻辑码流顺序；不宣称跨不同硬件浮点实现逐位一致。相同模型/数值模式的两端独立实际解码需测试一致。

截断 API 解析完整帧并按新 K 重新打包、更新长度与 CRC。同一先前状态可解码该前缀；截断改变参考，后续 P 帧必须基于新参考重新编码，不能直接拼接原高预算流。理论 bpp=13K/(WH)，实际 bpp=8×len(packet)/(WH)；本版固定长度，无熵压缩。

## 数据、训练与复现约束

只复用既有 Manifest 与 OpenCV BGR→RGB /255 读取。独立索引排除小于 1920×1080 的源；更高分辨率只裁剪不缩放。按原视频源、路径、内容 ID 与源哈希检查 split。固定 Validation；从旧 Test 中未被官方/CP 1080p 对照使用的高分辨率源单独建立 HR 最终 Test 并封存。旧 cows/goat/parkour 仅作已知历史绝对质量对照，不能称新盲测。外部数据通过配置导入且必须给出许可/使用条件，不自动下载。

联合训练 AdamW lr1e-4、wd.01、BF16、batch1、clip1、GOP32、TBPTT2（可4）。损失 RGB L1+MSE、LPIPS-Alex、VQ、可选重建时间差损失、预算单调正则。随机主预算；按配置周期增加同状态额外预算。训练计划记录实际 completed optimizer steps 与 frames，不以计划代替结果。

Checkpoint 在完成 optimizer step 后原子保存模型、optimizer、scheduler、Python/NumPy/Torch/CUDA RNG、独立采样 RNG、配置、manifest/source identity、完成步数和累计帧；恢复以 checkpoint 为唯一提交点，回滚未提交日志，拒绝配置或数据身份改变。

先实际 1080p 前后向/多帧/码流/显存探测，再 Train 单段拟合，再正式 300 步。只有 Validation 质量与数值/显存稳定才继续 1000 步；5000 由真实结果决定。Scorer 在 tokenizer 稳定后冻结全部参数，用同状态各 K 的真实 ID 重建 MSE 标签；在线仅看当前帧编码特征和过去状态。未达门槛时保留实现及未运行标记，不提前声称已训练。

## 评估与验收

独立各预算 64+帧闭环，完整 RGB 像素计算 global-MSE PSNR、逐帧 PSNR mean、LPIPS-Alex、DISTS；记录实际 bytes/bpp、码本 histogram/perplexity，按 I/P、GOP早晚、源运动量分组。运动量仅为事后评估标签，不输入在线决策。额外同状态预算检查用于分析 Nested，不能混用独立历史的差值作为纯前缀增益。

独立 profiler 进程、固定预热和重复次数、batch1、CUDA synchronize：conv encoder、window attention、token/VQ、scatter、decoder fusion/upscale、state、pack/unpack；编码器与解码器吞吐分开，联合吞吐包括发送端本地参考解码、独立接收端、码流处理，模型加载/读盘/质量指标排除并声明。FLOPs 用 PyTorch FlopCounterMode，乘加=2；报告不支持算子范围，不当作硬件指令数。

自动行为测试覆盖位置、边界 padding、局部注意力操作形状、因果性、Nested、缺失 token 隔离、状态/GOP、13-bit 往返与损坏处理、真实参数更新、冻结 scorer、精确恢复；GPU 测试必须实际执行完整 1080p。消融开关支持 conv-only/local/local+temporal、fixed/nested，匹配训练预算；未运行结果保持缺失。
