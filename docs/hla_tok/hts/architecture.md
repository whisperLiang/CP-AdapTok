# HT-S 原生骨干审计

首帧使用官方 DMCI。P 单元使用官方 HT-S 的八帧时域宽度。各帧 RGB 按 ITU-R BT.709 转 YCbCr，再转为 FP16 并减 0.5。P 输入沿通道拼接，CUDA 内核做 16 的倍数空间 padding。尾块复用最后一帧，率失真分母取真实帧数。 [官方测试入口](https://github.com/microsoft/DCVC/blob/cbdae87a5445114cdc7f48816da63ea80bdeac40/test_video.py)

原生长序列配置使用单 I 帧、128 帧参考特征重置间隔与 0.15 的 entropy skip threshold。历史对齐配置将 LD 的 GOP 设为 32 帧。HT-S 与 HT-L 使用 33 帧的独立 GOP，即 1 个 I 帧接 4 个八帧 P 单元。重置判定从各 I 帧起算，skip threshold 取历史 V3 的 0。32 帧 Calibration 裁段内，两条高吞吐骨干均只有一个 I 帧。

HT-S 官方训练定义的主 latent 为 256 通道、16 倍下采样，hyperprior 为 128 通道、64 倍下采样，解码器特征、上下文和预测记忆均为 512 通道。CUDA 编码器在一个 P 单元后准备下一个参考上下文。解码器在下一个 P 单元开始时由已解码参考计算对应上下文。审计核对每条实际依赖上的状态哈希。 [官方 HT 模型](https://github.com/microsoft/DCVC/blob/cbdae87a5445114cdc7f48816da63ea80bdeac40/src/models/video_model_ht.py)

用于核对的 CUDA 构建增添了 clone-only 观察入口。官方计算、rANS 调用和参考写入路径沿用上游代码。连续表示实验将当前单元未量化的 y 送入官方 PyTorch 重建头，并锁定原生 Base 历史。连续表示表独立保留，无连续码率。

旧工作区的 V2 Seed 2、V3 A4、Original AdapTok 权重和引用结果使用 SHA256 清单。数据源按视频隔离 Train 与 Calibration。Validation 和 Test 的像素权限由各阶段准入条件控制。
