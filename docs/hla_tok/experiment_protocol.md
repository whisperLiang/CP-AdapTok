# HLA-Tok 实验协议

## 样本与指标

正式 Validation 固定为 Joint V1 Seed 1 第 40,000 步 CLOSED 结果的 174 帧。按清单依次使用 blackswan 的 0–49 帧、car-roundabout 的 0–63 帧及 dog 的 0–59 帧。Joint V1 逐帧表中的 `input_rgb_sha256` 实际是源 JPEG 文件 SHA-256。仓库另保存本次 OpenCV BGR 转 RGB 后的原生 RGB24 SHA-256。

训练和 Validation 读取前核对冻结清单身份、来源和文件哈希。Test 清单仅参与元数据交集检查。原版 AdapTok 历史评估曾使用 cows、goat、parkour，本次封存规则涵盖这些视频。官方权重、Joint V1 检查点、感知指标权重和旧资产哈希另行校验。 [冻结队列](../../configs/hla_tok/validation_cohort.json) · [审计](../../results/hla_tok/audit/cohort_audit.json)

在原生 RGB [0,1] 上计算 PSNR、LPIPS-Alex、LPIPS-VGG、DISTS 和 MS-SSIM。全局 PSNR 从整队列像素均方误差推导，逐帧 PSNR 另算平均值。边界条带沿 128 像素网格线取左右或上下各 2 像素。RGB 感知网络权重按文件 SHA-256 固定。

## Phase 0

- 原版 AdapTok 分别运行 stride128 与 stride96 的 FP32 和 BF16。每个 16 帧窗口运行一次官方编码，各预算分别用实际 Fixed13 载荷接收、查表并解码。每 4 帧时间组的每 Tile 预算为 96、128、256、384、480、512。末尾填充窗口只计真实帧质量，完整索引载荷计入实际 bpp。
- 验证 Joint V1 的 Seed 1 第 40,000 步、CLOSED 状态、相同源帧、参考指纹和 rANS 含头部字节账本。历史测得的编码延迟与本次重放补测的 RGB 指标注明各自来源。
- 原版性能诊断比较 4、8、16 Tile batch，并检查 4 帧因果重算、KV 重用及 PyTorch 编译候选的等价性。候选需要证据后才可用于基线。
- T0 随机权重的 1080p Batch 1 性能使用 64 帧，每档两次。报告同步 Wall Clock 与 CUDA Events，分别记录纯神经、含 Fixed13 载荷、RGB 主机与设备传输。K8192 的 BF16 串行神经吞吐继续训练门槛为 15 FPS。

## Phase 1

Phase 0 完整通过后运行固定 K16320，先用 256×256 Crop 连续训练 500 步，再从兼容模型初始化标准 VQ。使用 Train 的连续表示确定性拟合码本，之后训练 256×256 Crop 1500 步、512×512 Crop 2000 步和原生 1080p 1000 步。每 500 步保存检查点和固定 Validation 子集结果，训练结束使用完整 174 帧。训练配方固定于 [Phase 1 配置](../../configs/hla_tok/phase1.yaml)。

各预算画质以实发索引字节除以真实源像素数调整。固定预算门槛使用 stride128 FP32 的实际 bpp 测试点对目标实际 bpp 作区间内线性插值。相近实际 bpp 下，LPIPS-Alex 和 DISTS 上限为原版的 1.10 倍，PSNR 下限为原版减 1.0 dB。码本利用率、整数解码、梯度及计算吞吐参与工程验收。阶段门槛以 [验收记录](../../results/hla_tok/acceptance.json) 为准。
