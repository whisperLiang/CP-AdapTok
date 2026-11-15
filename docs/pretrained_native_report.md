# 官方骨干与原生码流后续报告

官方权重、原生 CUDA 码流和真实 decoded-latent 蒸馏已跑通，完成100步 Stage1及实际1080p评估。**当前码本塌缩，压缩质量不合格，30/60 FPS目标仍未达成。** 本报告记录短程工程实验，不代表研究目标或全数据集训练已经完成。历史随机骨干结果保留在 [原报告](implementation_report.md)。

## 环境与 teacher 修复

独立 `.venv-native`：Python3.12.13、PyTorch2.9.1+cu130、nvcc13.0.88、CUTLASS4.4.1、H100 PCIe。环境版本、补丁及编译源 SHA256 见 `results/native_environment.json`，依赖见 `native_environment_freeze.txt`。原有 Python3.10/torch2.6 环境及 InfoTok 项目没有修改。

原先未执行过的 PyTorch prior 重算 teacher 在真实 native 核对中失败：最大像素误差0.05914，大于0.02门限。已删除这条近似路径，保留失败日志 `native_gate_red.txt`。现在独立 native encoder 生成 SPS/IP 码流，独立 native decoder 消费这些字节，构建副本上的 [最小补丁](../native_patches/decoded_latent.patch) 返回解码器实际使用的 `m_y_hat` 的 detached clone。没有改变原生熵编码、量化或参考算法；Git submodule 源码保持原样。I/P目标精确等于原生 latent；修改返回副本不会改变内部状态；重置后 I/P 字节完全一致。

## 训练与 checkpoint

- 官方 LD375个和 DMCI398个 state tensor 均与下载文件逐项完全一致；仅训练新增 queries、VQ、统一 decoder。
- N_max2048、dim192、6heads、2layers、VQ8192×16、crop256、每次4帧、bf16、TBPTT2、LPIPS和真实 native latent loss；100次更新共400个 teacher 帧。
- MCL-JCV01训练64帧、UVG Beauty验证16帧、BasketballDrive测试32帧。无 resize；全部112帧的 RGB 与原始 YUV 经显式 BT.709 limited-range 转换所得字节哈希一致。原始 YUV 无颜色元数据，该转换规则作为假设记录。
- 使用checkpointed generator采样连续帧窗口，100步访问47个起点；记录每帧源位置。全1080p验证已执行。
- 保存 `checkpoints/pretrained_s1/step100.pt`，SHA256 `dbf0832d3275c7ecde8a8efd117ac5a8152bd677f7c950b021c70d8341473f92`。`latest.pt` 已恢复到101步，100步文件不变。
- 两个独立进程从同一100步checkpoint恢复到101步后，采样/CPU/CUDA/Python RNG全部一致。CUDA bf16的模型和optimizer未达到逐位相同，模型最大绝对差 `9.39518213e-06`；CPU恢复行为测试通过。不能声称 GPU 训练逐位可复现。

## 实际质量：同一独立8帧，五档真实字节重建

| K | 初始 PSNR | 100步 PSNR | LPIPS | DISTS | 实际 bpp |
|---:|---:|---:|---:|---:|---:|
| 128 | 14.6020 | 14.5612 | 0.8721 | 0.7250 | 0.00137346 |
| 256 | 14.6021 | 14.5613 | 0.8722 | 0.7250 | 0.00231481 |
| 512 | 14.6021 | 14.5613 | 0.8724 | 0.7250 | 0.00419753 |
| 1024 | 14.6021 | 14.5613 | 0.8722 | 0.7250 | 0.00796296 |
| 2048 | 14.6022 | 14.5613 | 0.8722 | 0.7250 | 0.01549383 |

最大档初始 LPIPS 1.0641→0.8722，PSNR 14.6022→14.5613dB。预算增加而 MSE 变差的相邻对比例为43.75%；微小差异不能当作有效渐进质量改善。

第100步四帧的 VQ perplexity均为1、使用率1/8192。当前新增瓶颈已塌缩，所有前缀的重建近乎相同；延长训练前应优先解决该训练问题，并重新检查实际 RD。Scorer和熵模型尚未做这一预训练骨干版本的 Stage3校准。

官方原生 QP32同8帧实际 SPS/IP码流 PSNR **32.9784dB**、bpp **0.02081067**。这是单个不同码率工作点，不能计算 BD-rate或声称等率比较。CP列统计真实 length-framed records，native列统计 SPS/IP bytes，未混入假设的物理链路开销。

32帧检查的全部五档经过真实字节解码，缓存/完整重建最大像素差 **0**，独立基础参考状态检查通过。该检查证明时序与缓存正确，不证明画质。

## 性能与 TCP

CP基于同一100步checkpoint，1920×1080、8测量帧、2 warmup，无coverage插桩、串行测量：

| 路径 | 整体 FPS | 稳态P FPS | P95 ms |
|---|---:|---:|---:|
| CP基础+最大 | 15.23 | 17.77 | 105.22 |
| CP五档显示 | 10.15 | 12.67 | 181.05 |

CP计入完整序列编码、真实熵码流、发送端本地基础解码/提交和接收端所有显示点重建。没有达到1080p30/60FPS。

原生独立encoder/decoder控制，8测量帧、4 warmup，整体120.24FPS、稳态P 154.33FPS；计入RGB转换、SPS/IP封装解析和原生编解码，排除初始化、源I/O和指标。原生使用官方FP16 fused CUDA算法与一次重建，CP使用PyTorch和多显示点，二者工作量不同；原生速度不代表CP达标。

另有2帧五档PNG真实TCP，以及8帧 Rate-only、当前32/128/512/2048/128/512/2048/512kbps、1200ms源截止时间、增强最多512的真实TCP实验。后者两端各提交8个相同基础参考，省去117个增强包，7个基础ACK超时，真实显示迟到7次。准确记录冷启动和积压，不保证实时交付；应用层整形不代表物理丢包实验。

## 测试、审计与复现

- native环境完整 pytest：94 passed；旧环境结果见 `results/pretrained_native/pytest_legacy.txt`。
- 当前源码独立覆盖率数据库：88.82%；最大生产函数复杂度9；CRAP≥10条目0；AST重复≥0.8条目0。未合并改行前数据库。命名审计工具不可用，仍使用明确记录的替代工具。
- 隔离行为变异15/15 killed。真实CLI及性能命令、退出码分别见 `final_commands.json`、`measurements_commands.json`；覆盖率插桩结果不用于性能声明。

```bash
cd /home/whisperliang/CP-AdapTok
bash scripts/setup_native.sh
source .venv-native/bin/activate
python scripts/prepare_pretrained_data.py
python -m pytest cp_adaptok/tests -q
# 首次新训练；已完成的路径应使用下面的resume命令继续
python -m cp_adaptok.cli.train --config configs/pretrained_s1.yaml --max-steps 100
python -m cp_adaptok.cli.train --config configs/pretrained_s1.yaml --checkpoint checkpoints/pretrained_s1/step100.pt --resume --max-steps 101
python -m cp_adaptok.cli.eval --config configs/pretrained_s1.yaml --checkpoint checkpoints/pretrained_s1/step100.pt --max-frames 8 --output results/pretrained_s1_test
python -m cp_adaptok.cli.eval --config configs/pretrained_s1.yaml --baseline native --max-frames 8 --output results/pretrained_native_baseline
python -m cp_adaptok.cli.benchmark --config configs/pretrained_s1.yaml --checkpoint checkpoints/pretrained_s1/step100.pt --resolution 1920 1080 --measure actual-bitstream --frames 8 --warmup 2 --output results/pretrained_benchmark_two
python -m scripts.benchmark_native
python scripts/build_pretrained_report.py
```

首次训练会读取已有 `checkpoint` 路径作为warm start；复现实验时应保留并使用选定的100步文件，或另设空的checkpoint/输出路径。当前训练量不足，码本塌缩、后续阶段、完整RD与空间/AdapTok对照、Scorer泛化、CP实时优化仍待完成。详见 [ub-mode操作日志](work_log.md)。

![短程训练与码本塌缩](/home/whisperliang/CP-AdapTok/results/pretrained_native/figures/pretrained_quality.png)
