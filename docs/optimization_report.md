# CP-AdapTok 诊断、训练与优化验收报告

本轮已经实际修改代码、运行多组训练和真实码流实验，**整体研究验收未通过**。码本长期 Perplexity=1 的训练阻塞已得到可复现实验和实现修复；异历史 P-frame teacher 默认已禁用；但有效的多预算压缩表示尚未在独立 1080p 视频上建立。不能将本轮 checkpoint 推荐为已收敛压缩模型。

在同一新模型、同一32帧1920×1080输入、相同基础/最终两次显示下，整数指纹融合使完整实际码流流水线从 **16.66 FPS 提升至 18.38 FPS**，32帧 Packet/Header 字节完全一致，严格状态验证保留。质量未达标，30/60 FPS 也均未达到。

## 版本、环境与保留资产

审查起点是 `911c9cc5a206f90b529ff65299350248a04b4b12`，开始时工作区干净。使用已有 `.venv-native`，Python3.12 / PyTorch2.9.1+cu130、H100 PCIe 80GB、官方 LD/DMCI 权重与现有 native decoded-latent 扩展。没有修改上游 CUDA 核心算法或系统 CUDA。环境及原配置、checkpoint SHA256 见 `results/optimization_20261115/baseline_manifest.json`，四个原始权重文件哈希复核见 `original_assets_unchanged.json`。

工作从2026-11-15持续到11-16（Asia/Shanghai）。设备上还有其他用户进程，未独占GPU；性能对照采用本任务串行运行，报告单次实测及其P50/P95，不宣称跨负载稳定的吞吐保证。

旧 `step100.pt` 的 SHA256 仍为 `dbf0832d3275c7ecde8a8efd117ac5a8152bd677f7c950b021c70d8341473f92`；旧 `latest.pt` 仍保留101步版本。所有新增实验写入 `results/optimization_20261115` 或 `checkpoints/optimization_20261115`。失败、被中断的实验和旧报告未覆盖。

代码按小提交保存：`a1d8473`（VQ/teacher）、`f816959`（Query/数据/curriculum）、`977584f`（padding/stream边界）、`1bd0b17`（逐位一致指纹融合及验收工具）。各实验还记录 config、源码 hash、Git 状态及训练 checkpoint 元数据；不能把这些短程结果标为正式收敛。

`d5c34b5`仅简化实验工具，17组配置、全部统计表、真实两帧双精度指标与状态字段在重构前后完全一致。最终报告及原始JSON/CSV均保留失败结论。

## 根因与直接证据

1. 原实现仅有硬最近邻。旧checkpoint的真实256 crop诊断中，I帧使用2个code，P帧使用1个；P-frame pre-VQ token cosine约0.999，token variance约5.6e-5。LayerNorm后的全局 std正常并不表示 Token 间具有内容差异。见 `baseline_diagnostics.json`。
2. 改随机采样可以提高训练分布的使用率，却不能保证推理使用率。固定温度1的训练 Perplexity约1700，确定性P帧仍为1。已保存每个code的频次，分别记录实际训练分布、同特征上的最近邻分布和确定性闭环推理分布。
3. 小幅初始化的 learned-query / decoder-grid-query 与公共注意力输出存在尺度失衡。保持相同结构，连续4帧实验的 encoder-query初始化尺度修正使最大档均值22.23→25.01 dB；在相同VQ方案中，decoder-grid-query尺度修正使22.20→28.01 dB。**后者前缀收益仍失败**，说明码本多样性和单片过拟合质量不能替代内容压缩能力。额外的content-only readout也未建立渐进收益，默认关闭。
4. Native decoded latent 提取本身正确。原生I目标由CP骨干解码，与native图像MSE为1.69e-7；把native P目标放入CP Base-only历史下，图像MSE约0.10。原始encoder与decoded latent还包含学习量化尺度差异，不能只凭原始latent MSE归因。见 `teacher_conditions.json` 和 `intra_teacher_decode_check.json`。
5. 540p暴露额外padding域错误：native latent高度34、CP为36。教师现在在与CP一致的64倍复制填充域内产生真实decoded latent，保留原始尺寸，不做resize。
6. Native构造函数原先修改调用方CUDA stream。繁忙GPU和跨stream输入测试复现了异步边界问题。现在保留调用方stream、显式等待输入、记录借用tensor生命周期，并在导出前完成owned-stream工作。没有放宽任何fingerprint校验。早期失败和最终成功日志均保留。

## 连续表示与 VQ 消融

A–E统一100次更新、同一固定真实MCL-JCV01四帧、256原像素crop、相同五档预算、相同LR。连续单帧、I/P双帧及4帧均实际训练；损失下降证明 Query→投影→统一Decoder梯度路径可学习，但前缀收益不充分。单帧/双帧与4帧的计算量不同，不能混作严格同预算消融。

| group | max_psnr | delta_psnr | deterministic_perplexity | deterministic_distinct_codes | quality_drop_ratio |
| --- | --- | --- | --- | --- | --- |
| A continuous | 22.23139 | 0.09722 | 1.34396 | 8 | 0.62500 |
| B hard | 22.14516 | 0.00290 | 1.00000 | 2 | 0.68750 |
| C stochastic T=1 | 8.37185 | 1.43144 | 1.88240 | 6 | 0.50000 |
| C annealed | 21.61648 | 0.06413 | 1.27072 | 5 | 0.31250 |
| D warmup | 17.71400 | -0.00090 | 1.40032 | 4 | 0.56250 |
| E initialized | 18.68580 | 0.04244 | 2.01477 | 7 | 0.50000 |
| query-scale + annealed VQ | 22.20512 | 0.01859 | 43.70420 | 397 | 0.75000 |
| decoder-query-scale | 28.00595 | -0.03636 | 35.48653 | 456 | 0.75000 |
| content-only readout | 27.97094 | -0.00519 | 61.59706 | 704 | 0.37500 |

PSNR、LPIPS和DISTS完整表见 `vq_ablations.csv`。`delta_psnr` 是2048相对128 Token的逐帧PSNR差均值；quality_drop_ratio统计所有相邻前缀MSE变差比例，微小退化也计入。连续模式不产生ID、不允许实际码流评估；其nearest统计仅为反事实诊断，不能当作量化有效性。

随机策略使用同一codebook与project-in/out，训练按归一化cosine/温度全码本采样，推理始终确定性。全码本分布用行为测试直接验证；本轮不使用top-k近似。温度采用绝对step调度，可恢复。数据驱动初始化使用真实连续特征的可复现球面K-means，不足独立样本会报错。

Dead-code恢复默认关闭。窗口统计及优化器moment行重置已通过可复现测试；本轮全码本随机训练的高采样覆盖与低确定性使用率表明主要矛盾是训练/推理分布差距，未据此大规模重置码本，F组未作正式效果结论。没有添加强均匀熵正则。

训练日志保存step0/10/50/100及间隔的Query层分布、pre-VQ方差/有效秩、top1/2差距、最近距离、完整code直方图、loss、梯度范数与prefix latent差异。`train.jsonl` 的step表示刚完成的更新，forward分布是在该次更新前采集；额外的 `post_update_quantizer.json` 明确记录checkpoint完成更新后的训练采样/最近邻探针，不执行额外更新。恢复保存optimizer、VQ窗口、sampling/CPU/CUDA/Python/NumPy RNG；GPU训练仍不宣称逐位复现。

## Teacher 策略消融

三组均100更新、相同四帧数据和预算，使用相同query尺度、预热、初始化及温度策略。

| experiment | max_psnr | delta_psnr | deterministic_perplexity | quality_drop_ratio |
| --- | --- | --- | --- | --- |
| teacher_T1 | 25.02708 | -0.02247 | 16.41412 | 0.37500 |
| teacher_T2 | 22.18992 | -0.02063 | 46.33339 | 0.50000 |
| teacher_T3 | 21.16817 | -0.00084 | 110.06487 | 0.50000 |

| experiment | frame_type | frames | psnr | lpips | dists |
| --- | --- | --- | --- | --- | --- |
| teacher_T1 | I | 1 | 24.74539 | 0.58765 | 0.27924 |
| teacher_T1 | P | 3 | 25.12098 | 0.59522 | 0.28369 |
| teacher_T2 | I | 1 | 21.69469 | 0.62665 | 0.34704 |
| teacher_T2 | P | 3 | 22.35500 | 0.58563 | 0.35769 |
| teacher_T3 | I | 1 | 19.63055 | 0.69401 | 0.43041 |
| teacher_T3 | P | 3 | 21.68071 | 0.63140 | 0.37703 |

T1的单片过拟合PSNR更高，但使用异历史P逐元素latent目标，不能据此证明监督条件正确。T2为I-only native latent + P实际重建；T3无native latent。默认采用T2，并支持后期教师权重衰减。T1只可通过明确的diagnostic标志启用。T4同CP上下文量化teacher未获数值验证，因而禁用；没有替换为近似重算目标。

`teacher_conditions.json`还记录相同特征下0/0.1/1/10教师权重对各模块梯度的实际影响。这是梯度敏感性探针，不是四组正式训练；T1–T3长期教师策略对比仍未完成。

## 数据、分辨率与容量

从[官方DAVIS下载页](https://davischallenge.org/davis2017/code.html)按官方2017 train清单提取8条1080p/4K原始序列、每条16帧，共128帧。只下载ZIP选定条目，验证CRC并记录内容SHA256；没有把DAVIS val/test混入train。来源及尺寸见 `data/davis2017/manifest.json`。MCL-JCV01、UVG Beauty、BasketballDrive继续作为原有工程过拟合/验证/测试来源。

该数据是pilot子集：没有建立正式全场景覆盖，特别是镜头切换覆盖仍不足。`dataset_audit.json` 的thumbnail仅用于内容统计，训练与评估输入不缩放；原始图像序列无时间戳，30fps是明确标注的名义值。

实际运行D0 256单帧/短序列、D1 512 crop 300更新、D2 960×540→1280×720原像素crop 100更新、D3 1920×1080 crop 100更新、D4 1080p 4→8帧100更新，另做1000更新1080p多场景延长训练。D4配置支持16帧继续训练，但本轮100步没有执行到16帧调度点。均保留原始尺寸、64倍padding/crop、TBPTT与checkpoint。

| experiment | updates | crop | n_max | codebook_size | grid | max_psnr | deterministic_perplexity |
| --- | --- | --- | --- | --- | --- | --- | --- |
| c1024_256 | 50 | 256 | 2048 | 1024 | [17, 30] | 21.25732 | 66.97671 |
| c2048_256 | 50 | 256 | 2048 | 2048 | [17, 30] | 20.75542 | 485.51097 |
| c4096_256 | 50 | 256 | 2048 | 4096 | [17, 30] | 21.36323 | 71.41401 |
| grid34_512 | 100 | 512 | 2048 | 8192 | [34, 60] | 26.16471 | 805.76570 |
| learned_512 | 100 | 512 | 2048 | 8192 | [17, 30] | 25.89684 | 391.85880 |
| n4096_256 | 50 | 256 | 4096 | 8192 | [17, 30] | 20.85884 | 1337.74424 |
| n8192_256 | 50 | 256 | 8192 | 8192 | [17, 30] | 20.86348 | 280.21179 |
| spatial_512 | 100 | 512 | 2048 | 8192 | [17, 30] | 26.15419 | 358.02364 |

512 learned / fixed-spatial / grid34对照为同100更新；其余容量试验为同50更新256 crop，不能跨组直接归因。34×60 grid对训练crop有约0.27 dB收益，但没有解决有效前缀问题，也没有被标为1080p收敛结果。4096/8192查询及1024/2048/4096/8192码本均复用通用实现。固定空间Tokenizer仅用于显式对照，主模型仍为Learned Queries与同一个Decoder。

以下是这三个同条件100步模型在同8帧真实1080p测试上的结果，均经过实际rANS码流。更大grid的crop过拟合收益未迁移到1080p，不能把空间grid作为全部问题的解释：

| model | tokens | psnr | lpips | dists | bpp |
| --- | --- | --- | --- | --- | --- |
| learned_512 | 128 | 11.39685 | 0.98826 | 0.59338 | 0.00137 |
| learned_512 | 256 | 11.40569 | 0.98744 | 0.59328 | 0.00231 |
| learned_512 | 512 | 11.40327 | 0.98707 | 0.59340 | 0.00420 |
| learned_512 | 1024 | 11.40454 | 0.98697 | 0.59347 | 0.00796 |
| learned_512 | 2048 | 11.40835 | 0.98718 | 0.59344 | 0.01549 |
| spatial_512 | 128 | 13.65185 | 1.05343 | 0.66794 | 0.00137 |
| spatial_512 | 256 | 13.65575 | 1.05090 | 0.66373 | 0.00231 |
| spatial_512 | 512 | 13.78132 | 1.02345 | 0.64819 | 0.00420 |
| spatial_512 | 1024 | 13.79951 | 1.02448 | 0.64875 | 0.00796 |
| spatial_512 | 2048 | 13.79433 | 0.97438 | 0.63275 | 0.01549 |
| grid34_512 | 128 | 10.03217 | 0.89736 | 0.62672 | 0.00137 |
| grid34_512 | 256 | 10.07926 | 0.89903 | 0.62831 | 0.00231 |
| grid34_512 | 512 | 10.10634 | 0.90015 | 0.62934 | 0.00420 |
| grid34_512 | 1024 | 10.11868 | 0.90071 | 0.62988 | 0.00796 |
| grid34_512 | 2048 | 10.12153 | 0.90112 | 0.62950 | 0.01549 |

## 真实 1080p 码率—失真

以下为重训entropy/scorer后，独立BasketballDrive同8帧的实际rANS字节重建。包括CP Packet及应用length framing；源RGB1920×1080，无隐式resize。

| tokens | psnr | lpips | dists | ms-ssim | bpp | kbps |
| --- | --- | --- | --- | --- | --- | --- |
| 128 | 13.03577 | 0.83053 | 0.69307 | 0.48846 | 0.00120 | 124.60000 |
| 256 | 13.00912 | 0.83002 | 0.69258 | 0.48881 | 0.00198 | 205.45000 |
| 512 | 13.00212 | 0.82968 | 0.69276 | 0.48898 | 0.00355 | 367.80000 |
| 1024 | 13.00600 | 0.82979 | 0.69310 | 0.48893 | 0.00667 | 691.50000 |
| 2048 | 13.00669 | 0.82945 | 0.69309 | 0.48911 | 0.01292 | 1339.40000 |

| frame_type | frames | psnr | lpips | dists | ms-ssim | bpp |
| --- | --- | --- | --- | --- | --- | --- |
| I | 1 | 12.29203 | 0.81465 | 0.61331 | 0.43153 | 0.01336 |
| P | 7 | 13.10878 | 0.83157 | 0.70449 | 0.49733 | 0.01285 |

相邻预算质量倒退比例为53.12%；最大预算均值13.0067 dB，旧100步checkpoint在0.01549383 bpp约14.5613 dB。新模型约0.01291860 bpp，码率不同，**不能宣称等码率质量提升**。延长训练没有建立有效多预算泛化，不能选择Perplexity最大的模型宣布成功。

| qp | psnr | lpips | dists | ms-ssim | bpp |
| --- | --- | --- | --- | --- | --- |
| 16 | 30.78346 | 0.37346 | 0.20481 | 0.91557 | 0.00978 |
| 32 | 32.97841 | 0.31697 | 0.18086 | 0.94716 | 0.02081 |
| 48 | 34.71068 | 0.27336 | 0.16252 | 0.96469 | 0.04770 |
| 63 | 35.79997 | 0.24649 | 0.14534 | 0.97318 | 0.11181 |

原生采用FP16 fused CUDA与SPS/IP字节，CP采用PyTorch、多次显示及应用记录，工作量和开销口径不同。码率共同区间约0.00978–0.01292 bpp，质量区间完全不重叠，因此标准BD-rate为null，禁止外推。仅在实测共同码率内作PCHIP积分，BD-PSNR约-18.22 dB，明确保留两种framing差异。逐点数据见 `rd_comparison.csv` / `native_controls/rd_summary.json`。

![VQ诊断与实际RD](/home/whisperliang/CP-AdapTok/results/optimization_20261115/optimization_summary.png)

![真实训练曲线](/home/whisperliang/CP-AdapTok/results/optimization_20261115/training_curves.png)

## 因果、熵模型、Scorer与数值安全

所有正式码流仍满足EncodeOnce产生N_max ID、预算只截取IDs[:K]、旧Packet字节不变。收发端只用K_ref=128的真实基础码流重建提交状态，增强不改变下一帧历史；没有额外Base/Enhancement Decoder或高分辨率残差编解码器。

使用延长训练模型的**确定性ID**重新训练entropy/scorer 300更新；旧模型的entropy/scorer没有被直接复用为最终结论。Fixed-K、Rate-only、Scorer与Offline Oracle的逐帧选择见 `final_evaluation/eval/evaluation.json` 的allocation字段。Oracle明确noncausal，仅离线；在线策略只读当前因果预测和实际字节，行为测试将真实失真标签翻转后，在线选择不变。本轮质量失败，不能据此宣称内容自适应分配已在有效RD模型上获益。

同checkpoint的32帧因果视频继续使用真实码流，平均PSNR/相邻帧重建变化误差如下。temporal_error定义为原图帧差与重建帧差之差的MSE，不是码流状态漂移。相邻前缀质量倒退仍为54.69%，严格状态一致不能替代长期画质验证。本轮固定K_ref=128，尚未执行不同K_ref的长期压缩效率消融。

| tokens | psnr | bpp | temporal_error | cache_max_error |
| --- | --- | --- | --- | --- |
| 128 | 13.50851 | 0.00119 | 0.00928 | 0.00000 |
| 256 | 13.50331 | 0.00193 | 0.00929 | 0.00000 |
| 512 | 13.50057 | 0.00344 | 0.00929 | 0.00000 |
| 1024 | 13.50054 | 0.00644 | 0.00929 | 0.00000 |
| 2048 | 13.50079 | 0.01244 | 0.00929 | 0.00000 |

FP32/FP64同权重32帧检查：跨精度Token差异12个，最大像素差0.00131737，参考feature最大差0.00093082；FP32缓存/完整解码最大差0.00104702，FP64为0.00000000。各自收发端状态保持严格一致，但跨设备未经验证，FP32未成为默认。保留FP64数值参考，没有放宽参考指纹。

更新代码还完整解码了基线留下的32帧 `.cpat`，包括后续帧的严格历史校验；见 `historical_stream.json`。新增可选实验字段默认false时保持旧model_id，避免无行为变化破坏旧码流。

## 性能 Profile 与优化

先profile，再按开销选择优化。下表是同步后的CPU wall与CUDA Events，采用同一checkpoint。子项存在嵌套，不能相加；CUDA event区间包括等待CPU的空闲，不能冒充纯kernel时间。

| module | calls | wall_ms | cuda_event_ms | fused_wall_ms |
| --- | --- | --- | --- | --- |
| encoder.reference_context | 8 | 7.37983 | 7.37330 | 7.30035 |
| encoder.preprocess | 8 | 0.68103 | 0.67619 | 0.53530 |
| encoder.ld_dmci | 8 | 5.04930 | 5.04321 | 7.38576 |
| encoder.query_feature_projection | 8 | 0.73026 | 0.72437 | 0.62581 |
| encoder.query_cross_attention | 8 | 0.92607 | 0.92074 | 0.81512 |
| encoder.learned_queries | 8 | 1.78233 | 1.77807 | 1.55300 |
| encoder.vq_search | 8 | 1.44792 | 1.44186 | 1.24741 |
| encoder.ids_to_cpu | 8 | 0.10654 | 0.10253 | 0.09379 |
| encoder.rans_packetize | 8 | 1.27235 | 1.26744 | 1.16838 |
| encoder.local_rans_decode | 8 | 0.19619 | 0.19158 | 0.16738 |
| decoder.embedding_lookup | 16 | 0.38707 | 0.38240 | 0.29682 |
| decoder.token_to_latent | 16 | 1.02662 | 1.01945 | 0.82981 |
| decoder.ld_dmci | 24 | 9.43514 | 9.42807 | 9.39314 |
| decoder.postprocess | 24 | 0.29911 | 0.29440 | 0.24341 |
| encoder.local_base_decode | 8 | 11.31763 | 11.31316 | 10.95663 |
| encoder.state_fingerprint | 8 | 3.35967 | 3.35490 | 0.96516 |
| encoder.scorer | 8 | 1.15372 | 1.14859 | 1.05064 |
| decoder.packet_parse_rans | 16 | 0.98821 | 0.98359 | 0.83970 |
| decoder.state_commit | 8 | 3.36118 | 3.35592 | 0.92357 |
| decoder.prefix_attention_cache | 8 | 1.70251 | 1.69498 | 1.38657 |
| decoder.grid_to_latent | 8 | 0.36241 | 0.35738 | 0.27153 |

整数指纹的混合/归约来自同一个纯整数函数，compiled为可选融合执行，语义仍为mixed_int64_v1。Triton编译曾暴露64位索引常量被压成int32的问题；将位置序列显式作为int64输入后，通过原生状态尺寸随机位模式的逐位一致检查。只创建仓库内64位driver链接，不修改系统库。冷编译与构造开销不计入warm FPS，不能声称首帧实时。

| backend | fps | mean_ms | p50_ms | p95_ms | peak_mib |
| --- | --- | --- | --- | --- | --- |
| eager | 16.66415 | 60.00906 | 57.02049 | 66.00639 | 1009.11719 |
| compiled | 18.38288 | 54.39845 | 51.74277 | 53.53941 | 1009.11719 |

该对比为同32帧、warmup2、相同FP64模型质量、实际编码+rANS+发送端本地基础解码+接收端基础/最大显示+状态更新。融合前后32帧Header/Packet完全一致。包含基础与最终增强两次显示，仍支持所有中间合法前缀独立解码；不强制每个packet都重建LD。

下面单列8帧五档质量评估中同步计时的编码、基础解码、最大完整重建与增量刷新。它们处于指标计算穿插的评估环境，不能相加或替代上述无指标计算的完整FPS；缓存增强这一行是在已缓存1024 Token后刷新到2048。基础行包括真实追加/rANS/重建/提交，不应把已显示基础图像的重复读取时间当成基础解码时间。

| stage | mean_ms | p50_ms | p95_ms |
| --- | --- | --- | --- |
| encoder_total | 46.72826 | 40.31975 | 75.00315 |
| receiver_base_append_decode_commit | 19.53719 | 15.85327 | 37.26137 |
| receiver_max_dense_after_base | 16.45462 | 11.90050 | 38.57839 |
| receiver_cache_1024_to_2048 | 16.37374 | 12.05339 | 37.03096 |

较早8帧5档研究显示约9.78 FPS、基础/最终两档约14.82 FPS；不同帧数不能与32帧结果直接归因。VQ大批搜索逐包一致但8帧实测14.72 FPS，未胜过14.82，不作为默认。FP32两档约14.95 FPS，收益很小且32帧数值门槛失败。rANS约1.27ms，主要热点仍是LD/DMCI与参考上下文；本轮没有为小热点另建C++生产双实现，也没有重写原生Proxy。

## 测试、实现文件与验收状态

最终pytest：**125 passed**。包含连续表示可学习、采样/确定性推理、初始化/窗口恢复/完整RNG、前缀/Decoder共享、教师帧型与padding、原像素分辨率、真实rANS追加、严格因果状态、数值缓存、在线标签隔离、实际码流benchmark。编译路径用真实CUDA状态尺寸逐位对照；历史流与32帧视频另做实测。

| check | exit_code | log |
| --- | --- | --- |
| pytest | 0 | results/optimization_20261115/acceptance/pytest.log |
| train | 0 | results/optimization_20261115/acceptance/train.log |
| resume | 0 | results/optimization_20261115/acceptance/resume.log |
| eval | 0 | results/optimization_20261115/acceptance/eval.log |
| benchmark | 0 | results/optimization_20261115/acceptance/benchmark.log |
| coverage | 0 | results/optimization_20261115/acceptance/coverage.log |
| audit | 0 | results/optimization_20261115/acceptance/audit.log |
| mutations | 0 | results/optimization_20261115/acceptance/mutations.log |
| lint | 0 | results/optimization_20261115/acceptance/lint.log |
| whitespace | 0 | results/optimization_20261115/acceptance/whitespace.log |
| eval_all_metrics | 0 | results/optimization_20261115/acceptance/eval_all_metrics.log |
| coverage_all_metrics | 0 | results/optimization_20261115/acceptance/coverage_all_metrics.log |
| audit_all_metrics | 0 | results/optimization_20261115/acceptance/audit_all_metrics.log |

工具链缺少skill命名的crapper/dryer/mutator，采用仓库已有radon、coverage、AST相似度及隔离行为变异脚本。当前覆盖率88.01%，最大生产函数复杂度9，CRAP≥10条目0，AST相似度≥0.8条目0，行为变异15/15 killed；输出与边界见 `acceptance/quality_audit.json`。性能结果来自独立无coverage串行测量，不能引用instrumented benchmark的FPS。

报表/实验工具的最终lint、数值表再生成和复杂度检查另存 `acceptance/delivery_checks.json`、`experiment_tool_complexity.json`；生产CRAP/覆盖率统计不包含这些离线工具。关键JSON/CSV和两张图随报告入库；完整训练日志、逐帧诊断、视频、数据和checkpoint仍保留本地独立实验目录。

实现文件分组：

| 文件 | 改动与目的 |
| --- | --- |
| `tokenizer/vector_quantizer.py`, `vq_initialization.py` | 一套VQ策略、确定性推理、真实初始化、窗口恢复与优化器行状态 |
| `tokenizer/learned_queries.py`, `lightweight_attention.py` | Query分层诊断、显式初始化尺度、默认关闭的readout消融 |
| `training.py`, `training_initialization.py`, `diagnostics.py` | 帧型监督、连续诊断、采样、梯度与分布、可恢复训练 |
| `backbones/native_baseline.py` | 正确padding域与native CUDA所有权边界，保留真实latent提取 |
| `backbones/causal_state.py` | 可选逐位一致的整数指纹融合 |
| `data.py`, `runtime.py` | 原生图像序列、矩形crop、内容隔离、RNG/checkpoint与可选执行后端 |
| `models/causal_progressive_codec.py`, `streaming/sender.py`, `receiver.py`, `profiling.py` | 保留原协议与严格状态；增加按阶段wall/event profile |
| `evaluation.py` | 保留全部真实字节与指标，可避免大量诊断PNG I/O |
| `configs/diagnostic_continuous.yaml`, `d1...d4`, `train_multiscene.yaml` | 分阶段独立配置，数据与预算明确 |
| `tests/*`, `scripts/*optimization*`, 其他本轮脚本 | 行为回归、实际训练/容量/数值/历史流/性能/CSV与图表复现 |

上述路径相对 `cp_adaptok/`，configs/scripts位于仓库根目录。操作与决策日志见 `docs/optimization_work_log.md`。

验收结论：训练机制、教师条件、padding、协议与状态一致性工程门槛通过；码本多样性恢复但有效渐进表示失败；正式多场景数据覆盖与收敛未完成；1080p质量失败；30/60FPS失败。高分辨率pilot关闭estimated-rate/perceptual项以隔离表示学习，尚未执行有效模型上的联合RD微调；正式配置可显式恢复这些项。没有正式BD-rate，没有未经训练验证的质量提升结论。

## 推荐配置与恢复命令

研究训练起点使用 `configs/train_multiscene.yaml`：官方train数据、512原像素crop、Learned Queries、统一Decoder、连续预热后初始化/随机退火、I-only教师、后期闭环衰减。这是继续研究的配置，**不是已验证生产模型**。正式扩展数据、内容变化信息传递和前缀收益仍需优先解决。

```bash
cd /home/whisperliang/CP-AdapTok
# 复现本轮消融（已有实验目录会自动按checkpoint恢复）
.venv-native/bin/python -m scripts.run_optimization_experiments --experiments continuous_4 hard stochastic annealed warmup initialized --steps 100
# 在已有1000步延长训练checkpoint上继续；不覆盖原始pretrained_s1
.venv-native/bin/python -m cp_adaptok.cli.train --config results/optimization_20261115/multiscene_extended/config.json --checkpoint results/optimization_20261115/multiscene_extended/latest.pt --resume --max-steps 2000
# 新的多场景研究训练，以及从其checkpoint恢复
.venv-native/bin/python -m cp_adaptok.cli.train --config configs/train_multiscene.yaml --max-steps 1000
.venv-native/bin/python -m cp_adaptok.cli.train --config configs/train_multiscene.yaml --checkpoint checkpoints/multiscene_research/latest.pt --resume --max-steps 2000
# 可选指纹融合；系统驱动不变，首次编译有启动开销
.venv-native/bin/python -m scripts.prepare_compile_env
export TRITON_LIBCUDA_PATH=/home/whisperliang/CP-AdapTok/.build/driver
.venv-native/bin/python -m cp_adaptok.cli.benchmark --config results/optimization_20261115/optimized_config.json --checkpoint results/optimization_20261115/auxiliary/latest.pt --measure actual-bitstream --frames 32 --warmup 2 --output results/optimization_20261115/reproduce_fused
```

完整可执行命令与实际退出码见 `acceptance/commands.json`；训练、各阶段、capacity、native QP、缓存与性能的JSON/CSV保留在独立results目录。图表由真实记录生成。仍未完成的研究目标不能由延长步数、更多Token或更高Perplexity代替验收。

仓库图表下载：[VQ与RD图](../results/optimization_20261115/optimization_summary.png)、[训练曲线](../results/optimization_20261115/training_curves.png)；机器可读总表：[summary.json](../results/optimization_20261115/summary.json)、[真实码率表](../results/optimization_20261115/rd_comparison.csv)、[同条件FPS](../results/optimization_20261115/fused_fingerprint_pipeline.json)、[验收命令](../results/optimization_20261115/acceptance/commands.json)。
