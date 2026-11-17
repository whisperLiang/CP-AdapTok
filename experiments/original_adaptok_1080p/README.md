# 官方 AdapTok 原生 1080p 独立实验

本目录只调用未修改的 `third_party/AdapTok`，不训练，不调用 CP/DCVC。
配置为 [original_adaptok_1080p.yaml](../../configs/original_adaptok_1080p.yaml)。

## 运行

在仓库根目录执行：

```bash
bash experiments/original_adaptok_1080p/setup.sh
.venv-adaptok-official/bin/python -m experiments.original_adaptok_1080p.download_checkpoint \
  --output checkpoints/original_adaptok_official/c514b22615a140c06f4c9792e93cbef9e5c79f32
bash experiments/original_adaptok_1080p/run.sh verify --output results/original_adaptok_1080p/verification
bash experiments/original_adaptok_1080p/run.sh suite
.venv-adaptok-official/bin/python -m experiments.original_adaptok_1080p.benchmark
.venv-native/bin/python -m experiments.original_adaptok_1080p.verify_cp_inputs
.venv-adaptok-official/bin/python -m experiments.original_adaptok_1080p.report
```

下载器只下载固定 Revision，最终 SHA256 必须与官方 Hub LFS 相同。
正常环境也可用 Hub 的 `snapshot_download`，然后 `--checkpoint /absolute/local/model_dir`；
适配器仍会核对本地 config/weights 与指定官方 Revision 的哈希，再调用原生严格加载接口。
配置中的 checkpoint 置为 `null` 可直接使用固定 Revision 的 Hub 缓存。

`suite` 执行 E0、单视频首窗 E1/E2，以及两种 stride × 三预算 × 三视频各64帧的 E3/E4。
每个完整 Encoder 输出供同 Tile 所有预算解码，保存真实数值，不强制质量单调。
独立 Benchmark 每种 stride 测三视频各16帧，不加载感知指标网络；请等待 suite 完成后运行。
结果根目录使用进程锁，完成的窗口校验身份后复用；输入或配置不同会拒绝覆盖。
中断窗口的旧尝试保留，新尝试写入独立子目录。

单视频或图像序列：

```bash
bash experiments/original_adaptok_1080p/run.sh evaluate \
  --input /absolute/path/video.mp4 --frames 17 --start 0 --stride 96 \
  --budgets 512 1024 2048 --tile-batch-size 4 \
  --output results/original_adaptok_1080p/my_clip
```

不指定 `--frames` 时，单个容器/图像序列使用所有剩余帧。源必须为1920×1080；拒绝隐式缩放或裁剪。
清单批量评估用 `--manifest path.json --content-ids DAVIS2017:cows DAVIS2017:goat DAVIS2017:parkour`。
裸YUV额外传 `--raw-width 1920 --raw-height 1080 --fps 30 --raw-pixel-format yuv420p`、
`--raw-color-range limited` 与 `--raw-color-matrix bt709`，并填写真实源元数据，不能猜测范围/矩阵。
图像/容器完全复用 CP OpenCV→RGB 规则；裸YUV为显式FFmpeg转换，元数据单独记录。

## 验证

```bash
OMP_NUM_THREADS=4 .venv-adaptok-official/bin/python -m pytest experiments/original_adaptok_1080p/tests -q
ADAPTOK_OFFICIAL_CHECKPOINT=checkpoints/original_adaptok_official/c514b22615a140c06f4c9792e93cbef9e5c79f32 \
ADAPTOK_OFFICIAL_VIDEO=/absolute/manifest/source/path \
OMP_NUM_THREADS=4 .venv-adaptok-official/bin/python -m pytest \
  experiments/original_adaptok_1080p/tests/test_official_equivalence.py -q
.venv-adaptok-official/bin/python -m experiments.original_adaptok_1080p.mutation_check
```

无GPU/权重时，CPU测试可运行，真实GPU一致性测试明确跳过；不会用随机模型代替官方实测。
`IdentityAdapter` 仅为分块和共享Encoder的管道测试夹具，不能生成实验结果。

## 术语与结果定义

| 术语 | 定义 |
|---|---|
| 时间窗口 | 原生checkpoint帧数；当前16连续帧，尾窗重复最后真实帧，仅真实帧计分 |
| Tile | 同一空间坐标跨整个时间窗口的 `[3,T,H,W]` 子视频 |
| 时间Group | checkpoint原生4组，每组512 latent；保留每组前K/4个，不截整个序列前K个 |
| K | 每Tile、每原生时间窗口的有效VQ Token总数，不是每1080p帧Token数 |
| quality_only | 未做模型熵编码；bpp=null，观看用MP4不代表模型码率 |
| PSNR逐帧均值 | 每真实完整帧PSNR的算术平均 |
| 全局MSE PSNR | `-10 log10(mean(frame_mse))`，与上一项分别保存 |
| 边界Mask | 默认在x/y=128n处宽4像素条带并集；两种stride使用同一个Mask |
| NN FPS | 真实源帧数 / Encode+Decode总时间；另列CUDA Event和同步wall口径 |
| 计算FPS | 加分块、传入GPU、拼接、分配及分派开销 |
| 流水线FPS | 计算加源读取、原始/重建视频和样图写盘；不包含指标计算 |

感知指标必须全部成功加载，失败直接报错。所有指标在拼接后的1080p RGB帧计算。
LPIPS-Alex/DISTS复用CP网络及预处理，LPIPS-VGG对照官方评估。源与权重SHA256跨两个环境检查。
视频采用RGB无损`libx264rgb -crf 0`保存；指标在显示量化之前计算。

结果默认位于 `results/original_adaptok_1080p`，操作日志与最终报告位于 `docs/`。
历史B0/B1只读复用，比较称为绝对重建质量比较，不能称为同码率RD实验。
