# 复现

## 准备

```bash
git submodule update --init
uv venv --python 3.12 .venv-native
uv pip install --python .venv-native/bin/python --index-url https://download.pytorch.org/whl/cu130   torch==2.9.1 torchvision==0.24.1
uv pip install --python .venv-native/bin/python -r configs/baselines/native-requirements.txt -e '.[dev]'
bash baselines/adaptok/setup.sh
```

在 H100 上核验过的工具链为 PyTorch 2.9.1+cu130、CUDA Toolkit 13.0.88 和 CUTLASS `4370102f9dacab813282e1d67722fceb0b90a019`。AdapTok 另用 Python 3.10、PyTorch 2.5.1+cu121，锁定依赖见 [AdapTok 环境](../baselines/adaptok/requirements-lock.txt)。需要 FFmpeg、C++17 编译器和系统 zlib 开发头文件。感知指标使用原 LPIPS、DISTS 及 AlexNet、VGG 权重，PHT 启动时将哈希与冻结 Validation 配置核对。

```bash
export PYTHON_BIN="$PWD/.venv-native/bin/python"
export CUDA_HOME=/path/to/cuda-13.0
export CUTLASS_HOME=/path/to/pinned-cutlass
export CP_ADAPTOK_DATA_ROOT="$PWD/data"
export OMP_NUM_THREADS=4 OMP_WAIT_POLICY=PASSIVE
bash scripts/build_native.sh official
bash scripts/build_native.sh pht
bash scripts/build_native.sh d1
```

将 [官方发布](https://1drv.ms/f/c/2866592d5c55df8c/IgAalzb_985lQ79GkXyW2P5OASPpZHHcrcGWEVQxO-mQCVg?e=qyvMN6) 的 image、LD 和 HT-S 权重放入同一个 `checkpoints/` 目录。文件名、预期长度和 SHA256 见 [DCVC-UF 锁定清单](../configs/baselines/dcvc_uf.json)。D1 的推理检查点随仓库提供，含原 Seed 2 的 55 个权重张量。官方 AdapTok 固定 Hub revision 与下载命令如下。

```bash
adpython="$PWD/.venv-adaptok-official/bin/python"
adaptok="$PWD/checkpoints/original_adaptok_official/c514b22615a140c06f4c9792e93cbef9e5c79f32"
"$adpython" -m baselines.adaptok.download_checkpoint --output "$adaptok"
```

Train、Validation 和 Test 的 [清单](../data/manifests)保留原始来源、文件哈希、划分和相对数据布局。换主机时保持数据根下 `large_scale/` 的子目录结构，设置 `CP_ADAPTOK_DATA_ROOT` 指向新位置。Test 的封存由 manifest loader 和 AdapTok 输入边界检查执行。本轮所有评估仅打开 Validation、Calibration 或外部 UVG 开发样本。采用原始 [UVG ShakeNDry](https://ultravideo.fi/dataset.html) 的 1080p/120 FPS YUV420，验证哈希后设置 `PHT_UVG`。Verizon LTE 原轨迹与归档哈希见 [网络轨迹](../data/traces/verizon-lte.down) 和 [上游证据](../results/dcvc_uf/upstream_manifest.json)。

## 单项评估

每段命令使用自己的原生扩展路径，通过同一入口运行。输出使用独立的 `.build` 目录。

```bash
export PHT_WEIGHTS="$PWD/checkpoints"
export PHT_UVG="$CP_ADAPTOK_DATA_ROOT/uvg/ShakeNDry_1920x1080_120fps_420_8bit_YUV.yuv"

export PYTHONPATH="$PWD/.build/pht-tok/official/site:$PWD"
"$PYTHON_BIN" -m benchmarks dcvc --weights "$PHT_WEIGHTS" --models ld hts   --work .build/dcvc-calibration --output .build/dcvc-calibration/results.json

export PYTHONPATH="$PWD/.build/pht-tok/pht/site:$PWD"
"$PYTHON_BIN" -m benchmarks pht-codec --weights "$PHT_WEIGHTS"   --output .build/examples
"$PYTHON_BIN" -m benchmarks pht equivalence --weights "$PHT_WEIGHTS" --sources 1 --qps 32
"$PYTHON_BIN" -m benchmarks pht fixed --weights "$PHT_WEIGHTS" --sources 1 --qps 32
"$PYTHON_BIN" -m benchmarks pht-validation --weights "$PHT_WEIGHTS"
"$PYTHON_BIN" -m benchmarks pht-latency profile --uvg "$PHT_UVG" --weights "$PHT_WEIGHTS"
"$PYTHON_BIN" -m benchmarks network --uvg "$PHT_UVG" --weights "$PHT_WEIGHTS"   --verizon data/traces/verizon-lte.down

export PYTHONPATH="$PWD/.build/uf_v2/native:$PWD/.build/uf_v2/records:$PWD/.build/uf_v2/legacy:$PWD/.build/pht-tok/official/site:$PWD"
"$PYTHON_BIN" -m benchmarks d1 --weights "$PHT_WEIGHTS"   --quality 5 10 20 40 --repetitions 3 --output .build/d1-validation
"$PYTHON_BIN" -m benchmarks d1-closure --weights "$PHT_WEIGHTS"   --checkpoint checkpoints/hla_tok_uf_v2/d1-seed2.pt --output .build/d1-closure
"$PYTHON_BIN" -m benchmarks d1-incremental --weights "$PHT_WEIGHTS"   --checkpoint checkpoints/hla_tok_uf_v2/d1-seed2.pt --output .build/d1-incremental

PYTHONPATH="$PWD" "$adpython" -m benchmarks adaptok suite   --checkpoint "$adaptok" --manifest data/manifests/cpat_validation.json   --content-ids DAVIS2017:car-roundabout DAVIS2017:dog --frames 16 --output .build/adaptok
```

官方 DCVC 运行未修改的 `test_video.py`。历史对齐配置及公开 UVG 对照仍可从 [Calibration 归档](../results/dcvc_uf/calibration.md)查阅。PHT 直接对照的动态 QP 路径为 S1，使用与 S4 相同的 P-Chunk 采集、带宽、包头和截止时间。D1 依次校验 5/10/20/40 Mbps 的真实硬预算，Full-max 配置先生成完整码流再执行前缀接收。AdapTok 产生官方 Token 预算下的重建，原始和重建 MP4 仅供观看。

## 共同 Validation 与回归

实际像素来自 `DAVIS2017:car-roundabout` 的前 17 帧。AdapTok 在首个 16 帧窗口及一帧尾窗重建，时延只测等长完整窗口。PHT 的连续 16 个 P-Chunk 和恢复复测另使用 UVG 开发样本。各方法的 `smoke-*` 入口写出输入摘要、物理码流或重建哈希、质量和性能。按上一节设定每个进程的 `PYTHONPATH`。最后运行：

```bash
mkdir -p .build/replay
# 在官方环境中分别运行 LD 和 HT-S
"$PYTHON_BIN" -m benchmarks smoke-native ld --weights "$PHT_WEIGHTS" --output .build/replay/ld.json
"$PYTHON_BIN" -m benchmarks smoke-native hts --weights "$PHT_WEIGHTS" --output .build/replay/hts.json
# 切换 PHT 环境
"$PYTHON_BIN" -m benchmarks smoke-native pht --weights "$PHT_WEIGHTS" --uvg "$PHT_UVG"   --profile results/pht_tok/profiling.json --rates configs/pht_tok/network_calibration.json   --trace data/traces/verizon-lte.down --output .build/replay/pht.json
# 切换 D1 环境
"$PYTHON_BIN" -m benchmarks smoke-d1 "$PHT_WEIGHTS" checkpoints/hla_tok_uf_v2/d1-seed2.pt .build/replay/d1.json
PYTHONPATH="$PWD" "$adpython" -m benchmarks smoke-adaptok data/manifests/cpat_validation.json "$adaptok"   .build/replay/adaptok.json
"$PYTHON_BIN" -m benchmarks compare .build/replay --output .build/replay/comparison
```

率失真表按相同的 17 帧比较，并分别记录实际字节、PSNR、LPIPS、DISTS 和 AdapTok Token 数。性能记录保留同步核心时间、串行时间、P50/P95、显存和加载耗时。网络表保留仿真的实际送达字节和丢帧。吞吐与按采集时间计的显示延迟分别报告。
