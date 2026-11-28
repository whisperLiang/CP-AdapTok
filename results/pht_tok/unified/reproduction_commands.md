# 复现

在原始 `experiment/pht-tok` 的工作树执行。官方 DCVC 子模块固定在 `cbdae87a5445114cdc7f48816da63ea80bdeac40`。H100 实测环境使用 PyTorch 2.9.1+cu130、CUDA Toolkit 13.0.88、Python 3.12 和固定 CUTLASS `4370102f9dacab813282e1d67722fceb0b90a019`。检查点目录同时放入官方 I、LD 和 HT-S 发布权重。数据根沿用原 Validation 布局。 [冻结权重](../baseline_replay/freeze.json)

```bash
git submodule update --init third_party/DCVC third_party/AdapTok
export PYTHON_BIN=/path/to/native-venv/bin/python
export CUDA_HOME=/path/to/cuda-13.0
export CUTLASS_HOME=/path/to/cutlass
export PHT_WEIGHTS=/path/to/checkpoints
export CP_ADAPTOK_DATA_ROOT=/path/to/data
export PHT_UVG="$CP_ADAPTOK_DATA_ROOT/uvg/ShakeNDry_1920x1080_120fps_420_8bit_YUV.yuv"
export OMP_NUM_THREADS=4 OMP_WAIT_POLICY=PASSIVE
export PYTHONPATH="$PWD/.build/pht-tok/dual/site:$PWD"
bash experiments/pht_tok/native/build.sh dual
bash experiments/pht_tok/native/build_entropy.sh dual
bash experiments/pht_tok/native/build.sh official
```

CPU 可保留双后端目录里的 rANS 二进制，或仅构建 CPU 扩展。完整模型的 CUDA 推理在单卡上串行运行，避免其他任务影响性能。

```bash
CUDA_VISIBLE_DEVICES="" "$PYTHON_BIN" -m pytest -q -m 'not gpu'
"$PYTHON_BIN" -m pytest -q -m gpu
ruff check pht_tok baselines common experiments/pht_tok benchmarks tests
mypy
```

## 正式数据与运行时

```bash
out="$PWD/.build/pht-tok/reproduction"
mkdir -p "$out" "$out/legacy"
"$PYTHON_BIN" -m benchmarks pht-evaluate validation --weights "$PHT_WEIGHTS" --output "$out"
"$PYTHON_BIN" -m benchmarks pht-evaluate runtime --weights "$PHT_WEIGHTS" --uvg "$PHT_UVG" --output "$out"
"$PYTHON_BIN" -m benchmarks pht-evaluate network --weights "$PHT_WEIGHTS" --output "$out"
"$PYTHON_BIN" -m benchmarks pht-validation --weights "$PHT_WEIGHTS" --work "$out/legacy"
"$PYTHON_BIN" -m benchmarks pht-evaluate official-control --weights "$PHT_WEIGHTS" --output "$out/patched"
PYTHONPATH="$PWD/.build/pht-tok/official/site:$PWD" "$PYTHON_BIN" -m benchmarks pht-evaluate official-control \
  --weights "$PHT_WEIGHTS" --output "$out/stock"
cmp "$out"/{patched,stock}/native_stock_wire.json
```

Validation 输出目录的 `parts/` 按后端、QP 和片段进行原子写入。复跑使用同一配置和权重时可以把输出指向同一个目录继续。调整模型、数据、环境或设置时使用新目录。率失真使用实际 Header、各层、记录长度和熵编码终止字节。网络回放在精确字节数上按模拟 IPv4/UDP/RTP 规则附加开销。

## 原始 AdapTok 和图表

AdapTok 使用独立的官方 Python 3.10 和 PyTorch 2.5.1+cu121 环境。检查点是固定 Hub revision 的完整快照。

```bash
export ADAPTOK_PYTHON=/path/to/adaptok-venv/bin/python
export ADAPTOK_CHECKPOINT=/path/to/checkpoints/original_adaptok_official/c514b22615a140c06f4c9792e93cbef9e5c79f32
PYTHONPATH="$PWD" "$ADAPTOK_PYTHON" -m benchmarks pht-evaluate adaptok --checkpoint "$ADAPTOK_CHECKPOINT" --output "$out"
PYTHONPATH="$FIGURE_AUDIT_SCRIPTS:$PYTHONPATH" "$PYTHON_BIN" -m experiments.pht_tok.figures --results "$out"
```

`FIGURE_AUDIT_SCRIPTS` 指向提供 `audit_panel_alignment.py` 的 Python 图表审核目录。图使用 PDF 和 SVG 矢量导出及 600 dpi PNG 预览。 [脚本](../../../experiments/pht_tok/figures.py)

基础 PHT1 重放从原始 `54674cc` 提交运行，保留冻结的原生权重、源及结果哈希。 [冻结证据](../baseline_replay/freeze.json)
