# PHT-Tok

PHT-Tok 在 Microsoft DCVC-UF HT-S 的八帧联合编解码上添加渐进真实码流。发送端分析每块一次。接收端可以逐层接收和熵解码，并用独立的参考前缀维持跨块状态。

## 运行环境

- `third_party/DCVC` 子模块固定在 `cbdae87a5445114cdc7f48816da63ea80bdeac40`
- NVIDIA H100，Python 3.12，PyTorch 2.9.1+cu130，CUDA Toolkit 13.0.88
- CUTLASS `4370102f9dacab813282e1d67722fceb0b90a019`
- 官方 image、LD、HT-S 和 HT-L checkpoint。发布入口、版本及 SHA256 见实验产物。
- 开发使用锁定的项目 Calibration 清单和 [UVG ShakeNDry](https://ultravideo.fi/dataset.html)。正式 Validation 仅进行冻结配置的独立重放。

## 构建与重放

在本分支工作树根目录运行。提前设置 `PYTHON_BIN`、`CUDA_HOME`、`CUTLASS_HOME` 和 `PHT_WEIGHTS`。已有同工具链的原生缓存时，可将 `PHT_NATIVE_CACHE` 指向缓存，在逐文件核对后重用相同编译单元。

```bash
git submodule update --init third_party/DCVC third_party/AdapTok
bash experiments/pht_tok/native/build.sh official
bash experiments/pht_tok/native/build.sh pht
bash experiments/pht_tok/native/build_entropy.sh

export PYTHONPATH="$PWD/.build/pht-tok/official/site:$PWD"
"$PYTHON_BIN" -m experiments.pht_tok.baseline \
  --weights "$PHT_WEIGHTS" --output .build/pht-tok/calibration.json

export PYTHONPATH="$PWD/.build/pht-tok/pht/site:$PWD"
"$PYTHON_BIN" -m experiments.pht_tok.study equivalence --weights "$PHT_WEIGHTS"
"$PYTHON_BIN" -m experiments.pht_tok.study fixed --weights "$PHT_WEIGHTS"
"$PYTHON_BIN" -m experiments.pht_tok.study closed --qps 32 --weights "$PHT_WEIGHTS" --uvg "$PHT_UVG"
cp .build/pht-tok/profiling.json .build/pht-tok/profiling-before.json
"$PYTHON_BIN" -m experiments.pht_tok.measure profile --weights "$PHT_WEIGHTS" --uvg "$PHT_UVG"
"$PYTHON_BIN" -m experiments.pht_tok.measure fixed-uvg --weights "$PHT_WEIGHTS" --uvg "$PHT_UVG"
"$PYTHON_BIN" -m experiments.pht_tok.network --weights "$PHT_WEIGHTS" \
  --uvg "$PHT_UVG" --verizon .build/pht-tok/verizon-lte.down
"$PYTHON_BIN" -m experiments.pht_tok.validation --weights "$PHT_WEIGHTS"

"$PYTHON_BIN" -m pytest -q --import-mode=importlib tests/pht_tok
ruff check pht_tok experiments/pht_tok tests/pht_tok
mypy --python-version 3.12 --follow-imports=skip --ignore-missing-imports pht_tok experiments/pht_tok
```

`PHT_UVG` 指向官方发布的 `ShakeNDry_1920x1080_120fps_420_8bit_YUV.yuv`。该文件只在本地实验依赖目录保存。公开网络迹来自固定版本的 [Mahimahi Verizon LTE 下载迹](https://github.com/ravinet/mahimahi/blob/f1346c38c415abfc98626dd82450f49a57033603/traces/Verizon-LTE-short.down)。

Python 入口由调用方在独占的非默认 CUDA stream 中执行。每个代理有独立的活动解码状态。实验 runner 给出加载权重、首帧初始化和逐块传输的完整示例。

- [位流与参考架构](architecture.md)
- [最终报告](../../results/pht_tok/final_report.md)
- [源文件、资产与工具链清单](../../results/pht_tok/upstream_manifest.json)
