# HLA-Tok-T 复现

实验从 HLA-Tok-CP 的 `533ac3b` 创建 `experiment/hla-tok-temporal`。冻结 C0 使用 R2、AQT、LocalTokenDecoder 和 D0。当前原生帧经过全 32D Rank-aware PCA、共享整数格点和 HLT v1 码流，接收端用完整基础层更新因果参考并按已接收前缀恢复 1920×1080 RGB。

单卡 H100 正式环境使用 Python 3.12、PyTorch 2.9.1+cu130。继承 CP 的数据、checkpoint、指标、排序及依赖环境。请在 Temporal worktree 中使用仓库原有 Gitlink 初始化两个子模块，并让 CP 数据清单中的原路径指向已验证的冻结数据。运行时输入为 GPU 上的单张完整 RGB 帧，报告的码率为 HLT 应用层字节。

```bash
git submodule update --init --recursive
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4
export PYTHONPATH=/path/to/archived/compact/native-deps
TEMPORAL_PYTHON=/path/to/frozen/python
TEMPORAL_V3=/path/to/hla-tok-v3/CP-AdapTok
TEMPORAL_OUTPUT=results/hla_tok/temporal
TEMPORAL_REPLAY=.build/temporal/baseline

"$TEMPORAL_PYTHON" -m scripts.hla_tok.continuous.evaluate phase0 --v3 "$TEMPORAL_V3" --output "$TEMPORAL_REPLAY"
"$TEMPORAL_PYTHON" -m scripts.hla_tok.temporal.baselines --v3 "$TEMPORAL_V3" --native-replay "$TEMPORAL_REPLAY" --output "$TEMPORAL_OUTPUT"
for action in calibrate diagnostics evaluate; do
  "$TEMPORAL_PYTHON" -m "scripts.hla_tok.temporal.$action" --v3 "$TEMPORAL_V3" --output "$TEMPORAL_OUTPUT" || exit
done
"$TEMPORAL_PYTHON" -m scripts.hla_tok.temporal.selection --output "$TEMPORAL_OUTPUT"
"$TEMPORAL_PYTHON" -m scripts.hla_tok.temporal.diagnostics --matched-support --v3 "$TEMPORAL_V3" --output "$TEMPORAL_OUTPUT"
"$TEMPORAL_PYTHON" -m scripts.hla_tok.temporal.dynamic --v3 "$TEMPORAL_V3" --output "$TEMPORAL_OUTPUT"
"$TEMPORAL_PYTHON" -m scripts.hla_tok.temporal.runtime --v3 "$TEMPORAL_V3" --output "$TEMPORAL_OUTPUT"
cc -O3 -std=c11 -Wall -Wextra -Werror -fPIC -shared hla_tok/temporal/records.c -o .build/temporal/records.so -lz
"$TEMPORAL_PYTHON" -m scripts.hla_tok.temporal.runtime --v3 "$TEMPORAL_V3" --output "$TEMPORAL_OUTPUT" --wire-library .build/temporal/records.so
"$TEMPORAL_PYTHON" -m scripts.hla_tok.temporal.mutations --output "$TEMPORAL_OUTPUT"
"$TEMPORAL_PYTHON" -m scripts.hla_tok.temporal.audit history --output "$TEMPORAL_OUTPUT"
"$TEMPORAL_PYTHON" -m scripts.hla_tok.temporal.audit engineering --output "$TEMPORAL_OUTPUT"
"$TEMPORAL_PYTHON" -m scripts.hla_tok.temporal.report --output "$TEMPORAL_OUTPUT"
NATURE_FIGURE_SCRIPTS=/path/to/nature-figure/scripts "$TEMPORAL_PYTHON" -m scripts.hla_tok.temporal.figures --output "$TEMPORAL_OUTPUT"
```

基础输入包括仓库归档的 `PCA32-BF16_rank.npz` 和合成 checkpoint。Temporal 拟合输出 `lattice.npz` 和资产 SHA-256。C11 Record 后端需要系统 C 编译器及 zlib 开发文件。构建产物放在本地 `.build`。应用解码入口接收静态资产、checkpoint、逐帧 `.hlt` 文件和可选的 `--wire-library`，通过 `python -m scripts.hla_tok.temporal.receive --help` 查看参数。

工程回归与 CP 使用相同的 CPU/CUDA 标记和容差。GPU 回归依赖 CP 的 DCVC 权重和 `scripts/prepare_compile_env.py` 生成的驱动目录。测试报告应写入 `.build/temporal/logs`，文件格式在 [工程审计入口](../../../scripts/hla_tok/temporal/audit.py) 定义。

[协议](protocol.md) · [实现](implementation.md) · [工作记录](work_log.md) · [最终报告](../../../results/hla_tok/temporal/final_report.md)
