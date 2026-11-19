# AIP-VTok

这是隔离的、按 Gate 放行的研究实现。当前可执行范围是 Gate 0：官方 DCVC-UF 原生控制、修复 I 连续尺度、128 通道对齐、直接参考投影、真实 H100 诊断与短程训练。后续 Tokenizer、AIP rANS 码流及实时编解码器未完成，不对这些功能作通过声明。

从仓库根目录运行：

```bash
bash scripts/aip_vtok/run_gate0.sh
```

脚本复用 `.venv-native`，拒绝占用其他 GPU 作业，自动恢复本实验未完成 checkpoint；Gate 不通过以非零状态退出。配置位于 `configs/aip_vtok/gate0.yaml`。结果与 checkpoint 仅写入 `results/aip_vtok/`、`checkpoints/aip_vtok/`。

报告由 `python -m aip_vtok.evaluation.report` 从原始 JSON 生成，见 `docs/aip_vtok_report.md`。不能把可逆的 P 归一化、I 的低秩近似、原生连续直通与官方量化码流当成相同实验；不能把参考模块的延迟当成完整编解码 FPS。
