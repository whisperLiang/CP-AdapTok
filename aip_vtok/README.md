# AIP-VTok

这是隔离的、按 Gate 放行的研究实现。Gate 0 已通过实测；Gate 1 包含真实 FSQ 三段残差 Tokenizer、增量潜在画布、单次渲染、配对损失及双种子训练/验证入口。正式结果以机器可读 Gate 判定为准。AIP rANS 码流和完整实时编解码器尚未启用，不对这些功能作通过声明。

从仓库根目录运行：

```bash
bash scripts/aip_vtok/run_gate0.sh
python -m aip_vtok.training.smoke
python -m aip_vtok.training.formal --seed 1
python -m aip_vtok.training.formal --seed 2
python -m aip_vtok.evaluation.formal --seed 1
python -m aip_vtok.evaluation.formal --seed 2
python -m aip_vtok.evaluation.gate1
```

脚本复用 `.venv-native`，拒绝占用其他 GPU 作业，自动恢复本实验未完成 checkpoint；Gate 不通过以非零状态退出。配置位于 `configs/aip_vtok/gate0.yaml`。结果与 checkpoint 仅写入 `results/aip_vtok/`、`checkpoints/aip_vtok/`。

报告由 `python -m aip_vtok.evaluation.report` 从原始 JSON 生成，见 `docs/aip_vtok_report.md`。不能把可逆的 P 归一化、I 的低秩近似、原生连续直通与官方量化码流当成相同实验；不能把参考模块的延迟当成完整编解码 FPS。

本次实测 Gate 1 已熔断，依据是长期预测反馈发散、FSQ 后期饱和与绝对质量不达标。Gate 2/3 未实施；当前不提供 AIP rANS 二进制码流或完整实时 FPS 声明。已保存的实验请用 `--resume` 继续，不能覆盖旧检查点。
