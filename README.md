# PHT-Tok

PHT-Tok 复用 Microsoft DCVC-UF 的官方 LD 与 HT-S 权重，在原生整数残差上生成可按字节截断的实际 rANS 码流。LD 每次编码一个 P 帧。HT-S 每次联合编码八个 P 帧，尾块只重复末帧，输出裁回有效帧数。

```python
from pht_tok import PHTTok

codec = PHTTok("ld", checkpoint="/path/to/checkpoints", device="cuda")
anchor = codec.encode(first_frame, qp=32)
codec.decode(anchor.data)
encoded = codec.encode(next_frame, qp=32)
payload = encoded.prefix(max_bytes=encoded.prefix_boundaries[0])
decoded = codec.decode(payload)
```

RGB 输入形状为 `T,3,H,W`，支持 `uint8` 以及范围为 `[0,1]` 的浮点张量。编码和解码的参考状态独立，可以分别由发送端和接收端的实例持有。HT-S 默认生成四档 PHT1 前缀。设置 `layers=8` 可启用八档 PHT2 前缀。参考层设置为全部层时，完整传输按官方参考闭环运行。可变预算链路须交付配置的固定参考层。

```bash
python -m pht_tok --backend ld --mode progressive --input /path/to/clip.mp4 \
  --checkpoint /path/to/checkpoints --qp 32 --device cuda --benchmark --output /path/to/streams
python -m benchmarks pht-evaluate --help
```

- [统一版验收](results/pht_tok/unified/final_report.md)
- [后端与码流契约](results/pht_tok/unified/backend_contract.md)
- [构建、复测和图表命令](results/pht_tok/unified/reproduction_commands.md)
- [原始 HT-S 研究报告](results/pht_tok/final_report.md)
