# AIP-VTok Joint V1 实验报告

- 源提交 c5cd06dfc8cb6afd19c6da668d7664eb3f32c2ea
- 分支 experiment/aip-vtok-joint-v1，生成报告时的代码 HEAD c5cd06dfc8cb6afd19c6da668d7664eb3f32c2ea
- IMPLEMENTED。架构和可见信息边界见 [architecture_contract.md](architecture_contract.md)。
- I/P 的独立 128D 正交变换与 A2 FSQ 产生 8 组 × 6D 的 16-bit 可表示整数符号，共享扫描支持八个前缀。高、低两尺度 SLD 在变换域保持真实观察值。
- 单包上限 128 Token，各自维护 32-bit rANS 状态，使用 12-bit 整数 CDF 和 CRC32。I2048/P510 的固定参考前缀计入实际传输，分包和头部计费见 rate_ledger.csv。
- 训练配置见 [joint_v1.yaml](../../../configs/aip_vtok/joint_v1.yaml)。每帧预算去重等权，参考项按帧计一次。学习率使用 1,000 步 warmup 和余弦衰减至峰值的 10%。A/B 窗口 4 帧，C 窗口 8 帧。全部训练保留真实 GOP 相位和当期模型的无梯度 burn-in。

## 工程验证与训练

- TESTED。工程证据见 [engineering_tests.json](engineering_tests.json)，通过 22 项。
- Seed 1 INCOMPLETE，尚无正式 checkpoint。
- Seed 2 INCOMPLETE，尚无正式 checkpoint。

## validation

INCOMPLETE。正式测量尚未产生。

## test

INCOMPLETE。正式测量尚未产生。

旧方案的高预算标为 N/A。旧方案的 raw 或估计载荷采用原协议计量。逐帧的参考指纹、后段质量、额外长序列结果和同步计时保存在相应评估目录。

## 复现

```bash
python scripts/aip_vtok/run_joint_v1.py --config configs/aip_vtok/joint_v1.yaml --seed 1 --train
python scripts/aip_vtok/run_joint_v1.py --config configs/aip_vtok/joint_v1.yaml --seed 1 --resume
python scripts/aip_vtok/evaluate_joint_v1.py --config configs/aip_vtok/joint_v1.yaml --seed 1 --split validation
python scripts/aip_vtok/run_joint_v1.py --config configs/aip_vtok/joint_v1.yaml --seed 2 --train
python scripts/aip_vtok/evaluate_joint_v1.py --config configs/aip_vtok/joint_v1.yaml --seed 1 --split test
python scripts/aip_vtok/benchmark_joint_v1.py --config configs/aip_vtok/joint_v1.yaml --seed 1
```
