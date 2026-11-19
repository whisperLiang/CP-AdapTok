# AIP-VTok C1 工作记录

时间使用 Asia/Shanghai。源分支基线为 `3cb3641`，子分支为 `experiment/aip-vtok-c1`。

- 按 ub-mode 先运行基线回归并记录 C1 测试的初次失败，随后实现固定扫描、物理 P 坐标旁路、FSQ 整数往返和参考状态边界。
- 核对 DMC-LD 接入、I/P 对齐契约、Fix-A 第 1000 步和 Gate 0 W128 封存权重。所有新权重归入 `checkpoints/aip_vtok/c1/`。
- 2026-11-19T17:29:36+08:00 完成 Train 拟合与校准。在任何验证像素读取前封存了扫描表和门槛。
- 2026-11-19T17:38:32+08:00 完成 Diag-0.5 的测量和门控记录。
- diag05_calibrate 使用 NVIDIA H100 PCIe，前后核对 751 个历史文件及投影 SHA256。`3b1bd715427bd177393e9552a99576e614ab7dd1ed307155129cf62896d5eb94`。开始 2026-11-19T17:24:09+08:00，结束 2026-11-19T17:29:36+08:00。
- diag05_all 使用 NVIDIA H100 PCIe，前后核对 751 个历史文件及投影 SHA256。`3b1bd715427bd177393e9552a99576e614ab7dd1ed307155129cf62896d5eb94`。开始 2026-11-19T17:29:53+08:00，结束 2026-11-19T17:38:32+08:00。
- c1a 2 帧短闭环更新 10 步，固定裁剪损失 0.10582 → 0.10475，步时中位数 0.337 s，峰值分配 5371 MiB。
- c1a 4 帧短闭环更新 10 步，固定裁剪损失 0.10640 → 0.10402，步时中位数 0.531 s，峰值分配 5371 MiB。

复现命令依阶段顺序运行，可使用已有的原子检查点续跑。

```bash
OMP_NUM_THREADS=4 .venv-native/bin/python scripts/aip_vtok/run_diag05.py
OMP_NUM_THREADS=4 .venv-native/bin/python scripts/aip_vtok/run_c1a.py
OMP_NUM_THREADS=4 .venv-native/bin/python scripts/aip_vtok/run_c1b.py
OMP_NUM_THREADS=4 .venv-native/bin/python scripts/aip_vtok/run_diag05.py --stage report
```
