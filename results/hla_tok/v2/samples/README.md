# 原生 1080p 同帧重建

所有样例来自 Validation 的 `DAVIS2017:car-roundabout` 第 32–47 帧，1920×1080、30 FPS、16 帧。PNG 对应第 32 帧，裁块使用同一组半开像素坐标 `[768, 432, 1152, 648)`。完整 PNG 和 FFV1 视频保存在实验 worktree，文件摘要、裁块核对和视频探测记录随分支提交。 [逐文件校验](verification.json)

| 图像或模型 | 条件 | 原生 PNG | 像素裁块 | FFV1 视频 |
|---|---|---|---|---|
| Original RGB | 冻结 Validation 原生 RGB | [PNG](pilots/source.png) | [裁块](pilots/source_crop.png) | [MKV](pilots/source.mkv) |
| Original AdapTok | stride128 FP32，480 Token/Tile/4 帧 | [PNG](official/original_adaptok.png) | [裁块](official/original_adaptok_crop.png) | [MKV](official/original_adaptok.mkv) |
| AIP-VTok Joint V1 | CLOSED K16320 档案重放 | [PNG](joint/joint_v1.png) | [裁块](joint/joint_v1_crop.png) | [MKV](joint/joint_v1.mkv) |
| HLA-Tok R0 | 旧检查点，K16320 Fixed13 | [PNG](legacy/R0_fixed.png) | [裁块](legacy/R0_fixed_crop.png) | [MKV](legacy/R0_fixed.mkv) |
| HLA-Tok R0 | 第 1,000 步，连续 K16320 | [PNG](pilots/R0.png) | [裁块](pilots/R0_crop.png) | [MKV](pilots/R0.mkv) |
| HLA-Tok R1 | 第 1,000 步，连续 K16320 | [PNG](pilots/R1.png) | [裁块](pilots/R1_crop.png) | [MKV](pilots/R1.mkv) |
| HLA-Tok R2 | 第 1,000 步，连续 K16320 | [PNG](pilots/R2.png) | [裁块](pilots/R2_crop.png) | [MKV](pilots/R2.mkv) |
| HLA-Tok R3 | 第 1,000 步，连续 K16320 | [PNG](pilots/R3.png) | [裁块](pilots/R3_crop.png) | [MKV](pilots/R3.mkv) |
| HLA-Tok R0 | 第 2,000 步，连续 K16320 | [PNG](extension/R0.png) | [裁块](extension/R0_crop.png) | [MKV](extension/R0.mkv) |
| HLA-Tok R2 | 第 2,000 步，连续 K16320 | [PNG](extension/R2.png) | [裁块](extension/R2_crop.png) | [MKV](extension/R2.mkv) |
