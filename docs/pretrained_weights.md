# 官方 LD/DMCI 权重

2026-11-15 已从固定版本 DCVC README 指向的官方 OneDrive `CVPR2026` 目录下载两份文件。来源：[官方说明](https://github.com/microsoft/DCVC/blob/cbdae87a5445114cdc7f48816da63ea80bdeac40/README.md#-usage)、[官方目录](https://1drv.ms/f/c/2866592d5c55df8c/IgAalzb_985lQ79GkXyW2P5OASPpZHHcrcGWEVQxO-mQCVg?e=qyvMN6)。

| 文件（位于项目 checkpoints/） | 用途 | 实际字节 | 官方 QuickXorHash |
|---|---|---:|---|
| cvpr2026_image.pth.tar | DMCI / I 帧骨干 | 168845375 | `djlEVMhDjexrJb/on44uFwIjRik=` |
| cvpr2026_video_ld.pth.tar | DMC LD / P 帧骨干 | 38724071 | `JCP2tLHKkC9NEEacsZDf2wUjLmo=` |

SHA256：

```text
b3b900de23f30e4fc437010ddffe6a5413a56d6cfd97fb6235adbcd2b6973302  cvpr2026_image.pth.tar
263172f25e04f3db6fd95487e1895df6f0fc2a5084e6e21d9016934fcdabc799  cvpr2026_video_ld.pth.tar
```

校验已实际执行：

- 文件长度与官方目录元数据完全一致。
- 按 [Microsoft QuickXorHash 算法](https://learn.microsoft.com/en-us/onedrive/developer/code-snippets/quickxorhash?view=odsp-graph-online) 重算，均与官方值一致；同时记录本地 SHA256。
- PyTorch ZIP archive CRC 全部通过。
- `torch.load(..., weights_only=True)` 读取并通过固定上游骨干的 `strict=True` 参数加载；图像398个 state_dict tensor，LD375个，所有浮点 Tensor 有限。
- 下载完成时原生 prerequisite 仍缺两个扩展；后续已在独立 `.venv-native` 中构建成功，并使用实际 decoded-latent teacher 完成100步训练，见 [后续报告](pretrained_native_report.md)。旧环境和历史 smoke checkpoint 保持原样。

机器可读清单：`checkpoints/official_weights_manifest.json`；下载/校验详情和剩余依赖：`results/weight_download/`。大文件保留在本机，按现有 Git ignore 规则不提交到源码仓库。

本次遇到旧共享 API 401、分享网页403，通过 Microsoft 匿名浏览器会话和 `my.microsoftpersonalcontent.com/_api/v2.0/shares/` 取得官方文件列表及签名下载地址；接口方式参考并核对了[公开实现](https://gist.github.com/NTFSvolume/e0395fe6eeaa9b47a5c8874ec8133987)。没有使用用户账号凭据；匿名临时令牌及签名下载 URL 不写入版本控制。

下载文件与完成 CP-AdapTok 训练是独立的里程碑。现有 smoke checkpoint 保存完整随机骨干 state_dict，加载它会覆盖初始化骨干；后续正式实验需要以官方权重初始化骨干，训练新增 queries/VQ/前缀 Decoder。原生 CUDA baseline/teacher 仍需构建扩展。本次仅完成资源取得与兼容性验证，未启动新一轮正式训练，既有未训练实验标记保持有效。
