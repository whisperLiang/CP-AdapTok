# 仓库清理

审计确认原始 PHT-Tok 为 `experiment/pht-tok`，原始基线提交 `54674cc164cc29b449c1682d3d4b05ec774baa70`。主工作树原先检出的 `pht-tok` 与 `main` 同指 `a1c1997c2117321d26ffbedcbe2d433d560904df`。清理后两个本地分支分别在主工作树和原 PHT-Tok 工作树检出。

| 删除项 | 范围 |
|---|---|
| 本地分支 | `experiment/aip-vtok-joint-v1`、`experiment/aip-vtok-sparse-decoder`、`experiment/pht-tok-adaptive`、`pht-tok` |
| 工作树 | Joint V1、Sparse Decoder、Adaptive 三个专属工作树 |
| 实验资产 | 上述工作树中的代码、结果、检查点、运行文件、外链、缓存和独立子模块检出，以及主工作树的 AIP 专属权重和孤立字节码 |
| PHT-Tok 当前快照 | 删除旧 CP-AdapTok、HLA 路线的专属训练代码、配置、结果和测试。公共数据读取、官方评测、原 AdapTok 对照和测试接入保留的公共实现 |

Joint V1 唯一的未提交文件是 AIP 专属实验报告。逐行核对出的变化对应当时的运行快照、Seed 1 检查点、源码提交和训练计时。Sparse Decoder 的公共权重和数据路径均解析到保留的主工作树，AIP 检查点链接仅供退役实验使用。校验时未发现占用删除路径的训练、评估或推理进程。删除前后通过了 29 项公共资产 SHA256 核验，包括权重、划分清单、UVG、指标权重和扩展样本。数据目录核验位置，Test 图像读取 0。 [资产清单](retained_assets.json) [逐路径执行记录](cleanup_manifest.json)

按文件系统已分配块核算，删除工作树及实验资产释放 **2,502,983,680 B**，独立工作树元数据和子模块检出再释放 **141,348,864 B**，合计 **2,644,332,544 B（2.46 GiB）**。这个值按 inode 去重并排除了仍有外链的文件。前后 Git 公共对象目录均是 **995,962,880 B**。PHT-Tok 当前快照的 Git 跟踪删除及孤立字节码另列于清单。

主分支完整保留，其现有 D1/HLA 推理代码和权重仍在。官方 LD、HT-S、HT-L、I 权重和 AdapTok 权重、公开数据及原始 PHT 实测证据继续保留。远端清理由用户单独授权后运行以下精确命令。

```bash
git push origin --delete experiment/aip-vtok-joint-v1
git push origin --delete experiment/aip-vtok-sparse-decoder
git push origin --delete pht-tok
```
