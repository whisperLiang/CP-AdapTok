# HLA-Tok-T 实现

`NativeLayout` 统一产生原生 Rank/Region 索引、实际末 Rank mask、O0/O1/O2 视图和 LocalTokenDecoder 的接收支持计划。120 个完整 Rank 之后只存在 120 个有效区域。原生有效 Token 一共 16320。LocalTokenDecoder 按原生 Rank 升序接收每个 Region 的有效 Token。任意合法支持由连续同构 Region 分组恢复，完整支持调用冻结 C0 的 `recover_grid`，D0 输出裁剪回 1080×1920。

共享 Rank-aware 投影沿用已归档的 PCA32 资产，FP32 计算按 Rank 做批处理。以 Train 峰值校准格点，以来源互斥的 Train holdout 选择 I/P 整数量化步长。舍入使用精确的最近偶数规则，中间整数运算使用较宽类型，超出 int16 格点时拒绝。I 从零开始。P 只预测已经提交的基础扫描坐标。复制和有界线性模式都用同一整数解码核心提交基础状态。

基础扫描从 O2 Rank 列表按完整 Rank 与固定分量宽度建立。B1/B2/B3 的最坏字节预算分别为 4166、10416、20833。不同位宽和块大小的独立候选在该上限内选取完整 Rank。S0 先扫固定 16 个分量然后扫其余 16 个。S1 是 Train Rank/分量能量排序，核心 2×2 对照共享同一绝对能量扫描。S2 使用 Train RGB MSE 的一阶下降估计和 Train holdout 独立短块的实际字节。评分权重、候选和停止规则记录于配置及矩阵身份清单。

DAVIS 图像序列使用清单的 nominal FPS 给出时间，存储键保留源视频、原帧编号和 SHA-256。Train 片段从 CP 原拟合和留出来源采样。长序列使用从头到尾的真实连续片段。场景切换压力测试直接切换真实源片段，单独报告参考状态，质量时间证据沿用原视频边界。

真实接收进程只初始化 LocalTokenDecoder 和 D0 模块。跨设备指纹约束作用于相同 int16 输入的整数核心。CUDA 浮点编码和合成的支持范围由实际 BF16/FP32 运行及旁路容差记录。
