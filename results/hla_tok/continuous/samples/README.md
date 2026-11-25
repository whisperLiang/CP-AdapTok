# 原生重建样例

固定使用 V3 的 `DAVIS2017:car-roundabout` 第 32 帧。图像为 1920×1080 RGB PNG。按同一坐标直接裁取移动汽车区域（左上角 x=768, y=432）和文字标志区域（x=1320, y=260），两处均为 384×216。汽车裁块展示车窗反射、车身亮度与边缘。标志裁块展示停车牌文字、背景石墙纹理与阴影。所有路径使用同一源帧和两块相同像素区域。 [文件与传输清单](manifest.json)

C0 表中使用主排序 O2 的真实 BF16 浮点包。C1 来源于原 V3 10000 步终点的实际 Fixed13 序列化和接收端解码。相同 Tensor 预算下两者的字节数可在清单中核对。 [C1 检查点与字节](c1_sample.json)

| 样例 | 原生 PNG | 汽车裁块 | 文字裁块 |
|---|---|---|---|
| 源帧 | [RGB](source.png) | [局部](source_crop.png) | [局部](source_sign_crop.png) |
| K510 | [RGB](O2_bf16_K510.png) | [局部](O2_bf16_K510_crop.png) | [局部](O2_bf16_K510_sign_crop.png) |
| K2040 | [RGB](O2_bf16_K2040.png) | [局部](O2_bf16_K2040_crop.png) | [局部](O2_bf16_K2040_sign_crop.png) |
| K8160 | [RGB](O2_bf16_K8160.png) | [局部](O2_bf16_K8160_crop.png) | [局部](O2_bf16_K8160_sign_crop.png) |
| K12240 | [RGB](O2_bf16_K12240.png) | [局部](O2_bf16_K12240_crop.png) | [局部](O2_bf16_K12240_sign_crop.png) |
| K16320 | [RGB](O2_bf16_K16320.png) | [局部](O2_bf16_K16320_crop.png) | [局部](O2_bf16_K16320_sign_crop.png) |
| C1 Fixed13 K16320 | [RGB](C1_fixed13_K16320.png) | [局部](C1_fixed13_K16320_crop.png) | [局部](C1_fixed13_K16320_sign_crop.png) |

实验工作区还保留 O0 和 O1 的同帧原生结果及两处裁块。Git 保留表中样例及全部样例清单。PNG 由解码 RGB 按统一规则舍入至 8 位。裁块文件逐像素核对原生 PNG。
