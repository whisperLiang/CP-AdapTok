"""Build the report from retained measurements, including unsuccessful quality gates."""

import csv
import json
import re
from statistics import mean

from cp_adaptok.runtime import ROOT, metadata, write_json

RESULTS = ROOT / "results/prefix_utilization"
LABELS = ["original", "C0", "C1", "C2", "fixed_spatial"]


def read(path):
    return json.loads((RESULTS / path).read_text())


def formatted(value):
    if isinstance(value, float) and 0 < abs(value) < 1e-4:
        return f"{value:.3e}"
    return f"{value:.6f}" if isinstance(value, float) else str(value)


def table(rows, columns):
    headers = "| " + " | ".join(label for _, label in columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    values = ["| " + " | ".join(formatted(row[key]) for key, _ in columns) + " |" for row in rows]
    return "\n".join([headers, divider, *values])


def export_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def native_rows(records, qp, fps):
    rows = []
    for kind in ("all", "I", "P"):
        selected = [row for row in records if kind == "all" or row["frame_type"] == kind]
        row = {"model": "native_dcvc_uf_ld", "qp": qp, "frame_type": kind, "frames": len(selected),
               "bpp": mean(r["total_bytes"] * 8 / (1920 * 1080) for r in selected),
               "kbps": mean(r["total_bytes"] * 8 * fps[r["source"]] / 1000 for r in selected),
               "packet_framing": "native SPS/IP"}
        row.update({metric: mean(r[metric] for r in selected)
                    for metric in ("psnr", "mse", "lpips", "dists", "ms-ssim")})
        rows.append(row)
    return rows


def native_summary():
    specs = read("full_1080p/evaluation_config.json")["data"]["test"]
    fps = {spec["path"]: spec["fps"] for spec in specs}
    rows = []
    for qp in (16, 32, 48, 63):
        rows.extend(native_rows(read(f"full_1080p/native_qp{qp}/native_baseline.json")["records"], qp, fps))
    return rows


def latency_summary(reports):
    rows = []
    for label, report in reports.items():
        records = report["records"]
        rows.append({"model": label,
                     "encoder_total_ms": mean(r["encode_timing_ms"]["encode_total_ms"] for r in records),
                     "base_append_decode_commit_ms": mean(r["base_decode_ms"] for r in records),
                     "dense_max_ms": mean(r["prefixes"][-1]["dense_decode_ms"] for r in records),
                     "cache_1024_to_2048_ms": mean(r["prefixes"][-1]["cache_decode_ms"] for r in records),
                     "cache_aggregation_ms": mean(r["prefixes"][-1]["attention_update_ms"] for r in records),
                     "cache_max_image_error": max(p["cache_max_error"] for r in records for p in r["prefixes"])})
    return rows


def intervention_mean(measurements, kind, mode):
    selected = [r for r in measurements if r["tokens"] == 2048 and r["frame_type"] == kind
                and r["intervention"] == mode]
    fields = {"psnr": "psnr", "image_rmse": "delta_image_rmse", "latent_rmse": "delta_latent_rmse",
              "lpips": "lpips", "dists": "dists"}
    return {"frame_type": kind, "mode": mode,
            **{key: mean(r[field] for r in selected) for key, field in fields.items()}}


def intervention_summary(path):
    measurements = read(path)
    return [intervention_mean(measurements, kind, mode) for kind in ("I", "P")
            for mode in ("original", "other_frame", "shuffle", "constant", "all_constant")]


def group_statistic_row(values, kind):
    fields = {"group_perplexity": "perplexity", "content_variance": "content_variance",
              "content_cosine": "content_cosine"}
    return {"frame_type": kind, "group_unique_min": min(g["unique_ids"] for g in values),
            "group_unique_max": max(g["unique_ids"] for g in values),
            **{key: mean(g[field] for g in values) for key, field in fields.items()}}


def diagnostic_frame_summary(groups, attention, kind):
    selected = [r for r in groups if r["frame_type"] == kind]
    row = group_statistic_row([g for frame in selected for g in frame["groups"]], kind)
    row["attention_query_norm_ratio"] = mean(r["prefixes"][-1]["attention_to_query_ratio"]
                                              for r in attention if r["frame_type"] == kind)
    return row


def diagnostic_summary():
    groups = read("diagnostics/per_group_statistics.json")
    attention = read("diagnostics/attention_mass.json")
    gradients = read("diagnostics/gradient_by_group.json")
    rows = [diagnostic_frame_summary(groups, attention, kind) for kind in ("I", "P")]
    p = gradients[2]
    by_group = []
    for index in range(16):
        row = {"group": index + 1}
        for budget in p["budgets"]:
            row[str(budget["tokens"])] = budget["decoder_embedding"][index]["rms"]
        by_group.append(row)
    return rows, by_group


def plot_series(axes, label, rows, marker):
    for ax, metric in zip(axes, ("psnr", "lpips", "dists")):
        ax.plot([r["bpp"] for r in rows], [r[metric] for r in rows], marker, label=label)


def plot_comparison(summary, native):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for label, values in summary["models"].items():
        plot_series(axes, label, values["absolute"], "o-")
    selected = [r for r in native if r["frame_type"] == "all"]
    plot_series(axes, "native LD", selected, "s-")
    for ax, metric in zip(axes, ("psnr", "lpips", "dists")):
        ax.set(xscale="log", xlabel="Actual bpp (different framing declared)", ylabel=metric)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS / "full_1080p/rd_all_controls.png", dpi=180)
    plt.close(fig)


def acceptance_summary():
    audit = read("acceptance/quality_audit.json")
    coverage = read("acceptance/coverage.json")
    tests = re.search(r"(\d+) passed", (RESULTS / "acceptance/pytest_final.log").read_text())
    mutations = read("acceptance/mutations/summary.json")
    return {"tests_passed": int(tests.group(1)), "coverage_percent": coverage["totals"]["percent_covered"],
            "max_complexity": audit["max_complexity"], "crap_over_10": len(audit["crap_over_10"]),
            "duplicates_over_08": len(audit["duplicates_over_08"]),
            "mutations_killed": sum(row["killed"] for row in mutations), "mutations_total": len(mutations)}


def build_data():
    summary = read("full_1080p/comparison/summary.json")
    reports = {label: read(f"full_1080p/{label}/evaluation.json") for label in LABELS}
    native = native_summary()
    latency = latency_summary(reports)
    diagnostic, gradients = diagnostic_summary()
    data = {"metadata": metadata(status="quality_gate_failed"), "selection": read("selection.json"),
            "comparison": summary, "native": native, "latency": latency, "diagnostic": diagnostic,
            "nested": read("nested_training/comparison.json"),
            "decoders": read("decoder_ablations/comparison.json"),
            "baseline_interventions": intervention_summary("diagnostics/token_interventions.json"),
            "nested_interventions": intervention_summary("diagnostics/nested_C1/token_interventions.json"),
            "progressive_interventions": intervention_summary("diagnostics/progressive_C2/token_interventions.json"),
            "acceptance": acceptance_summary(),
            "gradient_example": gradients,
            "engineering_training": {"updates": 18, "frame_presentations": 168, "formal_convergence": False},
            "formal_training": {"started": False, "reason": "local and validation quality gate failed"},
            "native_bd_rate": {"value": None, "reason": "native and CP measured quality/rate ranges do not overlap"}}
    write_json(RESULTS / "summary.json", data)
    write_json(RESULTS / "full_1080p/native_summary.json", native)
    export_csv(RESULTS / "full_1080p/native_rd.csv", native)
    export_csv(RESULTS / "performance/latency.csv", latency)
    write_json(RESULTS / "performance/scope.json", {
        "latencies": "existing full-resolution RD evaluation with interleaved metrics and shared device",
        "warm_end_to_end_fps": None, "reason": "quality gate failed; throughput optimization not promoted",
        "coverage_benchmark": "single-frame instrumented entrypoint check only; FPS excluded"})
    plot_comparison(summary, native)
    return data


def gain_rows(summary):
    rows = []
    for label in LABELS:
        point = summary["models"][label]
        gain = point["gains"][-1]
        ci = gain["psnr"]["sequence_bootstrap_ci95"]
        rows.append({"model": label, "base": point["absolute"][0]["psnr"],
                     "max": point["absolute"][-1]["psnr"], "delta": gain["psnr"]["mean_delta"],
                     "median": gain["psnr"]["median_delta"], "ci": f"[{ci[0]:.6f}, {ci[1]:.6f}]",
                     "regression": gain["psnr"]["regression_fraction"],
                     "lpips_delta": gain["lpips"]["mean_delta"], "dists_delta": gain["dists"]["mean_delta"]})
    return rows


def write_report(data):
    scalar = data["comparison"]
    absolute = [dict(model=label, **point) for label in LABELS for point in scalar["models"][label]["absolute"]]
    frame_types = [dict(model=label, frame_type=kind, **point)
                   for label in LABELS for kind in ("I", "P")
                   for point in scalar["models"][label]["by_frame_type"][kind]]
    export_csv(RESULTS / "full_1080p/rd_by_frame_type.csv", frame_types)
    baseline = read("baseline_manifest.json")
    checkpoint = baseline["candidates"][0]
    report = f'''# Prefix Information Utilization 专项验收报告

实际执行了诊断、代码重构、五组训练目标消融、三组 Decoder 消融、有限原生分辨率训练、五条独立1080p序列的真实码流及四个原生LD QP对照。**工程链路通过，研究效果未通过。默认保留原版全局 Softmax；本轮不发布新质量有效模型，不启动大规模正式长训或实时性能推广。**

Nested C1 的2048/128平均PSNR差只有0.022217 dB；区间为正但幅度很小，不能称为显著渐进提升。新聚合器C2的差为0.163736 dB，却以大幅降低基础质量、恶化LPIPS/DISTS为代价，且后缀内容干预不支持有效信息恢复。0.5 dB目标未达到。

## 1. 版本、模型与数据

审查HEAD：`{baseline['metadata']['git_commit']}`，起始工作区干净。A/B实现提交 `77f5eef`；结果元数据记录各次运行的Git SHA、dirty状态和生产源码SHA256。本报告生成时HEAD为`{data['metadata']['git_commit']}`，源码SHA256为`{data['metadata']['source_sha256']}`，最终文档提交在其后。

分析主基线为上轮最终离散模型 `{checkpoint['path']}`，SHA256 `{checkpoint['sha256']}`，stage3熵/Scorer已训练{checkpoint['step']}步，表示来自此前1000步模型。时间更新的`capacity/learned_512/latest.pt`也实际诊断（256原像素crop、2帧），用于避免遗漏新资产；它不是已验证正式收敛模型。没有使用continuous checkpoint生成码流。原四个候选及四个原始权重的哈希复核均未改变，见`acceptance/original_assets_unchanged.json`；新模型哈希见`checkpoint_manifest.json`。

使用现有Python3.12、PyTorch2.9.1/cu130、H100 PCIe 80GB及官方LD/DMCI权重。没有更改上游CUDA、网络调度或Packet格式。设备未承诺独占。

固定片段训练：MCL-JCV01，源16帧范围内固定seed19取得4帧，256 crop，保留原像素。选择与门槛判断只使用这个固定片段和独立Beauty验证。此后不再根据新增测试调参。

测试：BasketballDrive及官方DAVIS val中的blackswan、bmx-trees、breakdance、camel，共5条独立序列，每条8帧，均1920×1080、没有resize。新增4条通过HTTP range提取、ZIP CRC和SHA核验；与8条官方train内容ID分离。DAVIS图像序列fps取名义30，Basketball取50。只覆盖40帧、5条序列，不能代表完整数据集泛化。

## 2. 实测诊断与根因边界

A主诊断为原生1080p Basketball前4帧，五档真实离散ID；干预共享同一个FrozenFrameContext，每帧只提交一次正常K_ref。替换来源是同一片段另一帧在同一旧历史下的独立诊断编码，不进入正常因果流；相邻帧相关性是该对照的限制。常量控制在所有帧固定合法Code 0，不根据当前视频选码。全部ΔLatent/ΔImage定义为相对正常同长度输出的RMSE。

{table(data['diagnostic'], [('frame_type','帧型'),('group_unique_min','组内Code最少'),('group_unique_max','组内Code最多'),('group_perplexity','平均组Perplexity'),('content_variance','Code输出Embedding方差'),('content_cosine','组内Cosine'),('attention_query_norm_ratio','Attention/Query范数')])}

Code统计使用共享VQ的project_out输出，明确不含Decoder Position Embedding；另存positioned方差与position cosine。Encoder固定Query仍可导致Code随位置变化，所以仅有不同ID、Perplexity或Embedding方差不能证明视频信息不同。P/P的同位置ID变化与I/P切换分别解释；新补充诊断`acceptance/diagnostics_measured`另标记同帧型比较并记录pre-VQ有效秩。

主根因判断为**多因素共同作用**：Token内容具有高度公共分量，序列没有形成可验证的互补组织；Decoder对P后缀真实内容极弱敏感，固定查询/位置/历史的联合先验占主导；全局归一化稀释单Token梯度是重要限制。不能由这些数据独立量化“固定Query”和“历史”各自的图像贡献。相邻预算在共享前缀上的梯度余弦约0.999，不支持“梯度冲突”为此次主要原因。去除跨组Softmax仍失败，说明归一化并非唯一根因。

## 3. Decoder内容依赖与无内容对照

{table(data['baseline_interventions'], [('frame_type','帧型'),('mode','干预'),('psnr','PSNR'),('lpips','LPIPS'),('dists','DISTS'),('image_rmse','ΔImage RMSE'),('latent_rmse','ΔLatent RMSE')])}

这些是K=2048诊断，不能纳入正式RD。other_frame/shuffle/constant仅替换K_ref之后；all_constant有意替换全部ID为固定Code 0，单独检验无Token内容。P帧同片段跨帧或打乱后缀时图像变化约4e-5至7e-5；固定Code 0仍能引起约0.003–0.004的图像RMSE，说明Decoder不是完全不读取内容，但正常后缀缺乏有益增量方向。I帧对常量内容显著敏感，不等于后缀组织正确。此前“每帧首码”自适应常量对照未作为最终无内容证据，其原结果保留在`diagnostics/per_frame_constant_control`。所有K及same_frame/base_only完整记录保存在`diagnostics/token_interventions.json`和CSV。

## 4. 注意力与真实位置梯度

旧聚合器新增半段Token获得约0.5注意力总质量，各组近似均匀；这会同时重新分配旧Token权重，不能视为新增信息的独立贡献。固定Query、out(attended)、FFN范数、无位置对照、前缀Grid变化和每个Grid Query的组分布均保存于`attention_mass.json`。

下表为诊断第2个P帧（frame2）的**实际重建MSE autograd Embedding梯度RMS**。未接收位置梯度为0是正确边界；最大预算中单Token梯度约为基础预算的1/16。Query及VQ输入梯度遵循现有VQ的straight-through估计，Embedding梯度来自真实下游重建计算图，均不是从注意力权重推算。完整逐组梯度、方向余弦另见`gradient_by_group.json`。

{table(data['gradient_example'], [('group','组'),('128','K128'),('256','K256'),('512','K512'),('1024','K1024'),('2048','K2048')])}

![Attention](../results/prefix_utilization/diagnostics/attention_visualization.png)
![Gradients](../results/prefix_utilization/diagnostics/gradient_distribution.png)

## 5. Nested训练的实际消融

{table(data['nested'], [('variant','方案'),('updates','更新'),('frames','帧次'),('train_base_psnr','训练Base'),('train_gain','训练Max-Base'),('validation_base_psnr','1080p验证Base'),('validation_max_psnr','1080p验证Max'),('validation_gain','验证增益')])}

B0为旧2–4档采样与双向约束；B1强制端点+随机中间档，保留旧约束；B2增加显式曝光均衡权重（端点0.6、中间1.8）；B3使用单向detach-low约束、margin=0；scheduled前半程等权，后半程曝光均衡。每组相同初始化、数据、100更新、lr=3e-4、骨干和离散VQ。实际预算监督次数保存在comparison.json，等步数不冒充完全等FLOPs。所有分支共享一次完整Encoder/VQ；每档绝对失真仍被监督。为隔离表示问题，训练失真为MSE，VQ权重0.1、单调项0.1，感知训练项及estimated-rate项为0；本轮未完成非零码率权重的联合RD优化。

固定片段PSNR可约35 dB但增益只有0.010–0.015 dB，因此未通过进入正式长训的质量门槛。B4的detached-low残差L2与普通高预算MSE实际值和梯度等价，未发现独立辅助作用，未添加重复目标；证据见`incremental_alignment_equivalence.json`。

## 6. 全局Softmax与共享增量聚合

{table(data['decoders'], [('variant','方案'),('updates','更新'),('train_base_psnr','训练Base'),('train_max_psnr','训练Max'),('train_gain','训练差'),('validation_base_psnr','验证Base'),('validation_max_psnr','验证Max'),('validation_gain','验证差')])}

C0为旧聚合+旧目标，C1为旧聚合+Nested均衡/单向目标，C2仅替换聚合方式并采用同一C1目标。三者同一离散初始化、相同100更新和相同片段；GPU bf16不承诺跨进程逐位一致，B0/C0及B3/C1重复测量的小差异均保留。

C2的每个原Packet组使用完全共享的q/kv/out/FFN和194参数内容/顺序gate，累加attention值后进入同一Grid/Latent/LD Decoder。gate为2*sigmoid/sqrt(max_groups)，初始化alpha=1/4，分组尺寸取配置而不硬编码128。没有额外Encoder、增强Decoder、残差流、连续发送特征或预算专属参数。显式迁移仅新增gate.weight/bias，所有可复用参数枚举在migration.json；完整合成后strict=True加载，optimizer未恢复。

不同Model ID绑定`global_softmax_v1`与`shared_progressive_v2`；历史默认配置字段不改变旧身份字节。标准加载和收流严格校验，未添加静默兼容fallback。

## 7. 独立1080p绝对质量与真实RD

每条序列独立重置收发状态，EncodeOnce、真实rANS Packet、离散ID lookup、完整LD/DMCI和K_ref闭环，实际header/CRC/rANS/application length framing均计入码率。全部模型的熵表重新使用相同8条官方train中的32帧、512 crop作加一平滑多项式MLE，没有使用验证/测试Token；原checkpoint不变。Scorer未随这些表示重新拟合，预测仅保留为接口标签，未用于RD选择或准确率声明。FixedSpatial为已有独立训练对照，优化历史不同，不能把它当作唯一结构差异的严格消融。

{table(absolute, [('model','模型'),('tokens','K'),('psnr','PSNR'),('ms-ssim','MS-SSIM'),('lpips','LPIPS'),('dists','DISTS'),('mse','MSE'),('bpp','实际bpp'),('kbps','实际kbps')])}

I/P分别及整体所有指标见`full_1080p/rd_by_frame_type.csv`，逐帧实际字节、状态指纹和缓存误差保存在各evaluation.json。Basketball旧基线128/2048实测13.0358/13.0067 dB，复现原问题。C0/C1的小片段过拟合使多场景平均质量退步，不能推荐这些checkpoint取代原模型。

## 8. 渐进收益及统计可靠性

{table(gain_rows(scalar), [('model','模型'),('base','Base PSNR'),('max','Max PSNR'),('delta','平均ΔPSNR'),('median','中位ΔPSNR'),('ci','按序列bootstrap 95% CI'),('regression','PSNR倒退帧比例'),('lpips_delta','平均ΔLPIPS'),('dists_delta','平均ΔDISTS')])}

采用独立视频为主要单位的2000次bootstrap，seed19，每次保留该视频全部连续帧。CI为5条序列的经验区间，样本小且没有多训练seed验证，不能视作完整benchmark结论。相邻预算全部gain、三指标的均值/中位/改善比例/倒退比例/区间在summary.json；不能把首尾都正误解为所有相邻档单调。单序列Beauty的CI为null。C1小幅正差不能替代同码率绝对质量；C2增加Token的正差同时伴随感知退步及严重基础损害，未满足信息利用门槛。

## 9. 新聚合器内容因果干预

Nested C1另作两帧原生1080p干预，同样固定Code 0。它的P帧同片段跨帧/打乱后缀只引起约9e-5/2e-5图像RMSE；I帧正常后缀也未优于全部对照。该小样本结果不支持把其0.022 dB解释为可靠的增量内容恢复。

{table(data['nested_interventions'], [('frame_type','C1帧型'),('mode','干预'),('psnr','PSNR'),('lpips','LPIPS'),('dists','DISTS'),('image_rmse','ΔImage RMSE')])}

C2结果如下：

{table(data['progressive_interventions'], [('frame_type','帧型'),('mode','干预'),('psnr','PSNR'),('lpips','LPIPS'),('dists','DISTS'),('image_rmse','ΔImage RMSE')])}

该表为C2同样的Basketball四帧诊断。I帧把后缀或全部ID改为常量可大幅改善PSNR，P帧打乱后缀几乎无效。相似公共分量按组累积导致K本身改变输出，不能把长度曲线改善当作内容恢复证据。C2没有通过采用门槛，保留为有证据的失败消融；原全局Softmax维持默认路径。

## 10. 原生LD与码率边界

{table([r for r in data['native'] if r['frame_type']=='all'], [('qp','QP'),('frames','帧数'),('psnr','PSNR'),('ms-ssim','MS-SSIM'),('lpips','LPIPS'),('dists','DISTS'),('bpp','bpp'),('kbps','kbps')])}

原生字节为SPS/IP，CP字节另含其应用framing，均明确列出，未假设物理网络头。四个QP的质量及码率区间与CP都不重叠，BD-rate=null，不作外推。CP间非单调曲线或无共同质量区间也返回不可计算；没有以排序掩盖质量倒退。绝对质量差距不能由更低bpp包装为压缩优势。

![RD](../results/prefix_utilization/full_1080p/rd_all_controls.png)

## 11. 实际训练量、门槛与性能边界

五个B组共500更新/2000帧次，三个C组共300更新/1200帧次。正式长训门槛失败后，只对C0/C1/C2各做512 crop×4帧、720原像素crop×8帧、原生1080×16帧三阶段，每阶段2更新，共18更新/168帧次，用8条官方train核对形状、梯度及基础闭环；每次实际选中的源/窗口在train.jsonl，不能声称18步覆盖全部训练集。总研究/原生检查818更新/3368帧次；另有CLI验收3更新/6帧次，不计入研究效果。所有正式消融teacher=none，未用异历史P latent。

没有因失败而无限加步数。正式512→720→1080大规模收敛、联合RD微调、多训练seed仍未执行。原生检查的微量更新没有加入最终等100步RD比较，也没有据此改选测试模型。

{table(data['latency'], [('model','模型'),('encoder_total_ms','编码全流程ms'),('base_append_decode_commit_ms','接收基础/提交ms'),('dense_max_ms','已有基础后最大完整解码ms'),('cache_aggregation_ms','缓存追加/聚合ms'),('cache_1024_to_2048_ms','缓存追加及完整后端ms'),('cache_max_image_error','最大缓存图像误差')])}

上述耗时取自实际RD评估，穿插感知网络且设备共享，不是独占暖机性能对照，不能相加或换算端到端FPS。固定100步的训练耗时分别见各summary.json。新缓存只投影新增组，单元行为检查直接计数KV调用；不会因显示多个K而重新编码。每个显示点仍需完整LD后端，未声称LD本身增量化。LearnedQuery/VQ/纯attention的无插桩分项及完整暖机FPS本轮未推广；coverage单帧benchmark只验证入口，其插桩FPS不引用为性能。

完整与缓存均按同一Packet顺序作FP64加法；实际C2两帧全部16档cache误差0。独立逆序加法诊断：FP64 Grid/Image最大差0；FP32 Grid差1.505e-6、图像差0.00373149，不能改用FP32作为严格默认。证据见`performance/accumulation_order.json`。这不是所有输入的误差上界。

## 12. 工程验收与质量工具

{data['acceptance']['tests_passed']}项最终测试结果见`acceptance/pytest_final.log`；真实旧模型32帧流成功解码且保留Model ID `d31f2c342136a7ce4e38d654ebb79c1c`。新增Model ID拒绝错配，EncodeOnce前缀、仅已到达Token、统一Decoder、单次基础提交、增强不影响后续参考、缓存一致性及全视频固定常量Code均有行为测试。旧身份测试按新增默认字段的历史序列化规则更新，并由真实历史文件独立核验。

ub-mode的命名crapper/dryer/mutator不可用，替代为radon+coverage+AST相似度和隔离AST行为变异。覆盖率{data['acceptance']['coverage_percent']:.2f}%，最大生产复杂度{data['acceptance']['max_complexity']}，CRAP≥10条目{data['acceptance']['crap_over_10']}，AST相似度≥0.8条目{data['acceptance']['duplicates_over_08']}。{data['acceptance']['mutations_killed']}/{data['acceptance']['mutations_total']}关键行为变异被测试杀死，涵盖单向detach、端点、组softmax轴及追加顺序。详见`acceptance/quality_audit.json`。初次全套中旧预期哈希测试失败的日志也保留，没有隐瞒测试调整。工具插桩不进入性能结论。

## 13. 最终采用与未完成目标

工程实现正确：通过。训练能够更新并在固定片段收敛：通过局部证据，非正式多场景收敛。Token内容被有效利用：I帧有内容敏感性，P后缀及增量方向未通过。独立视频有效渐进质量：未通过，Nested只有约0.022 dB且绝对质量退步。端到端实时性能：本轮不作新达标声明。

采用可复现诊断、显式Nested/权重/单向约束、按序列统计及严格模型版本/缓存接口；它们均不等于推荐新checkpoint。全局Softmax继续作为默认模型，Nested保持研究配置，C2只保留必要失败消融。没有固定空间主方案、独立增强流、连续旁路、损害低预算来制造差值的训练目标或静默fallback。旧模型、数据和失败记录未删除。

## 14. 直接复现

在当前仓库及现有`.venv-native`、官方权重、原始视频可用的条件下运行：

```bash
cd /home/whisperliang/CP-AdapTok
.venv-native/bin/python -m scripts.diagnose_prefix_utilization
.venv-native/bin/python -m scripts.compare_prefix_training --steps 100
.venv-native/bin/python -m scripts.compare_prefix_decoders --steps 100
.venv-native/bin/python -m scripts.prepare_prefix_validation --sequences 4 --frames 16
.venv-native/bin/python -m scripts.run_prefix_validation --engineering-steps 2
.venv-native/bin/python -m scripts.measure_prefix_accumulation
.venv-native/bin/python -m scripts.verify_historical_stream \\
  --source results/pretrained_long_eval/eval/basketball \\
  --output results/prefix_utilization/acceptance/historical_stream.json
.venv-native/bin/python -m pytest cp_adaptok/tests -q
.venv-native/bin/python -m scripts.mutation_check --output results/prefix_utilization/acceptance/mutations
.venv-native/bin/python -m scripts.build_prefix_utilization_report
```

新实验均在`results/prefix_utilization`，checkpoint在本地保留；Git保存关键JSON/CSV/图、配置及报告，未将大权重或视频打入仓库。机器可读总表`summary.json`、操作记录`docs/prefix_utilization_work_log.md`以及独立训练/验证/干预原始记录可逐项审查。结果不支持的新方案没有被推荐为生产默认。
'''
    (ROOT / "docs/prefix_utilization_report.md").write_text(report)


def main():
    data = build_data()
    write_report(data)
    print("Wrote docs/prefix_utilization_report.md and measured summary/CSV/figures")


if __name__ == "__main__":
    main()
