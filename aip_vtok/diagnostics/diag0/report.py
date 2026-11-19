"""Generate the Chinese Diag-0 decision report from the measured artifacts."""

from collections import defaultdict

from aip_vtok.runtime import resolve, write_json

from .closed import FIX_A
from .fit import NATIVE_ORDER, ORDERS, RANKS
from .io import FIX, OUTPUT, PROBES, json_at, native_control, read_probe
from .matched import CLEAN, LOW
from .metrics import summarize
from .packets import COARSE, FREE, FULL, HYBRID, LAYOUTS
from .sparse import BUDGETS

PSNR = "psnr_global_mse"
FRAME_PSNR = "psnr_frame_mean"
CORE = (PSNR, "lpips", "dists", "latent_mse")
NAMES = {FREE: "自由 Top-K", FULL: "全分辨率前缀", COARSE: "2×2 共享深度", HYBRID: "Hybrid", FIX_A: "Fix-A"}
ORDER_NAMES = {ORDERS[0]: "方差", ORDERS[1]: "RGB 敏感度", NATIVE_ORDER: "原生敏感度"}
SOURCE = "../results/aip_vtok/diag0/"


def select(rows, **condition):
    return [row for row in rows if all(row.get(key) == value for key, value in condition.items())]


def one(rows, **condition):
    row, = select(rows, **condition)
    return row


def markdown_table(headers, rows):
    data = [headers, tuple("---" for _ in headers), *rows]
    return "\n".join("| " + " | ".join(str(cell) for cell in row) + " |" for row in data)


def numeric(row, fields, digits=3):
    return tuple(f"{row[key]:.{digits}f}" for key in fields)


def cell_values(rows, field, digits=3):
    return " / ".join(f"{row[field]:.{digits}f}" for row in rows)


def pool(rows, keys):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    return [{**dict(zip(keys, key)), **summarize(group)} for key, group in groups.items()]


def introduction(meta, native):
    frames = meta["validation_frame_ids"]
    train = meta["train_windows"]
    projector = meta["identity"]["projector_sha256"]
    bitrate = native["bpp"]
    lines = [
        "# AIP-VTok Diag-0 冻结潜在空间诊断",
        f"基线提交 `{meta['identity']['base_commit']}`。QP{meta['identity']['qp']}、"
        f"GOP{meta['identity']['gop']}、原生 1920×1080，单卡 H100。"
        "本仓库的 I 帧为 DMCI，沿用 Gate 0 的 256→128 白化 PCA。P 帧使用 `video_model_ld.DMC` 的单参考低延迟路径，"
        "对原生 128 通道做可逆逐通道标准化。所有探针沿用冻结的 I/P Decoder 与对齐权重。"
        "RGB 潜在梯度只用于闭式通道排序。",
        f"训练统计来自 {len(train)} 个训练视频的四帧窗口，共 {sum(p['frames'] for p in train)} 帧。"
        f"每种帧型取 {sum(p['sensitivity'] for p in train)} 帧计算 RGB MSE 输入梯度平方均值。"
        "I/P 通道 PCA 及块 PCA 分别用完整训练窗口的 FP64 流式协方差和精确特征分解。"
        "FSQ 尺度取训练绝对值第 99.5 百分位除以 1.5，以已解量化的锚点残差另行校准 Hybrid。"
        "训练更新与封存 Test 像素读取均为 0。",
        f"A–D 共用 {sum(map(len, frames.values()))} 帧 R0 对齐直通冻结历史及原 Fix-A 同历史 6 帧。"
        "A 另在这 30 帧对应的 Fix-A I2048/P510 冻结历史复测。"
        "下表每个视频的首两帧构成原六帧 I/P 诊断集。",
        markdown_table(("验证视频", "原生帧索引"), [(video, ", ".join(map(str, ids))) for video, ids in frames.items()]),
        f"官方 DMC-LD 原生实际码流为 **{bitrate:.5f} bpp**，"
        f"约 **{native['kbit_per_frame']:.1f} kbit/帧**，"
        f"64 帧全局 PSNR **{native['summary'][PSNR]:.3f} dB**。"
        "来源为 Gate 0 在相同视频/GOP/QP 的封存 SPS/IP 字节测量。"
        "LPIPS-Alex 与 DISTS 按整幅原生 1080p 计算并逐帧平均。色度为 Gate 1-Fix 的 BT.709 Cb/Cr 方差保留率、MSE 和均值绝对偏差。"
        "全局 PSNR 由帧均 MSE 计算，帧均 PSNR 先对每帧换算再平均。",
        f"Gate 0 `projector_w128` 运行前后 SHA256 均为 `{projector}`。"
        f"[配置、检查点 SHA256、完整性与拟合样本]({SOURCE}provenance.json)",
    ]
    return "\n\n".join(lines)


def block_label(method, block, rank):
    labels = {
        "r0": "R0 直通", "channel_pca": f"1/16 通道 PCA {rank}D",
        "bilinear": f"1/{16 * block} 双线性 128C", "block_pca": f"1/{16 * block} 块 PCA {rank}D",
        "fix_a_lift_oracle": "Fix-A 冻结 lift 连续最小二乘", "fix_a_pass0": "Fix-A 实际 Pass 0 / K510",
        "fix_a_final": "Fix-A 实际三段 / K8192",
    }
    return labels[method]


def block_table(rows):
    data = []
    for r in rows:
        centered = r.get("latent_energy_train_centered")
        zero = 100 * r["latent_mse"] / r["latent_energy_zero"]
        data.append((
            block_label(r["method"], r["block"], r["rank"]), f"{zero:.2f}",
            f"{100 * r['latent_mse'] / centered:.2f}" if centered else "—",
            *numeric(r, ("latent_mse", PSNR, FRAME_PSNR, "lpips", "dists")),
        ))
    return markdown_table(("路径", "残余/原始能量 %", "残余/训练中心化能量 %", "潜在 MSE", "PSNR", "帧均 PSNR", "LPIPS", "DISTS"), data)


def attribution(rows):
    mapping = (
        ("R0 对齐直通", "r0", 128, 1), ("1/64 块空间约束与 128D 带宽", "block_pca", 128, 4),
        ("同网格 128D→6D", "block_pca", 6, 4), ("替换为 Fix-A 冻结 lift 与其偏置", "fix_a_lift_oracle", 6, 4),
        ("实际 Pass 0 编码、FSQ 与 Fusion", "fix_a_pass0", 6, 4),
        ("追加 Pass 1/2 直至 K8192", "fix_a_final", 6, 1),
    )
    previous, table = None, []
    gap = one(rows, method="r0")[PSNR] - one(rows, method="fix_a_final")[PSNR]
    for label, method, rank, block in mapping:
        val = one(rows, method=method, rank=rank, block=block)[PSNR]
        delta = None if previous is None else previous - val
        table.append((label, f"{val:.3f}", "—" if delta is None else f"{delta:+.3f}",
                      "—" if delta is None else f"{100 * delta / gap:+.1f}%"))
        previous = val
    return markdown_table(("依序替换的路径", "全局 PSNR / dB", "本步损失 / dB", "占完整缺口"), table)


def capacity_section(b):
    keys = ("method", "block", "rank")
    matched = pool(select(b["rows"], history=LOW, matched_six=True), keys)
    r0 = one(matched, method="r0")
    block6 = one(matched, method="block_pca", block=4, rank=6)
    lift = one(matched, method="fix_a_lift_oracle")
    final = one(matched, method="fix_a_final")
    before = r0["latent_energy_zero"]
    oracle = select(json_at(FIX / "tokenizer_oracle.json")["rows"], **{"pass": 0})
    historic = sum(r["raw_mse_after_oracle"] for r in oracle) / sum(r["raw_mse_before"] for r in oracle)
    gap = r0[PSNR] - final[PSNR]
    residual = block6["latent_mse"] / before
    recovered = (lift["latent_mse"] - block6["latent_mse"]) / before
    capacity_finding = (
        "训练 PCA 回收了部分冻结 lift 的损失，剩余误差仍主要来自低秩块表示容量。"
        if residual > recovered > 0 else "留出帧的块 PCA 与冻结 lift 排名反映拟合分布和仿射中心的共同影响。"
    )
    summaries = [
        "## Q1 · 块容量、空间和 lift",
        f"同一 Fix-A 六帧上，冻结 lift 连续最小二乘保留 **{100 * lift['latent_mse'] / before:.2f}%** 原始能量，"
        f"封存结果为 **{100 * historic:.2f}%**。训练最优的 1/64 块 6D 保留 **{100 * block6['latent_mse'] / before:.2f}%**，"
        f"同口径少 **{100 * (lift['latent_mse'] - block6['latent_mse']) / before:.2f} 个百分点**。"
        f"以训练块均值中心化后，该 6D 在六帧上仍剩 **{100 * block6['latent_mse'] / block6['latent_energy_train_centered']:.2f}%**。"
        f"{capacity_finding}块 PCA 相对冻结 lift 的 RGB PSNR 变化为 **{block6[PSNR] - lift[PSNR]:+.3f} dB**。"
        "闭式 PCA 优化的是潜在平方误差，此结果刻画训练最优基底迁移到六帧时的误差。",
        block_table(matched),
        f"同历史复测的完整缺口为 **{gap:.3f} dB**。下面沿同一序列逐项替换，损失列可加总到该缺口。"
        "第一个约束保持低分辨率网格并给每块 128 个系数，双线性 128C 与全分辨率通道截断的独立对照见上表。"
        "替换 lift 的差额包含仿射中心，实际 Pass 0 的差额包含编码器、FSQ 和 Fusion。"
        "后续阶段追回画质时，可使前一步损失占最终缺口的比例超过 100%。",
        attribution(matched),
    ]
    clean = select(b["summary"], history=CLEAN, cohort="all")
    for kind in ("I", "P"):
        summaries.extend((f"### R0 冻结历史的 {kind} 帧空间与通道对照", block_table(select(clean, frame_type=kind))))
    return "\n\n".join(summaries)


def rank_rows(rows, context, frame_type):
    selection = select(rows, history=context, frame_type=frame_type, cohort="all")
    return {
        order: [one(selection, order=order, rank=rank) for rank in RANKS]
        for order in (*ORDERS, NATIVE_ORDER)
    }


def rank_control(variants):
    base = one(variants[ORDERS[0]], rank=24)
    summary = []
    for order in (*ORDERS, NATIVE_ORDER):
        rows = variants[order]
        rank = next((r["rank"] for r in rows if r[PSNR] >= base[PSNR] and r["lpips"] <= base["lpips"] and r["dists"] <= base["dists"]), None)
        summary.append(f"{ORDER_NAMES[order]} {rank if rank is not None else '未命中'}")
    return "、".join(summary)


def rank_table(variants):
    metrics = (PSNR, FRAME_PSNR, "latent_mse", "lpips", "dists")
    rows = [
        (rank, " / ".join(f"{100 * row['variance_explained_train']:.1f}" for row in entries),
         *(cell_values(entries, field) for field in metrics))
        for rank, entries in zip(RANKS, zip(*variants.values()))
    ]
    return markdown_table(("秩", "训练解释方差 %", "PSNR / dB", "帧均 PSNR / dB", "潜在 MSE", "LPIPS", "DISTS"), rows)


def rank_shift(summaries, order):
    clean = rank_rows(summaries, CLEAN, "P")[order]
    low = rank_rows(summaries, LOW, "P")[order]
    deltas = [(r["rank"], q[PSNR] - r[PSNR]) for r, q in zip(clean, low) if r["rank"] in (6, 24, 128)]
    return f"{ORDER_NAMES[order]}在相同 P 帧的低预算历史下，" + "、".join(
        f"{k}D 的全局 PSNR 变化为 {delta:+.3f} dB" for k, delta in deltas
    ) + "。"


def sensitivity_findings(summaries):
    scopes = ((CLEAN, "I", "R0 I"), (CLEAN, "P", "R0 P"), (LOW, "P", "Fix-A P"))
    comparisons = []
    for history, kind, label in scopes:
        rows = rank_rows(summaries, history, kind)
        variance, pca, native = (one(rows[order], rank=24) for order in (*ORDERS, NATIVE_ORDER))
        comparisons.append(f"{label} 为 {pca[PSNR] - variance[PSNR]:+.3f} dB（原生敏感坐标 {native[PSNR]:.3f} dB）")
    return "保留 24 个方向时，PCA 内改用 RGB 敏感排序对全局 PSNR 的影响如下。" + "，".join(comparisons) + "。"


def sensitivity_section(a):
    checks = [r for r in a["rows"] if "full_rank_max_abs" in r]
    max_abs = max(r["full_rank_max_abs"] for r in checks)
    psnr_delta = max(abs(r["full_rank_psnr_delta"]) for r in checks)
    lines = [
        "## Q2 · I/P 有效通道与历史偏移",
        f"FP32 128D 往返覆盖 {len(checks)} 个帧与排序组合，最大潜在绝对误差 **{max_abs:.3e}**，"
        f"与 R0 的最大单帧 PSNR 差 **{psnr_delta:.3e} dB**。"
        "RGB 敏感度来自训练 RGB MSE 在 PCA 或原生坐标的梯度平方均值。"
        "下面各数值按方差 PCA / RGB 敏感度 PCA / 原生 RGB 敏感度列出。",
        sensitivity_findings(a["summary"]),
    ]
    for history, kind, label in ((CLEAN, "I", "R0 I"), (CLEAN, "P", "R0 P"), (LOW, "P", "Fix-A P510 历史 P")):
        variants = rank_rows(a["summary"], history, kind)
        lines.extend((
            f"### {label}（{variants[ORDERS[0]][0]['frames']} 帧）",
            rank_table(variants),
            f"以该组方差 PCA 的实测 24D 为参照，同时达到其 PSNR、LPIPS 和 DISTS 的最小已测秩为 {rank_control(variants)}。",
        ))
    lines.extend(rank_shift(a["summary"], order) for order in (*ORDERS, NATIVE_ORDER))
    return "\n\n".join(lines)


SPARSE_KEYS = ("layout", "order", "version", "budget")


def sparse_section(c, d, e):
    rows = pool(select(c["rows"] + d["rows"], history=CLEAN), SPARSE_KEYS)
    quality, ledger = [], []
    for layout in (FREE, *LAYOUTS):
        for order in ORDERS:
            for k in BUDGETS:
                labels = {"layout": layout, "order": order, "budget": k}
                cont = one(rows, **labels, version="continuous")
                fsq = one(rows, **labels, version="fsq_quantized")
                key = (NAMES[layout], ORDER_NAMES[order], k)
                quality.append((
                    *key, *(cell_values((cont, fsq), field) for field in CORE),
                    f"{fsq[PSNR] - cont[PSNR]:+.3f}", f"{100 * fsq['tanh_saturation_ratio']:.2f}",
                ))
                if layout != FREE:
                    ledger.append((
                        *key, f"{fsq['tokens_used']:.0f}",
                        *numeric(fsq, ("R_tokens_raw", "R_tokens_entropy_lb", "R_depth_raw", "R_depth_compressed_ref",
                                       "R_depth_entropy_lb", "R_headers_est", "R_total_raw", "R_total_entropy_ref"), 1),
                        *numeric(fsq, ("bpp_raw", "bpp_entropy_ref"), 5),
                        *numeric(fsq, ("side_info_raw_pct", "side_info_entropy_ref_pct"), 1),
                    ))
    return "\n\n".join((
        "## Q3 · 自由槽位、连续组前缀与真实 FSQ",
        "每个 I/P 独立 PCA 基底使用前 126 个有序坐标，分成 21 组，每帧共 171,360 个槽位。"
        "方差序用槽位平方能量，敏感度序用训练梯度方差加权能量。前缀布局按单位 Token 的边际收益逐步升级，"
        "固定预算里程碑保留较低预算的全部槽位和 ID。Hybrid 在 1/64 上先付 510 个锚点，再为同一锚点的残差分配全分辨率深度。"
        "真实 FSQ 使用 `[8,8,8,5,5,5]` 和离散 `codes → ids → codes`，反量化使用固定 tanh 分箱中心的逆变换。"
        "逐组 tanh 饱和率和从计数池化的六维熵见 D0-C/D0-D JSON。",
        "质量列为连续 / FSQ，Δ 为 FSQ 全局 PSNR 减去连续值。所有值使用 R0 同上下文的 30 帧。",
        sparse_findings(rows, e),
        markdown_table(("布局", "排序", "K", "PSNR / dB", "LPIPS", "DISTS", "潜在 MSE", "Δ / dB", "FSQ 饱和 %"), quality),
        "### D0-D 逐项字节账本",
        "原始 token 按 **2K B** 计费。共享深度每次消耗 4 个 Token，上限可能留空。"
        "经验六维零阶边际熵是假设已知每帧概率表的理论熵下界，未计概率表字节。"
        "深度图对 `{0,1,2,4,8,12}` 编号做固定 3-bit 打包后，用 zlib level 9 计无损参考长度。"
        "Hybrid 在 K510 同样计入全零深度图。帧头为固定估算。"
        "自由 Top-K 的坐标另按每槽 18 bit 估算。"
        "连续路径的原始字节是与对应 FSQ 共享的槽位记账口径，连续系数本身未编码。"
        "两种估算 bpp 均为实验账本，官方 DMC-LD 一行使用实际码流。比例是逐帧侧信息分数的平均。",
        markdown_table(("布局", "排序", "K", "已用", "Token 原始 B", "Token 熵 B", "深度原始 B", "深度 zlib B",
                        "深度熵 B", "帧头 B", "总原始 B", "总参考 B", "bpp 原始", "bpp 熵参考", "侧信息原始 %", "侧信息熵参考 %"), ledger),
    ))


def sparse_findings(rows, e):
    selected = {row["layout"]: row["order"] for row in e["candidates"]}
    shared_order = e["candidate_ranking"]["pure"][0]["order"]
    findings = []
    for layout in (FREE, *LAYOUTS):
        order = selected.get(layout, shared_order)
        group = select(rows, layout=layout, order=order)
        low, high = (one(group, budget=k, version="fsq_quantized") for k in (510, 8192))
        raw = one(group, budget=8192, version="continuous")
        detail = (
            f"两档原始侧信息分别为 {low['side_info_raw_pct']:.1f}% 和 {high['side_info_raw_pct']:.1f}%，"
            f"熵参考侧信息分别为 {low['side_info_entropy_ref_pct']:.1f}% 和 {high['side_info_entropy_ref_pct']:.1f}%。"
            if layout != FREE else "自由槽位的位置按原始坐标开销计费。"
        )
        findings.append(
            f"- {NAMES[layout]}（{ORDER_NAMES[order]}序）FSQ 在 K510、K8192 分别达到 {low[PSNR]:.3f}、"
            f"{high[PSNR]:.3f} dB，K8192 连续值为 {raw[PSNR]:.3f} dB。{detail}"
        )
    return "\n".join(findings)


def reference_findings(e):
    summary = e["summary"]
    pure_layout = e["candidate_ranking"]["pure"][0]["layout"]
    pure, hybrid, fix = (one(summary, layout=layout, budget=510) for layout in (pure_layout, HYBRID, FIX_A))
    pure_late, hybrid_late = (r["late_48_63"][FRAME_PSNR] for r in (pure, hybrid))
    pure_drop = pure_late - pure["early_p_01_16"][FRAME_PSNR]
    quality = [one(summary, layout=HYBRID, budget=k) for k in (4096, 8192)]
    gain = "、".join(f"{row[PSNR] - hybrid[PSNR]:+.3f}" for row in quality)
    pure_p = select(e["rows"], layout=pure_layout, budget=510, frame_type="P")
    rms = [row["reference_latent_rms"] for row in pure_p]
    return (
        f"纯前缀 P510 平均显式覆盖 **{pure['P']['reference_spatial_coverage_pct']:.2f}%** 的潜在位置。"
        f"晚期帧均 PSNR 为 **{pure_late:.3f} dB**，较自身早期变化 **{pure_drop:+.3f} dB**，"
        f"帧 31/63 的参考 RMS 为 **{pure['reference_latent_rms_31']:.3f}/{pure['reference_latent_rms_63']:.3f}**。"
        f"其全程 P 参考 RMS 在 **{min(rms):.3f}–{max(rms):.3f}** 之间，闭环主要问题表现为持续低画质。"
        f"Hybrid 晚期为 **{hybrid_late:.3f} dB**，Fix-A 同档为 **{fix['late_48_63'][FRAME_PSNR]:.3f} dB**。"
        f"Hybrid 显示扩展到 K4096/K8192 后，全局 PSNR 相对 K510 的增量分别为 **{gain} dB**，沿用相同 P510 提交。"
    )


def closed_section(e):
    values = e["summary"]
    display = []
    for row in values:
        display.append((
            NAMES[row["layout"]], row["budget"], *numeric(row, (PSNR, FRAME_PSNR, "lpips", "dists")),
            f"{row['chroma_var_retention_pct']:.1f}", *numeric(row, ("chroma_mse", "chroma_mean_bias"), 6),
            f"{row['late_48_63'][FRAME_PSNR]:.3f}", *numeric(row, ("reference_latent_rms_31", "reference_latent_rms_63")),
        ))
    p510 = []
    for row in select(values, budget=510):
        p510.append((
            NAMES[row["layout"]], f"{row['P']['reference_tokens_used']:.0f}",
            f"{row['P']['reference_spatial_coverage_pct']:.2f}",
            *numeric(row["P"], (PSNR, "lpips", "dists", "reference_latent_mse")),
            *numeric(row, ("reference_to_native_rms_31", "reference_to_native_rms_63")),
            f"{row['late_48_63'][FRAME_PSNR] - row['early_p_01_16'][FRAME_PSNR]:+.3f}",
        ))
    candidates = "、".join(
        f"{NAMES[r['layout']]}（{ORDER_NAMES[r['order']]}序）" for r in e["candidates"] if r["layout"] != FIX_A
    )
    replay = e["historical_replay_max_abs"]
    return "\n\n".join((
        "## Q4 · car-roundabout 64 帧自回归参考",
        f"闭环候选为 {candidates} 与封存 Fix-A。两类新方案各自在 D0-D 内按 I/P、五个 K、"
        "PSNR、LPIPS、DISTS 与潜在 MSE 的平均名次选出。每个候选单独推进 GOP32。"
        "两个接收闭环分别显示到 K2048 与 K8192 并提交 I2048/P510，逐帧对比潜在、投影特征和时序记忆，所有张量逐位相等。"
        "I 帧在低显示档仍接收 I2048 用于提交。",
        f"Fix-A K8192 复跑与封存逐帧记录的最大差为 RGB MSE `{replay['mse']:.3e}`、"
        f"参考 RMS `{replay['reference_latent_rms']:.3e}`、原生潜在 RMS `{replay['native_aligned_latent_rms']:.3e}`。",
        reference_findings(e),
        markdown_table(("布局", "显示 K", "全局 PSNR", "帧均 PSNR", "LPIPS", "DISTS", "CbCr 方差保留 %", "CbCr MSE",
                        "CbCr 均值偏差", "48–63 帧均 PSNR", "RMS 帧 31", "RMS 帧 63"), display),
        "### P510 参考与晚期变化",
        markdown_table(("布局", "P 实用参考 Token", "P 位置覆盖 %", "P 全局 PSNR", "P LPIPS", "P DISTS", "P 参考潜在 MSE",
                        "RMS/同历史原生 帧 31", "RMS/同历史原生 帧 63", "晚期减早期 PSNR / dB"), p510),
        "早期为第 1–16 帧，晚期为第 48–63 帧。RMS 与参考潜在误差同时衡量幅度和失真，"
        "该长期比较覆盖一个验证视频。",
    ))


def quality_wins(left, right):
    return sum(
        (left[key] > right[key]) if key == PSNR else (left[key] < right[key])
        for key in CORE
    )


def decision_metrics(b, c, d, e):
    summary = e["summary"]
    pure, hybrid = (e["candidate_ranking"][family][0] for family in ("pure", "hybrid"))
    at_510 = [one(summary, layout=layout, budget=510) for layout in (pure["layout"], HYBRID, FIX_A)]
    at_8192 = [one(summary, layout=layout, budget=8192) for layout in (pure["layout"], HYBRID, FIX_A)]
    block6 = one(
        pool(select(b["rows"], history=CLEAN), ("method", "block", "rank")), method="block_pca", block=4, rank=6,
    )
    sparse = pool(select(c["rows"] + d["rows"], history=CLEAN, version="continuous"), SPARSE_KEYS)
    best_continuous = max(select(sparse, budget=8192), key=lambda row: row[PSNR])
    fsq = pool(select(d["rows"], history=CLEAN, version="fsq_quantized"), SPARSE_KEYS)
    costs = [one(fsq, layout=r["layout"], order=r["order"], budget=8192) for r in (pure, hybrid)]
    r0 = summarize(select(b["rows"], history=CLEAN, method="r0"))
    return at_510, at_8192, block6, best_continuous, costs, r0


def decision_section(b, c, d, e):
    low, high, block6, best, costs, r0 = decision_metrics(b, c, d, e)
    hybrid_stability = (
        low[1]["P"]["reference_latent_mse"] < low[0]["P"]["reference_latent_mse"]
        and low[1]["late_48_63"][FRAME_PSNR] > low[0]["late_48_63"][FRAME_PSNR]
    )
    hybrid_display = quality_wins(high[1], high[0]) >= 3 and quality_wins(high[1], high[2]) >= 3
    decision = "B" if hybrid_stability and hybrid_display else "A" if quality_wins(block6, best) >= 3 else "C"
    verdicts = {
        "A": "优先保留三段式 6D，下一步聚焦初始化和局部非线性重建。",
        "B": "优先验证情景 B 的 Hybrid 设计，下一阶段应验证参考稳定性与字节账本。",
        "C": "优先评估情景 C 的可逆通道变换和空间重建，随后再评估骨干微调的必要性。",
    }
    matrix = []
    for index, (lower, upper) in enumerate(zip(low, high)):
        cost = costs[index] if index < len(costs) else None
        matrix.append((
            NAMES[lower["layout"]], f"{lower['P'][PSNR]:.3f}", f"{lower['late_48_63'][FRAME_PSNR]:.3f}",
            *numeric(upper, (*CORE, "reference_latent_rms_31", "reference_latent_rms_63")),
            "—" if cost is None else f"{cost['side_info_entropy_ref_pct']:.2f}",
        ))
    report = "\n\n".join((
        "## Q5 · 决策矩阵",
        f"**建议走情景 {decision}。** {verdicts[decision]} 本次执行截至 Diag-0。",
        markdown_table(("布局", "闭环 P510 PSNR", "K510 晚期帧均 PSNR", "K8192 PSNR", "K8192 LPIPS", "K8192 DISTS",
                        "K8192 潜在 MSE", "RMS 帧 31", "RMS 帧 63", "K8192 同历史侧信息熵参考 %"), matrix),
        markdown_table(("情景", "实测证据"), [
            ("A · 保留三段式 6D", f"冻结 R0 历史的 1/64 块 6D 为 {block6[PSNR]:.3f} dB、潜在 MSE {block6['latent_mse']:.4f}。"
             f"K8192 连续上限中全局 PSNR 最高的路径为 {NAMES[best['layout']]}（{ORDER_NAMES[best['order']]}序），"
             f"{best[PSNR]:.3f} dB、潜在 MSE {best['latent_mse']:.4f}。"),
            ("B · Hybrid 锚点与稀疏增强", f"Hybrid 相对纯前缀将 P510 参考潜在 MSE 从 {low[0]['P']['reference_latent_mse']:.4f} "
             f"变为 {low[1]['P']['reference_latent_mse']:.4f}，晚期 K510 帧均 PSNR 变化 "
             f"{low[1]['late_48_63'][FRAME_PSNR] - low[0]['late_48_63'][FRAME_PSNR]:+.3f} dB。"
             f"K8192 对纯前缀与 Fix-A 分别改善 {quality_wins(high[1], high[0])} / {quality_wins(high[1], high[2])} 个质量指标（各 4 项）。"),
            ("C · 先改表示与空间重建", f"最佳 K8192 连续路径距 R0 直通仍有 {r0[PSNR] - best[PSNR]:.3f} dB，"
             f"LPIPS 为 {best['lpips']:.3f} 对 {r0['lpips']:.3f}，DISTS 为 {best['dists']:.3f} 对 {r0['dists']:.3f}。"),
        ]),
        "选择依据包括 P510 的参考误差与晚期质量、K8192 的四项重建指标，以及同历史侧信息与参考 RMS 的实测代价。"
        "闭环视频也参加了同历史候选比较，长程结果来自这条 64 帧序列。",
    ))
    return report, {
        "scenario": decision,
        "hybrid_beats_pure_on_reference_mse_and_late_psnr": hybrid_stability,
        "hybrid_wins_at_least_three_high_budget_metrics_against_both_controls": hybrid_display,
        "best_continuous_k8192": best,
        "best_block6": block6,
        "r0_direct": r0,
    }


def source_section():
    section = ["## 图表、原始数据与复现"]
    for name, label in (
        ("rank", "I/P 通道秩与敏感度"), ("spatial", "空间与通道容量"),
        ("free_quality_bytes", "自由 Top-K 质量与字节"), ("depth_quality_bytes", "深度前缀质量与字节"),
        ("closed_loop", "64 帧 PSNR 与参考 RMS"), ("crop_f063", "第 63 帧原尺寸像素裁剪"),
    ):
        section.append(f"![{label}]({SOURCE}figures/{name}.png)")
    section.append("\n".join(
        f"- D0-{probe} [{stem}.json]({SOURCE}{stem}.json) · [逐帧 CSV]({SOURCE}{stem}.csv)"
        for probe, stem in PROBES.items()
    ))
    section.extend((
        f"[训练拟合参数]({SOURCE}train_fit.pt) · [机器决策]({SOURCE}decision.json) · [完整测试]({SOURCE}verification.json) · "
        f"[工作记录]({SOURCE}work_log.md)",
        "```bash\nOMP_NUM_THREADS=4 .venv-native/bin/python scripts/aip_vtok/run_diag0.py\n```",
    ))
    return "\n\n".join(section)


def generate_report():
    from .plots import build_figures

    a, b, c, d, e = (read_probe(k) for k in PROBES)
    meta = json_at(OUTPUT / "provenance.json")
    if not {"matched", "closed"} <= meta["integrity"].keys():
        raise ValueError("report requires complete audited matched and closed-loop probes")
    native = native_control()
    for label, artifact in (("D", d), ("E", e)):
        if artifact.get("dmc_ld_actual_bitstream") != native:
            artifact["dmc_ld_actual_bitstream"] = native
            write_json(OUTPUT / f"{PROBES[label]}.json", artifact)
    decision, machine = decision_section(b, c, d, e)
    report = "\n\n".join((
        introduction(meta, native), capacity_section(b), sensitivity_section(a), sparse_section(c, d, e),
        closed_section(e), decision, source_section(),
    ))
    build_figures(a, b, c, d, e, native)
    write_json(OUTPUT / "decision.json", machine)
    path = resolve("docs/aip_vtok_diag0_report.md")
    temp = path.with_suffix(".md.tmp")
    temp.write_text(report + "\n", encoding="utf-8")
    temp.replace(path)
