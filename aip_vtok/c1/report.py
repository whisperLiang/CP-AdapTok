"""Generate the C1 Markdown reports and figures from raw JSON result records."""

from statistics import mean

from aip_vtok.diagnostics.diag0.io import json_at, native_control
from aip_vtok.diagnostics.diag0.report import select
from aip_vtok.runtime import ROOT

from .artifacts import LOCKED, OUTPUT, THRESHOLD_DB, group_summary

REPORT = ROOT / "docs/aip_vtok_c1_report.md"
WORK_LOG = ROOT / "docs/aip_vtok_c1_work_log.md"
PSNR = "psnr_global_mse"
RESULT_LINK = "../results/aip_vtok/c1/"


def load(name):
    path = OUTPUT / f"{name}.json"
    return json_at(path) if path.is_file() else {}


def table(headers, rows):
    return ["| " + " | ".join(headers) + " |",
            "| " + " | ".join("---" for _ in headers) + " |",
            *("| " + " | ".join(map(str, row)) + " |" for row in rows), ""]


def quality(row):
    return (row["frames"], f'{row[PSNR]:.3f}', f'{row["psnr_frame_mean"]:.3f}', f'{row["lpips"]:.4f}',
            f'{row["dists"]:.4f}', f'{row["chroma_var_retention_pct"]:.1f}',
            f'{row["chroma_mse"]:.6f}', f'{row["chroma_mean_bias"]:.6f}')


QUALITY_HEADER = ("帧数", "全局 PSNR / dB", "帧均 PSNR / dB", "LPIPS-Alex", "DISTS", "CbCr 方差 %", "CbCr MSE",
                  "CbCr 均值绝对偏差")


def link(name, label=None):
    return f"[{label or name}]({RESULT_LINK}{name})"


def heading():
    return ["# AIP-VTok C1 实验报告", "",
            "基线为 `3cb3641`。冻结 DMCI / DMC-LD、I 帧 256→128 白化 PCA、P 帧对角标准化和 Gate 0 W128 投影。",
            "原始数据为 1920×1080，潜在网格为 68×120，QP32、GOP32。封存 Test 图像读取数为零。",
            "174 帧验证集曾用于同一研究的历史开发。本文将它作为固定验证队列，五个候选校准视频来自 Train。",
            "", "B0/B1 使用 K=32,640 做稠密通道诊断。B2/B3 使用 K=510、1024、2048、4096、8192 做稀疏重建。",
            "全局 PSNR 从跨帧平均 RGB MSE 计算。LPIPS-Alex、DISTS 和 BT.709 Cb/Cr 在完整图像上计算。",
            ""]


def gate_table(diag, dense, sparse):
    def gate(data, yes, no):
        return (yes if data["passes_for_c1b"] else no) if data else "待完成"

    return table(("阶段", "结论"), [
        ("Diag-0.5", gate(diag, "完成，达到 C1-B 的前置门槛", "完成，未达到 C1-B 的前置门槛")),
        ("C1-A", gate(dense, "双种子通过", "双种子证据未达门槛")),
        ("C1-B", "双种子训练与评估完成" if sparse else (
            "门控停止，训练 0 步" if (diag and not diag["passes_for_c1b"]) or (dense and not dense["passes_for_c1b"])
            else "待完成")),
    ])


def lock_section(lock):
    winner = lock["winner"]
    rows = []
    for budget in (510, 1024, 2048, 4096, 8192):
        slots = lock["slots"][:budget]
        rows.append((budget, *(sum(s // 8160 == g for s in slots) for g in range(4))))
    return [
        "## Phase 1 · Train 校准和扫描表", "",
        f"P 帧锁为 **{winner['p_coordinate']}**，扫描表 **{winner['schedule']}**，"
        f"核宽 **σ={winner['sigma']:.1f}**。I 帧始终使用封存 PCA 对齐坐标。",
        f"候选按 I/P × 两种参考历史 × 五档 K 的 FSQ B2 全局 PSNR 等权平均排名。优胜得分为 "
        f"{winner['mean_slice_global_psnr']:.3f} dB，Train 校准的 Fix-A 历史 P510 插值收益为 "
        f"{winner['matched_low_p510_gain_db']:+.3f} dB。",
        f"拟合遍历 {len(lock['fit_plan'])} 个训练视频。另预留 {len(lock['calibration_plan'])} 个训练视频做候选筛选，"
        "它们也与 Gate 0 PCA 拟合视频互斥。",
        "S1 先填满 G0。S2a 首档给 G0，S2b 首档按同坐标成对交错 G0/G1，后续按终点剩余配额交错。"
        "S3 以 I/P 等权归一化后的 Train 敏感度能量固定配额，共用一个扫描表。",
        "Halton 离散坐标保留首次命中，尝试 16,320 个样本后按行主序补齐。",
        "", *table(("K", "G0", "G1", "G2", "G3"), rows), *calibration_table(),
        f"{link('scan_lock.json', '扫描表、拟合统计与视频清单')} · "
        f"{link('diag05/calibration.json', '候选排名与校准原始 JSON')} · {link('diag05/calibration.csv', '校准 CSV')}",
        "",
    ]


def calibration_table():
    ranked = load("diag05/calibration")["ranking"]
    cases = {}
    for row in ranked:
        cases.setdefault((row["p_coordinate"], row["schedule"]), row)
    return table(("P 坐标", "扫描表", "最佳 σ", "校准均分 / dB", "同历史 P510 对零填充 / dB"),
                 ((p, scan, r["sigma"], f'{r["mean_slice_global_psnr"]:.3f}',
                   f'{r["matched_low_p510_gain_db"]:+.3f}') for (p, scan), r in cases.items()))


def rank_gaps(summary):
    full = {(r["history"], r["frame_type"], r["coordinate"]): r[PSNR]
            for r in select(summary, rank=128, version="cont")}
    for row in select(summary, rank=24, version="cont"):
        key = row["history"], row["frame_type"], row["coordinate"]
        value = row[PSNR]
        yield (*key, row["fill"], f"{value:.3f}", f"{full[key]:.3f}", f"{value - full[key]:+.3f}")


def capacity_section(rank, dense):
    fields = ("coordinate_max_abs", "aligned_max_abs", "native_max_abs")
    maximum = max(row[field] for row in rank["rows"] for field in fields if field in row)
    return [
        "### D0.5-A · 原生 P 旁路与 Top-24 画质差", "",
        f"复用 Diag-0 的 60 个上下文。P 原生潜在直接取自 DMC-LD 编码器，敏感度梯度为 "
        f"`dL/dY_native = (dL/dY_aligned) / p_std`。连续 128D 往返 FP32 最大绝对误差为 **{maximum:.2e}**。",
        "连续与 FSQ 各扫七档秩。下表给出连续 Top-24 对完整 128D 的同历史差。"
        "FSQ 128D 最后一个 6D 组的四个补位只参加量化成组。",
        "", *table(("历史", "帧型", "坐标", "补值", "Top-24 / dB", "128D / dB", "Δ / dB"), rank_gaps(rank["summary"])),
        f"![P 帧秩曲线]({RESULT_LINK}figures/rank_p.png)",
        "",
        "图中各点为该历史下 24 个 P 帧的全局 MSE PSNR，曲线描述固定帧队列。",
        "",
        "### D0.5-B · 稠密通道截断", "",
        *table(("历史", "帧型", "坐标", "方法", *QUALITY_HEADER),
               ((r["history"], r["frame_type"], r["coordinate"], r["method"], *quality(r)) for r in dense["summary"])),
        *dense_health(dense["rows"]),
        f"{link('diag05/rank.json', '全部秩、补值与质量指标')} · {link('diag05/rank.csv', '秩 CSV')} · "
        f"{link('diag05/dense.json', 'B0 JSON')} · {link('diag05/dense.csv', 'B0 CSV')}", "",
    ]


def dense_health(rows):
    out = []
    keys = dict.fromkeys((r["history"], r["frame_type"]) for r in rows)
    for history, kind in keys:
        scope = select(rows, history=history, frame_type=kind, version="fsq")
        for group in range(4):
            out.append((history, kind, group, *group_fsq_health([r["fsq"][group] for r in scope])))
    return ["按帧宏平均的 B0-FSQ 码分布。饱和率按量化前实际 tanh 导数小于 0.05 定义，熵单位为 bit。",
            "", *table(("历史", "帧型", "组", "边界码", "tanh 饱和", "各维熵"), out)]


def group_fsq_health(rows):
    entropy = (mean(values) for values in zip(*(r["dim_entropy_bits"] for r in rows)))
    return (f'{mean(r["boundary_code_ratio"] for r in rows):.2%}',
            f'{mean(r["tanh_saturation_ratio"] for r in rows):.2%}', " / ".join(f"{v:.2f}" for v in entropy))


def drift_stats(rows):
    by_frame = {r["frame"]: r for r in rows}
    late = mean(by_frame[i]["psnr"] for i in range(48, 64))
    early = mean(by_frame[i]["psnr"] for i in range(1, 17))
    return (f'{mean(r["reference_latent_mse"] for r in rows):.4f}',
            f'{by_frame[31]["reference_latent_rms"]:.4f}', f'{by_frame[63]["reference_latent_rms"]:.4f}',
            f"{late:.3f}", f"{late - early:+.3f}")


def reference_drift(rows):
    p_rows = [r | {"seed": r.get("seed", 0)} for r in select(rows, budget=510, frame_type="P")]
    keys = sorted({(r["method"], r["seed"]) for r in p_rows})
    return table(("方法", "种子", "P 参考潜在 MSE", "帧 31 RMS", "帧 63 RMS", "晚期帧均 PSNR / dB",
                  "晚期减早期 / dB"),
                 ((method, seed or "—", *drift_stats(select(p_rows, method=method, seed=seed)))
                  for method, seed in keys))


def prefix_table(backslides):
    return table(("方法", "帧型", "K 变化", "PSNR 回退次数 / 比较数", "回退比例", "最大回退 / dB"),
                 ((r["method"], r.get("frame_type", "全部"), f'{r["lower"]}→{r["upper"]}',
                   f'{r["violations"]}/{r["pairs"]}', f'{r["fraction"]:.1%}', f'{r["max_backslide"]:.3f}')
                  for r in backslides if r["metric"] == "psnr" and r.get("frame_type", "P") == "P"))


def sparse_section(closed, diag):
    rows = closed["rows"]
    summary = group_summary(rows, ("method", "budget"))
    return [
        "### D0.5-C · 0 字节扫描表与低预算独立闭环", "",
        f"两条独立 64 帧闭环均按 I2048/P510 提交，发送端与接收端参考特征、记忆及潜在逐帧相等。"
        f"P510 上 B2 比 B2-ZeroFill 高 **{diag['closed_p510_gain_db']:+.3f} dB**，"
        f"门槛为 Train 同历史和 64 帧独立闭环各 ≥{diag['minimum_gain_db']:.1f} dB。",
        "接收端逐组插值并硬保留实收值。连续与 FSQ 同历史对照覆盖锁定扫描的全部档位。",
        "", *table(("方法", "显示 K", *QUALITY_HEADER), ((r["method"], r["budget"], *quality(r)) for r in summary)),
        *reference_drift(rows),
        "P 帧相邻显示档的 PSNR 回退如下。严格的整数 FSQ 前缀一致性逐帧核对。"
        "质量回退按自然输出统计，完整记录也含 LPIPS 和 DISTS。",
        "", *prefix_table(closed["quality_backslides"]),
        f"![固定参考独立闭环]({RESULT_LINK}figures/diag05_closed.png)",
        "",
        "图中各线包含同一段 car-roundabout 的 64 帧，统计量为全局 MSE PSNR。",
        "",
        "扫描表提前共享，每帧 0 字节。账本为 16 B 帧头、每包 8 B 头、每 ID 2 B，包最多 128 个 ID，"
        "各档在包边界结束。I 帧显示 K 小于 2048 时，实际接收仍含 2048 个槽位。",
        "原始字段分别列出显示与实际接收的未压缩 bpp，以及逐帧 ID 经验熵估算加包头的 bpp。",
        "",
        f"{link('diag05/matched_sparse.json', '同历史连续与 FSQ')} · {link('diag05/matched_sparse.csv', '同历史 CSV')} · "
        f"{link('diag05/closed.json', '闭环与包头账本')} · {link('diag05/closed.csv', '闭环 CSV')}", "",
    ]


def training_rows(phase):
    records = []
    for seed in (1, 2):
        run = load(f"{phase}/seed{seed}_training")
        if run:
            records.append((seed, run["steps"], run["target"], run["identity"]["rollout"],
                            f'{run["seconds"]:.1f}', f'{run["updates_per_second"]:.3f}',
                            f'{run["peak_allocated_mib"]:.0f}', run["videos"], str(run["start_phases"])))
    return records


def training_table(phase):
    return table(("种子", "已完成步数", "目标步数", "展开", "更新时间 / s", "步 / s", "峰值分配 / MiB", "Train 视频",
                  "GOP 窗口相位"), training_rows(phase))


def dense_section(dense, data):
    return [
        "## Phase 2 · C1-A 稠密通道补全", "",
        "两种种子均从零输出头开始，输入为量化后的稠密 Top-24，网络只写入缺失通道。"
        "P 帧的条件输入取自接收端已有的冻结因果上下文。",
        "训练使用 RGB L1/MSE、BT.709 多尺度 Cb/Cr、四个局部裁剪加全局 LPIPS-Alex、"
        "同历史敏感度加权潜在锚定和同历史原生特征监督。参考在每帧提交时截断梯度。",
        "原生特征目标由同一接收历史下的完整潜在解码而来，I 帧使用 256D，P 帧使用标准化前的 128D。"
        "参考损失经冻结 W128 投影器回传至恢复网络，下一帧读取独立保存的 detached clone。",
        "损失权重为 RGB L1 1.0、RGB MSE 1.0、Cb/Cr 0.2、LPIPS 0.15、潜在锚定 0.05、参考 0.15。"
        "冻结 Decoder 的梯度路径使用 BF16 autocast 与非重入检查点，参考生成和验证使用 FP32。",
        "", *training_table("c1a"),
        f"晋级依据为两个种子在相同 B0-FSQ 历史和各自独立闭环上均改善至少 {THRESHOLD_DB:.1f} dB，"
        "且 LPIPS 均下降。B1-cont-diag 使用连续输入做额外诊断。",
        "", *table(("种子", "历史", "ΔPSNR / dB", "LPIPS 改善", "DISTS 改善", "通过"),
                   ((r["seed"], r["history"], f'{r["psnr_gain_db"]:+.3f}', f'{r["lpips_gain"]:+.4f}',
                     f'{r["dists_gain"]:+.4f}', "是" if r["passed"] else "否") for r in dense["gains"])),
        *table(("历史", "方法", "种子", *QUALITY_HEADER),
               ((r["history"], r["method"], r["seed"] or "—", *quality(r)) for r in data["summary"])),
        f"![稠密补全的配对增益]({RESULT_LINK}figures/dense_gain.png)",
        "",
        "图中每个符号为一个种子在 174 帧队列上的全局 PSNR 差。两种符号对应两种参考历史。",
        "",
        f"{link('c1a/validation.json', '174 帧逐帧验证')} · {link('c1a/validation.csv', '验证 CSV')}",
        ""]


def stop_section(data, diag):
    same = select(data["summary"], history="same_context_b0_fsq_history", seed=0)
    scores = {r["method"]: r[PSNR] for r in same}
    loss = scores["B0-cont"] - scores["B0-fsq"]
    return [
        "### 门控停靠点", "",
        f"同历史下，稠密 B0 的 FSQ 输入比连续输入低 **{loss:.3f} dB**。"
        "恢复网络保留实收 FSQ 值，两个种子改善了 LPIPS 与 DISTS，"
        "像素误差的改善幅度低于预定门槛。",
        f"稀疏端的独立闭环显示，高斯插值在 P510 比零填充高 **{diag['closed_p510_gain_db']:.3f} dB**。"
        "这项证据来自固定插值基线，属于空间采样和重建诊断。",
        "实验停在 C1-A。C1-B 训练步数为 0。",
        "",
        f"{link('c1a/decision.json', '双种子门控原始记录')} · {link('c1b_gate.log', 'C1-B 入口拒绝记录')}",
        "",
    ]


def sparse_training_section():
    data = load("c1b/validation")
    closed = load("c1b/closed")
    if not data or not closed:
        return []
    lines = ["## Phase 3 · C1-B 联合空间与通道补全", "",
             "重新零初始化与 C1-A 同规格的恢复网络。各训练帧监督参考档、中间档和 K8192，"
             "同历史原生特征监督只作用于 I2048/P510。GOP 内的训练窗口从前导真帧暖启动。",
             "", *training_table("c1b")]
    for title, artifact in (("174 帧同历史", data), ("64 帧独立闭环", closed)):
        lines += [f"### {title}", "",
                  *table(("方法", "种子", "显示 K", *QUALITY_HEADER),
                         ((r["method"], r["seed"] or "—", r["budget"], *quality(r)) for r in artifact["summary"])),
                  *table(("对照", "种子", "K", "ΔPSNR / dB", "LPIPS 改善", "DISTS 改善"),
                         ((f'B3 − {r["baseline"]}', r["seed"], r["budget"], f'{r["psnr_gain_db"]:+.3f}',
                           f'{r["lpips_gain"]:+.4f}', f'{r["dists_gain"]:+.4f}') for r in artifact["gains"]))]
    return lines + [
        *reference_drift(closed["rows"]),
        "固定参考闭环的逐帧质量回退比例与幅度列在原始 JSON 的 `quality_backslides` 字段。",
        "",
        f"![稀疏补全独立闭环]({RESULT_LINK}figures/c1b_closed.png)", "",
        "图中每条曲线含同一段 64 帧，种子分别列示。Fix-A 复用封存的独立闭环。",
        "",
        f"{link('c1b/validation.json', '174 帧 JSON')} · {link('c1b/validation.csv', '174 帧 CSV')} · "
        f"{link('c1b/closed.json', '64 帧 JSON')} · {link('c1b/closed.csv', '64 帧 CSV')}", "",
    ]


def native_section():
    native = native_control()
    return ["## 原生 DMC-LD 实际码流", "",
            f"封存的 car-roundabout 64 帧原生 SPS/IP 码流为 **{native['summary'][PSNR]:.3f} dB @ "
            f"{native['bpp']:.5f} bpp**，共 {native['total_bytes']:,} B。"
            "C1 的未压缩账本和经验熵估算使用各自的率标签。",
            "",
            "[封存的原生逐帧字节账本](../results/aip_vtok/native_bitstream_control.json)", ""]


def work_log(diag, dense, sparse):
    steps = ["# AIP-VTok C1 工作记录", "",
             "时间使用 Asia/Shanghai。源分支基线为 `3cb3641`，子分支为 `experiment/aip-vtok-c1`。", "",
             "- 按 ub-mode 先运行基线回归并记录 C1 测试的初次失败，随后实现固定扫描、物理 P 坐标旁路、"
             "FSQ 整数往返和参考状态边界。",
             "- 核对 DMC-LD 接入、I/P 对齐契约、Fix-A 第 1000 步和 Gate 0 W128 封存权重。"
             "所有新权重归入 `checkpoints/aip_vtok/c1/`。"]
    lock = load("scan_lock")
    if lock:
        steps.append(f"- {lock['locked_at']} 完成 Train 拟合与校准。在任何验证像素读取前封存了扫描表和门槛。")
    for phase, record in (("Diag-0.5", diag), ("C1-A", dense), ("C1-B", sparse)):
        if record:
            steps.append(f"- {record['finished']} 完成 {phase} 的测量和门控记录。")
    steps.extend(audit_log())
    return steps + ["", "复现命令依阶段顺序运行，可使用已有的原子检查点续跑。", "", "```bash",
                    "OMP_NUM_THREADS=4 .venv-native/bin/python scripts/aip_vtok/run_diag05.py",
                    "OMP_NUM_THREADS=4 .venv-native/bin/python scripts/aip_vtok/run_c1a.py",
                    "OMP_NUM_THREADS=4 .venv-native/bin/python scripts/aip_vtok/run_c1b.py",
                    "OMP_NUM_THREADS=4 .venv-native/bin/python scripts/aip_vtok/run_diag05.py --stage report",
                    "```", ""]


def audit_log():
    steps = []
    for name in ("diag05_calibrate", "diag05_all", "c1a", "c1b"):
        audit = load(f"{name}_audit")
        if audit:
            n = audit["protected"]["files"]
            steps.append(f"- {name} 使用 {audit['gpu']['gpu'].split(',')[0]}，"
                         f"前后核对 {n} 个历史文件及投影 SHA256。`{audit['projector_after']['sha256']}`。"
                         f"开始 {audit['start']}，结束 {audit['end']}。")
    for phase in ("c1a", "c1b"):
        for length in (2, 4, 8):
            record = load(f"{phase}/smoke_{length}f")
            if record and "initial_loss" in record:
                steps.append(f"- {phase} {length} 帧短闭环更新 {len(record['steps'])} 步，"
                             f"固定裁剪损失 {record['initial_loss']:.5f} → {record['final_loss']:.5f}，"
                             f"步时中位数 {record['median_seconds_per_update']:.3f} s，"
                             f"峰值分配 {record['peak_allocated_mib']:.0f} MiB。")
    verification = load("verification")
    if verification:
        steps.append(f"- 联合回归 {verification['passed']} 项通过，原有 251 项。"
                     f"变异测试 {verification['mutations_killed']}/{verification['mutations_total']}。"
                     "命名工具未安装，检查采用覆盖率、Radon、AST 相似度和隔离变异。")
        steps.append(f"- {verification['fit_replay_equal_tensors']} 个 Train 拟合张量逐位复现。"
                     f"另逐条复现校准记录 {verification['fit_replay_calibration_rows']:,} 条，"
                     f"复核 C1-A 冒烟更新 {verification['dense_exact_smoke_replay_updates']} 步和原子续跑。"
                     "三张图的 PDF 字号和碰撞检查通过，图像经目视检查。")
        steps.append(f"- 代码检查保留各函数的 CRAP 明细，含门控停止的 C1-B 路径。"
                     f"{link('quality_audit.json', '复杂度与重复检查')} · {link('verification.json', '验收记录')}")
    if load("c1a/decision") and (OUTPUT / "c1b_gate.log").is_file():
        steps.append("- C1-B 入口在加载 H100 或创建训练权重前按 C1-A 门槛返回，确认训练步数为 0。")
    return steps


def render_dense(dense, diag):
    from .plots import dense_plot

    validation = load("c1a/validation")
    lines = dense_section(dense, validation)
    dense_plot(dense["gains"], THRESHOLD_DB)
    return lines if dense["passes_for_c1b"] else lines + stop_section(validation, diag)


def generate():
    from .plots import budget_plot, rank_plot

    lock = json_at(LOCKED) if LOCKED.is_file() else {}
    diag, dense, sparse = (load(f"{phase}/decision") for phase in ("diag05", "c1a", "c1b"))
    rank, dense_base, closed = (load(f"diag05/{name}") for name in ("rank", "dense", "closed"))
    lines = heading() + gate_table(diag, dense, sparse)
    if lock:
        lines += lock_section(lock)
    if rank:
        lines += capacity_section(rank, dense_base)
        rank_plot(rank["summary"])
    if diag and closed:
        lines += sparse_section(closed, diag)
        budget_plot(group_summary(closed["rows"], ("method", "budget")), "diag05_closed")
    if dense:
        lines += render_dense(dense, diag)
    if sparse:
        lines += sparse_training_section()
        budget_plot(load("c1b/closed")["summary"], "c1b_closed")
    lines += native_section()
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    WORK_LOG.write_text("\n".join(work_log(diag, dense, sparse)), encoding="utf-8")
