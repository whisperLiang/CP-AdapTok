# AIP-VTok C1-Followup 工作记录

使用 Asia/Shanghai 时间。基线提交 `a0f5eb82bf138ac9ed8fa44413261c6a8dfe9dd6`，分支 `experiment/aip-vtok-c1-followup`。

- 按 ub-mode 先写整数往返、重放、数据边界与门槛测试，记录初次失败后再实现。
- 对齐历史训练计划，按图像哈希找出能用于候选拟合的未用 Train 片段。
- 2026-11-19T19:10:07+08:00 锁定门槛。2026-11-19T19:11:28+08:00 锁定 Train 参数和候选。
- 基线逐帧匹配通过后对同一批 174 帧运行其他稠密候选。
- 每个训练里程碑使用独立目录和原子检查点。下一段更新依赖前一门槛，稀疏入口再次验证双种子记录。

| 阶段 | 开始 | 结束 | 实验阶段退出码 | Train 读取帧数 | Validation 读取帧数 | Test 读取帧数 |
| --- | --- | --- | --- | --- | --- | --- |
| [calibrate](../results/aip_vtok/c1_followup/audits/calibrate_01.json) | 2026-11-19T19:10:05+08:00 | 2026-11-19T19:11:29+08:00 | 0 | 34 | 0 | 0 |
| [replay](../results/aip_vtok/c1_followup/audits/replay_01.json) | 2026-11-19T19:12:55+08:00 | 2026-11-19T19:13:55+08:00 | 0 | 0 | 174 | 0 |
| [validate](../results/aip_vtok/c1_followup/audits/validate_01.json) | 2026-11-19T19:13:59+08:00 | 2026-11-19T19:16:05+08:00 | 0 | 0 | 174 | 0 |
| [train](../results/aip_vtok/c1_followup/audits/train_01.json) | 2026-11-19T19:16:09+08:00 | 2026-11-19T19:29:21+08:00 | 0 | 4000 | 348 | 0 |
| [all](../results/aip_vtok/c1_followup/audits/all_01.json) | 2026-11-19T19:32:52+08:00 | 2026-11-19T19:33:03+08:00 | 0 | 0 | 0 | 0 |
| [all](../results/aip_vtok/c1_followup/audits/all_02.json) | 2026-11-19T19:42:09+08:00 | 2026-11-19T19:42:20+08:00 | 0 | 0 | 0 | 0 |

## 执行命令

- 创建实验分支，退出码 0。

  `git switch -c experiment/aip-vtok-c1-followup a0f5eb82bf138ac9ed8fa44413261c6a8dfe9dd6`
- ub-mode 初次失败，退出码 2。

  `.venv-native/bin/python -m pytest -q tests/aip_vtok/test_c1_followup.py`
- 基础契约首次通过，退出码 0。

  `OMP_NUM_THREADS=4 .venv-native/bin/python -m pytest -q tests/aip_vtok/test_c1_followup.py`
- calibrate，退出码 0。

  `OMP_NUM_THREADS=4 .venv-native/bin/python -u scripts/aip_vtok/run_c1_followup.py --stage calibrate`
- replay，退出码 0。

  `OMP_NUM_THREADS=4 .venv-native/bin/python -u scripts/aip_vtok/run_c1_followup.py --stage replay`
- validate，退出码 0。

  `OMP_NUM_THREADS=4 .venv-native/bin/python -u scripts/aip_vtok/run_c1_followup.py --stage validate`
- train，退出码 0。

  `OMP_NUM_THREADS=4 .venv-native/bin/python -u scripts/aip_vtok/run_c1_followup.py --stage train`
- 隔离契约变异，退出码 1。

  `env PYTHONPATH=. OMP_NUM_THREADS=4 .venv-native/bin/python /tmp/c1_followup_mutations.py`
- 新增用例，退出码 0。

  `env OMP_NUM_THREADS=4 .venv-native/bin/python -m pytest -q tests/aip_vtok/test_c1_followup.py`
- Ruff，退出码 0。

  `.venv-native/bin/ruff check aip_vtok/c1_followup scripts/aip_vtok/run_c1_followup.py tests/aip_vtok/test_c1_followup.py`
- 隔离契约变异复验，退出码 0。

  `env PYTHONPATH=. OMP_NUM_THREADS=4 .venv-native/bin/python /tmp/c1_followup_mutations.py`
- 全流程续跑及最终稀疏门控，退出码 1。

  `env OMP_NUM_THREADS=4 COVERAGE_FILE=results/aip_vtok/c1_followup/.coverage.stages .venv-native/bin/coverage run --branch --source=aip_vtok.c1_followup scripts/aip_vtok/run_c1_followup.py --stage all`
- 未启动里程碑的报告复现，退出码 1。

  `env OMP_NUM_THREADS=4 .venv-native/bin/python -m pytest -q tests/aip_vtok/test_c1_followup.py -k report_renders_unstarted_seed_and_final_steps`
- 全流程和报告复验，退出码 0。

  `env OMP_NUM_THREADS=4 COVERAGE_FILE=results/aip_vtok/c1_followup/.coverage.stages .venv-native/bin/coverage run --branch --source=aip_vtok.c1_followup scripts/aip_vtok/run_c1_followup.py --stage all`
- 268 项历史回归及新增用例，退出码 0。

  `env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUBLAS_WORKSPACE_CONFIG=:4096:8 COVERAGE_FILE=results/aip_vtok/c1_followup/.coverage.regression .venv-native/bin/python -m pytest -q cp_adaptok/tests tests/aip_vtok --junitxml=results/aip_vtok/c1_followup/regression.xml --cov=aip_vtok.c1_followup --cov-report=`
- 代码提交，退出码 0。

  `git commit -m 'feat(aip-vtok): implement gated C1 follow-up' -m 'Preserve the sealed C1 experiment and compare groupwise FSQ against two-level integer reconstruction. Add Train-only locking, original-order RecoveryNet pilots and a dual-seed guard for sparse validation.' -m 'Validate with exact 174-frame legacy replay, two 500-step H100 pilots and the follow-up contract tests.'`
- 覆盖率、旧用例逐项比对和保护审计，退出码 0。

  `bash /tmp/c1_followup_quality.sh`
- 从原始文件生成报告和工作记录，退出码 0。

  `.venv-native/bin/python scripts/aip_vtok/run_c1_followup.py --stage report`
- 实验数据与报告提交，退出码 0。

  `git commit -m 'data(aip-vtok): publish C1 follow-up pilot stop' -m 'Record the 174-frame comparisons, both 500-step pilots, seal checks and the conditional sparse stop. Include the generated report, frame evidence and milestone checkpoints.' -m 'Verified exact historical MSE, 306 regression cases, eight contract mutations and zero sealed Test reads.'`
- 推送实验分支，退出码 0。

  `git push -u origin experiment/aip-vtok-c1-followup`

## 代码验证

使用覆盖率、Radon 和 AST 相似度执行 ub-mode 质量检查。CRAP ≥10 有 23 项，相似度 ≥0.8 有 0 对。[质量记录](../results/aip_vtok/c1_followup/quality.json)

在临时副本中杀死 8/8 个契约变异。[变异记录](../results/aip_vtok/c1_followup/mutations.json)
