# AIP‑VTok Sparse‑RFSQ 工作记录

基线 `a8474eefaf7bcdc9c66a229c87471eee67536513`。时间使用 Asia/Shanghai。

Train 像素读取前冻结协议和门槛于 2026-11-19T20:51:17+08:00。 [协议](../results/aip_vtok/sparse_rfsq/protocol.json)

| 阶段 | 开始 | 结束 | 退出码 | Train 帧 | Validation 帧 | Test 帧 |
| --- | --- | --- | --- | --- | --- | --- |
| calibrate | 2026-11-19T20:48:54+08:00 | 2026-11-19T20:48:54+08:00 | 1 | 0 | 0 | 0 |
| calibrate | 2026-11-19T20:51:14+08:00 | 2026-11-19T20:52:31+08:00 | 0 | 24 | 0 | 0 |
| s1 | 2026-11-19T21:00:35+08:00 | 2026-11-19T21:00:36+08:00 | 1 | 0 | 0 | 0 |
| s1 | 2026-11-19T21:02:54+08:00 | 2026-11-19T21:02:54+08:00 | 1 | 0 | 0 | 0 |
| s1 | 2026-11-19T21:03:13+08:00 | 2026-11-19T21:03:13+08:00 | 1 | 0 | 0 | 0 |
| s1 | 2026-11-19T21:05:57+08:00 | 2026-11-19T21:05:57+08:00 | 1 | 0 | 0 | 0 |
| s1 | 2026-11-19T21:07:01+08:00 | 2026-11-19T21:25:30+08:00 | 0 | 0 | 174 | 0 |
| test | 2026-11-19T21:25:34+08:00 | 2026-11-19T21:27:41+08:00 | 0 | 0 | 0 | 0 |
| s2 | 2026-11-19T21:27:44+08:00 | 2026-11-19T21:58:54+08:00 | 0 | 12486 | 174 | 0 |

| 任务 | 命令 | 退出码 |
| --- | --- | --- |
| calibrate | `.venv-native/bin/python -u scripts/aip_vtok/run_sparse_rfsq.py --stage calibrate` | 1 |
| calibrate | `.venv-native/bin/python -u scripts/aip_vtok/run_sparse_rfsq.py --stage calibrate` | 0 |
| s1 | `.venv-native/bin/python -u scripts/aip_vtok/run_sparse_rfsq.py --stage s1` | 1 |
| s1 | `.venv-native/bin/python -u scripts/aip_vtok/run_sparse_rfsq.py --stage s1` | 1 |
| s1 | `.venv-native/bin/python -u scripts/aip_vtok/run_sparse_rfsq.py --stage s1` | 1 |
| s1 | `.venv-native/bin/python -u scripts/aip_vtok/run_sparse_rfsq.py --stage s1` | 1 |
| s1 | `.venv-native/bin/python -u scripts/aip_vtok/run_sparse_rfsq.py --stage s1 --wait-for-gpu` | 0 |
| test | `.venv-native/bin/python -u scripts/aip_vtok/run_sparse_rfsq.py --stage test --wait-for-gpu` | 0 |
| s2 | `.venv-native/bin/python -u scripts/aip_vtok/run_sparse_rfsq.py --stage s2 --wait-for-gpu` | 0 |

## 实现与验证命令

| 任务 | 命令 | 退出码 |
| --- | --- | --- |
| 从封存基线创建分支 | `git switch -c experiment/aip-vtok-sparse-rfsq a8474ee` | 0 |
| ub-mode 接收端 TDD 红灯 | `env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 .venv-native/bin/python -m pytest -q -p no:cacheprovider tests/aip_vtok/test_sparse_rfsq.py` | 2 |
| ub-mode 闭环和梯度 TDD 红灯 | `env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 .venv-native/bin/python -m pytest -q -p no:cacheprovider tests/aip_vtok/test_sparse_rfsq.py -k 'independent_ref or gradient_diagnostics'` | 1 |
| 27 项新增测试 | `env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 .venv-native/bin/python -m pytest -q -p no:cacheprovider tests/aip_vtok/test_sparse_rfsq.py` | 0 |
| 联合回归收集 | `env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 .venv-native/bin/python -m pytest -q -p no:cacheprovider --collect-only cp_adaptok/tests tests/aip_vtok` | 0 |
| 新增路径 lint | `.venv-native/bin/ruff check aip_vtok/sparse_rfsq scripts/aip_vtok/run_sparse_rfsq.py tests/aip_vtok/test_sparse_rfsq.py` | 0 |
| 9 项隔离契约变异 | `env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 .venv-native/bin/python /tmp/sparse_rfsq_mutations.py` | 0 |
| 契约覆盖率 | `env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 COVERAGE_FILE=results/aip_vtok/sparse_rfsq/.coverage.contracts .venv-native/bin/coverage run --branch --source=aip_vtok.sparse_rfsq -m pytest -q -p no:cacheprovider tests/aip_vtok/test_sparse_rfsq.py` | 0 |
| 实现提交 | `git commit -m 'feat(aip-vtok): add gated sparse residual FSQ study'` | 0 |
| S1 与联合回归产物提交 | `git commit -m 'data(aip-vtok): record the Train-locked sparse scan selection'` | 0 |
| 原始数据、门控与哈希复核 | `env PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 .venv-native/bin/python -B /tmp/verify_sparse_rfsq.py` | 0 |
| 自动汇总报告和图表 | `env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUBLAS_WORKSPACE_CONFIG=:4096:8 .venv-native/bin/python -u scripts/aip_vtok/run_sparse_rfsq.py --stage report` | 0 |
| 导出 SVG 的初次空白检查 | `git diff --cached --check` | 2 |

## 联合回归

| 任务 | 命令 | 退出码 |
| --- | --- | --- |
| pytest | `/home/whisperliang/CP-AdapTok/.venv-native/bin/python -m pytest -q -p no:cacheprovider cp_adaptok/tests tests/aip_vtok --junitxml=results/aip_vtok/sparse_rfsq/regression.xml --cov=aip_vtok.sparse_rfsq --cov-branch --cov-report=` | 0 |

306 项旧用例及 27 项新增用例。 [JUnit](../results/aip_vtok/sparse_rfsq/regression.xml) · [日志](../results/aip_vtok/sparse_rfsq/regression.log)

隔离变异检出 9/9 项。 [变异记录](../results/aip_vtok/sparse_rfsq/mutations.json) · [复杂度与重复审计](../results/aip_vtok/sparse_rfsq/quality.json)
