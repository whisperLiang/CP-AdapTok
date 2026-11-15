# ub-mode work log

2026-11-15 Asia/Shanghai. Actions and reasons are recorded per verifiable unit; command results live in results/.

1. Read the supplied task and ub-mode skill; identified required invariants and TDD workflow.
2. Inspected home, repository status, data, GPU and dependencies. Home is not a Git repository; InfoTok has ongoing H100 training and uncommitted results, so created an independent CP-AdapTok repository without touching them.
3. Located real 1080p UVG/MCL-JCV/HEVC-B sources and an existing PyTorch 2.6 CUDA 12.4 installation. Isolated our environment and reference existing installed packages read-only.
4. Cloned DCVC and AdapTok under third_party; inspected learned queries, VQ, masking, scorer/ILP, LD/DMCI encoders, decoders, state mutation, entropy interfaces and CUDA proxy. Pin exact source SHAs before implementation.
5. No crapper/dryer/mutator executable or tool is available in this session. Use radon + coverage CRAP audit, AST duplication audit and targeted behavior mutation checks, documenting limits rather than claiming unavailable tools ran.
6. Wrote contract tests before implementation; captured expected missing-module failures in results/tdd_initial_red.txt and results/tdd_state_red.txt.
7. Implemented full learned queries, deterministic normalized VQ, shared prefix decoder and stable running-softmax cache. Implemented real packet-terminated rANS, canonical CRC wire records and strict append order. Initial 17 tests passed.
8. Added pure official LD/DMCI adapters, centered BT.709 RGB conversion, 64-pixel padding and crop, one frozen context per frame, single base commit, serialized sender-local decoding, independent receiver and online/oracle allocators. 22 CPU tests passed, including multi-frame state identity and periodic I resets.
9. Attempted official OneDrive weight retrieval; public share API returned HTTP 401. Neither official CUDA extension nor nvcc is present. CP rANS and PyTorch adapters do not require that extension; native baseline and verified native latent distillation must report missing prerequisites.
10. Started full 1920x1080 I/P structural test with official networks, random weights explicitly labeled; this is not an RD result.
