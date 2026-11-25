# HLA-Tok-UF work log (Asia/Shanghai)

## November 25, 2026

- Verified that the requested unpublished temporal branch points to the exact supplied base, and that the original active checkout is clean. Created a separate worktree and new local experiment branch; did not update historical refs.
- Initialized the existing official DCVC gitlink. Reused and rehashed the two files downloaded from Microsoft's published OneDrive directory, and inspected the pinned source and existing native build provenance. Official RT listing returned HTTP 403; direct public share API returned HTTP 401 for the primary and backup directories.
- Added failing frame sequencing tests first, then wrapped official native codecs with independent state and NAL endpoints. The synthetic native smoke exposed an invalid assumption about rANS parallelism; inspected upstream `MAX_EC_PARALLEL` and reran native Train I/P/reset with valid framing and correct native shapes.
- Snapshotted protected worktree file SHA-256 values and local branch refs. Reused the frozen 174-frame approval cohort and disjoint archived Train/Calibration windows. Started the real QP scan before implementing enhancement.
- Added failing tests for dependency order, byte caps, fragmented input, damaged records, and integer entropy round trips. Verified the new bounded segment decoder and the segment encoder byte-for-byte against the official 16-bit rANS encoder (including escapes). No Test pixels read.
