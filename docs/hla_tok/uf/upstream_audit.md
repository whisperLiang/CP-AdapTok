# DCVC-UF source and runtime audit

## Identity

The existing `third_party/DCVC` gitlink already pins official commit
`cbdae87a5445114cdc7f48816da63ea80bdeac40`; it was initialized in the UF
worktree without changing that gitlink or the vendor sources. The worktree
fork point is local `experiment/hla-tok-temporal` at
`502eaf64f6fe8693dc90ddf6164c87799f1aebfe`. Both UF checkpoints are read-only
inputs, checked by size and SHA-256 before loading, and loaded with strict
state dictionaries. The pinned source/OneDrive identities and hashes are in
`configs/hla_tok/uf/upstream.json`. See `docs/pretrained_weights.md` for the
original official size, QuickXor and checkpoint archive CRC verification.

DCVC ships under MIT (`third_party/DCVC/LICENSE.txt`, `NOTICE.txt`), with
file-specific notices, including the InterDigital Apache-2.0 stream helper.
UF does not redistribute weights or vendor binaries. `NOTICE.md` records
the parent repository's attribution. The existing build-only patch
`native_patches/decoded_latent.patch` copies the *actual* decoder input for
instrumentation; the baseline methods do not call it during timed runs.

## Mapped paths and invariants

| Path under the gitlink | What matters to UF |
| --- | --- |
| `src/models/video_model_ld.py` | `g_frame_delay=1`, `DMC`, PyTorch `forward_one_frame`, `get_recon_and_feature`, and the Python entry points to `DMCLDProxy`. Per-QP feature/encoder/decoder gains are learned channel vectors, not scalar QP divisions. |
| `src/models/image_model.py` | `DMCI` Intra pairs with LD; four ordered spatial masks, not the P-frame's two. The encoder and decoder latent scales are distinct parameter arrays. |
| `src/models/common_model.py` | `forward_prior_2x` clamps the prior's decoder step below at 0.5, scales the latent and predicts the second mask's means after reconstructing the first. The two masks form a full complementary spatial/channel support. `forward_prior_4x` is the corresponding Intra path. |
| `src/models/entropy_models.py` | Building CDFs quantizes the per-QP/per-channel hyperprior and Gaussian scale classes to a 16-bit rANS CDF; `update` caches the tables before proxy initialization. Registry 0 is z and 1 is y. `MLCodec_extensions_cpp` ships the CDF builder and the CPU coder. |
| `src/layers/extensions/inference/{bind.cpp,dmc_ld_proxy.cpp,dmci_proxy.cpp}` | Official FP16 production path. The C++ proxies capture CUDA graphs and own scratch allocations/events and CPU rANS workers. The wrapper loads proxies separately for each endpoint. |
| `src/cpp/py_rans/{rans.cpp,py_rans.cpp}` | Native encoder flushes z and y into the same set of rANS states (up to eight substreams), one real payload. The payload includes hyperprior, terminal state and internal parallel stream offsets. There is no unique additive z-vs-y byte attribution for the resulting stream. |
| `test_video.py` | Uses the upstream SPS + I/P NAL helpers; native RGB is transformed to centered YCbCr before coding. The CLI default has no forced I period and a 32-frame memory-reset interval. Dataset configs can override the I-frame cadence; the main UF comparison instead explicitly inserts true I-frames at frames 0, 32, 64, etc. |
| `train_video.py`, `training.md` | DMC(LD) training with one-frame chunks/backpropagation through retained features. This path estimates bits, and is not substituted for the real or timed C++ path. |

At 1920×1080, padding to a multiple of 64 produces 1920×1088; source pixels
are not resized. The measured decoder input is `(1,256,68,120)` for Intra
and `(1,128,68,120)` for LD. C++ hyper latents have 30×17 spatial positions
(4× downscaled relative to y, with y padded to a multiple of four). The
channel counts come from the pinned source; the decoder crops the output
and uses the same conversion for both frame types.

## State ownership and precision

PyTorch `ref_feature`, `memory` and `ctx` belong to `DMC`'s forward/training
path; they are not the native state. In CUDA, `add_ref_feature_from_frame`
installs the pixel-unshuffled Intra reconstruction. The encoder eagerly
precomputes next-frame context; the decoder computes it on the next call.
Both routes use the actual base feature, with an optional reconstruction
and feature adaptor refresh that does **not** create a random access point.
`clear_dpb` on the Python object alone does not clear the C++ proxy.
Recovery is by decoding a real I-frame and seeding the proxy from its
pixels. Python sequence checks reject missing/out-of-order P-frames before
running native decode. Floating state parity is a same-build, same-device
claim only; no FP32/BF16 or cross-device equivalence is assumed for the
native base.

Local build reused read-only Python 3.12.13 / PyTorch 2.9.1+cu130 / CUDA
NVCC 13.0.88 / GCC 12.3.0 / pybind11 3.1.0, on H100 PCIe driver 580.159.03.
CUTLASS v4.4.1 resolves to `4370102f9dacab813282e1d67722fceb0b90a019`.
Upstream `MLCodec_extensions_cpp` and `inference_extensions_cuda` distribution
versions are both `0.0.0`; the gitlink, source patch and binary digests, not
those labels, identify the implementation.
