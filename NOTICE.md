# Attribution

AdapTok (VisionXLab), pinned in third_party/manifest.json, is MIT licensed. Its learned latent query, prefix masking, VQ, quality scorer and offline allocation designs informed this causal single-frame implementation. See third_party/AdapTok/LICENSE.

DCVC (Microsoft), pinned in third_party/manifest.json, supplies the original LD/DMCI modules and the independent native CUDA control. See third_party/DCVC/LICENSE.txt and individual source-file notices. In particular its stream helper retains an Apache-2.0 header credited to InterDigital; file-specific notices take precedence over the repository's default MIT label.

No upstream source files were edited. Native and fixed-spatial controls are explicit experiments, not alternate recovery paths for the main codec.
