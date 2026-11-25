# Figures

All formal rate–distortion points use 174 audited 1080p Validation frames from three videos and the Seed 1 C0 checkpoint. PSNR is pooled pixel error expressed in dB. LPIPS-Alex and DISTS are means over the same source frames. The figures are descriptive. Per-video and per-frame tables carry the paired observations. Timings report 3,000 repeated executions per representative condition over nine input frames. The complete Validation timing is a separate table.

| Figure | Evidence |
| --- | --- |
| 1 · Token spectrum | 96 fitted Train frames. The shading spans the 10th to 90th percentiles across native Ranks. [Spectrum](../pca_spectrum.csv), [effective dimensions](../rank_effective_dimension.csv) |
| 2 · Projection | Pure BF16 projection loss at full and progressive budgets. [Full matrix](../projection_only_rd.csv), [paired scope](../projection_only_per_frame.csv) |
| 3 · Equal precision budgets | Projection and integer coding separated at full length. Three pairs use identical physical bytes at each K. [Physical byte ablation](../equal_bits_ablation.csv), [extra scalar loss](../projection_quantization_ablation.csv) |
| 4 · Rate–distortion | C0, C1, Original and eligible Compact curves. Original uses full 16-frame input windows. [Static](../scalar_quantization_rd.csv), [Original](../original_rd.csv) |
| 5 · Bandwidth | Largest physical prefix under each cap. Original is averaged over valid source frames including partial-window transmitted padding. Points use the predeclared O2 metric selection across profiles. [Byte-matched observations](../bandwidth_baseline_comparison.csv) |
| 6 · Adapter | Observed validation steps from shared source frames. The status panel is used when a stage produces no training measurements. [Stage decision](../training_selection.json) |
| 7 · H100 | Serial path including CRC, packing, transfers and native synthesis. [Time decomposition](../latency_breakdown.csv), [throughput](../throughput_benchmark.csv) |
| 8 · Native reconstructions | Same source frame, dimensions and ROI for all panels. The exact crop starts at (768, 432) and is 384 × 216. The assets retain original RGB pixel values and spatial resolution. [Image SHA-256](../samples/manifest.json) |

PDF and SVG contain editable plot text. PNG is a 300 dpi preview. The plate displays original 1080p PNGs with unmodified crops. The companion JSONs record the rendered alignment, text sizes and visible collisions.
