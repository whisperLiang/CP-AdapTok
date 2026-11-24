# Native validation samples

The sample protocol uses the frozen native `DAVIS2017:car-roundabout` sequence, frames 32–47, at 30 FPS.
The source and reconstruction are stored locally as lossless FFV1, a full-resolution RGB PNG, and the same
384×216 crop at `(x=768,y=432)`. The JSON manifest hashes the weights and files and verifies byte-for-byte
that the decoded FFV1 first frame equals the PNG. No Test frame participates.

The A0 samples under `A0_step04000/` and, after the endpoint runs, `A0_step12000/` are dense and have
neither a token budget nor a rate. Even terminal A0 weights do not certify a paired Phase A pass.
Reproduce other artifacts with
`python -m scripts.hla_tok.v3.run samples --checkpoint /path/to/step.pt`.
