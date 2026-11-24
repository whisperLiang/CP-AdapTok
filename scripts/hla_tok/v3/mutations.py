"""Re-run V2 mutations and exercise new V3 training/Receiver boundaries."""

from hla_tok.evaluation.io import write_json
from scripts.hla_tok.mutation_check import mutate
from scripts.hla_tok.v2.mutations import CASES
from scripts.hla_tok.v2.protocol import ROOT
from scripts.hla_tok.v3.protocol import OUTPUT

V3 = (
    ("dense_sender_feature_in_receiver", "models/codec.py", "return self._decode(values, geometry)",
     "return self._decode(values, geometry) + self.encoder(values, geometry)[1].mean()",
     "v3/test_v3_training.py::test_independent_receiver_owns_only_synthesis_and_decodes_actual_bytes"),
    ("lpips_rgb_gradient_detach", "training/losses.py", "self.alex(rgb0 * 2 - 1, rgb1 * 2 - 1)",
     "self.alex(rgb0 * 2 - 1, rgb1.detach() * 2 - 1)",
     "v3/test_reconstruction.py::test_alex_preprocessing_does_not_detach_reconstruction"),
    ("hidden_native_resize", "models/spatial.py", "geometry.pad(rgb).contiguous",
     "geometry.pad(F.interpolate(rgb, size=rgb.shape[-2:])).contiguous",
     "v3/test_reconstruction.py::test_native_path_preserves_original_pixel_grid"),
    ("validation_as_training_source", "training/train.py", 'self.reader.read("train", sample["video"], sample["frame"])',
     'self.reader.read("validation", sample["video"], sample["frame"])',
     "v3/test_v3_training.py::test_training_plan_reads_only_train_at_the_predeclared_native_coordinates"),
)


def run_mutations():
    target = OUTPUT / "mutations"
    target.mkdir(parents=True, exist_ok=True)
    rows = [mutate(ROOT, target, case) for case in (*CASES, *V3)]
    write_json(target / "summary.json", {
        "status": "PASS" if all(row["killed"] for row in rows) else "FAIL",
        "observed": rows, "not_applicable": ["ard_uses_suffix", "ard_wrong_anchor_mapping"],
    })
    if not all(row["killed"] for row in rows):
        raise RuntimeError("a targeted mutation survived")
    return rows
