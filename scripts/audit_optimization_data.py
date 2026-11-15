"""Content-only training data audit. Thumbnailing is for statistics, never codec input."""

import json

import cv2
import numpy as np

from cp_adaptok.runtime import resolve, write_json


def main():
    manifest = json.loads(resolve("data/davis2017/manifest.json").read_text())
    records = []
    for spec in manifest["training_specs"]:
        previous, motion, brightness, texture, cuts = None, [], [], [], 0
        paths = sorted(resolve(spec["path"]).glob("*.jpg"))[: spec["frames"]]
        for path in paths:
            image = cv2.imread(str(path))
            gray = cv2.cvtColor(cv2.resize(image, (256, 144)), cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
            brightness.append(float(gray.mean()))
            texture.append(float(cv2.Laplacian(gray, cv2.CV_32F).var()))
            if previous is not None:
                difference = float(np.abs(gray - previous).mean())
                motion.append(difference)
                cuts += difference > 0.3
            previous = gray
        records.append(
            dict(
                content_id=spec["content_id"],
                frames=len(paths),
                brightness_mean=float(np.mean(brightness)),
                brightness_range=[min(brightness), max(brightness)],
                motion_mean=float(np.mean(motion)),
                texture_mean=float(np.mean(texture)),
                candidate_hard_cuts=cuts,
            )
        )
    write_json(
        "results/optimization_20261115/dataset_audit.json",
        {
            "source_manifest": "data/davis2017/manifest.json",
            "analysis_only_resize": [256, 144],
            "training_resize": False,
            "records": records,
            "limitations": [
                "8 scenes and 128 frames are a pilot training subset",
                "motion/cut heuristics are not human scene labels",
                "formal coverage of shot transitions and all scene categories is not established",
            ],
        },
    )


if __name__ == "__main__":
    main()
