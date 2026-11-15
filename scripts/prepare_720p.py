"""Lossless native-pixel center crop, explicitly not an independent 720p dataset."""

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
source = Path("/home/whisperliang/InfoTok/data/chp_vtok/basketball.mp4")
destination = ROOT / "data/basketball_center720.mp4"
destination.parent.mkdir(exist_ok=True)
command = [
    "ffmpeg",
    "-v",
    "error",
    "-y",
    "-i",
    str(source),
    "-vf",
    "crop=1280:720:(iw-1280)/2:(ih-720)/2",
    "-frames:v",
    "8",
    "-c:v",
    "libx264rgb",
    "-crf",
    "0",
    "-preset",
    "fast",
    str(destination),
]
subprocess.run(command, check=True)
(destination.parent / "center720_manifest.json").write_text(
    json.dumps(
        {
            "source": str(source),
            "output": str(destination),
            "command": command,
            "operation": "center crop of original pixels; no resize; engineering benchmark only",
        },
        indent=2,
    )
    + "\n"
)
