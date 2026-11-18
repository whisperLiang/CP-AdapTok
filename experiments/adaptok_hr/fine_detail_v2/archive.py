"""Package raw observations and actual packets with a reproducible member-level digest."""

import gzip
import json
from pathlib import Path
import shutil
import tarfile

from adaptok_hr.runtime import file_hash, write_json

from .run import CHECKPOINTS, PACKAGE, ROOT
from .study import read

DEST = PACKAGE / "evidence"


def archive_members(name, members):
    target = DEST / f"{name}.tar.gz"
    member_hashes = {}
    temporary = target.with_suffix(".tmp")
    with temporary.open("wb") as binary, gzip.GzipFile(fileobj=binary, mode="wb", mtime=0, filename="") as compressed:
        with tarfile.open(fileobj=compressed, mode="w|") as out:
            for key, path in sorted(members):
                info = tarfile.TarInfo(key)
                info.size = path.stat().st_size
                with path.open("rb") as handle:
                    out.addfile(info, handle)
                member_hashes[key] = file_hash(path)
    temporary.replace(target)
    return {"archive": target.name, "sha256": file_hash(target), "members": member_hashes}


def run():
    DEST.mkdir(exist_ok=True)
    manifest = {}
    base = [(str(p.relative_to(ROOT)), p) for p in ROOT.glob("*.json")
            if p.name not in ("coverage.json", "quality_audit.json")]
    base += [(str(p.relative_to(ROOT)), p) for d in ("configs", "logs") for p in (ROOT / d).glob("*") if p.is_file()]
    base += [(p.name, p) for p in (ROOT / "tests.xml", ROOT / "quality_audit.json", ROOT / "coverage.json")]
    for directory in sorted(ROOT.glob("S[12]_*")):
        if not directory.is_dir():
            continue
        base.extend((str(p.relative_to(ROOT)), p) for p in directory.glob("*") if p.is_file() and p.suffix in (".json", ".jsonl"))
        base.extend((str(p.relative_to(ROOT)), p) for p in (directory / "probes").glob("*.json"))
        base.extend(("checkpoint_metadata/" + str(p.relative_to(CHECKPOINTS)), p)
                    for p in (CHECKPOINTS / directory.name).glob("*.json"))
        for validation in sorted(directory.glob("validation_*_*/summary.json")):
            folder = validation.parent
            key = directory.name + "_" + folder.name
            members = [(str(p.relative_to(ROOT)), p) for p in folder.rglob("*") if p.is_file() and "samples" not in p.parts]
            manifest[key] = archive_members(key, members)
            if (folder / "samples").exists():
                sample_dest = DEST / "samples" / key
                shutil.copytree(folder / "samples", sample_dest, dirs_exist_ok=True)
    manifest["training"] = archive_members("training", base)
    for name in ("selection.json", "decision.json", "acceptance.json", "decoder_decision.json", "performance_decision.json"):
        if (ROOT / name).exists():
            shutil.copy2(ROOT / name, DEST / name)
    for p in sorted((ROOT / "figures").glob("*")):
        if p.is_file() and not p.name.endswith(".overlay.pdf"):
            (DEST / "figures").mkdir(exist_ok=True)
            shutil.copy2(p, DEST / "figures" / p.name)
    write_json(DEST / "archives.json", manifest)
    files = {str(p.relative_to(DEST)): file_hash(p) for p in sorted(DEST.rglob("*"))
             if p.is_file() and p != DEST / "manifest.json"}
    write_json(DEST / "manifest.json", files)
    print(json.dumps({"archives": len(manifest), "members": sum(len(a["members"]) for a in manifest.values()),
                      "files": len(files), "largest_bytes": max((DEST / k).stat().st_size for k in files)}))


if __name__ == "__main__":
    run()
