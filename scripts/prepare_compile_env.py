"""Create local linker names for the installed 64-bit NVIDIA driver; no system files change."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    drivers = [p for p in Path("/usr/lib").glob("*/libcuda.so.1") if p.read_bytes()[:5] == b"\x7fELF\x02"]
    if not drivers:
        raise RuntimeError("no installed 64-bit libcuda.so.1")
    output = ROOT / ".build/driver"
    output.mkdir(parents=True, exist_ok=True)
    for name in ["libcuda.so", "libcuda.so.1"]:
        link = output / name
        if link.exists() and link.resolve() != drivers[0].resolve():
            raise ValueError(f"existing driver link differs: {link}")
        if not link.exists():
            link.symlink_to(drivers[0])
    print(f"export TRITON_LIBCUDA_PATH={output}")


if __name__ == "__main__":
    main()
