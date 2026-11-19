#!/usr/bin/env python3
"""Run the five frozen native-resolution Diag-0 probes and render their report."""

import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from aip_vtok.diagnostics.diag0.run import main  # noqa: E402

if __name__ == "__main__":
    main()
