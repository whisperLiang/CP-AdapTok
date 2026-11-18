"""Use the immutable A protocol with the one shared native evaluator."""

import json

from . import evaluate
from .run import PACKAGE

if __name__ == "__main__":
    evaluate.PROTOCOL = json.loads((PACKAGE / "protocol_A_snapshot.json").read_text())
    evaluate.main()
