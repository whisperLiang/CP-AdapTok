"""Operations on measured rate-distortion points."""

from math import exp, log

import numpy as np

DIRECTION = {"global_psnr": 1, "lpips": -1, "dists": -1}


def at_quality(points, metric, target):
    value = target * DIRECTION[metric]
    if not points or not points[0][0] <= value <= points[-1][0]:
        return None
    quality, rates = zip(*points, strict=True)
    return exp(float(np.interp(value, quality, [log(rate) for rate in rates])))


def archive_target(metric, target, original):
    if metric == "lpips" and target == 0.28978 and original:
        endpoint = max(row["lpips"] for row in original)
        if round(endpoint, 5) == target:
            return endpoint
    return target
