import numpy as np
from typing import List
from scipy.special import expit


def postprocess_threshold_area(
        pred: np.array, threshold: List[float], min_area: List[float]
):
    res = []
    for row in pred:
        row_res = []
        for i, c in enumerate(row):
            if c.min() < 0:
                c = expit(c)

            c = (c > threshold[i]).astype(np.uint8)
            if c.sum() < min_area[i]:
                c = np.zeros(c.shape, dtype=c.dtype)
            row_res.append(c)
        res.append(row_res)

    return np.array(res)


def clear_segment_with_classifier(mask: np.array, clf_pred: np.array,
                                  threshold: float):
    for i in range(len(mask)):
        if clf_pred[i][1] > threshold:
            mask[i] = np.zeros(mask.shape, dtype=mask.dtype)
    return mask


class PostProcessMixin:
    def postprocess_threshold_area(
            self, pred: np.array, threshold: List[float], min_area: List[float]
    ):
        return postprocess_threshold_area(pred, threshold, min_area)

    def clear_segment_with_classifier(self, mask: np.array, clf_pred: np.array,
                                      threshold: float):
        return clear_segment_with_classifier(mask, clf_pred, threshold)