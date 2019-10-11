import pandas as pd
import cv2
import matplotlib.pyplot as plt
from os.path import join
import numpy as np

DEFECT_COLOR = [(0, 0, 0), (0, 0, 255), (0, 255, 0), (255, 0, 0),
                (0, 255, 255)]


def mask_to_inner_contour(mask):
    mask = mask > 0.5
    pad = np.lib.pad(mask, ((1, 1), (1, 1)), 'reflect')
    contour = mask & (
            (pad[1:-1, 1:-1] != pad[:-2, 1:-1]) \
            | (pad[1:-1, 1:-1] != pad[2:, 1:-1]) \
            | (pad[1:-1, 1:-1] != pad[1:-1, :-2]) \
            | (pad[1:-1, 1:-1] != pad[1:-1, 2:])
    )
    return contour


def draw_shadow_text(img, text, pt, fontScale, color, thickness, color1=None,
                     thickness1=None):
    if color1 is None: color1 = (0, 0, 0)
    if thickness1 is None: thickness1 = thickness + 2

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, text, pt, font, fontScale, color1, thickness1,
                cv2.LINE_AA)
    cv2.putText(img, text, pt, font, fontScale, color, thickness, cv2.LINE_AA)


def draw_contour_overlay(image, mask, color=(0, 0, 255), thickness=1):
    contour = mask_to_inner_contour(mask)
    if thickness == 1:
        image[contour] = color
    else:
        for y, x in np.stack(np.where(contour)).T:
            cv2.circle(image, (x, y), thickness // 2, color,
                       lineType=cv2.LINE_4)
    return image


def draw_mask_overlay(image, mask, color=(0, 0, 255), alpha=0.5):
    H, W, C = image.shape
    mask = (mask * alpha).reshape(H, W, 1)
    overlay = image.astype(np.float32)
    overlay = np.maximum(overlay, mask * color)
    overlay = np.clip(overlay, 0, 255)
    overlay = overlay.astype(np.uint8)
    return overlay


def draw_grid(image, grid_size=[32, 32], color=[64, 64, 64], thickness=1):
    H, W, C = image.shape
    dx, dy = grid_size

    for x in range(0, W, dx):
        cv2.line(image, (x, 0), (x, H), color, thickness=thickness)
    for y in range(0, H, dy):
        cv2.line(image, (0, y), (W, y), color, thickness=thickness)
    return image


def int_tuple(x):
    return tuple([int(round(xx)) for xx in x])


def draw_predict_result_attention(
        image, truth_mask, truth_label, probability_label,
        attention):
    probability_label = probability_label.reshape(-1)

    color = DEFECT_COLOR
    H, W, C = image.shape

    overlay = image.copy()
    result = []
    for c in range(4):
        r = np.zeros((H, W, 3), np.uint8)

        t = truth_mask[c]
        pa = attention[c]
        pa = cv2.resize(pa, dsize=(W, H), interpolation=cv2.INTER_NEAREST)

        r = draw_mask_overlay(r, pa, (255, 255, 255), alpha=1)
        r = draw_contour_overlay(r, t, color[c + 1], thickness=4)

        overlay = draw_contour_overlay(overlay, t, color[c + 1], thickness=4)

        draw_shadow_text(r, 'predict%d' % (c + 1), (5, 30), 1, color[c + 1], 2)
        if truth_label[c] > 0.5:
            draw_shadow_text(r, '[%d] ' % (truth_label[c]), (5, 60), 1,
                             color[c + 1], 2)
        else:
            draw_shadow_text(r, '[%d] ' % (truth_label[c]), (5, 60), 1,
                             [64, 64, 64], 2)
        draw_shadow_text(r, '%0.03f' % (probability_label[c]), (60, 60), 1,
                         int_tuple([probability_label[c] * 255] * 3), 2)

        result.append(r)

    # ---
    draw_shadow_text(overlay, 'truth', (5, 30), 1, [255, 255, 255], 2)
    result = [image, overlay, ] + result
    result = np.vstack(result)
    result = draw_grid(result, grid_size=[W, H], color=[255, 255, 255],
                       thickness=3)

    return result
