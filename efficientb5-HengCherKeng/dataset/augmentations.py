import numpy
import cv2
import math
import random
import torch


def do_flip_lr(image, mask):
    image = cv2.flip(image, 1)
    mask = mask[:, ::-1]
    return image, mask


def do_flip_ud(image, mask):
    image = cv2.flip(image, 0)
    mask = mask[::-1, :]
    return image, mask


def do_random_crop(image, mask, w, h):
    height, width = image.shape[:2]
    x, y = 0, 0
    if width > w:
        x = numpy.random.choice(width - w)
    if height > h:
        y = numpy.random.choice(height - h)
    image = image[y: y + h, x: x + w]
    mask = mask[y: y + h, x: x + w]
    return image, mask


def do_random_crop_rescale(image, mask, w, h):
    height, width = image.shape[:2]
    x, y = 0, 0
    if width > w:
        x = numpy.random.choice(width - w)
    if height > h:
        y = numpy.random.choice(height - h)
    image = image[y: y + h, x: x + w]
    mask = mask[y: y + h, x: x + w]
    if (w, h) != (width, height):
        image = cv2.resize(image, dsize=(width, height), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, dsize=(width, height), interpolation=cv2.INTER_NEAREST)
    return image, mask


def do_random_crop_rotate_rescale(image, mask, w, h):
    H, W = image.shape[:2]
    dangle = numpy.random.uniform(-8, 8)
    dshift = numpy.random.uniform(-0.1, 0.1, 2)
    dscale_x = numpy.random.uniform(-0.00075, 0.00075)
    dscale_y = numpy.random.uniform(-0.25, 0.25)
    cos = numpy.cos(dangle / 180 * math.pi)
    sin = numpy.sin(dangle / 180 * math.pi)
    sx, sy = 1 + dscale_x, 1 + dscale_y  # 1,1 #
    tx, ty = dshift * min(H, W)
    src = numpy.array(
        [[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]], numpy.float32
    )
    src = src * [sx, sy]
    x = (src * [cos, -sin]).sum(1) + W / 2
    y = (src * [sin, cos]).sum(1) + H / 2
    src = numpy.column_stack([x, y])
    dst = numpy.array([[0, 0], [w, 0], [w, h], [0, h]])
    s = src.astype(numpy.float32)
    d = dst.astype(numpy.float32)
    transform = cv2.getPerspectiveTransform(s, d)
    image = cv2.warpPerspective(
        image,
        transform,
        (W, H),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    mask = cv2.warpPerspective(
        mask,
        transform,
        (W, H),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0),
    )
    return image, mask


def do_random_log_contast(image, gain=[0.70, 1.30]):
    gain = numpy.random.uniform(gain[0], gain[1], 1)
    inverse = numpy.random.choice(2, 1)
    image = image.astype(numpy.float32) / 255
    if inverse == 0:
        image = gain * numpy.log(image + 1)
    else:
        image = gain * (2 ** image - 1)
    image = numpy.clip(image * 255, 0, 255).astype(numpy.uint8)
    return image


def do_random_noise(image, noise=8):
    H, W = image.shape[:2]
    image = image.astype(numpy.float32)
    image = image + numpy.random.uniform(-1, 1, (H, W, 1)) * noise
    image = numpy.clip(image, 0, 255).astype(numpy.uint8)
    return image


# https://github.com/chainer/chainercv/blob/master/chainercv/links/model/ssd/transforms.py
def do_random_contrast(image):
    beta = 0
    alpha = random.uniform(0.5, 2.0)
    image = image.astype(numpy.float32) * alpha + beta
    image = numpy.clip(image, 0, 255).astype(numpy.uint8)
    return image


# Сustomize
def do_random_salt_pepper_noise(image, noise=0.0005):
    height, width = image.shape[:2]
    num_salt = int(noise * width * height)
    # Salt mode
    yx = [numpy.random.randint(0, d - 1, num_salt) for d in image.shape[:2]]
    image[tuple(yx)] = [255, 255, 255]
    # Pepper mode
    yx = [numpy.random.randint(0, d - 1, num_salt) for d in image.shape[:2]]
    image[tuple(yx)] = [0, 0, 0]
    return image


def do_random_salt_pepper_line(image, noise=0.0005, length=10):
    height, width = image.shape[:2]
    num_salt = int(noise * width * height)
    # Salt mode
    y0x0 = numpy.array([numpy.random.randint(0, d - 1, num_salt) for d in image.shape[:2]]).T
    y1x1 = y0x0 + numpy.random.choice(2 * length, size=(num_salt, 2)) - length
    for (y0, x0), (y1, x1) in zip(y0x0, y1x1):
        cv2.line(image, (x0, y0), (x1, y1), (255, 255, 255), 1)
    # Pepper mode
    y0x0 = numpy.array([numpy.random.randint(0, d - 1, num_salt) for d in image.shape[:2]]).T
    y1x1 = y0x0 + numpy.random.choice(2 * length, size=(num_salt, 2)) - length
    for (y0, x0), (y1, x1) in zip(y0x0, y1x1):
        cv2.line(image, (x0, y0), (x1, y1), (0, 0, 0), 1)
    return image


def do_random_cutout(image, mask):
    height, width = image.shape[:2]
    u0 = [0, 1][numpy.random.choice(2)]
    u1 = numpy.random.choice(width)
    if u0 == 0:
        x0, x1 = 0, u1
    if u0 == 1:
        x0, x1 = u1, width
    image[:, x0:x1] = 0
    mask[:, x0:x1] = 0
    return image, mask


def train_augment(image, label, mask, infor):
    random_augment_flag = numpy.random.choice(3)  # TODO: Why is this random?
    if random_augment_flag == 0:
        pass
    elif random_augment_flag == 1:
        image, mask = do_random_crop_rescale(image, mask, 1600 - (256 - 180), 180)
    elif random_augment_flag == 2:
        image, mask = do_random_crop_rotate_rescale(
            image, mask, 1600 - (256 - 224), 224
        )
    image, mask = do_random_crop(image, mask, 400, 256)
    if numpy.random.rand() > 0.25:
        image, mask = do_random_cutout(image, mask)
    if numpy.random.rand() > 0.5:
        image, mask = do_flip_lr(image, mask)
    if numpy.random.rand() > 0.5:
        image, mask = do_flip_ud(image, mask)
    if numpy.random.rand() > 0.5:
        image = do_random_log_contast(image, gain=[0.70, 1.50])
    u = numpy.random.choice(3)
    if u == 0:
        pass
    if u == 1:
        image = do_random_noise(image, noise=8)
    if u == 2:
        image = do_random_salt_pepper_noise(image, noise=0.0001)
    return image, label, mask, infor


# TODO: find place for collate functions
def null_collate(batch):
    batch_size = len(batch)
    input = []
    truth_label = []
    truth_mask = []
    infor = []
    for b in range(batch_size):
        input.append(batch[b][0])
        truth_label.append(batch[b][1])
        truth_mask.append(batch[b][2])
        infor.append(batch[b][3])
    input = numpy.stack(input).astype(numpy.float32) / 255
    input = input.transpose(0, 3, 1, 2)
    truth_label = numpy.stack(truth_label)
    truth_mask = numpy.stack(truth_mask)
    input = torch.from_numpy(input).float()
    truth_label = torch.from_numpy(truth_label).float()
    truth_mask = torch.from_numpy(truth_mask).long().unsqueeze(1)
    return input, truth_label, truth_mask, infor