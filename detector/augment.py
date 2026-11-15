import numpy as np
from PIL import Image, ImageFilter

SCALE_RANGE = (0.8, 1.2)
SHIFT_FRACTION = 0.1
BRIGHTNESS_RANGE = (0.7, 1.3)
CONTRAST_RANGE = (0.8, 1.2)
BLUR_PROBABILITY = 0.3
BLUR_RADIUS_RANGE = (0.5, 1.5)
NOISE_PROBABILITY = 0.3
NOISE_SIGMA_RANGE = (3.0, 10.0)
MIN_VISIBLE_FRACTION = 0.5
MIN_BOX_SIDE = 2.0
FILL_VALUE = 114


def affine(image, boxes, scale, shift_x, shift_y):
    height, width = image.shape[:2]
    center_x, center_y = width / 2, height / 2
    coefficients = (
        1 / scale, 0, center_x - (center_x + shift_x) / scale,
        0, 1 / scale, center_y - (center_y + shift_y) / scale,
    )
    source = Image.fromarray(np.clip(image, 0, 255).round().astype('uint8'))
    moved = source.transform(
        (width, height), Image.Transform.AFFINE, coefficients,
        resample=Image.Resampling.BILINEAR, fillcolor=(FILL_VALUE,) * 3,
    )
    kept = []
    for left, top, right, bottom in boxes:
        left = (left - center_x) * scale + center_x + shift_x
        right = (right - center_x) * scale + center_x + shift_x
        top = (top - center_y) * scale + center_y + shift_y
        bottom = (bottom - center_y) * scale + center_y + shift_y
        full_area = (right - left) * (bottom - top)
        left, right = max(0.0, left), min(float(width), right)
        top, bottom = max(0.0, top), min(float(height), bottom)
        if right - left < MIN_BOX_SIDE or bottom - top < MIN_BOX_SIDE:
            continue
        if (right - left) * (bottom - top) < MIN_VISIBLE_FRACTION * full_area:
            continue
        kept.append((left, top, right, bottom))
    return np.asarray(moved, dtype='float32'), kept


def photometric(image, rng):
    brightness = rng.uniform(*BRIGHTNESS_RANGE)
    contrast = rng.uniform(*CONTRAST_RANGE)
    mean = image.mean()
    result = ((image - mean) * contrast + mean) * brightness
    if rng.random() < BLUR_PROBABILITY:
        radius = rng.uniform(*BLUR_RADIUS_RANGE)
        blurred = Image.fromarray(np.clip(result, 0, 255).round().astype('uint8'))
        result = np.asarray(blurred.filter(ImageFilter.GaussianBlur(radius)), dtype='float32')
    if rng.random() < NOISE_PROBABILITY:
        result = result + rng.normal(0.0, rng.uniform(*NOISE_SIGMA_RANGE), size=result.shape)
    return np.clip(result, 0, 255).astype('float32')


def augment(image, boxes, rng, size):
    scale = rng.uniform(*SCALE_RANGE)
    limit = SHIFT_FRACTION * size
    shift_x, shift_y = rng.uniform(-limit, limit, size=2)
    moved, kept = affine(image, boxes, scale, shift_x, shift_y)
    return photometric(moved, rng), kept
