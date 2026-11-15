import argparse
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

import augment as aug
from plotting import label_font

TILE = 320
COLUMNS = 3


def photometric_only(image, brightness=1.0, contrast=1.0, blur=None, noise=None, rng=None):
    mean = image.mean()
    result = ((image - mean) * contrast + mean) * brightness
    if blur:
        blurred = Image.fromarray(np.clip(result, 0, 255).round().astype('uint8'))
        result = np.asarray(blurred.filter(ImageFilter.GaussianBlur(blur)), dtype='float32')
    if noise:
        result = result + rng.normal(0.0, noise, size=result.shape)
    return np.clip(result, 0, 255).astype('float32')


def tile(image, boxes, label):
    picture = Image.fromarray(np.clip(image, 0, 255).round().astype('uint8'))
    draw = ImageDraw.Draw(picture)
    for left, top, right, bottom in boxes:
        draw.rectangle((left, top, right, bottom), outline=(0, 220, 0), width=2)
    face = label_font()
    draw.rectangle((0, 0, TILE, 22), fill=(0, 0, 0))
    draw.text((6, 3), label, fill='white', font=face)
    return picture


def sheet(tiles, columns):
    rows = -(-len(tiles) // columns)
    canvas = Image.new('RGB', (columns * TILE, rows * TILE), 'white')
    for index, picture in enumerate(tiles):
        canvas.paste(picture, ((index % columns) * TILE, (index // columns) * TILE))
    return canvas


def example_sheet(image, boxes, rng, variants):
    tiles = [tile(image, boxes, 'original')]
    for number in range(1, variants + 1):
        moved, kept = aug.augment(image, boxes, rng, TILE)
        tiles.append(tile(moved, kept, f'aumento aleatorio {number}'))
    return sheet(tiles, COLUMNS)


def transformations_sheet(image, boxes, rng):
    limit = aug.SHIFT_FRACTION * TILE
    entries = [('original', image, boxes)]
    for label, scale, shift_x, shift_y in (
        (f'escala {aug.SCALE_RANGE[0]}', aug.SCALE_RANGE[0], 0, 0),
        (f'escala {aug.SCALE_RANGE[1]}', aug.SCALE_RANGE[1], 0, 0),
        ('traslación -10 % x, -10 % y', 1.0, -limit, -limit),
        ('traslación +10 % x, +10 % y', 1.0, limit, limit),
    ):
        moved, kept = aug.affine(image, boxes, scale, shift_x, shift_y)
        entries.append((label, moved, kept))
    for label, options in (
        (f'brillo {aug.BRIGHTNESS_RANGE[0]}', {'brightness': aug.BRIGHTNESS_RANGE[0]}),
        (f'brillo {aug.BRIGHTNESS_RANGE[1]}', {'brightness': aug.BRIGHTNESS_RANGE[1]}),
        (f'contraste {aug.CONTRAST_RANGE[0]}', {'contrast': aug.CONTRAST_RANGE[0]}),
        (f'contraste {aug.CONTRAST_RANGE[1]}', {'contrast': aug.CONTRAST_RANGE[1]}),
        (f'desenfoque radio {aug.BLUR_RADIUS_RANGE[1]}', {'blur': aug.BLUR_RADIUS_RANGE[1]}),
        (f'ruido sigma {aug.NOISE_SIGMA_RANGE[1]}', {'noise': aug.NOISE_SIGMA_RANGE[1]}),
    ):
        entries.append((label, photometric_only(image, rng=rng, **options), boxes))
    combined, kept = aug.augment(image, boxes, rng, TILE)
    entries.append(('combinada (como en el entrenamiento)', combined, kept))
    return sheet([tile(picture, kept_boxes, label) for label, picture, kept_boxes in entries], 4)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--images', type=int, default=3)
    parser.add_argument('--variants', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    for variable in ('DATASET_DIR', 'EXAMPLES_OUTPUT_DIR'):
        if not os.environ.get(variable):
            parser.error(f'Falta la variable de entorno {variable}')
    if min(args.images, args.variants) < 1:
        parser.error('Imágenes y variantes deben ser positivas')
    root = Path(os.environ['DATASET_DIR']).resolve(strict=True)
    output = Path(os.environ['EXAMPLES_OUTPUT_DIR']).resolve()
    output.mkdir(parents=True, exist_ok=True)

    from pilot import sample

    images, boxes = sample(root, 'train', args.images, args.seed, TILE)
    rng = np.random.default_rng(args.seed)
    targets = [
        (output / f'aumento-datos-ejemplo-{index + 1:02d}.png', example_sheet(images[index], boxes[index], rng, args.variants))
        for index in range(len(images))
    ]
    targets.append((output / 'aumento-datos-transformaciones.png', transformations_sheet(images[0], boxes[0], rng)))
    existing = [path.name for path, _ in targets if path.exists()]
    if existing:
        parser.error(f'Ya existen: {", ".join(existing)}. Muévelos antes de generar de nuevo')
    for path, canvas in targets:
        canvas.save(path)
        print(path)


if __name__ == '__main__':
    main()
