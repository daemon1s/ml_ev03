import argparse
import json
import os
import time
from pathlib import Path

from PIL import Image, ImageDraw

from demo import detect, load_detector, read_threshold
from plate_reader import config_from_env, crop_plate, make_asker, read_plate
from plotting import label_font

IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
CELL_WIDTH = 480
CELL_HEIGHT = 200
LABEL_HEIGHT = 70


def require(parser, names):
    for name in names:
        if not os.environ.get(name):
            parser.error(f'Falta la variable de entorno {name}')


def sample_images(folder, count, offset):
    images = [path for path in sorted(folder.iterdir()) if path.suffix.lower() in IMAGE_SUFFIXES]
    step = max(1, len(images) // count)
    return images[offset::step]


def sheet(cells, path):
    columns = 2
    rows = (len(cells) + columns - 1) // columns
    canvas = Image.new('RGB', (columns * CELL_WIDTH, rows * (CELL_HEIGHT + LABEL_HEIGHT)), 'white')
    pen = ImageDraw.Draw(canvas)
    face = label_font(16)
    for index, (crop, lines) in enumerate(cells):
        x = (index % columns) * CELL_WIDTH
        y = (index // columns) * (CELL_HEIGHT + LABEL_HEIGHT)
        fitted = crop.copy()
        fitted.thumbnail((CELL_WIDTH - 10, CELL_HEIGHT))
        canvas.paste(fitted, (x + 5, y + 5))
        for number, line in enumerate(lines):
            pen.text((x + 5, y + CELL_HEIGHT + 8 + number * 20), line, fill='black', font=face)
    canvas.save(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('images', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--count', type=int, default=10)
    parser.add_argument('--offset', type=int, default=0)
    args = parser.parse_args()
    require(parser, ('TRAIN_OUTPUT_DIR', 'KERAS_BACKEND'))
    if not args.images.is_dir():
        parser.error(f'No existe {args.images}')
    if args.output.exists():
        parser.error(f'{args.output} ya existe: usa otro nombre')
    try:
        config = config_from_env()
    except ValueError as error:
        parser.error(str(error))
    run = Path(os.environ['TRAIN_OUTPUT_DIR']).resolve(strict=True)
    threshold = read_threshold(run)
    detector = load_detector(run / 'best.keras')
    raw_answers = []
    base_ask = make_asker(config)

    def ask(crop):
        started = time.perf_counter()
        text = base_ask(crop)
        raw_answers.append((text, time.perf_counter() - started))
        return text

    args.output.mkdir(parents=True)
    records = []
    cells = []
    for path in sample_images(args.images, args.count, args.offset):
        with Image.open(path) as opened:
            image = opened.convert('RGB')
        found = detect(detector, image, threshold)
        if not found:
            records.append({'archivo': path.name, 'detecciones': 0})
            continue
        best = max(found, key=lambda item: item['confianza'])
        before = len(raw_answers)
        reading = read_plate(image, best, config, ask)
        raw, seconds = raw_answers[before] if len(raw_answers) > before else (None, None)
        crop = crop_plate(image, best['caja'])
        crop.save(args.output / f'{len(records) + 1:02d}-{path.stem[:40]}.png')
        records.append({
            'archivo': path.name, 'detecciones': len(found), 'confianza': round(best['confianza'], 3),
            'ancho_caja': round(best['caja'][2] - best['caja'][0]), 'respuesta_cruda': raw,
            'segundos': None if seconds is None else round(seconds, 2), 'lectura': reading,
        })
        cells.append((crop, [
            f'{len(records):02d} · conf {best["confianza"]:.0%} · ancho {round(best["caja"][2] - best["caja"][0])} px',
            f'cruda: {raw!r}',
            f'estado: {reading["estado"]} {reading["texto"] or reading["motivo"] or ""}',
        ]))
    (args.output / 'lecturas.json').write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding='utf-8')
    if cells:
        sheet(cells, args.output / 'hoja.png')
    print(json.dumps(records, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
