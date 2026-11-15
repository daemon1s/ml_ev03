import csv
import json
import os
import shutil
from pathlib import Path

MANUAL_PREFIX = 'manual_'
TOLERANCE = 0.5


def required_env(name):
    value = os.environ.get(name, '').strip()
    if not value:
        raise ValueError(f'Falta configurar {name}')
    return value


def coco_to_yolo(bbox, width, height):
    x, y, w, h = bbox
    if w <= 0 or h <= 0:
        raise ValueError(f'Caja sin área: {bbox}')
    if x < -TOLERANCE or y < -TOLERANCE or x + w > width + TOLERANCE or y + h > height + TOLERANCE:
        raise ValueError(f'Caja fuera de la imagen: {bbox}')
    cx = (x + w / 2) / width
    cy = (y + h / 2) / height
    return f'0 {cx:.6f} {cy:.6f} {w / width:.6f} {h / height:.6f}'


def labels_by_image(coco):
    images = {image['id']: image for image in coco['images']}
    result = {image['file_name']: [] for image in coco['images']}
    for annotation in coco['annotations']:
        image = images[annotation['image_id']]
        result[image['file_name']].append(coco_to_yolo(annotation['bbox'], image['width'], image['height']))
    return result


def merge(source, manual, target):
    if target.exists() and any(target.iterdir()):
        raise ValueError(f'TARGET_DATASET_DIR ya existe y no está vacío: {target}')
    shutil.copytree(source, target, dirs_exist_ok=True)
    coco = json.loads((manual / 'annotations' / 'instances_default.json').read_text(encoding='utf-8'))
    labels = labels_by_image(coco)
    train_images = target / 'train' / 'images'
    train_labels = target / 'train' / 'labels'
    rows = []
    for name, lines in sorted(labels.items()):
        origin = manual / 'images' / name
        if not origin.is_file():
            raise ValueError(f'Falta la imagen {origin}')
        final = MANUAL_PREFIX + name
        if (train_images / final).exists():
            raise ValueError(f'{final} ya existe en train')
        shutil.copy2(origin, train_images / final)
        (train_labels / (Path(final).stem + '.txt')).write_text('\n'.join(lines) + '\n', encoding='utf-8')
        rows.append({'imagen': final, 'nombre_en_cvat': name, 'objetos': len(lines)})
    manifest = target / 'curacion' / 'muestra_manual_en_train.csv'
    manifest.parent.mkdir(exist_ok=True)
    with manifest.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['imagen', 'nombre_en_cvat', 'objetos'])
        writer.writeheader()
        writer.writerows(rows)
    return len(rows), sum(row['objetos'] for row in rows)


def main():
    source = Path(required_env('SOURCE_DATASET_DIR')).resolve(strict=True)
    manual = Path(required_env('MANUAL_SAMPLE_DIR')).resolve(strict=True)
    target = Path(required_env('TARGET_DATASET_DIR')).resolve()
    if target == source or target.is_relative_to(source) or source.is_relative_to(target):
        raise ValueError('TARGET_DATASET_DIR debe ser una carpeta distinta del corpus original')
    images, boxes = merge(source, manual, target)
    print(f'Agregadas {images} imágenes y {boxes} cajas a train en {target}')


if __name__ == '__main__':
    main()
