import argparse
import json
import math
import os
from pathlib import Path

LOSS_KEYS = ('loss', 'val_loss')


def merge_histories(histories):
    merged = {}
    for history in histories:
        for key, values in history.items():
            merged.setdefault(key, []).extend(float(value) for value in values)
    return merged


def write_history(history, output):
    (output / 'history.json').write_text(json.dumps(history, indent=2), encoding='utf-8')
    keys = sorted(history)
    rows = ['epoch,' + ','.join(keys)]
    for index in range(len(history['loss'])):
        rows.append(','.join([str(index + 1)] + [repr(history[key][index]) for key in keys]))
    (output / 'history.csv').write_text('\n'.join(rows) + '\n', encoding='utf-8')


def plot_curves(history, path, boundary=None):
    from PIL import Image, ImageDraw

    from plotting import label_font

    width, height, margin = 900, 520, 80
    face = label_font()
    image = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(image)
    epochs = len(history['loss'])
    values = [value for key in LOSS_KEYS for value in history[key]]
    if min(values) <= 0:
        raise ValueError('La escala logarítmica exige pérdidas positivas')
    low_log = math.floor(math.log10(min(values)) * 10) / 10
    high_log = math.ceil(math.log10(max(values)) * 10) / 10
    span = (high_log - low_log) or 1.0

    def point(index, value):
        x = margin + (width - 2 * margin) * index / max(1, epochs - 1)
        y = height - margin - (height - 2 * margin) * (math.log10(value) - low_log) / span
        return x, y

    for decade in range(math.ceil(low_log), math.floor(high_log) + 1):
        x, y = point(0, 10 ** decade)
        draw.line([(margin, y), (width - margin, y)], fill=(225, 225, 225))
        draw.text((10, y - 8), f'{10 ** decade:g}', fill='black', font=face)
    draw.line([(margin, margin), (margin, height - margin), (width - margin, height - margin)], fill='black')
    if boundary is not None and 0 < boundary < epochs:
        x = point(boundary - 1, min(values))[0]
        draw.line([(x, margin), (x, height - margin)], fill=(160, 160, 160))
        draw.text((x + 6, margin), 'backbone descongelado', fill=(100, 100, 100), font=face)
    for key, color in zip(LOSS_KEYS, ((31, 119, 180), (214, 39, 40))):
        points = [point(index, value) for index, value in enumerate(history[key])]
        if len(points) > 1:
            draw.line(points, fill=color, width=2)
        for x, y in points:
            draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=color)
    draw.text((margin, 15), 'Pérdida por época (escala logarítmica)', fill='black', font=face)
    draw.text((margin, height - 45), f'épocas 1 a {epochs} · mínimo validación {min(history["val_loss"]):.3f}', fill='black', font=face)
    draw.text((width - 330, 15), 'azul: entrenamiento · rojo: validación', fill='black', font=face)
    image.save(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--frozen-epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--patience', type=int, default=8)
    parser.add_argument('--train-images', type=int, default=0)
    parser.add_argument('--valid-images', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    for variable in ('DATASET_DIR', 'TRAIN_OUTPUT_DIR'):
        if not os.environ.get(variable):
            parser.error(f'Falta la variable de entorno {variable}')
    if min(args.epochs, args.batch_size, args.patience) < 1 or args.frozen_epochs < 0:
        parser.error('Épocas, lote y paciencia deben ser positivos y las épocas con backbone congelado no negativas')
    if args.frozen_epochs >= args.epochs:
        parser.error('--frozen-epochs debe ser menor que --epochs')
    root = Path(os.environ['DATASET_DIR']).resolve(strict=True)
    output = Path(os.environ['TRAIN_OUTPUT_DIR']).resolve()
    if output.is_relative_to(root):
        parser.error('TRAIN_OUTPUT_DIR debe estar fuera del corpus')
    if output.exists():
        parser.error('TRAIN_OUTPUT_DIR ya existe: usa una carpeta nueva')

    if not os.environ.get('KERAS_BACKEND'):
        parser.error('Falta la variable de entorno KERAS_BACKEND')
    import keras
    import keras_cv

    from pilot import augmented_batches, sample, targets

    keras.utils.set_random_seed(args.seed)
    size = 320
    available = {
        split: len(list((root / split / 'images').glob('*.jpg'))) for split in ('train', 'valid')
    }
    train_count = args.train_images or available['train']
    valid_count = args.valid_images or available['valid']
    training_images, training_boxes = sample(root, 'train', train_count, args.seed, size)
    validation_images, validation_boxes = sample(root, 'valid', valid_count, args.seed, size)
    maximum = max(1, *(len(boxes) for boxes in training_boxes + validation_boxes))
    validation_data = (validation_images, targets(validation_boxes, maximum))
    batches = augmented_batches(
        keras, training_images, training_boxes, maximum, batch_size=args.batch_size, seed=args.seed,
    )

    output.mkdir(parents=True)
    best_path = output / 'best.keras'

    class BestCheckpoint(keras.callbacks.Callback):
        def __init__(self):
            super().__init__()
            self.best = math.inf
            self.best_epoch = 0
            self.epoch_offset = 0

        def on_epoch_end(self, epoch, logs=None):
            value = logs['val_loss']
            if math.isfinite(value) and value < self.best:
                self.best = value
                self.best_epoch = self.epoch_offset + epoch + 1
                self.model.save(best_path)
                print(f'\nMejor modelo guardado: época {self.best_epoch}, val_loss {value:.4f}')

    checkpoint = BestCheckpoint()
    backbone = keras_cv.models.YOLOV8Backbone.from_preset('yolo_v8_xs_backbone_coco')
    detector = keras_cv.models.YOLOV8Detector(
        backbone=backbone, num_classes=1, bounding_box_format='xyxy', fpn_depth=1
    )

    def compile_detector(learning_rate):
        detector.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate, global_clipnorm=10),
            classification_loss='binary_crossentropy', box_loss='ciou', run_eagerly=True,
        )

    histories = []
    if args.frozen_epochs:
        backbone.trainable = False
        compile_detector(1e-3)
        histories.append(detector.fit(
            batches, validation_data=validation_data, epochs=args.frozen_epochs, callbacks=[checkpoint],
        ).history)
        checkpoint.epoch_offset = args.frozen_epochs
    backbone.trainable = True
    compile_detector(1e-4)
    stopper = keras.callbacks.EarlyStopping(monitor='val_loss', patience=args.patience)
    histories.append(detector.fit(
        batches, validation_data=validation_data, epochs=args.epochs - args.frozen_epochs,
        callbacks=[checkpoint, stopper],
    ).history)

    history = merge_histories(histories)
    if not all(math.isfinite(value) for key in LOSS_KEYS for value in history[key]):
        raise RuntimeError('El entrenamiento produjo una pérdida no finita')
    if not best_path.exists():
        raise RuntimeError('No se guardó ningún modelo')
    write_history(history, output)
    plot_curves(history, output / 'curvas.png', boundary=args.frozen_epochs or None)
    summary = {
        'imagenes_train': train_count,
        'imagenes_valid': valid_count,
        'epocas_ejecutadas': len(history['loss']),
        'mejor_epoca': checkpoint.best_epoch,
        'mejor_val_loss': checkpoint.best,
        'argumentos': vars(args),
        'keras': keras.__version__,
        'motor': os.environ['KERAS_BACKEND'],
    }
    (output / 'resumen.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'Mejor época: {checkpoint.best_epoch} · val_loss {checkpoint.best:.4f} · {best_path}')


if __name__ == '__main__':
    main()
