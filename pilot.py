import argparse
import math
import os
import random
from pathlib import Path


def box_xyxy(label, image_size):
    class_id, x, y, width, height = map(float, label.split())
    left, top = x - width / 2, y - height / 2
    right, bottom = x + width / 2, y + height / 2
    tolerance = 1e-5
    if not all(map(math.isfinite, (left, top, right, bottom))) or class_id != 0:
        raise ValueError('Etiqueta YOLO inválida')
    if not (-tolerance <= left < right <= 1 + tolerance and -tolerance <= top < bottom <= 1 + tolerance):
        raise ValueError('Caja YOLO fuera de la imagen o degenerada')
    return tuple(min(1, max(0, coordinate)) * image_size for coordinate in (left, top, right, bottom))


def sample(root, split, count, seed, size):
    import numpy as np
    from PIL import Image

    images_dir = root / split / 'images'
    labels_dir = root / split / 'labels'
    paths = sorted(path for path in images_dir.glob('*.jpg') if path.is_file())
    if len(paths) < count:
        raise ValueError(f'{split}: se solicitaron {count} imágenes y hay {len(paths)}')
    images, boxes = [], []
    for path in random.Random(seed).sample(paths, count):
        label = labels_dir / path.with_suffix('.txt').name
        if not path.resolve().is_relative_to(root) or not label.resolve().is_relative_to(root):
            raise ValueError('Una imagen o etiqueta apunta fuera de DATASET_DIR')
        with Image.open(path) as image:
            images.append(np.asarray(image.convert('RGB').resize((size, size), Image.Resampling.BILINEAR)))
        lines = label.read_text(encoding='utf-8').splitlines()
        boxes.append([box_xyxy(line, size) for line in lines if line.strip()])
    return np.asarray(images, dtype='float32'), boxes


def targets(boxes, maximum):
    import numpy as np

    coordinates = np.full((len(boxes), maximum, 4), -1, dtype='float32')
    classes = np.full((len(boxes), maximum), -1, dtype='float32')
    for index, image_boxes in enumerate(boxes):
        if image_boxes:
            coordinates[index, :len(image_boxes)] = image_boxes
            classes[index, :len(image_boxes)] = 0
    return {'boxes': coordinates, 'classes': classes}


def augmented_batches(keras, images, boxes, maximum, batch_size, seed):
    import numpy as np
    from augment import augment

    class AugmentedBatches(keras.utils.PyDataset):
        def __init__(self):
            super().__init__()
            self.rng = np.random.default_rng(seed)
            self.order = self.rng.permutation(len(images))

        def __len__(self):
            return math.ceil(len(images) / batch_size)

        def __getitem__(self, index):
            chosen = self.order[index * batch_size:(index + 1) * batch_size]
            pairs = [augment(images[i], boxes[i], self.rng, images.shape[1]) for i in chosen]
            return np.stack([image for image, _ in pairs]), targets([kept for _, kept in pairs], maximum)

        def on_epoch_end(self):
            self.order = self.rng.permutation(len(images))

    return AugmentedBatches()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('train', 'verify'))
    parser.add_argument('--train-images', type=int, default=16)
    parser.add_argument('--valid-images', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    for variable in ('DATASET_DIR', 'PILOT_OUTPUT_DIR'):
        if not os.environ.get(variable):
            parser.error(f'Falta la variable de entorno {variable}')
    root = Path(os.environ['DATASET_DIR']).resolve(strict=True)
    output = Path(os.environ['PILOT_OUTPUT_DIR']).resolve()
    if output.is_relative_to(root):
        parser.error('PILOT_OUTPUT_DIR debe estar fuera del corpus')
    if min(args.train_images, args.valid_images, args.epochs) < 1:
        parser.error('Los tamaños de muestra y el número de épocas deben ser positivos')

    os.environ.setdefault('KERAS_BACKEND', 'tensorflow')
    import keras
    import keras_cv
    import numpy as np
    from PIL import Image, ImageDraw

    keras.utils.set_random_seed(args.seed)
    size = 320
    validation_images, validation_boxes = sample(root, 'valid', args.valid_images, args.seed, size)
    model_path = output / 'pilot.keras'
    if args.action == 'train':
        training_images, training_boxes = sample(root, 'train', args.train_images, args.seed, size)
        maximum = max(1, *(len(boxes) for boxes in training_boxes + validation_boxes))
        training_batches = augmented_batches(
            keras, training_images, training_boxes, maximum, batch_size=4, seed=args.seed,
        )
        backbone = keras_cv.models.YOLOV8Backbone.from_preset('yolo_v8_xs_backbone_coco')
        backbone.trainable = False
        detector = keras_cv.models.YOLOV8Detector(
            backbone=backbone, num_classes=1, bounding_box_format='xyxy', fpn_depth=1
        )
        detector.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001, global_clipnorm=10),
            classification_loss='binary_crossentropy', box_loss='ciou', run_eagerly=True,
        )
        history = detector.fit(
            training_batches,
            validation_data=(validation_images, targets(validation_boxes, maximum)),
            epochs=args.epochs,
            callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)],
        )
        if not all(math.isfinite(loss) for key in ('loss', 'val_loss') for loss in history.history[key]):
            raise RuntimeError('El entrenamiento produjo una pérdida no finita')
        output.mkdir(parents=True, exist_ok=False)
        detector.save(model_path)
        print(f'Guardado: {model_path}')
        return

    detector = keras.saving.load_model(model_path, compile=False, safe_mode=True)
    detector.prediction_decoder = keras_cv.layers.NonMaxSuppression(
        bounding_box_format='xyxy', from_logits=False, confidence_threshold=0.0, max_detections=3,
    )
    predictions = detector.predict(validation_images, batch_size=4, verbose=0)
    image = Image.fromarray(validation_images[0].astype('uint8'))
    draw = ImageDraw.Draw(image)
    drawn = 0
    for box, confidence in zip(predictions['boxes'][0], predictions['confidence'][0]):
        if confidence < 0 or not np.isfinite(confidence) or not np.isfinite(box).all():
            continue
        left, top, right, bottom = np.clip(box, 0, size - 1)
        if left < right and top < bottom:
            draw.rectangle((left, top, right, bottom), outline='red', width=2)
            drawn += 1
    if not drawn:
        raise RuntimeError('El modelo recargado no produjo cajas candidatas dibujables')
    image.save(output / 'prediccion-local.png')
    print(f'Modelo recargado. Cajas candidatas dibujadas: {drawn}')


if __name__ == '__main__':
    main()
