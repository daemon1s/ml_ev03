import argparse
import json
import os
from pathlib import Path

import numpy as np

from .evaluate import DECODER_CONFIDENCE, DECODER_DETECTIONS, detections

MODEL_SIZE = 320
NMS_IOU = 0.5
MAX_DETECTIONS = 10
BOX_COLOR = (230, 30, 30)


def read_threshold(run):
    metrics = json.loads((run / 'evaluacion' / 'metricas.json').read_text(encoding='utf-8'))
    return float(metrics['umbral_elegido_en_validacion'])


def load_detector(model_path):
    import keras

    detector = keras.saving.load_model(model_path, compile=False, safe_mode=True)
    decoder = detector.prediction_decoder
    decoder.iou_threshold = NMS_IOU
    decoder.confidence_threshold = DECODER_CONFIDENCE
    decoder.max_detections = DECODER_DETECTIONS
    detector.make_predict_function(force=True)
    applied = (decoder.iou_threshold, decoder.confidence_threshold, decoder.max_detections)
    if applied != (NMS_IOU, DECODER_CONFIDENCE, DECODER_DETECTIONS):
        raise RuntimeError(f'El decodificador no tomó la configuración pedida: {applied}')
    return detector


def prepare(image):
    from PIL import Image

    resized = image.convert('RGB').resize((MODEL_SIZE, MODEL_SIZE), Image.Resampling.BILINEAR)
    return np.asarray(resized, dtype='float32')


def scale_boxes(boxes, width, height):
    factors = np.asarray([width, height, width, height], dtype='float64') / MODEL_SIZE
    return np.asarray(boxes, dtype='float64').reshape(-1, 4) * factors


def detect(detector, image, threshold):
    raw = detector.predict(prepare(image)[None], verbose=0)
    boxes, confidence = detections(raw, MODEL_SIZE, MAX_DETECTIONS)[0]
    keep = confidence >= threshold
    width, height = image.size
    return [
        {'caja': [float(value) for value in box], 'confianza': float(score)}
        for box, score in zip(scale_boxes(boxes[keep], width, height), confidence[keep])
    ]


def draw(image, found):
    from PIL import ImageDraw

    from .plotting import label_font

    canvas = image.convert('RGB')
    pen = ImageDraw.Draw(canvas)
    width = max(2, canvas.width // 200)
    face = label_font(max(14, canvas.width // 35))
    for item in found:
        left, top, right, bottom = item['caja']
        pen.rectangle((left, top, right, bottom), outline=BOX_COLOR, width=width)
        text = f'Patente {item["confianza"]:.0%}'
        anchor = (left, max(0, top - face.size - 4))
        box = pen.textbbox(anchor, text, font=face)
        pen.rectangle(box, fill=BOX_COLOR)
        pen.text(anchor, text, fill='white', font=face)
    return canvas


def require_environment(parser, names):
    for name in names:
        if not os.environ.get(name):
            parser.error(f'Falta la variable de entorno {name}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('image', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    require_environment(parser, ('TRAIN_OUTPUT_DIR', 'KERAS_BACKEND'))
    run = Path(os.environ['TRAIN_OUTPUT_DIR']).resolve(strict=True)
    if not args.image.is_file():
        parser.error(f'No existe {args.image}')
    if args.output.exists():
        parser.error(f'{args.output} ya existe: usa otro nombre')

    from PIL import Image

    threshold = read_threshold(run)
    detector = load_detector(run / 'best.keras')
    with Image.open(args.image) as opened:
        image = opened.convert('RGB')
    found = detect(detector, image, threshold)
    draw(image, found).save(args.output)
    print(json.dumps({'umbral': threshold, 'detecciones': found}, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
