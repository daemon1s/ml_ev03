import argparse
import json
import os
from pathlib import Path

import numpy as np

IOU_LEVELS = tuple(round(0.5 + 0.05 * step, 2) for step in range(10))
DECODER_CONFIDENCE = 0.001
DECODER_DETECTIONS = 100
THRESHOLD_CANDIDATES = tuple(round(0.05 + 0.01 * step, 2) for step in range(91))


def iou_matrix(first, second):
    first = np.asarray(first, dtype='float64').reshape(-1, 4)
    second = np.asarray(second, dtype='float64').reshape(-1, 4)
    left = np.maximum(first[:, None, 0], second[None, :, 0])
    top = np.maximum(first[:, None, 1], second[None, :, 1])
    right = np.minimum(first[:, None, 2], second[None, :, 2])
    bottom = np.minimum(first[:, None, 3], second[None, :, 3])
    intersection = np.clip(right - left, 0, None) * np.clip(bottom - top, 0, None)
    area_first = (first[:, 2] - first[:, 0]) * (first[:, 3] - first[:, 1])
    area_second = (second[:, 2] - second[:, 0]) * (second[:, 3] - second[:, 1])
    union = area_first[:, None] + area_second[None, :] - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)


def detections(output, size, max_detections):
    result = []
    for boxes, confidence in zip(output['boxes'], output['confidence']):
        boxes = np.asarray(boxes, dtype='float64').reshape(-1, 4)
        confidence = np.asarray(confidence, dtype='float64').reshape(-1)
        keep = np.isfinite(confidence) & (confidence >= 0) & np.isfinite(boxes).all(axis=1)
        boxes, confidence = np.clip(boxes[keep], 0, size), confidence[keep]
        valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
        boxes, confidence = boxes[valid], confidence[valid]
        order = np.argsort(-confidence, kind='stable')[:max_detections]
        result.append((boxes[order], confidence[order]))
    return result


def match_image(truth, boxes, level):
    flags = np.zeros(len(boxes), dtype=bool)
    if len(truth) == 0 or len(boxes) == 0:
        return flags
    overlaps = iou_matrix(boxes, truth)
    taken = np.zeros(len(truth), dtype=bool)
    for index in range(len(boxes)):
        candidates = np.where(taken, -1.0, overlaps[index])
        best = int(candidates.argmax())
        if candidates[best] >= level:
            taken[best] = True
            flags[index] = True
    return flags


def collect(detected, truths, level, threshold=0.0):
    scores, flags = [], []
    for (boxes, confidence), truth in zip(detected, truths):
        keep = confidence >= threshold
        boxes, confidence = boxes[keep], confidence[keep]
        scores.append(confidence)
        flags.append(match_image(truth, boxes, level))
    return np.concatenate(scores), np.concatenate(flags)


def pr_curve(scores, flags, total):
    order = np.argsort(-scores, kind='stable')
    flags = flags[order]
    true_positives = np.cumsum(flags)
    false_positives = np.cumsum(~flags)
    precision = true_positives / np.maximum(true_positives + false_positives, 1)
    recall = true_positives / total if total else np.zeros(len(flags))
    return recall, precision


def average_precision(scores, flags, total):
    if total == 0 or len(scores) == 0:
        return 0.0
    recall, precision = pr_curve(scores, flags, total)
    recall = np.concatenate(([0.0], recall, [1.0]))
    precision = np.concatenate(([0.0], precision, [0.0]))
    for index in range(len(precision) - 2, -1, -1):
        precision[index] = max(precision[index], precision[index + 1])
    changes = np.where(recall[1:] != recall[:-1])[0]
    return float(np.sum((recall[changes + 1] - recall[changes]) * precision[changes + 1]))


def count_truth(truths):
    return int(sum(len(truth) for truth in truths))


def precision_recall(detected, truths, threshold, level=0.5):
    _, flags = collect(detected, truths, level, threshold)
    true_positives = int(flags.sum())
    false_positives = int(len(flags) - true_positives)
    total = count_truth(truths)
    false_negatives = total - true_positives
    precision = true_positives / (true_positives + false_positives) if flags.size else 0.0
    recall = true_positives / total if total else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        'umbral': threshold, 'precision': precision, 'recall': recall, 'f1': f1,
        'verdaderos_positivos': true_positives, 'falsos_positivos': false_positives,
        'falsos_negativos': false_negatives,
    }


def best_threshold(detected, truths, level=0.5, candidates=THRESHOLD_CANDIDATES):
    scored = [precision_recall(detected, truths, threshold, level) for threshold in candidates]
    return max(scored, key=lambda row: (row['f1'], row['umbral']))


def mean_average_precision(detected, truths):
    total = count_truth(truths)
    values = {level: average_precision(*collect(detected, truths, level), total) for level in IOU_LEVELS}
    return values[0.5], float(np.mean(list(values.values())))


def plot_pr_curve(detected, truths, operating, path):
    from PIL import Image, ImageDraw

    from .plotting import label_font

    face = label_font()
    width, height, margin = 700, 520, 70
    image = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(image)
    scores, flags = collect(detected, truths, 0.5)
    recall, precision = pr_curve(scores, flags, count_truth(truths))

    def point(x, y):
        return margin + (width - 2 * margin) * x, height - margin - (height - 2 * margin) * y

    draw.line([point(0, 1), point(0, 0), point(1, 0)], fill='black')
    for tick in (0.0, 0.25, 0.5, 0.75, 1.0):
        x, y = point(tick, 0)
        draw.text((x - 12, y + 6), f'{tick:.2f}', fill='black', font=face)
        x, y = point(0, tick)
        draw.text((18, y - 8), f'{tick:.2f}', fill='black', font=face)
    if len(recall) > 1:
        draw.line([point(r, p) for r, p in zip(recall, precision)], fill=(31, 119, 180), width=2)
    x, y = point(operating['recall'], operating['precision'])
    draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=(214, 39, 40))
    draw.text((margin, 15), 'Curva precisión-recall en prueba (IoU 0,5)', fill='black', font=face)
    draw.text((margin, height - 35), 'Recall', fill='black', font=face)
    draw.text((8, margin - 28), 'Precisión', fill='black', font=face)
    draw.text((width - 280, 15), f'punto rojo: umbral {operating["umbral"]:.2f}', fill='black', font=face)
    image.save(path)


def draw_examples(images, truths, detected, threshold, path, columns=4):
    from PIL import Image, ImageDraw

    from .plotting import label_font

    face = label_font(14)
    size = images.shape[1]
    rows = -(-len(images) // columns)
    sheet = Image.new('RGB', (columns * size, rows * size), 'white')
    for index, image_array in enumerate(images):
        tile = Image.fromarray(np.clip(image_array, 0, 255).astype('uint8'))
        draw = ImageDraw.Draw(tile)
        for box in truths[index]:
            draw.rectangle(tuple(box), outline=(0, 200, 0), width=2)
        boxes, confidence = detected[index]
        for box, score in zip(boxes, confidence):
            if score >= threshold:
                draw.rectangle(tuple(box), outline=(230, 30, 30), width=2)
                draw.text((box[0] + 2, max(22, box[1] - 15)), f'{score:.2f}', fill=(255, 60, 60), font=face)
        draw.rectangle((0, 0, size, 20), fill=(0, 0, 0))
        draw.text((6, 2), 'verde: etiqueta · rojo: predicción', fill='white', font=face)
        sheet.paste(tile, ((index % columns) * size, (index // columns) * size))
    sheet.save(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-detections', type=int, default=10)
    parser.add_argument('--nms-iou', type=float, default=0.5)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--valid-images', type=int, default=0)
    parser.add_argument('--test-images', type=int, default=0)
    parser.add_argument('--examples', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    for variable in ('DATASET_DIR', 'TRAIN_OUTPUT_DIR', 'KERAS_BACKEND'):
        if not os.environ.get(variable):
            parser.error(f'Falta la variable de entorno {variable}')
    if min(args.max_detections, args.batch_size, args.examples) < 1:
        parser.error('Detecciones máximas, lote y ejemplos deben ser positivos')
    root = Path(os.environ['DATASET_DIR']).resolve(strict=True)
    run = Path(os.environ['TRAIN_OUTPUT_DIR']).resolve(strict=True)
    model_path = run / 'best.keras'
    if not model_path.is_file():
        parser.error(f'No existe {model_path}')
    output = run / 'evaluacion'
    if output.exists():
        parser.error(f'{output} ya existe: bórrala o muévela antes de evaluar de nuevo')

    import keras

    from .pilot import sample

    keras.utils.set_random_seed(args.seed)
    size = 320
    available = {split: len(list((root / split / 'images').glob('*.jpg'))) for split in ('valid', 'test')}
    valid_images, valid_boxes = sample(root, 'valid', args.valid_images or available['valid'], args.seed, size)
    test_images, test_boxes = sample(root, 'test', args.test_images or available['test'], args.seed, size)
    valid_truths = [np.asarray(boxes, dtype='float64').reshape(-1, 4) for boxes in valid_boxes]
    test_truths = [np.asarray(boxes, dtype='float64').reshape(-1, 4) for boxes in test_boxes]

    detector = keras.saving.load_model(model_path, compile=False, safe_mode=True)
    decoder = detector.prediction_decoder
    decoder.iou_threshold = args.nms_iou
    decoder.confidence_threshold = DECODER_CONFIDENCE
    decoder.max_detections = DECODER_DETECTIONS
    detector.make_predict_function(force=True)
    applied = (decoder.iou_threshold, decoder.confidence_threshold, decoder.max_detections)
    if applied != (args.nms_iou, DECODER_CONFIDENCE, DECODER_DETECTIONS):
        raise RuntimeError(f'El decodificador no tomó la configuración pedida: {applied}')

    def run_split(images):
        output_raw = detector.predict(images, batch_size=args.batch_size, verbose=0)
        return detections(output_raw, size, args.max_detections), int(output_raw['boxes'].shape[1])

    valid_detected, _ = run_split(valid_images)
    test_detected, decoder_slots = run_split(test_images)
    chosen = best_threshold(valid_detected, valid_truths)
    on_test = precision_recall(test_detected, test_truths, chosen['umbral'])
    on_valid_ap50, on_valid_ap = mean_average_precision(valid_detected, valid_truths)
    test_ap50, test_ap = mean_average_precision(test_detected, test_truths)
    all_confidences = np.concatenate([confidence for _, confidence in test_detected])

    output.mkdir()
    summary = {
        'modelo': str(model_path),
        'umbral_elegido_en_validacion': chosen['umbral'],
        'validacion': {
            'imagenes': len(valid_images), 'cajas': count_truth(valid_truths),
            'ap50': on_valid_ap50, 'ap50_95': on_valid_ap, **chosen,
        },
        'prueba': {
            'imagenes': len(test_images), 'cajas': count_truth(test_truths),
            'ap50': test_ap50, 'ap50_95': test_ap, **on_test,
        },
        'configuracion': {
            'iou_para_aciertos': 0.5, 'nms_iou': args.nms_iou, 'detecciones_maximas_por_imagen': args.max_detections,
            'ranuras_del_decodificador': decoder_slots,
            'confianza_minima_observada_en_prueba': float(all_confidences.min()) if all_confidences.size else None,
            'semilla': args.seed,
        },
    }
    (output / 'metricas.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    plot_pr_curve(test_detected, test_truths, on_test, output / 'curva-precision-recall-prueba.png')
    shown = min(args.examples, len(test_images))
    draw_examples(
        test_images[:shown], test_truths[:shown], test_detected[:shown], chosen['umbral'],
        output / 'ejemplos-prueba.png',
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
