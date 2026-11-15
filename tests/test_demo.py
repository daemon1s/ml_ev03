import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from detector.demo import MODEL_SIZE, detect, draw, prepare, read_threshold, scale_boxes


class FakeDetector:
    def __init__(self, boxes, confidence):
        self.output = {'boxes': np.asarray([boxes], dtype='float32'), 'confidence': np.asarray([confidence], dtype='float32')}
        self.received = None

    def predict(self, batch, verbose=0):
        self.received = batch
        return self.output


class DemoTest(unittest.TestCase):
    def test_prepare_resizes_to_model_size(self):
        array = prepare(Image.new('RGB', (640, 480)))
        self.assertEqual(array.shape, (MODEL_SIZE, MODEL_SIZE, 3))
        self.assertEqual(array.dtype, np.float32)

    def test_scale_boxes_maps_to_original_size(self):
        scaled = scale_boxes([[0, 0, 160, 80]], 640, 480)
        self.assertEqual(scaled.tolist(), [[0.0, 0.0, 320.0, 120.0]])

    def test_detect_filters_by_threshold_and_scales(self):
        detector = FakeDetector([[10, 10, 50, 30], [100, 100, 200, 150]], [0.9, 0.2])
        found = detect(detector, Image.new('RGB', (640, 640)), 0.39)
        self.assertEqual(len(found), 1)
        self.assertAlmostEqual(found[0]['confianza'], 0.9, places=5)
        self.assertEqual(found[0]['caja'], [20.0, 20.0, 100.0, 60.0])
        self.assertEqual(detector.received.shape, (1, MODEL_SIZE, MODEL_SIZE, 3))

    def test_detect_returns_empty_when_nothing_passes(self):
        detector = FakeDetector([[10, 10, 50, 30]], [0.1])
        self.assertEqual(detect(detector, Image.new('RGB', (320, 320)), 0.39), [])

    def test_detect_ignores_padding_slots(self):
        detector = FakeDetector([[10, 10, 50, 30], [-1, -1, -1, -1]], [0.8, -1])
        self.assertEqual(len(detect(detector, Image.new('RGB', (320, 320)), 0.39)), 1)

    def test_draw_keeps_size_and_marks_the_box(self):
        image = Image.new('RGB', (400, 300), 'white')
        drawn = draw(image, [{'caja': [100, 100, 200, 160], 'confianza': 0.93}])
        self.assertEqual(drawn.size, image.size)
        self.assertNotEqual(drawn.getpixel((100, 130)), (255, 255, 255))

    def test_read_threshold_from_metrics(self):
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder)
            (run / 'evaluacion').mkdir()
            (run / 'evaluacion' / 'metricas.json').write_text(json.dumps({'umbral_elegido_en_validacion': 0.39}), encoding='utf-8')
            self.assertEqual(read_threshold(run), 0.39)


if __name__ == '__main__':
    unittest.main()
