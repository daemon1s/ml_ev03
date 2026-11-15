import base64
import io
import os
import unittest
from unittest import mock

import numpy as np
from PIL import Image

from detector.plate_reader import OcrConfig
from detector.web_demo import process, reader_from_env


class FakeDetector:
    def predict(self, batch, verbose=0):
        return {
            'boxes': np.asarray([[[10, 10, 50, 30]]], dtype='float32'),
            'confidence': np.asarray([[0.9]], dtype='float32'),
        }


def jpeg_bytes(size):
    buffer = io.BytesIO()
    Image.new('RGB', size, 'white').save(buffer, format='JPEG')
    return buffer.getvalue()


class WebDemoTest(unittest.TestCase):
    def test_process_returns_detection_and_embedded_image(self):
        result = process(FakeDetector(), 0.39, jpeg_bytes((640, 480)))
        self.assertEqual(len(result['detecciones']), 1)
        self.assertEqual(result['umbral'], 0.39)
        self.assertGreaterEqual(result['milisegundos'], 0)
        self.assertTrue(result['imagen'].startswith('data:image/jpeg;base64,'))
        decoded = base64.b64decode(result['imagen'].split(',', 1)[1])
        self.assertEqual(Image.open(io.BytesIO(decoded)).size, (640, 480))

    def test_process_shrinks_large_images_for_display(self):
        result = process(FakeDetector(), 0.39, jpeg_bytes((2560, 1280)))
        decoded = base64.b64decode(result['imagen'].split(',', 1)[1])
        self.assertEqual(Image.open(io.BytesIO(decoded)).size, (1280, 640))

    def test_process_without_reader_marks_reading_as_disabled(self):
        result = process(FakeDetector(), 0.39, jpeg_bytes((640, 480)))
        self.assertEqual(result['detecciones'][0]['lectura']['estado'], 'desactivada')
        self.assertFalse(result['lectura_activa'])

    def test_process_attaches_reading_to_each_detection(self):
        config = OcrConfig(0.6, 10, 'modelo', 'http://127.0.0.1:1/api/chat')
        result = process(FakeDetector(), 0.39, jpeg_bytes((640, 480)), (config, lambda crop: 'BCDF12'))
        reading = result['detecciones'][0]['lectura']
        self.assertEqual(reading, {'estado': 'leida', 'texto': 'BCDF12', 'motivo': None})
        self.assertTrue(result['lectura_activa'])

    def test_process_reports_which_reader_was_used(self):
        config = OcrConfig(0.6, 10, 'modelo', 'http://127.0.0.1:1/api/chat', 'gemini')
        with_reader = process(FakeDetector(), 0.39, jpeg_bytes((640, 480)), (config, lambda crop: 'BCDF12'))
        without_reader = process(FakeDetector(), 0.39, jpeg_bytes((640, 480)))
        self.assertEqual(with_reader['lector'], {'backend': 'gemini', 'modelo': 'modelo'})
        self.assertIsNone(without_reader['lector'])

    def test_reader_from_env_is_none_when_reading_is_off(self):
        with mock.patch.dict(os.environ, {'OCR_BACKEND': 'none'}, clear=True):
            with mock.patch('detector.web_demo.make_asker') as make_asker:
                self.assertIsNone(reader_from_env())
        make_asker.assert_not_called()

    def test_reader_from_env_pairs_config_with_its_asker(self):
        env = {
            'OCR_BACKEND': 'fast_plate', 'OCR_MIN_CONFIDENCE': '0.6', 'OCR_MIN_CROP_WIDTH': '40',
            'OCR_MODEL': 'global-plates-mobile-vit-v2-model', 'OCR_MODEL_DIR': '/modelos/ocr',
        }
        with mock.patch.dict(os.environ, env, clear=True):
            with mock.patch('detector.web_demo.make_asker', return_value='ask') as make_asker:
                config, ask = reader_from_env()
        self.assertEqual(config.backend, 'fast_plate')
        self.assertEqual(ask, 'ask')
        make_asker.assert_called_once_with(config)

    def test_process_rejects_non_images(self):
        with self.assertRaises(ValueError):
            process(FakeDetector(), 0.39, b'esto no es una imagen')


if __name__ == '__main__':
    unittest.main()
