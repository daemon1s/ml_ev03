import base64
import io
import unittest

import numpy as np
from PIL import Image

from web_demo import process


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

    def test_process_rejects_non_images(self):
        with self.assertRaises(ValueError):
            process(FakeDetector(), 0.39, b'esto no es una imagen')


if __name__ == '__main__':
    unittest.main()
