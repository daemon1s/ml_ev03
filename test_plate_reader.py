import json
import os
import unittest
from unittest import mock

from PIL import Image

import plate_reader
from plate_reader import (
    OcrConfig,
    config_from_env,
    crop_plate,
    normalize,
    parse_answer,
    read_plate,
    valid_format,
)


def config(**changes):
    values = {
        'min_confidence': 0.6,
        'min_crop_width': 40,
        'model': 'qwen3-vl:4b-instruct',
        'endpoint': 'http://127.0.0.1:11434/api/chat',
    }
    values.update(changes)
    return OcrConfig(**values)


def found(box=(100, 100, 220, 140), confidence=0.9):
    return {'caja': list(box), 'confianza': confidence}


class NormalizeTest(unittest.TestCase):
    def test_uppercases_and_removes_separators(self):
        self.assertEqual(normalize(' bc-df. 12 '), 'BCDF12')

    def test_drops_country_word_printed_on_the_plate(self):
        self.assertEqual(normalize('SP·XL·83 CHILE'), 'SPXL83')

    def test_empty_text_stays_empty(self):
        self.assertEqual(normalize(' - . '), '')


class FormatTest(unittest.TestCase):
    def test_accepts_chilean_formats(self):
        for plate in ('BCDF12', 'AB1234', 'ABC12'):
            self.assertTrue(valid_format(plate), plate)

    def test_rejects_other_shapes(self):
        for plate in ('', 'BCDF1', 'BCDF123', 'ABCDE1', '123456', 'BC12DF', 'BCDFGH'):
            self.assertFalse(valid_format(plate), plate)


class ParseAnswerTest(unittest.TestCase):
    def test_extracts_text_field(self):
        self.assertEqual(parse_answer(json.dumps({'texto': 'BCDF12'})), 'BCDF12')

    def test_rejects_non_object(self):
        with self.assertRaises(ValueError):
            parse_answer('[]')

    def test_rejects_missing_or_non_string_text(self):
        for payload in ({}, {'texto': 5}):
            with self.assertRaises(ValueError):
                parse_answer(json.dumps(payload))

    def test_rejects_invalid_json(self):
        with self.assertRaises(ValueError):
            parse_answer('no es json')


class CropTest(unittest.TestCase):
    def test_adds_margin_and_clamps_to_image(self):
        image = Image.new('RGB', (300, 200), 'white')
        crop = crop_plate(image, (100, 100, 200, 140))
        self.assertEqual(crop.size, (120, 48))
        edge = crop_plate(image, (0, 0, 100, 40))
        self.assertEqual(edge.size, (110, 44))


class FitTest(unittest.TestCase):
    def test_shrinks_large_crops_keeping_aspect(self):
        fitted = plate_reader.fit_for_model(Image.new('RGB', (2400, 1200)))
        self.assertEqual(fitted.size, (768, 384))

    def test_keeps_small_crops_unchanged(self):
        self.assertEqual(plate_reader.fit_for_model(Image.new('RGB', (90, 40))).size, (90, 40))


class ReadPlateTest(unittest.TestCase):
    def setUp(self):
        self.image = Image.new('RGB', (400, 300), 'white')

    def test_valid_text_is_read(self):
        result = read_plate(self.image, found(), config(), lambda crop: 'bc-df 12')
        self.assertEqual(result, {'estado': 'leida', 'texto': 'BCDF12', 'motivo': None})

    def test_text_outside_format_is_illegible_and_not_shown(self):
        result = read_plate(self.image, found(), config(), lambda crop: 'HOLA')
        self.assertEqual(result['estado'], 'ilegible')
        self.assertIsNone(result['texto'])

    def test_empty_answer_is_illegible(self):
        result = read_plate(self.image, found(), config(), lambda crop: '')
        self.assertEqual(result['estado'], 'ilegible')

    def test_low_confidence_skips_reading_without_calling_the_model(self):
        calls = []
        result = read_plate(self.image, found(confidence=0.5), config(), lambda crop: calls.append(crop))
        self.assertEqual(result['estado'], 'omitida')
        self.assertEqual(result['motivo'], 'confianza_baja')
        self.assertEqual(calls, [])

    def test_confidence_equal_to_minimum_is_read(self):
        result = read_plate(self.image, found(confidence=0.6), config(), lambda crop: 'BCDF12')
        self.assertEqual(result['estado'], 'leida')

    def test_small_box_skips_reading(self):
        calls = []
        result = read_plate(self.image, found(box=(10, 10, 40, 20)), config(), lambda crop: calls.append(crop))
        self.assertEqual(result['estado'], 'omitida')
        self.assertEqual(result['motivo'], 'recorte_pequeno')
        self.assertEqual(calls, [])

    def test_model_failure_reports_unavailable(self):
        def broken(crop):
            raise OSError('sin conexión')

        result = read_plate(self.image, found(), config(), broken)
        self.assertEqual(result['estado'], 'no_disponible')
        self.assertIsNone(result['texto'])

    def test_invalid_model_answer_reports_unavailable(self):
        def broken(crop):
            raise ValueError('respuesta inválida')

        result = read_plate(self.image, found(), config(), broken)
        self.assertEqual(result['estado'], 'no_disponible')


class FakeResponse:
    def __init__(self, body):
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self, limit=-1):
        return json.dumps(self.body).encode()


class FakeOpener:
    def __init__(self, body):
        self.body = body
        self.requests = []

    def open(self, request, timeout=None):
        self.requests.append((request, timeout))
        return FakeResponse(self.body)


def gemini_config(**changes):
    values = {
        'model': 'gemini-3.5-flash-lite',
        'endpoint': 'https://api.example.test/v1beta/models/gemini-3.5-flash-lite:generateContent',
        'backend': 'gemini',
        'api_key': 'clave-de-prueba',
    }
    values.update(changes)
    return config(**values)


class AskerTest(unittest.TestCase):
    def setUp(self):
        self.crop = Image.new('RGB', (120, 40), 'white')

    def test_gemini_sends_key_in_header_and_image_inline(self):
        opener = FakeOpener({'candidates': [{'content': {'parts': [{'text': json.dumps({'texto': 'BCDF12'})}]}}]})
        ask = plate_reader.make_asker(gemini_config(), opener)
        self.assertEqual(ask(self.crop), 'BCDF12')
        request, timeout = opener.requests[0]
        self.assertEqual(request.full_url, gemini_config().endpoint)
        self.assertEqual(request.get_header('X-goog-api-key'), 'clave-de-prueba')
        self.assertNotIn('clave-de-prueba', request.full_url)
        self.assertEqual(timeout, plate_reader.REQUEST_TIMEOUT)
        sent = json.loads(request.data)
        parts = sent['contents'][0]['parts']
        self.assertEqual(parts[0]['text'], plate_reader.PROMPT)
        self.assertEqual(parts[1]['inline_data']['mime_type'], 'image/png')
        self.assertTrue(parts[1]['inline_data']['data'])
        self.assertEqual(sent['generationConfig']['temperature'], 0)
        self.assertEqual(sent['generationConfig']['responseMimeType'], 'application/json')

    def test_gemini_response_without_candidates_is_invalid(self):
        for body in ({}, {'candidates': []}, {'candidates': [{'finishReason': 'SAFETY'}]}):
            ask = plate_reader.make_asker(gemini_config(), FakeOpener(body))
            with self.assertRaises(ValueError):
                ask(self.crop)

    def test_ollama_keeps_its_payload(self):
        opener = FakeOpener({'message': {'content': json.dumps({'texto': 'AB1234'})}})
        ask = plate_reader.make_asker(config(), opener)
        self.assertEqual(ask(self.crop), 'AB1234')
        sent = json.loads(opener.requests[0][0].data)
        self.assertEqual(sent['model'], 'qwen3-vl:4b-instruct')
        self.assertEqual(len(sent['messages'][0]['images']), 1)
        self.assertIsNone(opener.requests[0][0].get_header('X-goog-api-key'))

    def test_config_repr_hides_the_key(self):
        self.assertNotIn('clave-de-prueba', repr(gemini_config()))


class GeminiConfigTest(unittest.TestCase):
    ENV = {
        'OCR_BACKEND': 'gemini',
        'OCR_MIN_CONFIDENCE': '0.6',
        'OCR_MIN_CROP_WIDTH': '40',
        'OCR_MODEL': 'gemini-3.5-flash-lite',
        'OCR_GEMINI_URL': 'https://generativelanguage.googleapis.com',
        'OCR_GEMINI_API_KEY': 'clave-de-prueba',
    }

    def test_reads_gemini_values_without_needing_ollama(self):
        with mock.patch.dict(os.environ, self.ENV, clear=True):
            loaded = config_from_env()
        self.assertEqual(loaded.backend, 'gemini')
        self.assertEqual(loaded.api_key, 'clave-de-prueba')
        self.assertEqual(
            loaded.endpoint,
            'https://generativelanguage.googleapis.com/v1beta/models/gemini-3.5-flash-lite:generateContent',
        )

    def test_missing_variable_is_an_explicit_error(self):
        for name in self.ENV:
            env = {key: value for key, value in self.ENV.items() if key != name}
            with mock.patch.dict(os.environ, env, clear=True):
                with self.assertRaisesRegex(ValueError, name):
                    config_from_env()

    def test_rejects_insecure_or_unsafe_endpoint_and_model(self):
        cases = (
            ('OCR_GEMINI_URL', 'http://generativelanguage.googleapis.com'),
            ('OCR_GEMINI_URL', 'https://user:pass@generativelanguage.googleapis.com'),
            ('OCR_GEMINI_URL', 'https://generativelanguage.googleapis.com/otra?x=1'),
            ('OCR_MODEL', '../models/otro'),
            ('OCR_MODEL', 'gemini 3'),
        )
        for name, value in cases:
            with mock.patch.dict(os.environ, {**self.ENV, name: value}, clear=True):
                with self.assertRaises(ValueError, msg=f'{name}={value}'):
                    config_from_env()


class ConfigTest(unittest.TestCase):
    ENV = {
        'OCR_BACKEND': 'ollama',
        'OCR_MIN_CONFIDENCE': '0.6',
        'OCR_MIN_CROP_WIDTH': '40',
        'OCR_MODEL': 'qwen3-vl:4b-instruct',
        'OCR_OLLAMA_URL': 'http://127.0.0.1:11434',
    }

    def test_reads_every_value_from_environment(self):
        with mock.patch.dict(os.environ, self.ENV, clear=True):
            loaded = config_from_env()
        self.assertEqual(loaded.min_confidence, 0.6)
        self.assertEqual(loaded.min_crop_width, 40)
        self.assertEqual(loaded.model, 'qwen3-vl:4b-instruct')
        self.assertEqual(loaded.endpoint, 'http://127.0.0.1:11434/api/chat')

    def test_missing_variable_is_an_explicit_error(self):
        for name in self.ENV:
            env = {key: value for key, value in self.ENV.items() if key != name}
            with mock.patch.dict(os.environ, env, clear=True):
                with self.assertRaisesRegex(ValueError, name):
                    config_from_env()

    def test_rejects_cloud_model_and_remote_endpoint(self):
        for name, value in (('OCR_MODEL', 'qwen3-vl:235b-cloud'), ('OCR_OLLAMA_URL', 'http://example.com:11434')):
            with mock.patch.dict(os.environ, {**self.ENV, name: value}, clear=True):
                with self.assertRaises(ValueError):
                    config_from_env()

    def test_rejects_confidence_outside_unit_range(self):
        with mock.patch.dict(os.environ, {**self.ENV, 'OCR_MIN_CONFIDENCE': '1.5'}, clear=True):
            with self.assertRaises(ValueError):
                config_from_env()

    def test_rejects_unknown_backend(self):
        with mock.patch.dict(os.environ, {**self.ENV, 'OCR_BACKEND': 'otro'}, clear=True):
            with self.assertRaisesRegex(ValueError, 'OCR_BACKEND'):
                config_from_env()

    def test_ollama_backend_is_default_in_config(self):
        with mock.patch.dict(os.environ, self.ENV, clear=True):
            loaded = config_from_env()
        self.assertEqual(loaded.backend, 'ollama')
        self.assertEqual(loaded.api_key, '')


if __name__ == '__main__':
    unittest.main()
