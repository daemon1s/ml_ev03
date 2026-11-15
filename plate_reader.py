import base64
import io
import json
import re
import urllib.parse
import urllib.request
from dataclasses import dataclass

from audit_review import NoRedirect, required_env

CROP_MARGIN = 0.1
MAX_MODEL_SIDE = 768
COUNTRY_WORD = 'CHILE'
FORMATS = (
    re.compile(r'[A-Z]{4}[0-9]{2}'),
    re.compile(r'[A-Z]{2}[0-9]{4}'),
    re.compile(r'[A-Z]{3}[0-9]{2}'),
)
LOCAL_HOSTS = {'localhost', '127.0.0.1', '::1'}
REQUEST_TIMEOUT = 90
PROMPT = (
    'La imagen es el recorte de una patente vehicular chilena. '
    'Transcribe solo los caracteres grandes de la placa, sin guiones, puntos ni espacios, '
    'e ignora la palabra pequeña CHILE y el escudo. '
    'Si no se distinguen todos los caracteres con claridad, responde con texto vacío. '
    'No adivines ni completes caracteres.'
)
ANSWER_SCHEMA = {
    'type': 'object',
    'properties': {'texto': {'type': 'string'}},
    'required': ['texto'],
    'additionalProperties': False,
}


@dataclass(frozen=True)
class OcrConfig:
    min_confidence: float
    min_crop_width: int
    model: str
    endpoint: str


def config_from_env():
    min_confidence = float(required_env('OCR_MIN_CONFIDENCE'))
    if not 0 <= min_confidence <= 1:
        raise ValueError('OCR_MIN_CONFIDENCE debe estar entre 0 y 1')
    min_crop_width = int(required_env('OCR_MIN_CROP_WIDTH'))
    if min_crop_width < 1:
        raise ValueError('OCR_MIN_CROP_WIDTH debe ser positivo')
    model = required_env('OCR_MODEL')
    if model.lower().endswith(('-cloud', ':cloud')):
        raise ValueError('OCR_MODEL debe ser un modelo descargado para inferencia local')
    parsed = urllib.parse.urlsplit(required_env('OCR_OLLAMA_URL'))
    if (parsed.scheme != 'http' or parsed.hostname not in LOCAL_HOSTS
            or parsed.port is None or parsed.username or parsed.password
            or parsed.path not in {'', '/'} or parsed.query or parsed.fragment):
        raise ValueError('OCR_OLLAMA_URL debe ser una dirección HTTP local con puerto')
    endpoint = parsed.geturl().rstrip('/') + '/api/chat'
    return OcrConfig(min_confidence, min_crop_width, model, endpoint)


def normalize(text):
    return re.sub(r'[^A-Z0-9]', '', text.upper()).replace(COUNTRY_WORD, '')


def fit_for_model(crop):
    from PIL import Image

    fitted = crop.convert('RGB')
    if max(fitted.size) > MAX_MODEL_SIDE:
        fitted.thumbnail((MAX_MODEL_SIDE, MAX_MODEL_SIDE), Image.Resampling.LANCZOS)
    return fitted


def valid_format(plate):
    return any(pattern.fullmatch(plate) for pattern in FORMATS)


def parse_answer(content):
    try:
        answer = json.loads(content)
    except json.JSONDecodeError as error:
        raise ValueError('Ollama no devolvió JSON') from error
    if not isinstance(answer, dict) or not isinstance(answer.get('texto'), str):
        raise ValueError('La respuesta de lectura no trae el campo texto')
    return answer['texto']


def crop_plate(image, box):
    left, top, right, bottom = box
    pad_x = (right - left) * CROP_MARGIN
    pad_y = (bottom - top) * CROP_MARGIN
    bounds = (
        max(0, int(left - pad_x)), max(0, int(top - pad_y)),
        min(image.width, int(right + pad_x + 0.999999)), min(image.height, int(bottom + pad_y + 0.999999)),
    )
    return image.crop(bounds)


def make_asker(config):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())

    def ask(crop):
        encoded = io.BytesIO()
        fit_for_model(crop).save(encoded, format='PNG')
        payload = {
            'model': config.model, 'stream': False, 'format': ANSWER_SCHEMA,
            'options': {'temperature': 0, 'num_ctx': 4096, 'num_predict': 60},
            'messages': [{
                'role': 'user', 'content': PROMPT,
                'images': [base64.b64encode(encoded.getvalue()).decode('ascii')],
            }],
        }
        request = urllib.request.Request(
            config.endpoint, json.dumps(payload).encode(), {'Content-Type': 'application/json'},
        )
        with opener.open(request, timeout=REQUEST_TIMEOUT) as response:
            body = json.loads(response.read(65536))
        try:
            return parse_answer(body['message']['content'])
        except (KeyError, TypeError) as error:
            raise ValueError('Ollama no devolvió una respuesta de lectura válida') from error

    return ask


def outcome(state, text=None, reason=None):
    return {'estado': state, 'texto': text, 'motivo': reason}


def read_plate(image, item, config, ask):
    if item['confianza'] < config.min_confidence:
        return outcome('omitida', reason='confianza_baja')
    left, _, right, _ = item['caja']
    if right - left < config.min_crop_width:
        return outcome('omitida', reason='recorte_pequeno')
    try:
        raw = ask(crop_plate(image, item['caja']))
    except (OSError, ValueError):
        return outcome('no_disponible')
    plate = normalize(raw)
    if not valid_format(plate):
        return outcome('ilegible')
    return outcome('leida', text=plate)
