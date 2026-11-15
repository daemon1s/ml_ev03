import base64
import io
import json
import re
import threading
import time
import urllib.parse
import urllib.request
from collections import deque
from dataclasses import dataclass, field

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
BACKENDS = ('ollama', 'gemini')
GEMINI_MODEL = re.compile(r'[A-Za-z0-9][A-Za-z0-9.\-]*')
GEMINI_SCHEMA = {
    'type': 'OBJECT',
    'properties': {'texto': {'type': 'STRING'}},
    'required': ['texto'],
}
MAX_RESPONSE_BYTES = 65536
MINUTE_SECONDS = 60.0
DAY_SECONDS = 86400.0
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
    backend: str = 'ollama'
    api_key: str = field(default='', repr=False)
    max_per_minute: int = 0
    max_per_day: int = 0


class QuotaExceeded(Exception):
    pass


class ReadBudget:
    def __init__(self, per_minute, per_day, clock=time.time):
        self.per_minute = per_minute
        self.per_day = per_day
        self.clock = clock
        self.taken = deque()
        self.lock = threading.Lock()

    def take(self):
        with self.lock:
            now = self.clock()
            while self.taken and now - self.taken[0] >= DAY_SECONDS:
                self.taken.popleft()
            recent = sum(1 for moment in self.taken if now - moment < MINUTE_SECONDS)
            if recent >= self.per_minute or len(self.taken) >= self.per_day:
                return False
            self.taken.append(now)
            return True


def positive_int(name):
    value = int(required_env(name))
    if value < 1:
        raise ValueError(f'{name} debe ser positivo')
    return value


def ollama_endpoint(model):
    if model.lower().endswith(('-cloud', ':cloud')):
        raise ValueError('OCR_MODEL debe ser un modelo descargado para inferencia local')
    parsed = urllib.parse.urlsplit(required_env('OCR_OLLAMA_URL'))
    if (parsed.scheme != 'http' or parsed.hostname not in LOCAL_HOSTS
            or parsed.port is None or parsed.username or parsed.password
            or parsed.path not in {'', '/'} or parsed.query or parsed.fragment):
        raise ValueError('OCR_OLLAMA_URL debe ser una dirección HTTP local con puerto')
    return parsed.geturl().rstrip('/') + '/api/chat'


def gemini_endpoint(model):
    if not GEMINI_MODEL.fullmatch(model):
        raise ValueError('OCR_MODEL no es un identificador de modelo válido')
    parsed = urllib.parse.urlsplit(required_env('OCR_GEMINI_URL'))
    if (parsed.scheme != 'https' or not parsed.hostname or parsed.username or parsed.password
            or parsed.path not in {'', '/'} or parsed.query or parsed.fragment):
        raise ValueError('OCR_GEMINI_URL debe ser una dirección HTTPS sin credenciales ni ruta')
    return f"{parsed.geturl().rstrip('/')}/v1beta/models/{model}:generateContent"


def config_from_env():
    backend = required_env('OCR_BACKEND')
    if backend not in BACKENDS:
        raise ValueError('OCR_BACKEND debe ser ollama o gemini')
    min_confidence = float(required_env('OCR_MIN_CONFIDENCE'))
    if not 0 <= min_confidence <= 1:
        raise ValueError('OCR_MIN_CONFIDENCE debe estar entre 0 y 1')
    min_crop_width = int(required_env('OCR_MIN_CROP_WIDTH'))
    if min_crop_width < 1:
        raise ValueError('OCR_MIN_CROP_WIDTH debe ser positivo')
    model = required_env('OCR_MODEL')
    if backend == 'gemini':
        endpoint = gemini_endpoint(model)
        api_key = required_env('OCR_GEMINI_API_KEY')
        per_minute = positive_int('OCR_MAX_PER_MINUTE')
        per_day = positive_int('OCR_MAX_PER_DAY')
    else:
        endpoint = ollama_endpoint(model)
        api_key = ''
        per_minute = per_day = 0
    return OcrConfig(min_confidence, min_crop_width, model, endpoint, backend, api_key, per_minute, per_day)


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
        raise ValueError('El lector no devolvió JSON') from error
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


def encode_crop(crop):
    encoded = io.BytesIO()
    fit_for_model(crop).save(encoded, format='PNG')
    return base64.b64encode(encoded.getvalue()).decode('ascii')


def ollama_request(config, image):
    payload = {
        'model': config.model, 'stream': False, 'format': ANSWER_SCHEMA,
        'options': {'temperature': 0, 'num_ctx': 4096, 'num_predict': 60},
        'messages': [{'role': 'user', 'content': PROMPT, 'images': [image]}],
    }
    return urllib.request.Request(
        config.endpoint, json.dumps(payload).encode(), {'Content-Type': 'application/json'},
    )


def ollama_text(body):
    return body['message']['content']


def gemini_request(config, image):
    payload = {
        'contents': [{'parts': [
            {'text': PROMPT},
            {'inline_data': {'mime_type': 'image/png', 'data': image}},
        ]}],
        'generationConfig': {
            'temperature': 0, 'maxOutputTokens': 60,
            'responseMimeType': 'application/json', 'responseSchema': GEMINI_SCHEMA,
        },
    }
    headers = {'Content-Type': 'application/json', 'x-goog-api-key': config.api_key}
    return urllib.request.Request(config.endpoint, json.dumps(payload).encode(), headers)


def gemini_text(body):
    return body['candidates'][0]['content']['parts'][0]['text']


READERS = {
    'ollama': (ollama_request, ollama_text),
    'gemini': (gemini_request, gemini_text),
}


def make_asker(config, opener=None):
    opener = opener or urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    build_request, extract_text = READERS[config.backend]
    budget = ReadBudget(config.max_per_minute, config.max_per_day) if config.max_per_minute and config.max_per_day else None

    def ask(crop):
        if budget and not budget.take():
            raise QuotaExceeded('Se agotó el cupo de lecturas')
        request = build_request(config, encode_crop(crop))
        with opener.open(request, timeout=REQUEST_TIMEOUT) as response:
            body = json.loads(response.read(MAX_RESPONSE_BYTES))
        try:
            return parse_answer(extract_text(body))
        except (KeyError, IndexError, TypeError) as error:
            raise ValueError('El lector no devolvió una respuesta de lectura válida') from error

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
    except QuotaExceeded:
        return outcome('omitida', reason='cupo_agotado')
    except (OSError, ValueError):
        return outcome('no_disponible')
    plate = normalize(raw)
    if not valid_format(plate):
        return outcome('ilegible')
    return outcome('leida', text=plate)
