import base64
import hashlib
import io
import json
import math
import os
import sqlite3
import tempfile
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw


REASONS = {
    'patente_completa': ('correcta', 'Patente visible y bien delimitada'),
    'patente_cortada': ('mal_delimitada', 'La caja corta parte de la patente'),
    'margen_excesivo': ('mal_delimitada', 'La caja incluye demasiado fondo'),
    'otro_objeto': ('no_es_patente', 'La región corresponde a otro objeto'),
    'detalle_insuficiente': ('dudosa', 'Falta detalle para decidir'),
    'objeto_ambiguo': ('dudosa', 'El objeto o su contorno no están claros'),
}
COLOR = '#ffb348'
AUDIT_CRITERIA = 'patente-encuadre-v1'


def required_env(name):
    value = os.environ.get(name, '').strip()
    if not value:
        raise ValueError(f'Falta configurar {name}')
    return value


def local_file(path, directory):
    if not path.is_file() or path.is_symlink() or path.resolve().parent != directory.resolve():
        raise ValueError(f'Archivo local no válido en {directory.name}')
    return path


@dataclass(frozen=True)
class Box:
    key: str
    xywh: tuple

    def pixels(self, size):
        x, y, width, height = self.xywh
        return (
            (x - width / 2) * size[0], (y - height / 2) * size[1],
            (x + width / 2) * size[0], (y + height / 2) * size[1],
        )


def make_box(key, values):
    coordinates = tuple(float(value) for value in values)
    if len(coordinates) != 4 or not all(math.isfinite(value) for value in coordinates):
        raise ValueError('Se requieren cuatro coordenadas finitas')
    box = Box(key, coordinates)
    left, top, right, bottom = box.pixels((1, 1))
    if not (-0.000001 <= left < right <= 1.000001 and -0.000001 <= top < bottom <= 1.000001):
        raise ValueError('La caja está vacía o sale de la imagen')
    return box


@dataclass
class Case:
    photo: str
    key: str
    images: list
    boxes: tuple
    errors: dict


def scan_train(train):
    images_path = (train / 'images').resolve(strict=True)
    labels_path = (train / 'labels').resolve(strict=True)
    if images_path.parent != train.resolve() or labels_path.parent != train.resolve():
        raise ValueError('Las carpetas train/images y train/labels deben estar dentro de train')
    cases = {}
    images = [path for path in sorted(images_path.iterdir()) if path.suffix.lower() in {'.png', '.jpg', '.jpeg', '.webp', '.bmp'}]
    if not images:
        raise ValueError('No hay imágenes compatibles en train/images')
    stems = {}
    for image in images:
        stem = image.stem.casefold()
        stems[stem] = stems.get(stem, 0) + 1
    for path in images:
        local_file(path, images_path)
        data = path.read_bytes()
        photo = hashlib.sha256(data).hexdigest()
        boxes = []
        errors = {}
        label = labels_path / f'{path.stem}.txt'
        try:
            with Image.open(io.BytesIO(data)) as image:
                image.verify()
        except (OSError, ValueError, Image.DecompressionBombError):
            errors['unreadable'] = 'No se pudo leer la imagen'
        if stems[path.stem.casefold()] > 1:
            errors['ambiguous'] = 'Dos imágenes tienen el mismo nombre base'
        try:
            lines = local_file(label, labels_path).read_text(encoding='utf-8-sig').splitlines()
        except (OSError, ValueError):
            lines = []
            errors['image'] = 'Etiqueta ausente o ilegible'
        if not any(line.strip() for line in lines) and 'image' not in errors:
            errors['image'] = 'La etiqueta está vacía'
        for line_number, line in enumerate(lines, 1):
            parts = line.split()
            if not parts:
                continue
            try:
                if len(parts) != 5 or parts[0] != '0':
                    raise ValueError('Se esperaba la clase 0 y cuatro coordenadas')
                boxes.append(make_box('', parts[1:]).xywh)
            except ValueError as error:
                errors[f'line-{line_number}'] = f'Línea {line_number} · {error}'
        sorted_boxes = tuple(Box(f'box-{index}', coords) for index, coords in enumerate(sorted(boxes), 1))
        invalid_lines = [lines[int(key.split('-')[1]) - 1] for key in errors if key.startswith('line-')]
        variant = json.dumps([sorted(boxes), errors, invalid_lines], sort_keys=True).encode()
        identity = f'{photo}:{hashlib.sha256(variant).hexdigest()}'
        case = cases.setdefault(identity, Case(photo, identity, [], sorted_boxes, errors))
        case.images.append(path)
    return list(cases.values())


def crop_evidence(path, box):
    with Image.open(path) as opened:
        image = opened.convert('RGB')
    left, top, right, bottom = box.pixels(image.size)
    pad_x = (right - left) * 0.5
    pad_y = (bottom - top) * 0.5
    bounds = (
        max(0, math.floor(left - pad_x)), max(0, math.floor(top - pad_y)),
        min(image.width, math.ceil(right + pad_x)), min(image.height, math.ceil(bottom + pad_y)),
    )
    crop = image.crop(bounds)
    crop.thumbnail((900, 900), Image.Resampling.LANCZOS)
    scale_x = crop.width / (bounds[2] - bounds[0])
    scale_y = crop.height / (bounds[3] - bounds[1])
    draw = ImageDraw.Draw(crop)
    draw.rectangle(
        ((left - bounds[0]) * scale_x, (top - bounds[1]) * scale_y,
         (right - bounds[0]) * scale_x, (bottom - bounds[1]) * scale_y),
        outline=COLOR, width=max(1, min(3, round(min(crop.size) / 30))),
    )
    return crop


def verdict(response):
    result = json.loads(response)
    if not isinstance(result, dict):
        raise ValueError('Respuesta visual inválida')
    state, reason = result.get('result'), result.get('reason')
    if not isinstance(reason, str) or reason not in REASONS or REASONS[reason][0] != state:
        raise ValueError('Veredicto visual inválido o incongruente')
    return state, reason


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, msg, headers, newurl):
        raise ValueError('El servidor de auditoría intentó redirigir la solicitud')


class Vision:
    def __init__(self):
        self.model = required_env('AUDIT_MODEL')
        if self.model.lower().endswith(('-cloud', ':cloud')):
            raise ValueError('AUDIT_MODEL debe ser un modelo descargado para inferencia local')
        parsed = urllib.parse.urlsplit(required_env('AUDIT_OLLAMA_URL'))
        if (parsed.scheme != 'http' or parsed.hostname not in {'localhost', '127.0.0.1', '::1'}
                or parsed.port is None or parsed.username or parsed.password
                or parsed.path not in {'', '/'} or parsed.query or parsed.fragment):
            raise ValueError('AUDIT_OLLAMA_URL debe ser una dirección HTTP local con puerto')
        self.endpoint = parsed.geturl().rstrip('/') + '/api/chat'
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())

    def inspect(self, case, box):
        crop = crop_evidence(case.images[0], box)
        encoded = io.BytesIO()
        crop.save(encoded, format='PNG')
        prompt = (
            'Audita una anotación para entrenamiento de un detector de patentes vehiculares. '
            'Observa el objeto dentro del rectángulo naranja y usa el margen para evaluar sus bordes. '
            'Aprueba solo si se ve claramente una patente vehicular completa y la caja la delimita bien. '
            'Si el contenido o el borde no están claros, responde dudosa. No leas ni transcribas la matrícula. '
            'Elige un par result/reason de esta lista: '
            + ', '.join(f'{state}/{reason}' for reason, (state, _) in REASONS.items())
        )
        schema = {
            'type': 'object',
            'properties': {
                'result': {'type': 'string', 'enum': sorted({state for state, _ in REASONS.values()})},
                'reason': {'type': 'string', 'enum': list(REASONS)},
            },
            'required': ['result', 'reason'],
            'additionalProperties': False,
        }
        payload = {
            'model': self.model, 'stream': False, 'format': schema,
            'options': {'temperature': 0, 'num_ctx': 4096, 'num_predict': 100},
            'messages': [{'role': 'user', 'content': prompt, 'images': [base64.b64encode(encoded.getvalue()).decode('ascii')]}],
        }
        request = urllib.request.Request(
            self.endpoint, json.dumps(payload).encode(), {'Content-Type': 'application/json'},
        )
        with self.opener.open(request, timeout=90) as response:
            body = json.loads(response.read(65536))
        try:
            return verdict(body['message']['content'])
        except (KeyError, TypeError) as error:
            raise ValueError('Ollama no devolvió un veredicto visual válido') from error


class Review:
    def __init__(self, path):
        self.db = sqlite3.connect(path)
        self.db.executescript('''
            CREATE TABLE IF NOT EXISTS decisions (
                case_key TEXT, target TEXT, action TEXT, coordinates TEXT,
                PRIMARY KEY (case_key, target)
            );
            CREATE TABLE IF NOT EXISTS inspections (
                case_key TEXT, target TEXT, model TEXT, result TEXT, reason TEXT,
                PRIMARY KEY (case_key, target, model)
            );
        ''')

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.db.close()

    def decisions(self, case):
        rows = self.db.execute(
            'SELECT target, action, coordinates FROM decisions WHERE case_key = ?', (case.key,),
        )
        return {target: (action, json.loads(coords)) for target, action, coords in rows}

    def save_decision(self, case, target, action, coordinates=None):
        with self.db:
            self.db.execute(
                'INSERT OR REPLACE INTO decisions VALUES (?, ?, ?, ?)',
                (case.key, target, action, json.dumps(coordinates)),
            )

    def inspections(self, case, model):
        rows = self.db.execute(
            'SELECT target, result, reason FROM inspections WHERE case_key = ? AND model = ?',
            (case.key, f'{model}/{AUDIT_CRITERIA}'),
        )
        return {target: (state, reason) for target, state, reason in rows}

    def save_inspection(self, case, box, model, result, reason):
        with self.db:
            self.db.execute(
                'INSERT OR REPLACE INTO inspections VALUES (?, ?, ?, ?, ?)',
                (case.key, box.key, f'{model}/{AUDIT_CRITERIA}', result, reason),
            )


def history_rows(cases, review, model):
    rows = []
    for case in cases:
        inspections = review.inspections(case, model)
        if not inspections:
            continue
        decisions = review.decisions(case)
        for box in case.boxes:
            if box.key in inspections:
                state, reason = inspections[box.key]
                action = decisions[box.key][0] if box.key in decisions else None
                rows.append((case, box.key, state, reason, action))
    return rows


def export_train(cases, review, destination, model=''):
    if destination.exists():
        raise ValueError('El destino ya existe. Elige una carpeta nueva')
    exports = []
    remaining = 0
    for case in cases:
        decisions = review.decisions(case)
        if decisions.get('case', ('', None))[0] == 'exclude':
            continue
        confirmed_negative = not case.boxes and decisions.get('image', ('', None))[0] == 'negative'
        invalid = set(case.errors) - decisions.keys() if not confirmed_negative else set()
        if invalid or {'unreadable', 'ambiguous'} & case.errors.keys():
            raise ValueError(f'{case.images[0].name} tiene hallazgos de formato pendientes')
        boxes = {box.key: box.xywh for box in case.boxes}
        inspections = review.inspections(case, model)
        remaining += sum(
            key not in decisions and inspections.get(key, ('pending', None))[0] != 'correcta'
            for key in boxes
        )
        for target, (action, coords) in decisions.items():
            if action == 'remove':
                boxes.pop(target, None)
            elif action in {'adjust', 'add'}:
                boxes[target] = make_box(target, coords).xywh
        labels = ''.join('0 ' + ' '.join(format(value, '.10g') for value in coords) + '\n' for coords in boxes.values())
        exports.append((case, labels))
    with tempfile.TemporaryDirectory(prefix='.audit-', dir=destination.parent) as temp:
        staging = Path(temp) / 'train'
        (staging / 'images').mkdir(parents=True)
        (staging / 'labels').mkdir()
        for case, label in exports:
            for image in case.images:
                data = local_file(image, image.parent).read_bytes()
                if hashlib.sha256(data).hexdigest() != case.photo:
                    raise ValueError(f'{image.name} cambió. Vuelve a abrir la revisión')
                (staging / 'images' / image.name).write_bytes(data)
                (staging / 'labels' / f'{image.stem}.txt').write_text(label, encoding='utf-8')
        summary = {'imagenes_exportadas': sum(len(case.images) for case, _ in exports), 'cajas_sin_resolver': remaining}
        (staging / 'resumen.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
        staging.rename(destination)
    return destination
