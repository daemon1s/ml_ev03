import argparse
import base64
import io
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError

from demo import detect, draw, load_detector, read_threshold

MAX_UPLOAD_BYTES = 15 * 1024 * 1024
DISPLAY_SIDE = 1280
PAGE_PATH = Path(__file__).parent / 'web' / 'index.html'


def process(detector, threshold, data):
    try:
        with Image.open(io.BytesIO(data)) as opened:
            image = ImageOps.exif_transpose(opened).convert('RGB')
    except (UnidentifiedImageError, OSError) as error:
        raise ValueError('El archivo no es una imagen válida') from error
    started = time.perf_counter()
    found = detect(detector, image, threshold)
    elapsed = time.perf_counter() - started
    shown = draw(image, found)
    shown.thumbnail((DISPLAY_SIDE, DISPLAY_SIDE))
    buffer = io.BytesIO()
    shown.save(buffer, format='JPEG', quality=90)
    return {
        'umbral': threshold,
        'detecciones': found,
        'milisegundos': round(elapsed * 1000),
        'imagen': 'data:image/jpeg;base64,' + base64.b64encode(buffer.getvalue()).decode('ascii'),
    }


def make_handler(detector, threshold, page):
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def reply(self, status, content_type, body):
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def reply_json(self, status, payload):
            self.reply(status, 'application/json; charset=utf-8', json.dumps(payload, ensure_ascii=False).encode('utf-8'))

        def do_GET(self):
            if self.path != '/':
                self.reply_json(404, {'error': 'No encontrado'})
                return
            self.reply(200, 'text/html; charset=utf-8', page)

        def do_POST(self):
            if self.path != '/detectar':
                self.reply_json(404, {'error': 'No encontrado'})
                return
            try:
                length = int(self.headers.get('Content-Length', ''))
            except ValueError:
                self.reply_json(411, {'error': 'Falta el tamaño del archivo'})
                return
            if not 0 < length <= MAX_UPLOAD_BYTES:
                self.reply_json(413, {'error': f'La imagen debe pesar entre 1 byte y {MAX_UPLOAD_BYTES // 1024 // 1024} MB'})
                return
            data = self.rfile.read(length)
            try:
                with lock:
                    result = process(detector, threshold, data)
            except ValueError as error:
                self.reply_json(400, {'error': str(error)})
                return
            self.reply_json(200, result)

    return Handler


def main():
    parser = argparse.ArgumentParser()
    parser.parse_args()
    for variable in ('TRAIN_OUTPUT_DIR', 'KERAS_BACKEND', 'DEMO_HOST', 'DEMO_PORT'):
        if not os.environ.get(variable):
            parser.error(f'Falta la variable de entorno {variable}')
    run = Path(os.environ['TRAIN_OUTPUT_DIR']).resolve(strict=True)
    host = os.environ['DEMO_HOST']
    port = int(os.environ['DEMO_PORT'])
    threshold = read_threshold(run)
    detector = load_detector(run / 'best.keras')
    page = PAGE_PATH.read_bytes()
    server = HTTPServer((host, port), make_handler(detector, threshold, page))
    print(f'Detector listo en http://{host}:{port} · umbral {threshold:.2f}', flush=True)
    server.serve_forever()


if __name__ == '__main__':
    main()
