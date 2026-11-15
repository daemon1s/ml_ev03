# EV03 · Detección de patentes

Detector de patentes vehiculares con YOLOv8 XS, entrenado con Keras 3, más una capa de lectura (OCR) separada que transcribe el texto del recorte. Se usa desde una web local.

- **Detección.** YOLOv8 XS (`keras-cv`) con aprendizaje por transferencia, entrenado sobre una clase, `Patente`. Ubica la patente; no lee texto.
- **Lectura.** Una etapa aparte, elegida con `OCR_BACKEND`: Qwen3-VL 4B por Ollama (local) o Gemini (nube). No forma parte del modelo entrenado.

## Resultados

Umbral de confianza 0,39, elegido en validación y medido una vez en las 139 imágenes de `test`:

| AP@0,5 | AP@0,5:0,95 | Precisión | Recall | F1 |
| ---: | ---: | ---: | ---: | ---: |
| 0,901 | 0,738 | 0,934 | 0,895 | 0,914 |

## Archivos

| Archivo | Para qué sirve |
| --- | --- |
| `detector/merge_manual.py` | Incorpora 40 imágenes anotadas a mano a `train` (convierte COCO a YOLO). |
| `detector/augment.py` | Aumento de datos solo para `train`: escala, traslación, brillo, contraste, desenfoque y ruido; sin volteo. |
| `detector/pilot.py` | Carga de imágenes y etiquetas, lotes con aumento y piloto corto de comprobación. |
| `detector/train.py` | Entrenamiento completo con parada temprana; guarda `best.keras`, `history.csv` y las curvas. |
| `detector/evaluate.py` | Elige el umbral en `valid` y mide en `test`: AP, precisión, recall y F1. |
| `detector/demo.py` | Detección sobre una foto desde la línea de comandos. |
| `detector/plate_reader.py` | Capa de lectura con reglas de confianza, tamaño y formato de patente chilena. |
| `detector/web_demo.py` y `web/index.html` | Web local que une detección y lectura. |
| `Dockerfile`, `docker-compose.yml`, `deploy/` | Despliegue en un servidor con nginx. |

El código está en el paquete `detector/` y se ejecuta con `python -m detector.<módulo>` desde la raíz. Los tests están en `tests/`.

## Modelo entrenado

La corrida final está en `runs/ev03-20261115/`: `best.keras` (el modelo), `history.csv` y `curvas.png` (entrenamiento), `resumen.json` y, en `evaluacion/`, `metricas.json` y la curva precisión-recall. Para usar el modelo sin entrenar, apunta `TRAIN_OUTPUT_DIR` a esa carpeta.

## Preparar el entorno

Requiere Python 3.11. Las dependencias de entrenamiento y evaluación están en `requirements-pilot.txt`; el entrenamiento usa Keras 3 con motor PyTorch para aprovechar la GPU en Windows (`torch` y `torchvision` con CUDA, instalados aparte).

```powershell
python -m venv .venv-torch
.\.venv-torch\Scripts\python.exe -m pip install -r requirements-pilot.txt
```

## Corpus

El corpus no está en el repositorio. `DATASET_DIR` apunta a una carpeta con `train`, `valid` y `test`, cada una con `images` y `labels` en formato YOLO.

| Partición | Imágenes |
| --- | ---: |
| `train` | 1.141 |
| `valid` | 289 |
| `test` | 139 |

El corpus deriva de [My First Project v2 en Roboflow Universe](https://universe.roboflow.com/luiss-workspace-jfeoo/my-first-project-yzokm/dataset/2), con licencia CC BY 4.0 según su `data.yaml`.

## Entrenar y evaluar

`TRAIN_OUTPUT_DIR` debe ser una carpeta nueva fuera del corpus.

```powershell
$env:KERAS_BACKEND = "torch"
$env:DATASET_DIR = "C:\ruta\al\dataset"
$env:TRAIN_OUTPUT_DIR = "$PWD\runs\mi-corrida"
.\.venv-torch\Scripts\python.exe -m detector.train
.\.venv-torch\Scripts\python.exe -m detector.evaluate
```

## Ejecutar la web local

La lectura se elige con `OCR_BACKEND`. Con Ollama:

```powershell
ollama pull qwen3-vl:4b-instruct
$env:KERAS_BACKEND = "torch"
$env:TRAIN_OUTPUT_DIR = "$PWD\runs\mi-corrida"
$env:DEMO_HOST = "127.0.0.1"
$env:DEMO_PORT = "8765"
$env:OCR_BACKEND = "ollama"
$env:OCR_MODEL = "qwen3-vl:4b-instruct"
$env:OCR_OLLAMA_URL = "http://127.0.0.1:11434"
$env:OCR_MIN_CONFIDENCE = "0.60"
$env:OCR_MIN_CROP_WIDTH = "80"
.\.venv-torch\Scripts\python.exe -m detector.web_demo
```

Con Gemini, `OCR_BACKEND=gemini` y las variables que lista `.env.example`, incluida la clave. Si falta una variable, el programa lo indica y no arranca. Abre `http://127.0.0.1:8765`.

## Tests

```powershell
.\.venv-torch\Scripts\python.exe -m unittest discover -s tests -t .
```
