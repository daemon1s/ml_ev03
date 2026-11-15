FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HOME=/tmp

RUN useradd --system --uid 10001 --no-create-home --shell /usr/sbin/nologin app

WORKDIR /app

COPY requirements-deploy.txt .
RUN pip install -r requirements-deploy.txt

ARG OCR_BACKEND
ARG OCR_MODEL
ARG OCR_MODEL_DIR
RUN if [ "$OCR_BACKEND" = "fast_plate" ]; then \
        test -n "$OCR_MODEL" && test -n "$OCR_MODEL_DIR" \
        && python -c "import os; from pathlib import Path; from fast_plate_ocr.inference.hub import download_model; download_model(os.environ['OCR_MODEL'], save_directory=Path(os.environ['OCR_MODEL_DIR']))"; \
    fi

RUN apt-get update \
    && apt-get install -y --no-install-recommends fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

COPY detector/ detector/
COPY web/index.html web/index.html

USER app

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD ["python", "-c", "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:' + os.environ['DEMO_PORT'] + '/', timeout=5)"]

CMD ["python", "-m", "detector.web_demo"]
