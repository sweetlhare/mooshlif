# ШЛИФ-Скан: single-image сборка (frontend + backend)
# Сборка:  docker build -t shlifscan .
# Запуск:  docker compose up  (см. docker-compose.yml)

# --- этап 1: фронтенд ---
FROM node:20-slim AS frontend
WORKDIR /build
COPY app/frontend/package*.json ./
RUN npm ci
COPY app/frontend/ ./
RUN npm run build

# --- этап 2: бэкенд ---
FROM python:3.11-slim

# libvips для быстрых DZI-пирамид гигапиксельных панорам
RUN apt-get update && apt-get install -y --no-install-recommends \
    libvips42 libglib2.0-0 libgl1 libglx-mesa0 fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /srv/shlifscan

COPY requirements.txt ./
# CPU-профиль по умолчанию; для CUDA соберите с BASE_TORCH=cu121
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir torch torchvision --index-url ${TORCH_INDEX} \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir fastapi "uvicorn[standard]" python-multipart \
       segmentation-models-pytorch pyvips joblib

COPY shlifscan/ shlifscan/
COPY app/backend/ app/backend/
COPY app/__init__.py app/__init__.py
COPY --from=frontend /build/dist app/frontend/dist

# модели монтируются volume-ом или кладутся в образ при наличии
COPY models/ models/

EXPOSE 8000
ENV PYTHONUNBUFFERED=1
CMD ["uvicorn", "app.backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
