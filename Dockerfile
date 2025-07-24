# syntax=docker/dockerfile:1

# --- Etapa 1: Builder ---
FROM python:3.9-slim AS builder

WORKDIR /app

# Instalar dependencias del sistema necesarias para la compilación
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements y crear wheels
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip wheel --no-cache-dir --wheel-dir=/app/wheels -r requirements.txt

# --- Etapa 2: Runtime ---
FROM python:3.9-slim

# Variables de entorno
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/home/appuser/.local/bin:$PATH"

WORKDIR /app

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    libgl1 \
    libglib2.0-0 \
    graphviz \
    git \
    && rm -rf /var/lib/apt/lists/*

# Crear usuario no root
RUN useradd --create-home appuser && \
    chown -R appuser:appuser /app

# Copiar wheels y establecer permisos
COPY --from=builder --chown=appuser:appuser /app/wheels /app/wheels

# Cambiar al usuario no root
USER appuser

# Instalar dependencias desde wheels
RUN pip install --no-cache-dir --user --no-index --find-links=/app/wheels /app/wheels/* && \
    rm -rf /app/wheels

# Copiar código de la aplicación (actualizar esta sección)
COPY --chown=appuser:appuser ./src /app/src
COPY --chown=appuser:appuser ./api.py /app/
COPY --chown=appuser:appuser ./models /app/models

# Asegurar permisos correctos
RUN chmod -R 755 /app/src && \
    chmod 755 /app/api.py && \
    chmod -R 755 /app/models

# Crear directorio para archivos temporales
RUN mkdir -p /app/temp_audio && \
    chmod 755 /app/temp_audio

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]