FROM python:3.10-slim

# ---------- env ----------
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    TZ=Asia/Shanghai

WORKDIR /app

# ---------- system deps (opencv / paddleocr / poppler) ----------
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    ca-certificates curl \
    libglib2.0-0 libsm6 libxext6 libxrender1 libgl1 \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# ---------- python deps (cache-friendly) ----------
COPY requirements.txt ./
RUN python -m pip install -U pip setuptools wheel && \
    (pip install -r requirements.txt --only-binary=:all: || pip install -r requirements.txt)

# ---------- app code ----------
COPY . .

# ---------- security: non-root ----------
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 5005

# ---------- production server ----------
# app:app  =>  app.py 里的 Flask 实例变量名是 app
CMD ["gunicorn","-w","2","-k","gthread","--threads","8","-b","0.0.0.0:5005","--timeout","120","--access-logfile","-","--error-logfile","-","app:app"]
