# Same API outside Modal: docker build -t reviewlens-api . && docker run -p 8000:8000 reviewlens-api
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 KERAS_BACKEND=torch TOKENIZERS_PARALLELISM=false HF_HOME=/cache/hf
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && rm -rf /var/lib/apt/lists/*

WORKDIR /srv
COPY requirements.txt .
RUN pip install --no-cache-dir torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY assets ./assets

RUN useradd -m api && mkdir -p /cache/hf && chown -R api /cache /srv
USER api
EXPOSE 8000
CMD ["uvicorn", "app.main:get_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
