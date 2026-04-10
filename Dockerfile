FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY env/ ./env/
COPY inference.py openenv.yaml ./
COPY server/ ./server/

ENV API_BASE_URL=""
ENV MODEL_NAME=""
ENV HF_TOKEN=""

EXPOSE 7860

# 🔧 FIXED: Use uvicorn directly (no 'python app.py' after it)
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
