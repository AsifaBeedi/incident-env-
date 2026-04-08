FROM python:3.11-slim

# --- system deps (nothing extra needed for this pure-Python env) ----------
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
    && rm -rf /var/lib/apt/lists/*

# --- working directory -----------------------------------------------------
WORKDIR /app

# --- Python dependencies ---------------------------------------------------
# Copy requirements first so Docker can cache this layer independently
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- application source ----------------------------------------------------
COPY inference.py openenv.yaml ./
COPY env/ ./env/

# --- runtime environment variables (override at docker run) ----------------
ENV API_BASE_URL=""
ENV MODEL_NAME=""
ENV HF_TOKEN=""

# --- OpenEnv validator entry point -----------------------------------------
# The validator calls:  docker run <image>
# inference.py writes the required [START]/[STEP]/[END] lines to stdout.
CMD ["python", "inference.py"]