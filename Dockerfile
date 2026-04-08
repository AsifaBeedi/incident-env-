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
COPY models.py tasks.py env.py inference.py app.py openenv.yaml ./

# --- runtime environment variables (override at docker run) ----------------
ENV API_BASE_URL=""
ENV MODEL_NAME=""
ENV HF_TOKEN=""

# --- OpenEnv validator entry point -----------------------------------------
# The validator calls:  docker run <image>
# inference.py writes the required [START]/[STEP]/[END] lines to stdout.
# Expose the OpenEnv API port
EXPOSE 7860

# Start the FastAPI server on 0.0.0.0:7860 (Hugging Face Spaces default port)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]