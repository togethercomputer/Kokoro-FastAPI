FROM python:3.11-slim

WORKDIR /app/ui

# Pick up the latest ca-certificates (CVE-2026-111943)
RUN apt-get update && apt-get install -y --no-install-recommends --only-upgrade ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies
# gradio caps pillow<13; the UI uses no image components, so take the patched pillow
RUN pip install --no-cache-dir gradio==6.16.0 requests==2.33.0 \
    && pip install --no-cache-dir --no-deps "pillow>=13.0.0"

# Create necessary directories
RUN mkdir -p data/inputs data/outputs

# Copy the application files
COPY . .

ENV API_HOST=kokoro-tts
ENV API_PORT=8880

# Run the Gradio app
CMD ["python", "app.py"]
