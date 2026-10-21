FROM python:3.10-slim

WORKDIR /app/ui

# Pick up the latest ca-certificates (CVE-2026-111943)
RUN apt-get update && apt-get install -y --no-install-recommends --only-upgrade ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies
RUN pip install gradio==5.9.1 requests==2.32.3

# Create necessary directories
RUN mkdir -p data/inputs data/outputs

# Copy the application files
COPY . .

ENV API_HOST=kokoro-tts
ENV API_PORT=8880

# Run the Gradio app
CMD ["python", "app.py"]
