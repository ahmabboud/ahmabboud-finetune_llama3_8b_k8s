FROM nvcr.io/nvidia/pytorch:24.12-py3

LABEL maintainer="Nebius AI Infrastructure"
LABEL description="GPU Cluster Acceptance Test"

WORKDIR /app

# Install additional dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy test scripts
COPY scripts/ /app/scripts/

# Make entrypoint executable
RUN chmod +x /app/scripts/entrypoint.sh

ENTRYPOINT ["/app/scripts/entrypoint.sh"]
