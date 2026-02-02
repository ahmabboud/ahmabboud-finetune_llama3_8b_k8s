# Training Docker image for Llama-3 function calling fine-tuning
# Based on NVIDIA PyTorch container with all dependencies pre-installed

FROM nvcr.io/nvidia/pytorch:24.07-py3

LABEL description="Llama-3 Function Calling Fine-tuning Environment"

# Set working directory
WORKDIR /app

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install additional dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Install flash-attn (requires special handling)
RUN pip install flash-attn --no-build-isolation

# Copy source code
COPY src/ /app/src/
COPY scripts/ /app/scripts/
COPY configs/ /app/configs/

# Make scripts executable
RUN chmod +x /app/scripts/*.py

# Set PYTHONPATH
ENV PYTHONPATH=/app:$PYTHONPATH

# Default command
CMD ["python", "-c", "print('Training container ready')"]
