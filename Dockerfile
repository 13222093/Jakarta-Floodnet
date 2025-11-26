# STAGE 1: Builder
FROM python:3.10-slim AS builder

WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install build dependencies
# We install libgl1 here just in case, but it's crucial in the final stage
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies to a specific location
COPY requirements.txt .
RUN pip install --prefix=/install --no-cache-dir -r requirements.txt

# STAGE 2: Final
FROM python:3.10-slim

WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install runtime system dependencies (CRITICAL for OpenCV)
# These must be installed in the final image to run
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed python packages from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Run command
CMD ["uvicorn", "services.api_gateway.src.main:app", "--host", "0.0.0.0", "--port", "8000"]
