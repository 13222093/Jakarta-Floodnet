FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY services/training_worker/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy Shared Libraries
COPY src/ml_core /app/src/ml_core
COPY src/data_acquisition /app/src/data_acquisition

# Copy Worker Code
COPY services/training_worker/src /app/src

# Set PYTHONPATH agar root /app terdeteksi
ENV PYTHONPATH="${PYTHONPATH}:/app"

# Jalankan main.py (Entry Point baru)
CMD ["python", "src/main.py"]