FROM python:3.9-slim

WORKDIR /app

# Install library berat (Tensorflow, Pandas)
COPY services/training_worker/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# COPY Shared Libraries & Data Utils
COPY src/ml_core /app/src/ml_core
COPY src/data_acquisition /app/src/data_acquisition

# COPY Source Code Worker
COPY services/training_worker/src /app/src

ENV PYTHONPATH="${PYTHONPATH}:/app"

# Worker langsung jalanin pipeline saat container nyala
CMD ["python", "src/pipeline.py"]