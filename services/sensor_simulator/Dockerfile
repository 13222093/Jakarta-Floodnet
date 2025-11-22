FROM python:3.9-slim

WORKDIR /app

# 1. Install dependencies
COPY services/sensor_simulator/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 2. Copy Source Code
COPY services/sensor_simulator/src /app/src

# 3. Setup ENV
ENV PYTHONPATH="${PYTHONPATH}:/app"

# 4. Run Simulator
CMD ["python", "src/main.py"]