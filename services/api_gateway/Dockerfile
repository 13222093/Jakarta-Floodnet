FROM python:3.9-slim

WORKDIR /app

# 1. Install dependencies
COPY services/api_gateway/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 2. COPY Shared Libraries (Ini kuncinya!)
# Mengambil folder src/ml_core dari Root Project ke dalam container
COPY src/ml_core /app/src/ml_core

# 3. COPY Source Code Service
COPY services/api_gateway/src /app/src
COPY services/api_gateway/src/main.py /app/

# 4. Setup Python Path
ENV PYTHONPATH="${PYTHONPATH}:/app"

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]