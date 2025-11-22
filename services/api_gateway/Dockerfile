FROM python:3.9-slim

WORKDIR /app

# 1. Install Dependencies
COPY services/api_gateway/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 2. COPY Shared Libraries (JANTUNG SYSTEM)
# Ini mengambil folder src/ml_core dari root project ke dalam container
COPY src/ml_core /app/src/ml_core

# 3. Copy Source Code API
COPY services/api_gateway/src /app/src

# 4. Set Python Path agar bisa import 'src.ml_core'
ENV PYTHONPATH="${PYTHONPATH}:/app"

# 5. Jalankan Server
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]