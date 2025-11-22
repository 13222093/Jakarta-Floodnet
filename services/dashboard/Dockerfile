FROM python:3.9-slim

WORKDIR /app

# 1. Install Dependencies
COPY services/dashboard/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 2. Copy Source Code
COPY services/dashboard/src /app/src

# 3. Expose Port Streamlit
EXPOSE 8501

# 4. Run Dashboard
CMD ["streamlit", "run", "src/app.py", "--server.port=8501", "--server.address=0.0.0.0"]