# Use official Python runtime as a parent image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set work directory
WORKDIR /app

# Install system dependencies (Required for OpenCV)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Expose the port (Railway uses PORT env var, but we expose 8000 as default)
EXPOSE 8000

# Run the application
# Using shell form to allow variable expansion if needed, but exec form is safer
CMD ["uvicorn", "services.api_gateway.src.main:app", "--host", "0.0.0.0", "--port", "8000"]
