FROM python:3.9-slim

WORKDIR /app

# Simulator butuh requests library
RUN pip install requests pandas numpy

COPY src/data_acquisition /app/src/data_acquisition
COPY services/sensor_simulator/sim_script.py /app/

ENV PYTHONPATH="${PYTHONPATH}:/app"

CMD ["python", "sim_script.py"]