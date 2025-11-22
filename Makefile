.PHONY: up down build logs clean train check

# 1. Jalankan Semua Service (Background)
up:
	docker-compose up -d
	@echo "🚀 FloodNet is RUNNING!"
	@echo "📊 Dashboard: http://localhost:8501"
	@echo "📡 API Docs:  http://localhost:8000/docs"

# 2. Matikan Semua Service
down:
	docker-compose down
	@echo "🛑 FloodNet Stopped."

# 3. Build Ulang (Kalau ada perubahan kode/requirements)
build:
	docker-compose build

# 4. Paksa Build Ulang Bersih (Kalau error aneh)
rebuild:
	docker-compose build --no-cache

# 5. Lihat Log (Semua Service)
logs:
	docker-compose logs -f

# 6. Lihat Log Spesifik (Misal cuma API)
log-api:
	docker-compose logs -f api_gateway

log-dash:
	docker-compose logs -f dashboard

log-sim:
	docker-compose logs -f sensor_simulator

# 7. Bersih-bersih Docker (Hati-hati!)
clean:
	docker-compose down -v
	docker system prune -f

# 8. Utilitas: Download Data (Jalankan script lokal jika perlu, opsional)
download-data:
	python src/data_acquisition/download_data2020.py