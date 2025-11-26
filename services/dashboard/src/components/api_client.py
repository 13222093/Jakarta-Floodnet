"""
API Client untuk komunikasi dengan Backend API Gateway
Jakarta FloodNet Dashboard
"""

import requests
from typing import Dict, Any, Optional
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Base URL untuk API Gateway
API_BASE_URL = "http://localhost:8000"

class APIClient:
    """Client untuk komunikasi dengan API Gateway"""
    
    def __init__(self, base_url: str = API_BASE_URL):
        self.base_url = base_url
        self.timeout = 30  # seconds
    
    def check_health(self) -> Dict[str, Any]:
        """
        Health check untuk memastikan API Gateway online
        
        Returns:
            Dictionary dengan status health check
        """
        try:
            response = requests.get(
                f"{self.base_url}/health",
                timeout=self.timeout
            )
            response.raise_for_status()
            
            return {
                'success': True,
                'status': 'online',
                'data': response.json(),
                'message': 'API Gateway terhubung'
            }
            
        except requests.exceptions.ConnectionError:
            logger.error("Connection error: API Gateway tidak dapat dijangkau")
            return {
                'success': False,
                'status': 'offline',
                'message': 'Gagal terhubung ke server. Pastikan API Gateway berjalan.'
            }
        except requests.exceptions.Timeout:
            logger.error("Timeout: API Gateway tidak merespons")
            return {
                'success': False,
                'status': 'timeout',
                'message': 'Server tidak merespons dalam waktu yang ditentukan.'
            }
        except Exception as e:
            logger.error(f"Unexpected error in health check: {str(e)}")
            return {
                'success': False,
                'status': 'error',
                'message': f'Error: {str(e)}'
            }
    
    def get_prediction(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Mendapatkan prediksi level air dari LSTM model
        
        Args:
            data: Dictionary dengan keys:
                - hujan_bogor: Curah hujan di Bogor (mm)
                - hujan_jakarta: Curah hujan di Jakarta (mm)
                - current_water_level: Ketinggian air saat ini (cm) - optional
        
        Returns:
            Dictionary dengan hasil prediksi
        """
        try:
            # Validasi input
            required_fields = ['hujan_bogor', 'hujan_jakarta']
            for field in required_fields:
                if field not in data:
                    return {
                        'success': False,
                        'message': f'Field {field} diperlukan'
                    }
            
            # Convert frontend field names to backend expected format
            # Backend expects: water_level_cm, rainfall_mm
            # Frontend sends: hujan_bogor, hujan_jakarta, current_water_level
            
            # Use Jakarta rainfall as primary rainfall value
            rainfall_mm = float(data['hujan_jakarta'])
            
            # Use current_water_level if provided, otherwise default to 100.0
            water_level_cm = float(data.get('current_water_level', 100.0))
            
            # Backend expected payload format
            payload = {
                'rainfall_mm': rainfall_mm,
                'water_level_cm': water_level_cm
            }
            
            logger.info(f"Sending prediction request with payload: {payload}")
            logger.info(f"Original frontend data: {data}")
            
            # Kirim POST request ke endpoint prediksi
            response = requests.post(
                f"{self.base_url}/predict",
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            result = response.json()
            
            return {
                'success': True,
                'data': result,
                'message': 'Prediksi berhasil didapatkan'
            }
            
        except requests.exceptions.ConnectionError:
            logger.error("Connection error in get_prediction")
            return {
                'success': False,
                'message': 'Gagal terhubung ke server prediksi'
            }
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error in get_prediction: {e}")
            return {
                'success': False,
                'message': f'Server error: {e.response.status_code}'
            }
        except Exception as e:
            logger.error(f"Unexpected error in get_prediction: {str(e)}")
            return {
                'success': False,
                'message': f'Error: {str(e)}'
            }
    
    def verify_image(self, image_file, filename: str = "image.jpg") -> Dict[str, Any]:
        """
        Verifikasi visual kondisi banjir menggunakan YOLO model
        
        Args:
            image_file: File gambar yang akan dianalisis (bytes atau file-like object)
            filename: Nama file (default: "image.jpg")
        
        Returns:
            Dictionary dengan hasil verifikasi visual
        """
        try:
            # Detect file type based on filename or default to jpeg
            if filename.lower().endswith('.png'):
                mime_type = 'image/png'
            elif filename.lower().endswith('.jpg') or filename.lower().endswith('.jpeg'):
                mime_type = 'image/jpeg'
            elif filename.lower().endswith('.gif'):
                mime_type = 'image/gif'
            else:
                mime_type = 'image/jpeg'  # default
            
            # Prepare file untuk upload dengan format yang benar
            if hasattr(image_file, 'read'):
                # File-like object
                file_content = image_file.read()
            else:
                # Bytes
                file_content = image_file
            
            files = {
                'file': (filename, file_content, mime_type)
            }
            
            logger.info(f"Sending image verification request for file: {filename}")
            
            # Kirim POST request dengan file gambar
            response = requests.post(
                f"{self.base_url}/verify-visual",
                files=files,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            result = response.json()
            
            return {
                'success': True,
                'data': result,
                'message': 'Analisis visual berhasil'
            }
            
        except requests.exceptions.ConnectionError:
            logger.error("Connection error in verify_image")
            return {
                'success': False,
                'message': 'Gagal terhubung ke server verifikasi visual'
            }
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error in verify_image: {e}")
            error_detail = ""
            try:
                if e.response.text:
                    error_detail = f" - {e.response.text}"
            except:
                pass
            return {
                'success': False,
                'message': f'Server error: {e.response.status_code}{error_detail}'
            }
        except Exception as e:
            logger.error(f"Unexpected error in verify_image: {str(e)}")
            return {
                'success': False,
                'message': f'Error: {str(e)}'
            }
    
    def verify_visual(self, image_file, filename: str = "image.jpg") -> Dict[str, Any]:
        """
        Alias untuk verify_image untuk konsistensi dengan backend API
        """
        return self.verify_image(image_file, filename)
    
    def get_history(self, limit: int = 100) -> Dict[str, Any]:
        """
        Mendapatkan data historis prediksi
        
        Args:
            limit: Jumlah data yang diambil
        
        Returns:
            Dictionary dengan data historis
        """
        try:
            response = requests.get(
                f"{self.base_url}/history",
                params={'limit': limit},
                timeout=self.timeout
            )
            response.raise_for_status()
            
            result = response.json()
            
            return {
                'success': True,
                'data': result,
                'message': 'Data historis berhasil didapatkan'
            }
            
        except requests.exceptions.ConnectionError:
            logger.error("Connection error in get_history")
            return {
                'success': False,
                'message': 'Gagal terhubung ke server',
                'data': []
            }
        except Exception as e:
            logger.error(f"Error in get_history: {str(e)}")
            return {
                'success': False,
                'message': f'Error: {str(e)}',
                'data': []
            }
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        Mendapatkan status sistem lengkap
        
        Returns:
            Dictionary dengan status semua komponen sistem
        """
        try:
            response = requests.get(
                f"{self.base_url}/status",
                timeout=self.timeout
            )
            response.raise_for_status()
            
            result = response.json()
            
            return {
                'success': True,
                'data': result,
                'message': 'Status sistem berhasil didapatkan'
            }
            
        except Exception as e:
            logger.error(f"Error in get_system_status: {str(e)}")
            return {
                'success': False,
                'message': f'Error: {str(e)}',
                'data': {
                    'lstm_model': 'unknown',
                    'yolo_model': 'unknown',
                    'api_gateway': 'error'
                }
            }

    def get_scenarios(self) -> Dict[str, Any]:
        """
        Mendapatkan daftar skenario demo
        """
        try:
            response = requests.get(
                f"{self.base_url}/scenarios",
                timeout=self.timeout
            )
            response.raise_for_status()
            return {
                'success': True,
                'data': response.json(),
                'message': 'Skenario berhasil didapatkan'
            }
        except Exception as e:
            logger.error(f"Error in get_scenarios: {str(e)}")
            return {
                'success': False,
                'message': f'Error: {str(e)}',
                'data': []
            }

    def predict_scenario(self, scenario_id: str) -> Dict[str, Any]:
        """
        Menjalankan prediksi berdasarkan skenario
        """
        try:
            response = requests.post(
                f"{self.base_url}/predict/scenario/{scenario_id}",
                timeout=self.timeout
            )
            response.raise_for_status()
            return {
                'success': True,
                'data': response.json(),
                'message': 'Prediksi skenario berhasil'
            }
        except Exception as e:
            logger.error(f"Error in predict_scenario: {str(e)}")
            return {
                'success': False,
                'message': f'Error: {str(e)}'
            }


# Singleton instance untuk digunakan di seluruh aplikasi
api_client = APIClient()
