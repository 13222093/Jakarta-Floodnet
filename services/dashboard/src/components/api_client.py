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
    
    def get_prediction(self, data: Dict[str, float]) -> Dict[str, Any]:
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
            
            # Kirim POST request ke endpoint prediksi
            response = requests.post(
                f"{self.base_url}/predict",
                json=data,
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
    
    def verify_image(self, image_file) -> Dict[str, Any]:
        """
        Verifikasi visual kondisi banjir menggunakan YOLO model
        
        Args:
            image_file: File gambar yang akan dianalisis
        
        Returns:
            Dictionary dengan hasil verifikasi visual
        """
        try:
            # Prepare file untuk upload
            files = {
                'file': ('image.jpg', image_file, 'image/jpeg')
            }
            
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
            return {
                'success': False,
                'message': f'Server error: {e.response.status_code}'
            }
        except Exception as e:
            logger.error(f"Unexpected error in verify_image: {str(e)}")
            return {
                'success': False,
                'message': f'Error: {str(e)}'
            }
    
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


# Singleton instance untuk digunakan di seluruh aplikasi
api_client = APIClient()
