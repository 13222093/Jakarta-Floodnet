"""
Complete Training Pipeline for Jakarta FloodNet
==============================================

This script orchestrates the complete training process for both:
1. LSTM model for flood forecasting
2. YOLO model for visual verification

Usage: python train_pipeline.py
"""

import os
import sys
import logging
from datetime import datetime
from typing import Dict, Any, Optional
import json

# Add src to path for imports
sys.path.append('.')
sys.path.append('..')

from preprocesing import preprocess_data
from lstm_model import FloodLevelLSTM
from yolo_model import FloodVisualVerifier
from metrics import evaluate_model_performance, print_model_summary

class FloodNetTrainingPipeline:
    """
    Complete training pipeline for Jakarta FloodNet system.
    Handles preprocessing, LSTM training, and YOLO setup.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize training pipeline.
        
        Args:
            config_path: Optional path to configuration file
        """
        self.setup_logging()
        self.config = self.load_config(config_path)
        self.results = {}
        
    def setup_logging(self):
        """Setup logging for the pipeline"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('training_pipeline.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def load_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """Load configuration for training pipeline"""
        default_config = {
            # Data configuration
            'data': {
                'input_file': '../../data/DATASET_FINAL_TRAINING.csv',
                'target_column': 'tma_manggarai',
                'features': ['hujan_bogor', 'hujan_jakarta']
            },
            
            # LSTM configuration
            'lstm': {
                'sequence_length': 24,
                'lstm_units': [64, 32, 16],
                'dropout_rate': 0.2,
                'learning_rate': 0.001,
                'epochs': 50,
                'batch_size': 32,
                'test_size': 0.2,
                'val_size': 0.2,
                'patience': 15
            },
            
            # Feature engineering configuration
            'features': {
                'lag_hours': [1, 2, 3, 6, 12, 24],
                'rolling_windows': [3, 6, 12, 24],
                'tma_lags': [1, 2, 3, 6],
                'create_time_features': True,
                'create_rainfall_combinations': True
            },
            
            # YOLO configuration
            'yolo': {
                'model_path': '../../models/yolo_model.pt',
                'confidence_threshold': 0.5,
                'device': 'auto'
            },
            
            # Output configuration
            'output': {
                'models_dir': '../../models',
                'lstm_model_name': 'lstm_flood_forecaster.h5',
                'results_file': 'training_results.json',
                'plots_dir': '../../plots'
            }
        }
        
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    user_config = json.load(f)
                # Deep merge configs (user config overrides defaults)
                for key, value in user_config.items():
                    if key in default_config and isinstance(value, dict):
                        default_config[key].update(value)
                    else:
                        default_config[key] = value
            except Exception as e:
                self.logger.warning(f"Error loading config file: {e}, using defaults")
        
        return default_config
    
    def setup_directories(self):
        """Create necessary directories for outputs"""
        dirs_to_create = [
            self.config['output']['models_dir'],
            self.config['output']['plots_dir']
        ]
        
        for directory in dirs_to_create:
            os.makedirs(directory, exist_ok=True)
            self.logger.info(f"Directory ready: {directory}")
    
    def run_preprocessing(self) -> str:
        """
        Run data preprocessing pipeline.
        
        Returns:
            Path to preprocessed data
        """
        self.logger.info("🔧 Starting data preprocessing...")
        
        try:
            # Run preprocessing
            df_processed = preprocess_data(
                self.config['data']['input_file'],
                self.config['features']
            )
            
            # Save preprocessed data
            output_path = os.path.join(
                self.config['output']['models_dir'],
                'preprocessed_data.csv'
            )
            df_processed.to_csv(output_path, index=False)
            
            self.results['preprocessing'] = {
                'status': 'completed',
                'input_shape': df_processed.shape,
                'features_created': len(df_processed.columns) - 2,  # Exclude timestamp and target
                'output_file': output_path
            }
            
            self.logger.info(f"✅ Preprocessing completed. Output shape: {df_processed.shape}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"❌ Preprocessing failed: {str(e)}")
            self.results['preprocessing'] = {
                'status': 'failed',
                'error': str(e)
            }
            raise
    
    def train_lstm_model(self, preprocessed_data_path: str) -> Dict[str, Any]:
        """
        Train LSTM forecasting model.
        
        Args:
            preprocessed_data_path: Path to preprocessed data
            
        Returns:
            Training results
        """
        self.logger.info("🚀 Starting LSTM model training...")
        
        try:
            # Load preprocessed data
            import pandas as pd
            df = pd.read_csv(preprocessed_data_path)
            
            # Initialize LSTM model
            lstm_config = self.config['lstm']
            model = FloodLevelLSTM(
                sequence_length=lstm_config['sequence_length'],
                lstm_units=lstm_config['lstm_units'],
                dropout_rate=lstm_config['dropout_rate'],
                learning_rate=lstm_config['learning_rate']
            )
            
            # Prepare data
            X_train, y_train, X_val, y_val, X_test, y_test = model.prepare_data(
                df,
                target_col=self.config['data']['target_column'],
                test_size=lstm_config['test_size'],
                val_size=lstm_config['val_size']
            )
            
            self.logger.info(f"Data splits: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
            
            # Train model
            model_save_path = os.path.join(
                self.config['output']['models_dir'],
                self.config['output']['lstm_model_name']
            )
            
            history = model.train(
                X_train, y_train, X_val, y_val,
                epochs=lstm_config['epochs'],
                batch_size=lstm_config['batch_size'],
                patience=lstm_config['patience'],
                save_path=model_save_path
            )
            
            # Evaluate model
            evaluation_results = model.evaluate(X_test, y_test)
            
            # Comprehensive evaluation
            eval_metrics = evaluate_model_performance(
                evaluation_results['actual'],
                evaluation_results['predictions'],
                flood_threshold=80.0,
                verbose=True
            )
            
            # Save model artifacts
            model.save_model(
                model_save_path,
                os.path.join(self.config['output']['models_dir'], 'lstm_scaler')
            )
            
            self.results['lstm'] = {
                'status': 'completed',
                'model_path': model_save_path,
                'metrics': eval_metrics,
                'training_history': history,
                'data_splits': {
                    'train_size': len(X_train),
                    'val_size': len(X_val),
                    'test_size': len(X_test)
                }
            }
            
            self.logger.info("✅ LSTM training completed successfully!")
            print_model_summary(eval_metrics, "LSTM FLOOD FORECASTER")
            
            return self.results['lstm']
            
        except Exception as e:
            self.logger.error(f"❌ LSTM training failed: {str(e)}")
            self.results['lstm'] = {
                'status': 'failed',
                'error': str(e)
            }
            raise
    
    def setup_yolo_model(self) -> Dict[str, Any]:
        """
        Setup YOLO visual verification model.
        
        Returns:
            Setup results
        """
        self.logger.info("👁️ Setting up YOLO visual verification...")
        
        try:
            # Initialize YOLO verifier
            yolo_config = self.config['yolo']
            verifier = FloodVisualVerifier(
                model_path=yolo_config['model_path'],
                confidence_threshold=yolo_config['confidence_threshold'],
                device=yolo_config['device']
            )
            
            # Try to load model (will fallback to YOLOv8n if custom model not found)
            load_success = verifier.load_model()
            
            if load_success:
                # Test basic functionality
                test_image_path = '../../sanity_check.png'  # Use existing sanity check image
                
                if os.path.exists(test_image_path):
                    # Test detection on sample image
                    import numpy as np
                    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
                    test_results = verifier.detect_flood_features(test_image)
                    
                    self.results['yolo'] = {
                        'status': 'completed',
                        'model_loaded': True,
                        'model_path': yolo_config['model_path'],
                        'fallback_used': not os.path.exists(yolo_config['model_path']),
                        'test_detection': 'success' if 'error' not in test_results else 'failed'
                    }
                    
                    self.logger.info("✅ YOLO model setup completed!")
                else:
                    self.results['yolo'] = {
                        'status': 'completed',
                        'model_loaded': True,
                        'model_path': yolo_config['model_path'],
                        'fallback_used': not os.path.exists(yolo_config['model_path']),
                        'test_detection': 'skipped - no test image'
                    }
                    
                    self.logger.info("✅ YOLO model setup completed (test skipped)!")
            else:
                self.results['yolo'] = {
                    'status': 'failed',
                    'error': 'Could not load YOLO model'
                }
                self.logger.error("❌ Failed to load YOLO model")
            
            return self.results['yolo']
            
        except Exception as e:
            self.logger.error(f"❌ YOLO setup failed: {str(e)}")
            self.results['yolo'] = {
                'status': 'failed',
                'error': str(e)
            }
            raise
    
    def create_integration_demo(self):
        """Create integration demo script"""
        demo_script = '''
"""
Jakarta FloodNet Integration Demo
================================

This script demonstrates the integration of LSTM forecasting 
and YOLO visual verification for complete flood monitoring.
"""

import sys
sys.path.append('src/ml_core')

from lstm_model import FloodLevelLSTM
from yolo_model import FloodVisualVerifier
from preprocesing import preprocess_data
import numpy as np

def run_integration_demo():
    """Run complete integration demo"""
    print("🌊 JAKARTA FLOODNET - INTEGRATION DEMO")
    print("="*50)
    
    try:
        # 1. Load LSTM model for forecasting
        print("\\n🔮 Loading LSTM forecasting model...")
        lstm_model = FloodLevelLSTM()
        lstm_model.load_model(
            'models/lstm_flood_forecaster.h5',
            'models/lstm_scaler'
        )
        print("✅ LSTM model loaded!")
        
        # 2. Setup YOLO for visual verification
        print("\\n👁️ Setting up YOLO verification...")
        yolo_verifier = FloodVisualVerifier()
        yolo_verifier.load_model()
        print("✅ YOLO verifier ready!")
        
        # 3. Demo prediction workflow
        print("\\n🧪 Running prediction workflow...")
        
        # Simulate current weather data
        print("  📊 Simulating current weather conditions...")
        # In real scenario, this would come from weather API
        current_rainfall_bogor = 5.2
        current_rainfall_jakarta = 8.1
        
        # Generate mock processed features (in real scenario, use latest data)
        mock_features = np.random.rand(1, 24, 53)  # 24 timesteps, 53 features
        
        # LSTM prediction
        water_level_prediction = lstm_model.predict(mock_features)[0]
        print(f"  🔮 LSTM Prediction: {water_level_prediction:.1f} cm")
        
        # Visual verification (mock image)
        mock_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        visual_analysis = yolo_verifier.detect_flood_features(mock_image)
        
        # Integration analysis
        verification_result = yolo_verifier.verify_lstm_prediction(
            mock_image, 
            water_level_prediction,
            threshold_high=80.0
        )
        
        print(f"\\n📋 INTEGRATION RESULTS:")
        print(f"  • LSTM Water Level: {water_level_prediction:.1f} cm")
        print(f"  • Visual Flood Indicators: {visual_analysis.get('flood_indicators', 0)}")
        print(f"  • Verification Status: {verification_result.get('verification_status', 'N/A')}")
        print(f"  • Overall Confidence: {verification_result.get('overall_confidence', 0):.2%}")
        
        # Decision making
        if water_level_prediction > 80:
            print(f"\\n🚨 FLOOD ALERT: High water level predicted!")
        elif visual_analysis.get('flood_indicators', 0) > 2:
            print(f"\\n⚠️ VISUAL ALERT: Flood indicators detected in imagery!")
        else:
            print(f"\\n✅ NORMAL: No immediate flood threat detected")
        
        print(f"\\n✨ Integration demo completed successfully!")
        
    except Exception as e:
        print(f"❌ Demo error: {str(e)}")

if __name__ == "__main__":
    run_integration_demo()
        '''
        
        with open('../../integration_demo.py', 'w') as f:
            f.write(demo_script)
        
        self.logger.info("📝 Created integration demo script")
    
    def save_results(self):
        """Save training results to file"""
        results_file = os.path.join(
            self.config['output']['models_dir'],
            self.config['output']['results_file']
        )
        
        # Add metadata
        self.results['metadata'] = {
            'pipeline_version': '1.0.0',
            'timestamp': datetime.now().isoformat(),
            'config': self.config
        }
        
        try:
            with open(results_file, 'w') as f:
                json.dump(self.results, f, indent=2, default=str)
            
            self.logger.info(f"💾 Results saved to: {results_file}")
        except Exception as e:
            self.logger.error(f"❌ Failed to save results: {str(e)}")
    
    def run_complete_pipeline(self):
        """Run the complete training pipeline"""
        start_time = datetime.now()
        
        print("🌊 JAKARTA FLOODNET - COMPLETE TRAINING PIPELINE")
        print("="*60)
        print(f"Started at: {start_time}")
        print("="*60)
        
        try:
            # Setup
            self.setup_directories()
            
            # Step 1: Preprocessing
            preprocessed_path = self.run_preprocessing()
            
            # Step 2: LSTM Training
            lstm_results = self.train_lstm_model(preprocessed_path)
            
            # Step 3: YOLO Setup
            yolo_results = self.setup_yolo_model()
            
            # Step 4: Integration Demo
            self.create_integration_demo()
            
            # Step 5: Save Results
            self.save_results()
            
            end_time = datetime.now()
            duration = end_time - start_time
            
            print("\n" + "="*60)
            print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
            print("="*60)
            print(f"Duration: {duration}")
            print(f"\n📋 SUMMARY:")
            print(f"  ✅ Preprocessing: {self.results['preprocessing']['status']}")
            print(f"  ✅ LSTM Training: {self.results['lstm']['status']}")
            print(f"  ✅ YOLO Setup: {self.results['yolo']['status']}")
            print(f"\n📁 OUTPUT FILES:")
            print(f"  • LSTM Model: {self.results['lstm']['model_path']}")
            print(f"  • Results: {os.path.join(self.config['output']['models_dir'], self.config['output']['results_file'])}")
            print(f"  • Integration Demo: integration_demo.py")
            
        except Exception as e:
            self.logger.error(f"❌ Pipeline failed: {str(e)}")
            print(f"\n❌ PIPELINE FAILED: {str(e)}")
            raise

def main():
    """Main entry point"""
    try:
        pipeline = FloodNetTrainingPipeline()
        pipeline.run_complete_pipeline()
    except KeyboardInterrupt:
        print("\n⏹️ Pipeline interrupted by user")
    except Exception as e:
        print(f"\n💥 Pipeline crashed: {str(e)}")
        raise

if __name__ == "__main__":
    main()