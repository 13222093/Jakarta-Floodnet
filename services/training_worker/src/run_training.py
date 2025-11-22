#!/usr/bin/env python3
"""
Jakarta FloodNet Training Launcher
=================================

Simple launcher script for the complete training pipeline.

Usage:
    python run_training.py                    # Use default config
    python run_training.py --config custom.json  # Use custom config
"""

import argparse
import sys
import os

def main():
    """Main launcher function"""
    parser = argparse.ArgumentParser(description='Jakarta FloodNet Training Pipeline')
    parser.add_argument(
        '--config', 
        type=str, 
        default='training_config.json',
        help='Path to configuration file (default: training_config.json)'
    )
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Run quick training with reduced epochs for testing'
    )
    
    args = parser.parse_args()
    
    print("🌊 JAKARTA FLOODNET - TRAINING LAUNCHER")
    print("="*50)
    
    # Check if we're in the right directory
    if not os.path.exists('train_pipeline.py'):
        print("❌ Error: train_pipeline.py not found!")
        print("Please run this script from src/ml_core/ directory.")
        sys.exit(1)
    
    # Import and run pipeline
    try:
        from services.training_worker.src.train_pipeline import FloodNetTrainingPipeline
        
        # Load config
        config_path = args.config if os.path.exists(args.config) else None
        
        if args.quick:
            print("🚀 Running in QUICK mode (reduced epochs)")
            # Override config for quick testing
            import json
            
            quick_config = {
                "lstm": {
                    "epochs": 10,  # Reduced epochs
                    "patience": 5
                },
                "features": {
                    "lag_hours": [1, 2, 3],  # Reduced features
                    "rolling_windows": [3, 6],
                    "tma_lags": [1, 2]
                }
            }
            
            # Save quick config
            with open('quick_config.json', 'w') as f:
                json.dump(quick_config, f, indent=2)
            
            config_path = 'quick_config.json'
        
        # Create and run pipeline
        pipeline = FloodNetTrainingPipeline(config_path)
        pipeline.run_complete_pipeline()
        
    except KeyboardInterrupt:
        print("\n⏹️ Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Training failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()