#!/usr/bin/env python3
"""
Quick runner for the complete project.
"""

import subprocess
import sys
from pathlib import Path

def run_command(command, description):
    print(f"\n{'='*60}")
    print(f"📋 {description}")
    print(f"💻 Command: {command}")
    print('='*60)
    
    try:
        result = subprocess.run(command, shell=True, check=True)
        print(f"✅ {description} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed!")
        print(f"Error: {e}")
        return False

def main():
    print("🚀 MARINE POLLUTION PREDICTION - QUICK START")
    print("="*60)
    
    # Step 1: Check if data exists
    data_dir = Path("data/processed")
    if not any(data_dir.glob("*.joblib")):
        print("\n📥 Step 1: Downloading data...")
        success = run_command(
            "python run_pipeline.py --steps download",
            "Download data"
        )
        if not success:
            print("Data download failed. Using existing data if available.")
    else:
        print("✅ Data already exists. Skipping download.")
    
    # Step 2: Train model
    print("\n🤖 Step 2: Training model...")
    # Find the latest processed data
    processed_files = list(data_dir.glob("*/*.joblib"))
    if processed_files:
        latest_data = max(processed_files, key=lambda x: x.stat().st_mtime)
        success = run_command(
            f'python complete_training_pipeline.py "{latest_data}" --output-dir "results/quick_run"',
            "Train model"
        )
    else:
        print("❌ No processed data found!")
        sys.exit(1)
    
    # Step 3: Deploy API
    print("\n🌐 Step 3: Starting API server...")
    print("Note: You need to manually run in a new terminal:")
    print("python deploy_model.py serve")
    print("\nOr press Enter to continue...")
    input()
    
    print("\n" + "="*60)
    print("🎉 COMPLETE!")
    print("="*60)
    print("Next steps:")
    print("1. Open new terminal")
    print("2. Run: python deploy_model.py serve")
    print("3. Open: http://localhost:8000/docs")
    print("="*60)

if __name__ == "__main__":
    main()
