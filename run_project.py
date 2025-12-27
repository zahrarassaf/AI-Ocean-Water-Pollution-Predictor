#!/usr/bin/env python3
"""
Ocean Water Pollution Prediction System - Quick Start
Run this script to execute the complete project pipeline.
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description):
    """Run a shell command with descriptive output"""
    print(f"\n{'='*60}")
    print(f"🚀 {description}")
    print(f"{'='*60}")
    print(f"Command: {command}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, 
                              capture_output=True, text=True)
        print(f"✅ Success")
        if result.stdout:
            print(f"Output: {result.stdout[:500]}...")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed with error: {e}")
        if e.stderr:
            print(f"Error details: {e.stderr}")
        return False

def main():
    """Main execution flow"""
    print("\n" + "="*60)
    print("🌊 OCEAN WATER POLLUTION PREDICTION SYSTEM")
    print("="*60)
    
    # Check Python version
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    print(f"Python version: {python_version}")
    
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher required")
        return
    
    # Create directory structure
    directories = [
        "data/raw",
        "data/processed",
        "models",
        "results",
        "visualizations"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"📁 Created: {directory}")
    
    # Install requirements
    if not os.path.exists("requirements.txt"):
        print("📝 Creating requirements.txt...")
        with open("requirements.txt", "w") as f:
            f.write("""numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
joblib>=1.1.0
plotly>=5.0.0
""")
    
    # Ask user for action
    print("\n📋 Available Actions:")
    print("1. Process data and train model")
    print("2. Run predictions only")
    print("3. Launch dashboard")
    print("4. Run complete pipeline")
    
    try:
        choice = int(input("\nSelect action (1-4): "))
    except:
        choice = 4
    
    # Execute based on choice
    if choice == 1:
        # Process data and train
        run_command("python process_data.py", "Processing ocean data")
        run_command("python train_model.py", "Training AI model")
        
    elif choice == 2:
        # Run predictions
        run_command("python predict.py", "Running pollution predictions")
        
    elif choice == 3:
        # Launch dashboard
        try:
            import streamlit
            run_command("streamlit run dashboard.py", "Launching AI Dashboard")
        except ImportError:
            print("Streamlit not installed. Installing...")
            run_command("pip install streamlit", "Installing Streamlit")
            run_command("streamlit run dashboard.py", "Launching AI Dashboard")
            
    elif choice == 4:
        # Complete pipeline
        print("\n" + "="*60)
        print("🏃‍♂️ RUNNING COMPLETE PIPELINE")
        print("="*60)
        
        steps = [
            ("python process_data.py", "Data Processing"),
            ("python train_model.py", "Model Training"),
            ("python predict.py", "Prediction Testing"),
            ("python -c \"import sys; sys.path.append('.'); from predict import OceanPollutionPredictor; p = OceanPollutionPredictor()\"", 
             "Model Validation")
        ]
        
        for cmd, desc in steps:
            if not run_command(cmd, desc):
                print(f"Pipeline stopped at: {desc}")
                break
        else:
            print("\n" + "="*60)
            print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
            print("="*60)
            print("\nNext steps:")
            print("1. Check 'models/' for trained models")
            print("2. Run 'python predict.py' for predictions")
            print("3. Run 'streamlit run dashboard.py' for dashboard")
            print("4. Add your real NetCDF data to 'data/raw/'")
    
    print("\n" + "="*60)
    print("🏁 Execution complete!")
    print("="*60)

if __name__ == "__main__":
    main()
