#!/usr/bin/env python3
"""
Quick start script for marine productivity predictor
"""

import subprocess
import sys
import os

def run_command(cmd, description):
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}")
    print(f"Running: {cmd}")
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✓ Success")
        if result.stdout:
            print(f"Output: {result.stdout[:500]}...")
    else:
        print("✗ Failed")
        print(f"Error: {result.stderr}")
        return False
    
    return True

def main():
    print("Marine Productivity Predictor - Quick Start")
    print("=" * 60)
    
    # 1. Install minimal requirements
    if not run_command(
        "pip install numpy pandas scikit-learn xarray netcdf4 gdown",
        "1. Installing core dependencies"
    ):
        return
    
    # 2. Create directory structure
    dirs = ["data/raw", "data/processed", "models", "results", "logs"]
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
    print("\n✓ Created directory structure")
    
    # 3. Run simplified download
    print("\n" + "="*60)
    print("2. Downloading test data")
    print("="*60)
    
    # Create simple download script
    download_script = """
import gdown
import os

urls = [
    "https://drive.google.com/file/d/17YEvHDE9DmtLsXKDGYwrGD8OE46swNDc/view?usp=sharing",
    "https://drive.google.com/file/d/16DyROUrgvfRQRvrBS3W3Y4o44vd-ZB67/view?usp=sharing",
    "https://drive.google.com/file/d/1c3f92nsOCY5hJv3zy0SAPXf8WsAVYNVI/view?usp=sharing",
    "https://drive.google.com/file/d/1JapTCN9CLn_hy9CY4u3Gcanv283LBdXy/view?usp=sharing"
]

filenames = ["chlorophyll.nc", "light_attenuation.nc", "water_clarity.nc", "productivity.nc"]

for url, filename in zip(urls, filenames):
    print(f"Downloading {filename}...")
    try:
        # Extract file ID
        file_id = url.split('/d/')[1].split('/')[0]
        gdown.download(f"https://drive.google.com/uc?id={file_id}", f"data/raw/{filename}", quiet=False)
        print(f"✓ Downloaded {filename}")
    except Exception as e:
        print(f"✗ Failed to download {filename}: {e}")
"""
    
    with open("quick_download.py", "w") as f:
        f.write(download_script)
    
    run_command("python quick_download.py", "Downloading data")
    
    # 4. Clean up
    if os.path.exists("quick_download.py"):
        os.remove("quick_download.py")
    
    print("\n" + "="*60)
    print("QUICK START COMPLETED")
    print("="*60)
    print("\nNext steps:")
    print("1. Check downloaded files in data/raw/")
    print("2. Run: python run_pipeline.py --data-dir data/raw")
    print("3. Or run individual scripts:")
    print("   - python scripts/download_data.py")
    print("   - python scripts/train_marine.py")

if __name__ == "__main__":
    main()
