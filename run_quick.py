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
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, encoding='utf-8')
    
    if result.returncode == 0:
        print("[SUCCESS]")
        if result.stdout:
            # Print only first 500 chars
            output_preview = result.stdout[:500]
            if len(result.stdout) > 500:
                output_preview += "..."
            print(f"Output: {output_preview}")
    else:
        print("[FAILED]")
        print(f"Error: {result.stderr}")
        return False
    
    return True

def main():
    print("Marine Productivity Predictor - Quick Start")
    print("=" * 60)
    
    # 1. Install minimal requirements (including bottleneck)
    if not run_command(
        "pip install numpy pandas scikit-learn xarray netcdf4 gdown bottleneck",
        "1. Installing core dependencies"
    ):
        return
    
    # 2. Create directory structure
    dirs = ["data/raw", "data/processed", "models", "results", "logs"]
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
    print("\n[SUCCESS] Created directory structure")
    
    # 3. Run simplified download
    print("\n" + "="*60)
    print("2. Downloading test data")
    print("="*60)
    
    # Create simple download script with ASCII characters only
    download_script = '''import gdown
import os
from pathlib import Path

urls = [
    "https://drive.google.com/file/d/17YEvHDE9DmtLsXKDGYwrGD8OE46swNDc/view?usp=sharing",
    "https://drive.google.com/file/d/16DyROUrgvfRQRvrBS3W3Y4o44vd-ZB67/view?usp=sharing",
    "https://drive.google.com/file/d/1c3f92nsOCY5hJv3zy0SAPXf8WsAVYNVI/view?usp=sharing",
    "https://drive.google.com/file/d/1JapTCN9CLn_hy9CY4u3Gcanv283LBdXy/view?usp=sharing"
]

filenames = ["chlorophyll.nc", "light_attenuation.nc", "water_clarity.nc", "productivity.nc"]

print("Starting download...")
for i, (url, filename) in enumerate(zip(urls, filenames)):
    print(f"Downloading file {i+1}/{len(urls)}: {filename}")
    
    try:
        # Extract file ID
        file_id = url.split('/d/')[1].split('/')[0]
        output_path = f"data/raw/{filename}"
        
        gdown.download(
            f"https://drive.google.com/uc?id={file_id}",
            output_path,
            quiet=False
        )
        
        # Check if file was downloaded
        if Path(output_path).exists():
            size_mb = Path(output_path).stat().st_size / (1024 * 1024)
            print(f"[SUCCESS] Downloaded: {filename} ({size_mb:.1f} MB)")
        else:
            print(f"[FAILED] File not created: {filename}")
            
    except Exception as e:
        print(f"[ERROR] Failed to download {filename}: {e}")

print("\\nDownload process completed.")'''
    
    # Write download script with explicit encoding
    try:
        with open("quick_download.py", "w", encoding="utf-8") as f:
            f.write(download_script)
    except Exception as e:
        print(f"[ERROR] Could not create download script: {e}")
        return
    
    # Run the download script
    if not run_command("python quick_download.py", "Downloading data"):
        print("[WARNING] Download had issues, but continuing...")
    
    # 4. Clean up
    if os.path.exists("quick_download.py"):
        try:
            os.remove("quick_download.py")
        except:
            pass
    
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
