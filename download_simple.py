import gdown
from pathlib import Path

# Create directory
Path("data/raw").mkdir(parents=True, exist_ok=True)

urls = [
    "https://drive.google.com/file/d/17YEvHDE9DmtLsXKDGYwrGD8OE46swNDc/view",
    "https://drive.google.com/file/d/16DyROUrgvfRQRvrBS3W3Y4o44vd-ZB67/view",
    "https://drive.google.com/file/d/1c3f92nsOCY5hJv3zy0SAPXf8WsAVYNVI/view",
    "https://drive.google.com/file/d/1JapTCN9CLn_hy9CY4u3Gcanv283LBdXy/view"
]

filenames = ["chlorophyll.nc", "light_attenuation.nc", "water_clarity.nc", "productivity.nc"]

print("Downloading marine data...")
for url, filename in zip(urls, filenames):
    try:
        # Extract file ID
        file_id = url.split('/d/')[1].split('/')[0]
        output = f"data/raw/{filename}"
        
        print(f"Downloading {filename}...")
        gdown.download(f"https://drive.google.com/uc?id={file_id}", output, quiet=False)
        
        if Path(output).exists():
            size = Path(output).stat().st_size / (1024 * 1024)
            print(f"Success: {filename} ({size:.1f} MB)")
        else:
            print(f"Failed: {filename}")
            
    except Exception as e:
        print(f"Error downloading {filename}: {e}")

print("\nDone!")
