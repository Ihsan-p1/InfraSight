"""
Download Indonesian Pothole Dataset from Roboflow
Workspace: deteksi-lubang-jalan
Project: jalan-berlubang-yufpf
Version: 8
Format: yolov8
"""
from roboflow import Roboflow
import os
from pathlib import Path

def download_dataset():
    print("Initializing Roboflow...")
    rf = Roboflow(api_key="BIuNP6BsX47uaOad322d")
    
    print("Accessing workspace and project...")
    project = rf.workspace("deteksi-lubang-jalan").project("jalan-berlubang-yufpf")
    version = project.version(8)
    
    # Download to data/raw/indonesian_pothole dataset for better organization
    target_dir = Path("data/raw/indonesian_pothole")
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # Set the current working directory to the target dir so Roboflow downloads it there
    # Or we can just let it download and move it, but Roboflow usually downloads to CWD / project_name
    original_cwd = os.getcwd()
    os.chdir(str(target_dir))
    
    try:
        print(f"Downloading dataset to {target_dir.absolute()}...")
        dataset = version.download("yolov8")
        print(f"Dataset downloaded successfully at {dataset.location}")
    except Exception as e:
        print(f"Error downloading dataset: {e}")
    finally:
        os.chdir(original_cwd)

if __name__ == "__main__":
    download_dataset()
