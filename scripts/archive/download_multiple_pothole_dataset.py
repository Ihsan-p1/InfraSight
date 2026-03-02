"""
Download Multiple Pothole Detection Dataset from Roboflow
Workspace: anurag-road-safety-project
Project: potholedetection-ak5kl
Version: 4
Format: yolov8
"""
from roboflow import Roboflow
import os
from pathlib import Path

def download_multiple_pothole_dataset():
    print("Initializing Roboflow...")
    # Using the API key provided by the user
    rf = Roboflow(api_key="REDACTED_API_KEY")
    
    print("Accessing workspace and project...")
    project = rf.workspace("anurag-road-safety-project").project("potholedetection-ak5kl")
    version = project.version(4)
    
    # Download to data/raw/multiple_pothole dataset
    target_dir = Path("data/raw/multiple_pothole")
    target_dir.mkdir(parents=True, exist_ok=True)
    
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
    download_multiple_pothole_dataset()
