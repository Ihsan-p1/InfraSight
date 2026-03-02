"""
Prepare Multiple Pothole Dataset
Organizes the downloaded Roboflow data and creates the YAML config.
"""
import os
import yaml
from pathlib import Path
import shutil

def prepare_data():
    raw_dir = Path("data/raw/multiple_pothole/PotHoleDetection-4")
    processed_dir = Path("data/processed/multiple_pothole")
    
    print(f"Checking for raw data in {raw_dir}...")
    
    if not raw_dir.exists():
        print(f"Error: Raw directory {raw_dir} not found. Ensure download is complete.")
        # Create a dummy structure if we want to test the script logic
        return

    # Create processed directory structure
    for split in ['train', 'valid', 'test']:
        (processed_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
        (processed_dir / 'labels' / split).mkdir(parents=True, exist_ok=True)

    print("Organizing files...")
    # Roboflow usually downloads with trial/valid/test folders
    for split in ['train', 'valid', 'test']:
        src_split = 'train' if split == 'train' else ('valid' if split == 'valid' else 'test')
        src_dir = raw_dir / src_split
        
        if not src_dir.exists():
            print(f"Warning: Source split {src_split} not found.")
            continue
            
        # Copy images
        img_src = src_dir / 'images'
        if img_src.exists():
            for img_file in img_src.glob('*'):
                shutil.copy(img_file, processed_dir / 'images' / split / img_file.name)
        
        # Copy labels
        lbl_src = src_dir / 'labels'
        if lbl_src.exists():
            for lbl_file in lbl_src.glob('*'):
                shutil.copy(lbl_file, processed_dir / 'labels' / split / lbl_file.name)

    # Create data.yaml
    data_config = {
        'path': str(processed_dir.absolute()),
        'train': 'images/train',
        'val': 'images/valid',
        'test': 'images/test',
        'names': {
            0: 'pothole'
        },
        'nc': 1
    }
    
    yaml_path = processed_dir / 'data.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(data_config, f, default_flow_style=False)
        
    print(f"Dataset preparation complete. YAML saved at {yaml_path}")

if __name__ == "__main__":
    prepare_data()
