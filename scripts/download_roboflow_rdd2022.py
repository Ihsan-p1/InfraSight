"""
Download Roboflow RDD2022 Dataset and Filter D40 (Pothole) Only
"""
from roboflow import Roboflow
from pathlib import Path
import shutil


def download_roboflow_dataset(api_key, output_dir="data/raw/roboflow_rdd2022"):
    """
    Download RDD2022 dataset from Roboflow
    """
    print("=" * 70)
    print("ROBOFLOW RDD2022 DATASET DOWNLOAD")
    print("=" * 70)
    
    # Initialize Roboflow
    print(f"\n🔑 Initializing Roboflow with API key...")
    rf = Roboflow(api_key=api_key)
    
    # Access the project
    print("📦 Accessing RDD2022 project...")
    project = rf.workspace("iitintern").project("rdd2022-7ybsh")
    
    # Get dataset version 3 (user confirmed working)
    print("ℹ️  Getting dataset version 3...")
    version = project.version(3)
    
    # Download in YOLOv8 format
    print("\n📥 Downloading dataset (YOLOv8 format)...")
    print("   This may take several minutes...")
    
    dataset = version.download("yolov8", location=output_dir)
    
    print(f"\n✅ Dataset downloaded to: {output_dir}")
    print(f"   Dataset path: {dataset.location}")
    
    return dataset


def filter_d40_potholes(dataset_path):
    """
    Filter annotations to keep only D40 (pothole) class
    RDD2022 class mapping:
    - 0: D00 (Longitudinal Crack)
    - 1: D10 (Transverse Crack)
    - 2: D20 (Alligator Crack)
    - 3: D40 (Pothole) <- Keep only this
    """
    print("\n" + "=" * 70)
    print("FILTERING D40 (POTHOLE) CLASS ONLY")
    print("=" * 70)
    
    dataset_path = Path(dataset_path)
    
    stats = {
        'total_images': 0,
        'images_with_potholes': 0,
        'images_removed': 0,
        'total_potholes': 0,
        'other_damages_filtered': 0
    }
    
    # Process train, valid, test splits
    for split in ['train', 'valid', 'test']:
        labels_dir = dataset_path / split / 'labels'
        images_dir = dataset_path / split / 'images'
        
        if not labels_dir.exists():
            continue
            
        print(f"\n📂 Processing {split} split...")
        
        for label_file in labels_dir.glob('*.txt'):
            stats['total_images'] += 1
            image_file = images_dir / f"{label_file.stem}.jpg"
            
            # Read annotations
            with open(label_file, 'r') as f:
                lines = f.readlines()
            
            # Filter only class 3 (D40 - Pothole)
            pothole_lines = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    if class_id == 3:
                        # Remap class 3 -> 0 (pothole becomes class 0)
                        pothole_lines.append(f"0 {' '.join(parts[1:])}\n")
                        stats['total_potholes'] += 1
                    else:
                        stats['other_damages_filtered'] += 1
            
            # Save or remove
            if pothole_lines:
                # Save filtered annotations
                with open(label_file, 'w') as f:
                    f.writelines(pothole_lines)
                stats['images_with_potholes'] += 1
            else:
                # Remove image and label (no potholes)
                label_file.unlink()
                if image_file.exists():
                    image_file.unlink()
                stats['images_removed'] += 1
    
    # Print statistics
    print("\n" + "=" * 70)
    print("FILTERING STATISTICS")
    print("=" * 70)
    print(f"Total images processed: {stats['total_images']}")
    print(f"Images with potholes (D40): {stats['images_with_potholes']}")
    print(f"Images removed (no potholes): {stats['images_removed']}")
    print(f"Total pothole instances: {stats['total_potholes']}")
    print(f"Other damage types filtered out: {stats['other_damages_filtered']}")
    print("=" * 70)
    
    return stats


def update_data_yaml(dataset_path):
    """
    Update data.yaml to reflect single pothole class
    """
    print("\n📝 Updating data.yaml...")
    
    yaml_path = Path(dataset_path) / 'data.yaml'
    
    yaml_content = f"""# RDD2022 - D40 Pothole Only
path: {Path(dataset_path).absolute()}
train: train/images
val: valid/images
test: test/images

nc: 1
names: ['pothole']

# Original RDD2022 had 4 classes:
# D00, D10, D20, D40
# We filtered to keep only D40 (Pothole)
"""
    
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    print(f"✅ data.yaml updated at: {yaml_path}")


if __name__ == "__main__":
    # API key (working key from user)
    API_KEY = "gYoNG4iczFaZ6gEeyV6S"
    
    # Download dataset
    dataset = download_roboflow_dataset(API_KEY)
    
    # Filter D40 only
    stats = filter_d40_potholes(dataset.location)
    
    # Update data.yaml
    update_data_yaml(dataset.location)
    
    print("\n✅ Dataset preparation complete!")
    print(f"\nDataset ready for training:")
    print(f"  - Location: {dataset.location}")
    print(f"  - Images with potholes: {stats['images_with_potholes']}")
    print(f"  - Total pothole instances: {stats['total_potholes']}")
    print(f"\nNext: Update training script to use this dataset")
