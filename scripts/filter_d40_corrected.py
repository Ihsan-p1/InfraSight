"""
Filter D40 (Pothole) from Roboflow RDD2022 - CORRECTED VERSION
D40 is CLASS 5, not class 3!
"""
from pathlib import Path
import shutil


def filter_d40_potholes_corrected(dataset_path, output_path):
    """
    Filter annotations to keep only CLASS 5 (D40 - Pothole)
    
    Roboflow RDD2022 class mapping:
    - 0: D00
    - 1: D01
    - 2: D10
    - 3: D11
    - 4: D20
    - 5: D40 (POTHOLE) <- Keep only this!
    - 6: D43
    - 7: D44
    - 8: D50
    """
    print("=" * 70)
    print("FILTERING D40 (POTHOLE) - CLASS 5")
    print("=" * 70)
    
    dataset_path = Path(dataset_path)
    output_path = Path(output_path)
    
    # Create output directories
    for split in ['train', 'valid', 'test']:
        (output_path / split / 'images').mkdir(parents=True, exist_ok=True)
        (output_path / split / 'labels').mkdir(parents=True, exist_ok=True)
    
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
        
        out_labels_dir = output_path / split / 'labels'
        out_images_dir = output_path / split / 'images'
        
        if not labels_dir.exists():
            continue
            
        print(f"\n📂 Processing {split} split...")
        
        for label_file in labels_dir.glob('*.txt'):
            stats['total_images'] += 1
            
            # Corresponding image
            # Label: India_000001_jpg.rf.xxx.txt
            # Image: India_000001_jpg.rf.xxx.jpg
            image_file = images_dir / f"{label_file.stem}.jpg"
            
            if not image_file.exists():
                continue
            
            # Read annotations
            with open(label_file, 'r') as f:
                lines = f.readlines()
            
            # Filter only class 5 (D40 - Pothole)
            pothole_lines = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    if class_id == 5:  # D40 pothole
                        # Remap class 5 -> 0 (pothole becomes class 0)
                        pothole_lines.append(f"0 {' '.join(parts[1:])}\n")
                        stats['total_potholes'] += 1
                    else:
                        stats['other_damages_filtered'] += 1
            
            # Save or skip
            if pothole_lines:
                # Copy image
                shutil.copy2(image_file, out_images_dir / image_file.name)
                
                # Save filtered annotations
                with open(out_labels_dir / label_file.name, 'w') as f:
                    f.writelines(pothole_lines)
                
                stats['images_with_potholes'] += 1
            else:
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


def create_data_yaml(output_path, stats):
    """
    Create data.yaml for pothole-only dataset
    """
    print("\n📝 Creating data.yaml...")
    
    yaml_path = Path(output_path) / 'data.yaml'
    
    yaml_content = f"""# RDD2022 - D40 Pothole Only (Corrected)
path: {Path(output_path).absolute()}
train: train/images
val: valid/images
test: test/images

nc: 1
names: ['pothole']

# Statistics:
# Total pothole instances: {stats['total_potholes']}
# Images with potholes: {stats['images_with_potholes']}
# Filtered from 9 RDD2022 damage classes
"""
    
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    print(f"✅ data.yaml created at: {yaml_path}")


if __name__ == "__main__":
    # Paths
    input_path = "data/raw/roboflow_raw"
    output_path = "data/processed/rdd2022_d40_only"
    
    # Filter D40 (class 5)
    stats = filter_d40_potholes_corrected(input_path, output_path)
    
    # Create data.yaml
    create_data_yaml(output_path, stats)
    
    print("\n✅ D40 Pothole dataset ready!")
    print(f"\nDataset location: {output_path}")
    print(f"Next: Train YOLOv8m with this dataset")
