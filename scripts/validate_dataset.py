"""
Validate and clean YOLO dataset
Remove images with empty or invalid annotations
"""
from pathlib import Path
import shutil


def validate_and_clean_dataset(data_dir="data/processed/rdd2022_yolo"):
    """
    Validate YOLO dataset and remove invalid entries
    """
    print("=" * 70)
    print("Dataset Validator")
    print("=" * 70)
    
    data_path = Path(data_dir)
    
    for split in ['train', 'val']:
        print(f"\nValidating {split} set...")
        
        img_dir = data_path / 'images' / split
        lbl_dir = data_path / 'labels' / split
        
        if not img_dir.exists():
            print(f"  {img_dir} not found, skipping")
            continue
        
        images = sorted(img_dir.glob('*.jpg'))
        print(f"  Found {len(images)} images")
        
        removed_count = 0
        valid_count = 0
        
        for img_file in images:
            lbl_file = lbl_dir / (img_file.stem + '.txt')
            
            # Check if label exists
            if not lbl_file.exists():
                print(f"  Removing {img_file.name} (no label)")
                img_file.unlink()
                removed_count += 1
                continue
            
            # Check if label is empty or has invalid content
            try:
                with open(lbl_file, 'r') as f:
                    lines = f.readlines()
                
                # Remove if empty
                if len(lines) == 0:
                    print(f"  Removing {img_file.name} (empty label)")
                    img_file.unlink()
                    lbl_file.unlink()
                    removed_count += 1
                    continue
                
                # Validate each line
                valid_lines = []
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) >= 5:  # class_id x y w h (minimum for bbox)
                        # Check values are valid floats
                        try:
                            class_id = int(parts[0])
                            x, y, w, h = map(float, parts[1:5])
                            
                            # Check ranges
                            if 0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1:
                                valid_lines.append(line)
                        except:
                            pass
                
                # If no valid lines, remove
                if len(valid_lines) == 0:
                    print(f"  Removing {img_file.name} (all invalid annotations)")
                    img_file.unlink()
                    lbl_file.unlink()
                    removed_count += 1
                    continue
                
                # If some lines were invalid, rewrite file with only valid ones
                if len(valid_lines) < len(lines):
                    with open(lbl_file, 'w') as f:
                        f.writelines(valid_lines)
                
                valid_count += 1
                
            except Exception as e:
                print(f"  Error processing {img_file.name}: {e}")
                img_file.unlink()
                if lbl_file.exists():
                    lbl_file.unlink()
                removed_count += 1
        
        print(f"\n  Valid: {valid_count}")
        print(f"  Removed: {removed_count}")
    
    print("\n" + "=" * 70)
    print("Validation complete!")
    print("=" * 70)


if __name__ == "__main__":
    validate_and_clean_dataset()
