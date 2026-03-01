"""
Quick script to check class distribution in Roboflow dataset
"""
from pathlib import Path
from collections import Counter

def analyze_dataset(dataset_path):
    dataset_path = Path(dataset_path)
    
    all_classes = Counter()
    
    for split in ['train', 'valid', 'test']:
        labels_dir = dataset_path / split / 'labels'
        if not labels_dir.exists():
            continue
            
        print(f"\n{split.upper()} split:")
        split_classes = Counter()
        
        for label_file in labels_dir.glob('*.txt'):
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        # First element is class ID
                        class_id = parts[0]
                        split_classes[class_id] += 1
                        all_classes[class_id] += 1
        
        for class_id, count in sorted(split_classes.items()):
            print(f"  Class {class_id}: {count} instances")
    
    print("\n" + "=" * 60)
    print("TOTAL DISTRIBUTION:")
    print("=" * 60)
    for class_id, count in sorted(all_classes.items()):
        print(f"Class {class_id}: {count} instances")

if __name__ == "__main__":
    analyze_dataset("data/raw/roboflow_raw")
