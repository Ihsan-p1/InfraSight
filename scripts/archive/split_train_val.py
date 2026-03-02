"""
Split converted YOLO dataset into train and val sets
"""
import shutil
from pathlib import Path
import random


def split_train_val(
    data_dir="data/processed/rdd2022_yolo",
    val_split=0.2,
    seed=42
):
    """Split training data into train and val sets"""
    print("Splitting dataset into train/val...")
    
    data_path = Path(data_dir)
    train_imgs = sorted((data_path / 'images' / 'train').glob('*.jpg'))
    
    random.seed(seed)
    random.shuffle(train_imgs)
    
    split_idx = int(len(train_imgs) * (1 - val_split))
    train_list = train_imgs[:split_idx]
    val_list = train_imgs[split_idx:]
    
    print(f"Total: {len(train_imgs)}, Train: {len(train_list)}, Val: {len(val_list)}")
    
    #Move val images and labels
    val_img_dir = data_path / 'images' / 'val'
    val_lbl_dir = data_path / 'labels' / 'val'
    
    for img in val_list:
        # Move image
        shutil.move(str(img), str(val_img_dir / img.name))
        
        # Move label
        lbl = data_path / 'labels' / 'train' / (img.stem + '.txt')
        if lbl.exists():
            shutil.move(str(lbl), str(val_lbl_dir / lbl.name))
    
    print("Split complete!")


if __name__ == "__main__":
    split_train_val()
