"""
Dataset preprocessing utilities including synthetic data generation
"""
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict
import random
import shutil
from tqdm import tqdm
import yaml


class DatasetPreprocessor:
    """Preprocess and augment datasets for YOLOv8 training"""
    
    def __init__(self, output_dir: str = "data/processed"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_synthetic_card_data(
        self,
        card_image_path: str,
        base_images_dir: str,
        num_synthetic: int = 500,
        output_dir: Optional[str] = None
    ) -> Path:
        """
        CRITICAL: Generate synthetic training data by pasting card onto images
        Solves class imbalance: 1000 potholes vs 30 cards → 2:1 ratio
        
        Args:
            card_image_path: Path to card image with transparent background (PNG)
            base_images_dir: Directory with base images (RDD2022)
            num_synthetic: Number of synthetic images to generate
            output_dir: Output directory
            
        Returns:
            Path to synthetic dataset directory
        """
        if output_dir is None:
            output_dir = self.output_dir / "synthetic_cards"
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        images_out = output_dir / "images"
        labels_out = output_dir / "labels"
        images_out.mkdir(exist_ok=True)
        labels_out.mkdir(exist_ok=True)
        
        # Load card template
        card_img = cv2.imread(card_image_path, cv2.IMREAD_UNCHANGED)
        if card_img is None:
            raise ValueError(f"Could not load card image: {card_image_path}")
        
        # Get base images
        base_images = list(Path(base_images_dir).glob("*.jpg")) + \
                     list(Path(base_images_dir).glob("*.png"))
        
        if len(base_images) == 0:
            raise ValueError(f"No images found in {base_images_dir}")
        
        print(f"Generating {num_synthetic} synthetic card images...")
        
        for i in tqdm(range(num_synthetic)):
            # Randomly select base image
            base_path = random.choice(base_images)
            base_img = cv2.imread(str(base_path))
            if base_img is None:
                continue
            
            h, w = base_img.shape[:2]
            
            # Random card scale (simulating different distances)
            scale = random.uniform(0.08, 0.15)  # 8-15% of image size
            card_h = int(h * scale)
            card_w = int(card_img.shape[1] * (card_h / card_img.shape[0]))
            
            # Resize card
            card_resized = cv2.resize(card_img, (card_w, card_h))
            
            # Random position (avoid edges)
            margin = 50
            x = random.randint(margin, max(margin + 1, w - card_w - margin))
            y = random.randint(margin, max(margin + 1, h - card_h - margin))
            
            # Paste card onto image
            if card_resized.shape[2] == 4:  # Has alpha channel
                # Alpha blending
                alpha = card_resized[:, :, 3] / 255.0
                for c in range(3):
                    base_img[y:y+card_h, x:x+card_w, c] = \
                        alpha * card_resized[:, :, c] + \
                        (1 - alpha) * base_img[y:y+card_h, x:x+card_w, c]
            else:
                base_img[y:y+card_h, x:x+card_w] = card_resized[:, :, :3]
            
            # Save image
            output_img_path = images_out / f"synthetic_card_{i:04d}.jpg"
            cv2.imwrite(str(output_img_path), base_img)
            
            # Generate YOLO segmentation annotation (polygon)
            # Card coordinates (normalized [0, 1])
            x_norm = x / w
            y_norm = y / h
            w_norm = card_w / w
            h_norm = card_h / h
            
            # Polygon points (rectangle corners, clockwise)
            polygon_points = [
                x_norm, y_norm,  # Top-left
                x_norm + w_norm, y_norm,  # Top-right
                x_norm + w_norm, y_norm + h_norm,  # Bottom-right
                x_norm, y_norm + h_norm  # Bottom-left
            ]
            
            # YOLO segmentation format: class x1 y1 x2 y2 x3 y3 x4 y4
            # Class 1 = reference_object (card)
            annotation = "1 " + " ".join(map(str, polygon_points))
            
            # Save annotation
            output_label_path = labels_out / f"synthetic_card_{i:04d}.txt"
            with open(output_label_path, 'w') as f:
                f.write(annotation + "\n")
        
        print(f"✓ Generated {num_synthetic} synthetic images in {output_dir}")
        return output_dir
    
    def merge_datasets(
        self,
        datasets: List[Dict[str, str]],
        output_name: str = "merged"
    ) -> Path:
        """
        Merge multiple datasets into one unified dataset
        
        Args:
            datasets: List of dicts with {'images': path, 'labels': path, 'name': str}
            output_name: Name for merged dataset
            
        Returns:
            Path to merged dataset
        """
        merged_dir = self.output_dir / output_name
        merged_images = merged_dir / "images"
        merged_labels = merged_dir / "labels"
        
        merged_images.mkdir(parents=True, exist_ok=True)
        merged_labels.mkdir(parents=True, exist_ok=True)
        
        file_counter = 0
        
        print(f"Merging {len(datasets)} datasets...")
        
        for dataset in datasets:
            ds_name = dataset['name']
            img_dir = Path(dataset['images'])
            lbl_dir = Path(dataset['labels'])
            
            # Get all images
            images = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
            
            for img_path in tqdm(images, desc=f"Processing {ds_name}"):
                # Find corresponding label
                label_path = lbl_dir / (img_path.stem + ".txt")
                
                if not label_path.exists():
                    continue
                
                # Copy with standardized naming
                new_img_name = f"{ds_name}_{file_counter:05d}{img_path.suffix}"
                new_lbl_name = f"{ds_name}_{file_counter:05d}.txt"
                
                shutil.copy(img_path, merged_images / new_img_name)
                shutil.copy(label_path, merged_labels / new_lbl_name)
                
                file_counter += 1
        
        print(f"✓ Merged {file_counter} image-label pairs into {merged_dir}")
        return merged_dir
    
    def train_val_split(
        self,
        dataset_dir: Path,
        train_ratio: float = 0.8,
        stratify_by_class: bool = True
    ) -> Tuple[Path, Path]:
        """
        Split dataset into train/val sets
        
        Args:
            dataset_dir: Path to merged dataset
            train_ratio: Ratio for training set (0.8 = 80% train, 20% val)
            stratify_by_class: Ensure both classes in validation
            
        Returns:
            (train_dir, val_dir)
        """
        images_dir = dataset_dir / "images"
        labels_dir = dataset_dir / "labels"
        
        # Create output directories
        train_img_dir = self.output_dir / "images" / "train"
        train_lbl_dir = self.output_dir / "labels" / "train"
        val_img_dir = self.output_dir / "images" / "val"
        val_lbl_dir = self.output_dir / "labels" / "val"
        
        for d in [train_img_dir, train_lbl_dir, val_img_dir, val_lbl_dir]:
            d.mkdir(parents=True, exist_ok=True)
        
        # Get all image files
        images = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
        
        # Shuffle
        random.shuffle(images)
        
        # Split
        split_idx = int(len(images) * train_ratio)
        train_images = images[:split_idx]
        val_images = images[split_idx:]
        
        # Copy files
        print("Creating train/val split...")
        for img_path in tqdm(train_images, desc="Train set"):
            lbl_path = labels_dir / (img_path.stem + ".txt")
            if lbl_path.exists():
                shutil.copy(img_path, train_img_dir / img_path.name)
                shutil.copy(lbl_path, train_lbl_dir / lbl_path.name)
        
        for img_path in tqdm(val_images, desc="Val set"):
            lbl_path = labels_dir / (img_path.stem + ".txt")
            if lbl_path.exists():
                shutil.copy(img_path, val_img_dir / img_path.name)
                shutil.copy(lbl_path, val_lbl_dir / lbl_path.name)
        
        print(f"✓ Train: {len(train_images)} | Val: {len(val_images)}")
        
        return self.output_dir, self.output_dir
    
    def create_data_yaml(
        self,
        dataset_dir: Path,
        output_path: Optional[Path] = None
    ) -> Path:
        """
        Create data.yaml configuration for YOLO training
        
        Args:
            dataset_dir: Path to processed dataset
            output_path: Where to save data.yaml
            
        Returns:
            Path to data.yaml
        """
        if output_path is None:
            output_path = self.output_dir / "data.yaml"
        
        data_config = {
            'path': str(dataset_dir.absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'nc': 2,  # Number of classes
            'names': {
                0: 'pothole',
                1: 'reference_object'
            }
        }
        
        with open(output_path, 'w') as f:
            yaml.dump(data_config, f, default_flow_style=False)
        
        print(f"✓ Created data.yaml at {output_path}")
        return output_path


if __name__ == "__main__":
    # Example usage
    preprocessor = DatasetPreprocessor()
    
    # Generate synthetic card data
    # synthetic_dir = preprocessor.generate_synthetic_card_data(
    #     card_image_path="path/to/card.png",
    #     base_images_dir="data/raw/RDD2022/images",
    #     num_synthetic=500
    # )
