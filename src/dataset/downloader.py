"""
Dataset downloader for RDD2022 and Roboflow datasets
"""
import os
from pathlib import Path
from typing import Optional
import dataset_tools as dtools


class DatasetDownloader:
    """Handle downloading of pothole datasets"""
    
    def __init__(self, data_dir: str = "data/raw"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def download_rdd2022(self) -> Path:
        """
        Download RDD2022 (Road Damage Dataset) using dataset-tools
        
        Returns:
            Path to downloaded dataset
        """
        print("Downloading RDD2022 dataset...")
        dst_path = self.data_dir / "RDD2022"
        
        try:
            dtools.download(
                dataset='RDD2022',
                dst_dir=str(dst_path)
            )
            print(f"✓ RDD2022 downloaded to {dst_path}")
            return dst_path
        except Exception as e:
            print(f"✗ Error downloading RDD2022: {e}")
            raise
    
    def download_roboflow_dataset(
        self,
        api_key: str,
        workspace: str,
        project: str,
        version: int = 1
    ) -> Path:
        """
        Download pothole segmentation dataset from Roboflow
        
        Args:
            api_key: Roboflow API key
            workspace: Roboflow workspace name
            project: Project name
            version: Dataset version
            
        Returns:
            Path to downloaded dataset
        """
        try:
            from roboflow import Roboflow
        except ImportError:
            raise ImportError("Please install roboflow: pip install roboflow")
        
        print(f"Downloading Roboflow dataset: {workspace}/{project}...")
        dst_path = self.data_dir / "roboflow_pothole"
        
        try:
            rf = Roboflow(api_key=api_key)
            project_obj = rf.workspace(workspace).project(project)
            dataset = project_obj.version(version).download(
                model_format="yolov8",  # YOLOv8 segmentation format
                location=str(dst_path)
            )
            print(f"✓ Roboflow dataset downloaded to {dst_path}")
            return Path(dataset.location)
        except Exception as e:
            print(f"✗ Error downloading Roboflow dataset: {e}")
            raise
    
    def verify_dataset_structure(self, dataset_path: Path) -> bool:
        """
        Verify that dataset has required YOLO structure
        
        Expected structure:
            dataset/
                images/
                    train/
                    val/
                labels/
                    train/
                    val/
                data.yaml
        
        Returns:
            True if valid structure
        """
        required_dirs = [
            dataset_path / "images" / "train",
            dataset_path / "images" / "val",
            dataset_path / "labels" / "train",
            dataset_path / "labels" / "val"
        ]
        
        for dir_path in required_dirs:
            if not dir_path.exists():
                print(f"✗ Missing directory: {dir_path}")
                return False
        
        print(f"✓ Dataset structure valid: {dataset_path}")
        return True


if __name__ == "__main__":
    # Example usage
    downloader = DatasetDownloader()
    
    # Download RDD2022
    rdd_path = downloader.download_rdd2022()
    
    # Download from Roboflow (requires API key)
    # rf_path = downloader.download_roboflow_dataset(
    #     api_key="YOUR_API_KEY",
    #     workspace="workspace-name",
    #     project="pothole-segmentation"
    # )
