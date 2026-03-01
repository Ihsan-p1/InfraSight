"""
Re-download Roboflow dataset WITHOUT filtering to check original class distribution
"""
from roboflow import Roboflow


def download_roboflow_raw(api_key, output_dir="data/raw/roboflow_raw"):
    """
    Download RDD2022 dataset WITHOUT filtering
    """
    print("=" * 70)
    print("ROBOFLOW RDD2022 RAW DOWNLOAD (No Filtering)")
    print("=" * 70)
    
    # Initialize Roboflow
    print(f"\n🔑 Initializing Roboflow...")
    rf = Roboflow(api_key=api_key)
    
    # Access the project
    print("📦 Accessing RDD2022 project...")
    project = rf.workspace("iitintern").project("rdd2022-7ybsh")
    
    # Get dataset version 3
    print("ℹ️  Getting dataset version 3...")
    version = project.version(3)
    
    # Download in YOLOv8 format
    print("\n📥 Downloading dataset (YOLOv8 format)...")
    print("   This will NOT be filtered - checking original classes")
    
    dataset = version.download("yolov8", location=output_dir)
    
    print(f"\n✅ Dataset downloaded to: {output_dir}")
    print(f"   Dataset path: {dataset.location}")
    
    return dataset


if __name__ == "__main__":
    API_KEY = "gYoNG4iczFaZ6gEeyV6S"
    
    # Download without filtering
    dataset = download_roboflow_raw(API_KEY)
    
    print("\n✅ Raw dataset ready!")
    print("\nNext: Run analyze_dataset_classes.py to check distribution")
