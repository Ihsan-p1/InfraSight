"""
Direct Download Script for RDD2022 India Dataset
Downloads from official S3 bucket with progress bar
"""
import urllib.request
import os
from pathlib import Path
from tqdm import tqdm
import zipfile


class DownloadProgressBar(tqdm):
    """Progress bar for urllib downloads"""
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_file(url, output_path):
    """Download file with progress bar"""
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)


def main():
    print("=" * 70)
    print("RDD2022 India Dataset - Direct Download")
    print("=" *70)
    
    # Setup paths
    data_dir = Path("data/raw")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    zip_file = data_dir / "RDD2022_India.zip"
    extract_dir = data_dir / "RDD2022"
    
    # Download URL
    url = "https://bigdatacup.s3.ap-northeast-1.amazonaws.com/2022/CRDDC2022/RDD2022/Country_Specific_Data_CRDDC2022/RDD2022_India.zip"
    
    print(f"\n📥 Downloading RDD2022 India Dataset...")
    print(f"   Source: {url}")
    print(f"   Size: 502.3 MB")
    print(f"   Output: {zip_file.absolute()}\n")
    
    # Download
    try:
        download_file(url, zip_file)
        print(f"\n✅ Download complete!")
    except Exception as e:
        print(f"\n❌ Download failed: {e}")
        return
    
    # Extract
    print(f"\n📦 Extracting dataset...")
    print(f"   Destination: {extract_dir.absolute()}\n")
    
    try:
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            # Get total file count for progress
            file_list = zip_ref.namelist()
            
            # Extract with progress bar
            for file in tqdm(file_list, desc="Extracting", unit="file"):
                zip_ref.extract(file, extract_dir)
        
        print(f"\n✅ Extraction complete!")
        
        # Optional: Remove zip file to save space
        print(f"\n🗑️  Removing zip file to save space...")
        zip_file.unlink()
        print(f"✅ Cleanup complete!")
        
    except Exception as e:
        print(f"\n❌ Extraction failed: {e}")
        return
    
    # Verify structure
    print(f"\n📁 Dataset Structure:")
    india_dir = extract_dir / "India"
    if india_dir.exists():
        print(f"   ✓ India/")
        for subdir in ["train", "test"]:
            subdir_path = india_dir / subdir
            if subdir_path.exists():
                print(f"   ✓ India/{subdir}/")
                for sub in subdir_path.iterdir():
                    if sub.is_dir():
                        count = len(list(sub.glob("*")))
                        print(f"     ✓ {subdir}/{sub.name}/ ({count} files)")
    
    print(f"\n✅ RDD2022 India dataset ready!")
    print(f"📍 Location: {extract_dir.absolute()}")
    print(f"\n🚀 Next step: python models/training/train_yolo.py")


if __name__ == "__main__":
    main()
