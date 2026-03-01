"""
RDD2022 to YOLO Format Converter
Converts PASCAL VOC XML annotations to YOLO TXT format
"""
import xml.etree.ElementTree as ET
from pathlib import Path
import shutil
from tqdm import tqdm
import yaml


# RDD2022 damage classes mapping
RDD_CLASSES = {
    'D00': 0,  # Longitudinal Crack
    'D10': 1,  # Transverse Crack  
    'D20': 2,  # Alligator Crack
    'D40': 3,  # Pothole
}

# We'll map all to single "pothole" class for simplicity
USE_SINGLE_POTHOLE_CLASS = True  # Set True to combine all as "pothole"


def convert_box(size, box):
    """Convert PASCAL VOC bbox to YOLO format"""
    dw = 1.0 / size[0]
    dh = 1.0 / size[1]
    
    x_center = (box[0] + box[1]) / 2.0
    y_center = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    
    x_center *= dw
    w *= dw
    y_center *= dh
    h *= dh
    
    return (x_center, y_center, w, h)


def convert_annotation(xml_file, output_file):
    """Convert single XML file to YOLO format"""
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    size = root.find('size')
    if size is None:
        return False  
        
    w = int(size.find('width').text)
    h = int(size.find('height').text)
    
    objects = root.findall('object')
    if not objects:
        return False  
    
    with open(output_file, 'w') as f:
        for obj in objects:
            name = obj.find('name').text
            
            if USE_SINGLE_POTHOLE_CLASS:
                class_id = 0  
            else:
                if name not in RDD_CLASSES:
                    continue  
                class_id = RDD_CLASSES[name]
            
            xmlbox = obj.find('bndbox')
            if xmlbox is None:
                continue
                
            b = (
                float(xmlbox.find('xmin').text),
                float(xmlbox.find('xmax').text),
                float(xmlbox.find('ymin').text),
                float(xmlbox.find('ymax').text)
            )
            
            bbox = convert_box((w, h), b)
            line = "{} {:.6f} {:.6f} {:.6f} {:.6f}\n".format(class_id, bbox[0], bbox[1], bbox[2], bbox[3])
            f.write(line)
    
    return True


def convert_rdd2022_to_yolo(
    rdd_dir="data/raw/RDD2022/India",
    output_dir="data/processed/rdd2022_yolo"
):
    """Convert entire RDD2022 dataset to YOLO format"""
    print("=" * 70)
    print("RDD2022 to YOLO Converter")
    print("=" * 70)
    
    rdd_path = Path(rdd_dir)
    output_path = Path(output_dir)
    
    for split in ['train', 'val']:
        (output_path / 'images' / split).mkdir(parents=True, exist_ok=True)
        (output_path / 'labels' / split).mkdir(parents=True, exist_ok=True)
    
    print("\n Input:", rdd_path.absolute())
    print("Output:", output_path.absolute())
    
    for split_name, yolo_split in [('train', 'train'), ('test', 'val')]:
        print("\nProcessing", split_name, "set...")
        
        xml_dir = rdd_path / split_name / 'annotations' / 'xmls'
        img_dir = rdd_path / split_name / 'images'
        
        if not xml_dir.exists():
            print("Warning:", xml_dir, "not found, skipping")
            continue
        
        xml_files = sorted(xml_dir.glob('*.xml'))
        print("   Found", len(xml_files), "XML files")
        
        converted_count = 0
        skipped_count = 0
        
        for xml_file in tqdm(xml_files, desc="Converting " + split_name):
            img_name = xml_file.stem + '.jpg'
            img_file = img_dir / img_name
            
            if not img_file.exists():
                skipped_count += 1
                continue
            
            label_file = output_path / 'labels' / yolo_split / (xml_file.stem + ".txt")
            success = convert_annotation(xml_file, label_file)
            
            if success:
                dest_img = output_path / 'images' / yolo_split / img_name
                shutil.copy2(img_file, dest_img)
                converted_count += 1
            else:
                skipped_count += 1
        
        print("   Converted:", converted_count)
        print("   Skipped (no objects):", skipped_count)
    
    if USE_SINGLE_POTHOLE_CLASS:
        class_names = ['pothole']
    else:
        class_names = ['D00_LongitudinalCrack', 'D10_TransverseCrack', 'D20_AlligatorCrack', 'D40_Pothole']
    
    data_yaml = {
        'path': str(output_path.absolute()),
        'train': 'images/train',
        'val': 'images/val',
        'nc': len(class_names),
        'names': class_names
    }
    
    yaml_file = output_path / 'data.yaml'
    with open(yaml_file, 'w') as f:
        yaml.dump(data_yaml, f, default_flow_style=False)
    
    print("\nConversion complete!")
    print("Data config:", yaml_file)
    
    train_imgs = len(list((output_path / 'images' / 'train').glob('*.jpg')))
    val_imgs = len(list((output_path / 'images' / 'val').glob('*.jpg')))
    
    print("\nDataset Summary:")
    print("   Classes:", data_yaml['nc'])
    print("   Names:", ', '.join(data_yaml['names']))
    print("   Train Images:", train_imgs)
    print("   Val Images:", val_imgs)
    print("\nReady for training!")
    print("   python models/training/train_yolo.py")
    
    return output_path


if __name__ == "__main__":
    convert_rdd2022_to_yolo()
