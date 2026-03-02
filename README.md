# InfraSight - Pothole Volumetric Analysis System

**AI-Powered Road Maintenance Assistant** - Computer Vision system for automated pothole volume measurement using monocular depth estimation and instance segmentation.

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6+-red.svg)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.4+-green.svg)](https://developer.nvidia.com/cuda-downloads)

## Project Overview

### Objectives

Build a comprehensive Computer Vision system capable of:
- Accurate pothole volume measurement using monocular depth estimation
- Road surface material type detection (asphalt/concrete/paving)
- Repair material recommendations with cost estimation
- GPS-based pothole location mapping for infrastructure planning


### Technical Strategy

#### Core Models
- **Custom Model**: YOLOv8-Seg (trained on pothole + reference object dataset)
- **Pretrained Model #1**: Depth Anything V2 Small (monocular depth estimation)
- **Pretrained Model #2** (Phase 7): MobileNetV3-Small (material classification)

#### Reference Object Approach
- **Primary**: Standard card (ATM/KTP/SIM - 8.5cm × 5.4cm) - Recommended
- **Secondary**: Rp500 coin (2.7cm diameter) - Backup option

## Features

### Core Features
- **Instance Segmentation**: High-precision pothole and reference object detection using YOLOv8-Seg.
- **Monocular Depth Tomography**: Depth map generation with Depth Anything V2.
- **Premium 3D Visualization**: Downward-projection 3D mesh rendering with Plotly.
- **Volumetric Analysis**: Accurate area (cm²), average depth (cm), and volume (cm³) estimation.
- **Web Interface**: Professional dashboard in Streamlit (Full English).

### Advanced Features
- **Surface Material Detection**: Automated classification (Asphalt, Concrete, Paving).
- **Maintenance Recommendations**: Material quantity (kg) and cost estimation (IDR).
- **Damage Map & GPS**: Extraction from EXIF metadata with interactive Folium maps.
- **Automated Reporting**: Instant PDF maintenance report generation.

## Project Structure

```
InfraSight/
├── config/              # Configuration files (config.yaml)
├── data/                # Database and processed images
├── scripts/             # Core utilities and archived dev scripts
├── src/
│   ├── core/            # Volumetric logic, severity, repair advisor
│   ├── models/          # Model inference (YOLO, Depth Anything, Material)
│   ├── utils/           # Shared utilities (logger, GPS, analysis)
│   └── visualization/   # 3D Mesh Engine (Plotly)
├── webapp/              # Streamlit dashboard
└── README.md            # Documentation
```

## Technology Stack

- **Deep Learning**: PyTorch 2.6+, Ultralytics (YOLOv8), Transformers (HuggingFace)
- **Computer Vision**: OpenCV
- **3D Visualization**: Plotly (browser-compatible surface plots)
- **Web Interface**: Streamlit + Streamlit-Folium
- **Datasets**: RDD2022, Roboflow Pothole Segmentation
- **GPS & Mapping**: Geopy, Folium

## Installation and Setup

### System Requirements
- **Python**: 3.10 or 3.11 (Note: Python 3.14 not supported by PyTorch CUDA)
- **GPU**: NVIDIA GPU with CUDA support (recommended for training)
  - Minimum: 6GB VRAM (e.g., RTX 3050)
  - Estimated training time: 4-5 hours for 50 epochs on RTX 3050

### Installation Steps

```bash
# Step 1: Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Step 2: Install PyTorch with CUDA support
# First, check your CUDA version using: nvidia-smi
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121  # For CUDA 12.1

# Step 3: Install remaining dependencies
pip install -r requirements.txt

# Step 4: Verify GPU detection
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"
```

## Usage Guide

### Phase 2: Dataset Preparation
```bash
# Download RDD2022 dataset
python -c "from src.dataset.downloader import DatasetDownloader; DatasetDownloader().download_rdd2022()"

# Collect 20-30 photos with CARD reference object
# Generate synthetic card data (500+ images)
# See src/dataset/preprocessor.py for details
```

### Phase 3: Model Training
```bash
# Train custom YOLOv8-Seg on local GPU
python models/training/train_yolo.py

# Expected output: 4-5 hours training time, mAP@50 > 0.60
```

### Phase 4-6: Core Development
```bash
# Integrate depth estimation, volumetric calculation, and 3D visualization
# Launch web application
cd webapp
streamlit run app.py
```

### Phase 7: Advanced Features (Optional)
```bash
# Train material classifier using MobileNetV3
# Implement GPS extraction and reverse geocoding
# Integrate repair recommendation engine
```

## Performance Metrics

- **Volume Accuracy**: ±30% (acceptable for monocular depth estimation MVP)
- **YOLO mAP@50**: > 0.60 (instance segmentation)
- **Material Classification**: > 95% accuracy (3 classes with distinctive textures)
- **Confidence Level**: High with card reference, Low with coin reference


## Technical Algorithms

### Volumetric Calculation (Surface vs Bottom Heuristic)
1. **Ground Plane Estimation**: Combine reference object depth with healthy asphalt measurements
2. **Pothole Bottom Detection**: Calculate average of bottom 10% deepest pixels
3. **Depth Difference**: Extract normalized relative depth
4. **Calibration**: Apply empirical constant (default: 30.0) for cm conversion
5. **Volume Calculation**: Multiply area (cm²) by depth (cm)

### Synthetic Data Generation (Class Imbalance Mitigation)
- Automated pasting of card PNG onto RDD2022 base images
- Random scaling, positioning, and alpha blending
- Automatic generation of YOLO polygon annotations
- Improves class ratio from 1000:30 to 2:1

## Acknowledgments

- **YOLOv8**: [Ultralytics](https://github.com/ultralytics/ultralytics)
- **Depth Anything V2**: [HuggingFace](https://huggingface.co/depth-anything/Depth-Anything-V2-Small)
- **RDD2022 Dataset**: [Road Damage Detection](https://github.com/sekilab/RoadDamageDetector)





