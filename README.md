# InfraSight - Pothole Volumetric Analysis MVP

🕳️ **AI-Powered Road Maintenance Assistant** - Computer Vision system for measuring pothole volume using monocular depth estimation and instance segmentation.

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6+-red.svg)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.4+-green.svg)](https://developer.nvidia.com/cuda-downloads)

## 🎯 Project Goals

Build a Computer Vision system that:
- Measures pothole volume accurately using monocular depth estimation
- Detects road surface material type (asphalt/concrete/paving)
- Recommends repair materials and estimates costs
- Maps pothole locations via GPS integration

**Target**: Outstanding (A+) grade through sophisticated multi-model integration

## 🧬 Strategy

### Core Models
- **Custom**: YOLOv8-Seg (trained on pothole + reference object dataset)
- **Pretrained #1**: Depth Anything V2 Small (monocular depth estimation)
- **Pretrained #2** (Phase 7): MobileNetV3-Small (material classification)

### Reference Object Approach
- **Primary**: Standard card (ATM/KTP/SIM - 8.5cm × 5.4cm) - Recommended
- **Secondary**: Rp500 coin (2.7cm diameter) - Backup

## ✨ Features

### Core MVP (Phase 1-6)
- ✅ Instance segmentation of potholes and reference objects
- ✅ Monocular depth map generation (Depth Anything V2)
- ✅ Volumetric calculation (area, depth, volume in cm³)
- ✅ Surface vs Bottom heuristic for depth estimation
- ✅ Interactive 3D visualization with Plotly
- ✅ Web interface with Streamlit

### Advanced Features (Phase 7 - A+ Enhancement)
- 🔧 **Material Classification**: Auto-detect road surface type (asphalt/concrete/paving)
- 💰 **Repair Recommendation**: Calculate required materials (kg) and estimated cost (Rp)
- 📍 **GPS Geolocation**: Extract location from photo EXIF + reverse geocoding
- 🗺️ **Interactive Maps**: Leaflet.js maps for infrastructure planning

## 📁 Project Structure

```
InfraSight/
├── config/              # Configuration files (config.yaml)
├── data/                # Dataset storage (raw, processed, annotations)
├── models/
│   ├── weights/         # Model checkpoints
│   └── training/        # Training scripts (train_yolo.py)
├── src/
│   ├── dataset/         # Dataset utilities (downloader, preprocessor)
│   ├── models/          # Model inference (YOLO, Depth Anything V2)
│   ├── core/            # Volumetric calculation logic (calibration, volumetric)
│   ├── visualization/   # 3D visualization (Plotly)
│   └── utils/           # Helpers (logger, geolocation)
├── webapp/              # Streamlit application
├── tests/               # Unit tests
└── notebooks/           # Exploration notebooks
```

## 🛠️ Tech Stack

- **Deep Learning**: PyTorch 2.6+, Ultralytics (YOLOv8), Transformers (HuggingFace)
- **Computer Vision**: OpenCV
- **3D Visualization**: Plotly (browser-compatible surface plots)
- **Web Interface**: Streamlit + Streamlit-Folium
- **Datasets**: RDD2022, Roboflow Pothole Segmentation
- **GPS & Mapping**: Geopy, Folium

## 🚀 Setup

### Requirements
- **Python**: 3.10 or 3.11 (3.14 not supported by PyTorch CUDA)
- **GPU**: NVIDIA GPU with CUDA support (recommended for training)
  - Minimum: 6GB VRAM (e.g., RTX 3050)
  - Training time: ~4-5 hours for 50 epochs on RTX 3050

### Installation

```bash
# 1. Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows

# 2. Install PyTorch with CUDA support
# Check your CUDA version: nvidia-smi
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121  # CUDA 12.1

# 3. Install other dependencies
pip install -r requirements.txt

# 4. Verify GPU detection
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"
```

## 📖 Usage

### Phase 2: Dataset Preparation
```bash
# Download RDD2022 dataset
python -c "from src.dataset.downloader import DatasetDownloader; DatasetDownloader().download_rdd2022()"

# Collect 20-30 photos with CARD reference object
# Generate synthetic card data (500+ images)
# See src/dataset/preprocessor.py
```

### Phase 3: Model Training
```bash
# Train custom YOLOv8-Seg on local GPU
python models/training/train_yolo.py

# Expected: ~4-5 hours, mAP@50 > 0.60
```

### Phase 4-6: Core Development
```bash
# Integrate depth estimation, volumetric calculation, 3D viz
# Run web application
cd webapp
streamlit run app.py
```

### Phase 7: Advanced Features (Optional)
```bash
# Train material classifier (MobileNetV3)
# Implement GPS extraction and mapping
# Add repair recommendation engine
```

## 🎯 Expected Performance

- **Volume Accuracy**: ±30% (acceptable for monocular depth estimation MVP)
- **YOLO mAP@50**: > 0.60 (segmentation)
- **Material Classification**: > 95% (3 classes with distinctive textures)
- **Confidence**: High with card, Low with coin

## 📊 Project Status

- [x] **Phase 1**: Project Setup + GPU Configuration ✅
- [x] **Phase 7**: Advanced Features Documented ✅
- [ ] **Phase 2**: Dataset Preparation (In Progress)
- [ ] **Phase 3**: YOLOv8-Seg Training
- [ ] **Phase 4**: Depth Estimation Integration
- [ ] **Phase 5**: Volumetric Calculation
- [ ] **Phase 6**: Streamlit MVP Deployment
- [ ] **Phase 7**: Material Classifier + GPS Integration

## 🔬 Key Algorithms

### Volumetric Calculation (Surface vs Bottom Heuristic)
1. **Ground Plane**: Combine reference object depth + healthy asphalt
2. **Pothole Bottom**: Average of bottom 10% deepest pixels
3. **Depth Difference**: Normalized relative depth
4. **Calibration**: Empirical constant (default: 30.0) to convert to cm
5. **Volume**: Area (cm²) × Depth (cm)

### Synthetic Data Generation (Class Imbalance Solution)
- Paste card PNG onto RDD2022 base images
- Random scaling, positioning, alpha blending
- Auto-generate YOLO polygon annotations
- Improves ratio from 1000:30 to 2:1

## 📝 Documentation

- **Implementation Plan**: See `implementation_plan.md` in artifacts
- **Task Breakdown**: See `task.md` in artifacts
- **Configuration**: `config/config.yaml`

## 🙏 Acknowledgments

- **YOLOv8**: [Ultralytics](https://github.com/ultralytics/ultralytics)
- **Depth Anything V2**: [HuggingFace](https://huggingface.co/depth-anything/Depth-Anything-V2-Small)
- **RDD2022 Dataset**: [Road Damage Detection](https://github.com/sekilab/RoadDamageDetector)

## 📄 License

Academic project for educational purposes.

---

**Status**: 🚧 Under Development | **Target**: A+ Grade

