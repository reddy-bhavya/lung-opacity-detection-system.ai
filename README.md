# AI-Based Lung Opacity Detection System

An AI-powered chest X-ray diagnostic tool that analyzes medical images through a 6-level pipeline — screening, disease classification, location detection, severity assessment, and clinical recommendations — delivering a complete diagnostic report in under 10 seconds.

> **Disclaimer:** This system is developed for academic research purposes only. It is not intended for clinical diagnosis or medical decision-making without review and confirmation by a licensed medical professional.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Features](#features)
- [Technologies](#technologies)
- [Infrastructure](#infrastructure)
- [Performance](#performance)
- [Deployment](#deployment)
- [Known Limitations](#known-limitations)
- [Future Enhancements](#future-enhancements)
- [Academic Context](#academic-context)

---

## Overview

### The Problem

Lung opacity — the appearance of white or hazy regions on chest X-rays — is a critical diagnostic indicator for serious conditions including pneumonia, COVID-19, tuberculosis, and lung cancer. Despite its clinical importance, the current state of radiology presents significant challenges:

- Radiologists spend **5–10 minutes** analyzing a single chest X-ray manually
- A projected global shortage of **2.3 million radiologists** by 2030
- **20–30% inter-observer disagreement** between doctors reviewing the same X-ray
- COVID-19 created a **300–400% surge** in chest imaging demand, overwhelming hospital capacity
- Delayed pneumonia diagnosis directly increases patient mortality from **8% to 20–30%**
- Over **70% of the world** lacks adequate access to trained radiologists

In emergency departments, rural clinics, and high-volume hospitals, these bottlenecks translate directly into delayed treatment and preventable deaths.

### The Solution

This system acts as a **first-line AI screening assistant** that instantly analyzes a chest X-ray and produces a structured 6-level diagnostic report. It is designed to:

- Screen abnormal X-rays instantly so radiologists focus only on cases that need attention
- Classify the likely disease with 98% accuracy across 4 categories
- Detect and localize opacity regions with bounding box overlays
- Calculate what percentage of lung tissue is affected
- Grade clinical severity and generate triage priority recommendations

The system uses **YOLOv8**, a state-of-the-art unified deep learning framework, for all three AI models — reducing development complexity while maintaining high accuracy. Unlike traditional approaches requiring separate frameworks for classification and detection, this system uses a single library throughout.

### Dataset Summary

| Dataset | Source | Images | Purpose |
|---------|--------|--------|---------|
| COVID-19 Radiography Database | Kaggle | 21,165 | Levels 1 and 2 training |
| RSNA Pneumonia Detection Challenge | Kaggle | 6,012 annotated | Level 3 training |

**Class Distribution:**

| Class | Images | Percentage |
|-------|--------|------------|
| Normal | 10,192 | 48.2% |
| Lung Opacity | 6,012 | 28.4% |
| COVID-19 | 3,616 | 17.1% |
| Viral Pneumonia | 1,345 | 6.3% |
| **Total** | **21,165** | **100%** |

**Data Split:** 70% Train / 15% Validation / 15% Test (stratified to maintain class balance)

---

## Architecture

### System Flow

```
┌──────────────────────────────────────────────────────────┐
│                     REACT FRONTEND                        │
│   User uploads chest X-ray → Analyze button → Results    │
└───────────────────────┬──────────────────────────────────┘
                        │  HTTP POST /api/analyze
                        │  (multipart/form-data)
                        ▼
┌──────────────────────────────────────────────────────────┐
│                   FASTAPI BACKEND                         │
│   Receives image → Preprocesses → Runs pipeline          │
│   Returns structured JSON response                        │
└───────────────────────┬──────────────────────────────────┘
                        │
           ┌────────────▼────────────┐
           │     6-LEVEL PIPELINE    │
           └────────────┬────────────┘
                        │
       ┌────────────────▼────────────────┐
       │  LEVEL 1 — Normal vs Abnormal   │
       │  Model: YOLOv8n-cls             │
       │  Input: Chest X-ray image       │
       │  Output: NORMAL / ABNORMAL      │
       │          + confidence %         │
       └────────────────┬────────────────┘
                        │ If NORMAL → Stop and return result
                        │ If ABNORMAL → Continue
                        ▼
       ┌────────────────────────────────┐
       │  LEVEL 2 — Disease             │
       │  Classification                │
       │  Model: YOLOv8l-cls            │
       │  Input: Chest X-ray image      │
       │  Output: COVID-19 /            │
       │          Lung Opacity /        │
       │          Viral Pneumonia /     │
       │          Normal                │
       │          + probability % per   │
       │          class                 │
       └────────────────┬───────────────┘
                        │
                        ▼
       ┌────────────────────────────────┐
       │  LEVEL 3 — Location Detection  │
       │  Model: YOLOv8s-det            │
       │  Input: Chest X-ray image      │
       │  Output: Bounding boxes drawn  │
       │          on affected regions   │
       │          Left / Right /        │
       │          Bilateral lung        │
       └────────────────┬───────────────┘
                        │
                        ▼
       ┌────────────────────────────────┐
       │  LEVEL 4 — Affected Area       │
       │  Type: Rule-based Math         │
       │  Input: Bounding boxes from    │
       │         Level 3                │
       │  Logic: Total box area /       │
       │         Lung area × 100        │
       │  Output: % of lung affected    │
       └────────────────┬───────────────┘
                        │
                        ▼
       ┌────────────────────────────────┐
       │  LEVEL 5 — Severity Assessment │
       │  Type: Rule-based Logic        │
       │  Input: Affected % from L4     │
       │         + bilateral flag       │
       │  Output: MILD /                │
       │          MODERATE / SEVERE     │
       └────────────────┬───────────────┘
                        │
                        ▼
       ┌────────────────────────────────┐
       │  LEVEL 6 — Clinical            │
       │  Recommendation                │
       │  Type: Decision Tree           │
       │  Input: Severity + disease     │
       │  Output: Priority color +      │
       │          Action + Timeline +   │
       │          Clinical notes        │
       └────────────────────────────────┘
                        │
                        ▼
       JSON response sent to React frontend
       Complete diagnostic report displayed
```

---

## Features

### 6-Level Diagnostic Pipeline

**Level 1 — Normal vs Abnormal Screening**
The first gate in the pipeline. Uses YOLOv8n-cls to perform binary classification on the input X-ray. If the image is classified as normal, the pipeline stops immediately and returns the result. This prevents unnecessary computation and reduces processing time for normal cases.

**Level 2 — Disease Classification**
If the X-ray is abnormal, this level identifies the specific disease. Uses YOLOv8l-cls (large model, 36 million parameters) trained on 14,814 images across 4 classes. Returns a probability distribution across all classes so the user can see not just the top prediction but the confidence for each disease.

**Level 3 — Opacity Location Detection**
Uses YOLOv8s-det to draw bounding boxes around detected opacity regions. Determines whether opacity is in the left lung, right lung, or bilateral (both lungs). Bilateral involvement is a critical clinical indicator — the system automatically upgrades severity when both lungs are affected.

**Level 4 — Affected Area Calculation**
Pure mathematical logic. Takes bounding box coordinates from Level 3 and calculates the total opacity area relative to the estimated lung area (60% of total image area for standard PA X-rays). Returns the percentage of lung tissue affected.

**Level 5 — Severity Assessment**
Rule-based classification applying standard radiological thresholds. Bilateral involvement automatically upgrades MILD to MODERATE, reflecting increased clinical concern.

| Affected Area | Base Severity | With Bilateral |
|---------------|--------------|----------------|
| Less than 25% | MILD | Upgraded to MODERATE |
| 25% to 50% | MODERATE | Stays MODERATE |
| More than 50% | SEVERE | Stays SEVERE |

**Level 6 — Clinical Recommendation**
Decision tree logic that maps severity to a structured clinical recommendation including priority color, recommended action, response timeline, and disease-specific notes.

| Severity | Priority | Action | Timeline |
|----------|----------|--------|----------|
| SEVERE | IMMEDIATE (RED) | Immediate radiologist review | Within 30 minutes |
| MODERATE | HIGH (ORANGE) | Urgent radiologist review | Within 2–4 hours |
| MILD | ROUTINE (YELLOW) | Routine radiologist review | Within 24 hours |

Additional clinical notes:
- COVID-19 detected → Isolation precautions recommended
- Bilateral MODERATE → Monitor oxygen saturation
- SEVERE → Consider ICU admission

---

## Technologies

### AI and Machine Learning

| Technology | Version | Purpose |
|-----------|---------|---------|
| YOLOv8 (Ultralytics) | 8.4.17 | Classification and detection models |
| PyTorch | 2.10.0+cu128 | Deep learning framework |
| CUDA | 12.8 | GPU acceleration |
| ImageNet Pretrained Weights | — | Transfer learning base |

**Why YOLOv8 over alternatives (EfficientNet + Faster R-CNN):**
Single unified framework for both classification and detection, built-in training pipeline, transfer learning from ImageNet, 70–80% faster training, proven in medical imaging research.

### Backend

| Technology | Version | Purpose |
|-----------|---------|---------|
| Python | 3.12.12 | Primary language |
| FastAPI | 0.100+ | REST API framework |
| Uvicorn | Latest | ASGI server |
| Pydantic | Latest | Data validation |
| Pillow (PIL) | Latest | Image preprocessing |
| python-multipart | Latest | File upload handling |

### Frontend

| Technology | Version | Purpose |
|-----------|---------|---------|
| React | 18.0+ | UI framework |
| JavaScript (ES6+) | — | Frontend language |
| Axios | Latest | HTTP client |
| CSS3 | — | Styling |
| Node.js | 25.1.0 | Runtime environment |
| npm | 11.6.2 | Package manager |

### Data Processing

| Technology | Purpose |
|-----------|---------|
| pydicom | DICOM to PNG conversion for RSNA dataset |
| pandas | Dataset management and CSV processing |
| scikit-learn | Train/val/test split (train_test_split) |
| NumPy | Numerical operations |

### Development and Training

| Technology | Purpose |
|-----------|---------|
| Google Colab Pro | Cloud GPU training environment |
| Google Drive | Dataset and model storage |
| VS Code | Local development IDE |
| Git | Version control |
| GitHub | Code repository |

---

## Infrastructure

### Training Environment

| Resource | Specification |
|----------|--------------|
| GPU | NVIDIA A100 SXM4 40GB |
| Platform | Google Colab Pro |
| Storage | Google Drive 100GB |
| Python | 3.12.12 |
| PyTorch | 2.10.0+cu128 |
| Ultralytics | 8.4.17 |

### Training Time

| Model | Epochs | Time |
|-------|--------|------|
| Level 1 — YOLOv8n-cls | 50 | ~4 hours |
| Level 2 — YOLOv8l-cls | 50 | ~1.4 hours |
| Level 3 — YOLOv8s-det | 150 | ~2 hours |

### Project Structure

```
lung-opacity-detection/
├── data/
│   ├── raw/
│   │   ├── covid19/               # COVID-19 Radiography Dataset
│   │   └── rsna/                  # RSNA Pneumonia Detection Dataset
│   └── processed/
│       ├── level1/                # Binary classification (normal/abnormal)
│       ├── level2/                # 4-class classification
│       └── level3_detection/      # Detection with YOLO format labels
├── models_trained/
│   ├── level1_binary.pt           # Level 1 trained weights (8.4 MB)
│   ├── level2_multiclass.pt       # Level 2 trained weights (72.6 MB)
│   └── level3_detection.pt        # Level 3 trained weights (22.5 MB)
├── scripts/
│   ├── main.py                    # FastAPI backend — 6-level pipeline
│   └── logic.py                   # Levels 4–6 rule-based logic
├── frontend/
│   └── src/
│       ├── App.js                 # Main React component
│       └── App.css                # UI styling
└── README.md
```

---

## Performance

### Model Accuracy Results

| Level | Model | Parameters | Accuracy | Target | Status |
|-------|-------|-----------|----------|--------|--------|
| Level 1 | YOLOv8n-cls | 1.4M | 92.9% | 90%+ | Passed |
| Level 2 | YOLOv8l-cls | 36.2M | 98.0% | 85%+ | Passed |
| Level 3 | YOLOv8s-det | 11.1M | 53.1% mAP50 | 70%+ | Below target |

### Level 2 Per-Class Accuracy (tested on 200 unseen images)

| Class | Correct | Total | Accuracy |
|-------|---------|-------|----------|
| Normal | 50 | 50 | 100% |
| COVID-19 | 49 | 50 | 98% |
| Lung Opacity | 48 | 50 | 96% |
| Viral Pneumonia | 49 | 50 | 98% |
| **Overall** | **196** | **200** | **98%** |

### Level 3 Detection Results

| Metric | Value |
|--------|-------|
| mAP50 | 53.1% |
| Precision | 53.8% |
| Recall | 56.1% |
| Training images | 4,809 |
| Validation images | 1,203 |

---

## Known Limitations

**Level 3 Detection Below Target**
Detection achieved 53.1% mAP50 against a 70% target. Opacity boundaries on chest X-rays are inherently ambiguous — even experienced radiologists sometimes disagree on exact boundaries. With only 6,012 training images (detection tasks typically require 50,000+), this performance is expected. This limitation is acknowledged and documented honestly.

**Standard PA X-Rays Only**
Models were trained exclusively on standard Posterior-Anterior (PA) chest X-rays taken in controlled conditions. Portable bedside X-rays, AP-view X-rays, and ICU X-rays with visible medical equipment may produce lower confidence scores, as demonstrated during testing.

**Dataset Scope**
Training data sourced from public Kaggle datasets. These may not represent the full diversity of real-world clinical imaging across different equipment, patient populations, and geographic regions.

---

## Deployment

### Prerequisites
- Python 3.9+
- Node.js 18+
- Git

### Local Setup

**1. Clone Repository**
```bash
git clone https://github.com/yourusername/lung-opacity-detection.git
cd lung-opacity-detection
```

**2. Backend Setup**
```bash
python3 -m venv venv
source venv/bin/activate
pip install fastapi uvicorn python-multipart ultralytics pillow
uvicorn scripts.main:app --reload
```
Backend running at: `http://127.0.0.1:8000`
API documentation at: `http://127.0.0.1:8000/docs`

**3. Frontend Setup**
```bash
cd frontend
npm install
npm start
```
Frontend running at: `http://localhost:3000`

### API Reference

| Method | Endpoint | Description | Response Time |
|--------|----------|-------------|---------------|
| GET | `/api/health` | Check server status | < 1 second |
| POST | `/api/analyze` | Analyze chest X-ray | < 10 seconds |

---

## Future Enhancements

- Expand training data to include portable, AP-view, and ICU X-rays
- Improve Level 3 detection accuracy with a larger annotated dataset
- Add PDF report generation for clinical handoff
- Deploy to cloud infrastructure (Render backend + Vercel frontend)
- Add DICOM format support for hospital PACS system compatibility
- Integrate with Electronic Health Records (EHR) systems
- Add batch processing for multiple X-rays simultaneously
- Implement user authentication for multi-user clinical environments
