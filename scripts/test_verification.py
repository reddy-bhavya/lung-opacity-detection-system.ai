"""
test_verification.py
Unit verification tests for AI-Based Lung Opacity Detection System
Standard: IEC 62304:2006 + A1:2015
Course: INFO7410 — Advanced Medical Device Software Engineering
Author: Bhavya Reddy

Run: python3 -m pytest scripts/test_verification.py -v -s
"""

import pytest
import numpy as np
import cv2
import os
from PIL import Image
from ultralytics import YOLO


# ─────────────────────────────────────────────────────────────
# FILE PATHS
# BASE_DIR points to the project root (one level up from scripts/)
# Images folder has a trailing space in the folder name
# ─────────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Real X-ray images used for testing
NORMAL_IMAGE  = os.path.join(BASE_DIR, "Images ", "normal ", "Normal-15.png")
OPACITY_IMAGE = os.path.join(BASE_DIR, "Images ", "lung_opacity", os.listdir(os.path.join(BASE_DIR, "Images ", "lung_opacity"))[0])

# Trained model weight file for Level 1 binary classifier
MODEL_L1 = os.path.join(BASE_DIR, "models_trained", "level1_binary.pt")


# ─────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# Copied directly from scripts/main.py
# Isolated here to avoid loading FastAPI and YOLO at import time
# ─────────────────────────────────────────────────────────────

def apply_clahe_numpy(image_np):
    """
    Enhances X-ray contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization).
    Converts RGB → grayscale → applies CLAHE → converts back to RGB.
    clipLimit=2.0 controls contrast enhancement strength.
    tileGridSize=(8,8) divides image into 8x8 tiles for local contrast enhancement.
    """
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)


def assess_severity(affected_percentage, is_bilateral=False):
    """
    Assigns severity level based on affected lung area percentage.
    Rules:
      - < 25%       → MILD
      - 25% to 50%  → MODERATE
      - > 50%       → SEVERE
    Bilateral upgrade rule:
      - If both lungs are affected AND base severity is MILD → upgrade to MODERATE
      - MODERATE and SEVERE are never downgraded by bilateral flag
    """
    if affected_percentage < 25:
        severity = "MILD"
    elif affected_percentage <= 50:
        severity = "MODERATE"
    else:
        severity = "SEVERE"
    if is_bilateral and severity == "MILD":
        severity = "MODERATE"
    return severity


# ─────────────────────────────────────────────────────────────
# TEST CLASS 1 — assess_severity()
# Verifies all severity thresholds and bilateral upgrade rule
# Functional Requirement: FR-08 — Severity Classification
# ─────────────────────────────────────────────────────────────

class TestSeverity:

    def test_mild(self):
        """20% affected, no bilateral → MILD"""
        result = assess_severity(20.0, False)
        print(f"\n  Test    : Affected area 20%, no bilateral involvement")
        print(f"  Expected: MILD")
        print(f"  Result  : {result}")
        print(f"  Verdict : {'PASS — 20% is below 25% threshold, correctly classified as MILD' if result == 'MILD' else 'FAIL'}")
        assert result == "MILD"

    def test_moderate(self):
        """35% affected, no bilateral → MODERATE"""
        result = assess_severity(35.0, False)
        print(f"\n  Test    : Affected area 35%, no bilateral involvement")
        print(f"  Expected: MODERATE")
        print(f"  Result  : {result}")
        print(f"  Verdict : {'PASS — 35% is between 25-50% threshold, correctly classified as MODERATE' if result == 'MODERATE' else 'FAIL'}")
        assert result == "MODERATE"

    def test_severe(self):
        """65% affected, no bilateral → SEVERE"""
        result = assess_severity(65.0, False)
        print(f"\n  Test    : Affected area 65%, no bilateral involvement")
        print(f"  Expected: SEVERE")
        print(f"  Result  : {result}")
        print(f"  Verdict : {'PASS — 65% is above 50% threshold, correctly classified as SEVERE' if result == 'SEVERE' else 'FAIL'}")
        assert result == "SEVERE"

    def test_bilateral_upgrade(self):
        """20% affected WITH bilateral → MODERATE (upgraded from MILD)"""
        result = assess_severity(20.0, True)
        print(f"\n  Test    : Affected area 20%, bilateral involvement = True")
        print(f"  Expected: MODERATE (upgraded from MILD — both lungs involved)")
        print(f"  Result  : {result}")
        print(f"  Verdict : {'PASS — bilateral upgrade rule correctly applied, MILD upgraded to MODERATE' if result == 'MODERATE' else 'FAIL'}")
        assert result == "MODERATE"


# ─────────────────────────────────────────────────────────────
# TEST CLASS 2 — Level 1 Binary Classifier (YOLOv8s-cls)
# Verifies real model predictions on real X-ray images
# Functional Requirement: FR-02 — Binary Classification
# ─────────────────────────────────────────────────────────────

class TestLevel1:

    def test_normal_image_returns_normal(self):
        """Normal-15.png must be classified as normal"""
        model = YOLO(MODEL_L1)
        image = Image.open(NORMAL_IMAGE).convert("RGB")
        processed = apply_clahe_numpy(np.array(image))
        result = model.predict(source=processed, verbose=False)
        label = result[0].names[result[0].probs.top1]
        confidence = round(result[0].probs.top1conf.item() * 100, 1)
        print(f"\n  Test    : Normal chest X-ray (Normal-15.png)")
        print(f"  Expected: NORMAL")
        print(f"  Result  : {label.upper()} ({confidence}% confidence)")
        print(f"  Verdict : {'PASS — model correctly identified a normal X-ray' if label == 'normal' else 'FAIL'}")
        assert label == "normal"

    def test_opacity_image_returns_abnormal(self):
        """Lung opacity image must be classified as abnormal"""
        model = YOLO(MODEL_L1)
        image = Image.open(OPACITY_IMAGE).convert("RGB")
        processed = apply_clahe_numpy(np.array(image))
        result = model.predict(source=processed, verbose=False)
        label = result[0].names[result[0].probs.top1]
        confidence = round(result[0].probs.top1conf.item() * 100, 1)
        print(f"\n  Test    : Lung opacity chest X-ray ({os.path.basename(OPACITY_IMAGE)})")
        print(f"  Expected: ABNORMAL")
        print(f"  Result  : {label.upper()} ({confidence}% confidence)")
        print(f"  Verdict : {'PASS — model correctly identified an abnormal X-ray' if label == 'abnormal' else 'FAIL'}")
        assert label == "abnormal"