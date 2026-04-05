from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from PIL import Image, ImageDraw
import io
import base64
import numpy as np
import cv2
import os

app = FastAPI(title="Lung Opacity Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_level1 = YOLO(f'{BASE_DIR}/models_trained/level1_binary.pt')
model_level2 = YOLO(f'{BASE_DIR}/models_trained/level2_multiclass.pt')
model_level3 = YOLO(f'{BASE_DIR}/models_trained/level3_detection.pt')


def apply_clahe_numpy(image_np):
    """Enhance X-ray contrast using CLAHE on a numpy array.
    Operates entirely in memory — no disk I/O required."""
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)


def segment_lungs(image_np):
    """Extract lung region mask from chest X-ray using multi-threshold OpenCV approach.
    Tries multiple threshold values and picks the one producing the largest valid mask.
    Returns a binary mask where True pixels represent lung tissue.
    Shape of returned mask matches input image_np shape (height, width)."""
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    h, w = gray.shape  # height, width — numpy convention
    best_mask = None
    best_area = 0

    for thresh_val in [60, 80, 100, 120, 140]:
        _, binary = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY_INV)

        kernel = np.ones((15, 15), np.uint8)
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]

        mask = np.zeros((h, w), dtype=np.uint8)
        for contour in contours:
            if cv2.contourArea(contour) > 5000:
                cv2.drawContours(mask, [contour], -1, 1, thickness=cv2.FILLED)

        area = mask.sum()
        max_allowed = h * w * 0.75
        if area > best_area and area < max_allowed:
            best_area = area
            best_mask = mask

    if best_mask is None or best_area < 100:
        # Fallback: use center region as approximate lung area
        fallback = np.zeros((h, w), dtype=np.uint8)
        fallback[int(h * 0.1):int(h * 0.9), int(w * 0.1):int(w * 0.9)] = 1
        return fallback.astype(bool)

    return best_mask.astype(bool)


def calculate_affected_area(boxes, img_w, img_h, lung_mask):
    """Calculate the percentage of actual lung area covered by detected opacity boxes.
    Uses intersection of opacity boxes with lung segmentation mask for medical accuracy.
    img_w = image width (PIL convention), img_h = image height (PIL convention)
    lung_mask shape is (img_h, img_w) — numpy (height, width) convention."""

    # Build opacity mask — shape must be (height, width) = (img_h, img_w)
    opacity_mask = np.zeros((img_h, img_w), dtype=bool)

    for box in boxes:
        cx, cy, bw, bh = box
        x1 = max(0, int((cx - bw / 2) * img_w))
        y1 = max(0, int((cy - bh / 2) * img_h))
        x2 = min(img_w, int((cx + bw / 2) * img_w))
        y2 = min(img_h, int((cy + bh / 2) * img_h))
        opacity_mask[y1:y2, x1:x2] = True

    # Ensure lung mask matches (img_h, img_w)
    if lung_mask.shape != (img_h, img_w):
        # cv2.resize takes (width, height) — note the swap
        lung_mask = cv2.resize(
            lung_mask.astype(np.uint8),
            (img_w, img_h),
            interpolation=cv2.INTER_NEAREST
        ).astype(bool)

    lung_area = lung_mask.sum()
    if lung_area < 100:
        return 0.0

    intersection = np.logical_and(lung_mask, opacity_mask)
    return round((intersection.sum() / lung_area) * 100, 1)


def assess_severity(affected_percentage, is_bilateral=False):
    """Assign severity level based on affected area percentage.
    Bilateral involvement with mild base severity is upgraded to moderate."""
    if affected_percentage < 25:
        severity = "MILD"
    elif affected_percentage <= 50:
        severity = "MODERATE"
    else:
        severity = "SEVERE"

    if is_bilateral and severity == "MILD":
        severity = "MODERATE"

    return severity


def generate_recommendation(severity, affected_percentage, disease=None):
    """Generate triage priority and clinical action based on severity and disease type."""
    recommendation = {}

    if severity == "SEVERE" or affected_percentage > 60:
        recommendation["priority"] = "IMMEDIATE (RED)"
        recommendation["action"] = "URGENT: Immediate radiologist review required"
        recommendation["timeline"] = "Within 30 minutes"
        recommendation["triage"] = "Emergency"
    elif severity == "MODERATE" or affected_percentage >= 30:
        recommendation["priority"] = "HIGH (ORANGE)"
        recommendation["action"] = "Flag for urgent radiologist review"
        recommendation["timeline"] = "Within 2-4 hours"
        recommendation["triage"] = "Urgent"
    else:
        recommendation["priority"] = "ROUTINE (YELLOW)"
        recommendation["action"] = "Routine radiologist review"
        recommendation["timeline"] = "Within 24 hours"
        recommendation["triage"] = "Standard"

    notes = []
    if disease == "covid":
        notes.append("Recommend isolation precautions")
    if severity == "MODERATE" and affected_percentage >= 30:
        notes.append("Monitor oxygen saturation")
    if severity == "SEVERE":
        notes.append("Consider ICU admission")

    recommendation["notes"] = notes
    return recommendation


@app.get("/api/health")
def health_check():
    return {"status": "System operational"}


@app.post("/api/analyze")
async def analyze_xray(file: UploadFile = File(...)):
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data)).convert("RGB")

    # PIL size returns (width, height)
    img_w, img_h = image.size

    image_np = np.array(image)  # shape: (height, width, 3)
    processed_np = apply_clahe_numpy(image_np)

    # segment_lungs returns mask of shape (height, width) = (img_h, img_w)
    lung_mask = segment_lungs(processed_np)

    # Level 1: Binary screening — normal vs abnormal
    result1 = model_level1.predict(source=processed_np, verbose=False)
    level1_class = result1[0].names[result1[0].probs.top1]
    level1_confidence = round(result1[0].probs.top1conf.item() * 100, 1)

    # Level 3: Run detection on every image as an independent safety check
    result3 = model_level3.predict(source=processed_np, verbose=False, conf=0.25)
    boxes_raw = result3[0].boxes if result3[0].boxes is not None else []

    left_lung = False
    right_lung = False
    box_list = []

    draw = ImageDraw.Draw(image)
    for box in boxes_raw:
        if box.conf[0].item() < 0.25:
            continue

        coords = box.xywhn[0].tolist()
        box_list.append(coords)

        cx, cy, bw, bh = coords
        x1 = max(0, int((cx - bw / 2) * img_w))
        y1 = max(0, int((cy - bh / 2) * img_h))
        x2 = min(img_w, int((cx + bw / 2) * img_w))
        y2 = min(img_h, int((cy + bh / 2) * img_h))

        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        draw.text((x1, y1 - 15), f"Opacity {round(box.conf[0].item() * 100)}%", fill="red")

        if cx < 0.5:
            left_lung = True
        else:
            right_lung = True

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    annotated_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    is_bilateral = left_lung and right_lung
    if is_bilateral:
        location = "Both lungs"
    elif left_lung:
        location = "Left lung"
    elif right_lung:
        location = "Right lung"
    else:
        location = "Not detected"

    detection_found = len(box_list) > 0
    is_abnormal = (
        level1_class == "abnormal"
        or (level1_confidence < 80 and detection_found)
        or len(box_list) >= 2
    )

    if not is_abnormal and level1_confidence > 85:
        return {
            "level1": {"result": "NORMAL", "confidence": level1_confidence},
            "level2": None,
            "level3": {"boxes_detected": 0, "location": "Not detected", "is_bilateral": False},
            "level4": None,
            "level5": None,
            "level6": None,
            "annotated_image": annotated_b64
        }

    # Level 2: Multi-class disease classification
    result2 = model_level2.predict(source=processed_np, verbose=False)
    level2_class = result2[0].names[result2[0].probs.top1]
    level2_confidence = round(result2[0].probs.top1conf.item() * 100, 1)
    probs = {
        result2[0].names[i]: round(p * 100, 1)
        for i, p in enumerate(result2[0].probs.data.tolist())
    }

    # Level 4: Quantify affected lung area using segmentation mask
    # Pass img_w and img_h separately — lung_mask shape is (img_h, img_w)
    affected_percentage = calculate_affected_area(box_list, img_w, img_h, lung_mask) if box_list else 0

    # Level 5: Assign clinical severity
    severity = assess_severity(affected_percentage, is_bilateral)

    # Level 6: Generate triage recommendation
    recommendation = generate_recommendation(severity, affected_percentage, level2_class)

    return {
        "level1": {"result": "ABNORMAL", "confidence": level1_confidence},
        "level2": {"disease": level2_class, "confidence": level2_confidence, "probabilities": probs},
        "level3": {"boxes_detected": len(box_list), "location": location, "is_bilateral": is_bilateral},
        "level4": {"affected_percentage": affected_percentage},
        "level5": {"severity": severity},
        "level6": recommendation,
        "annotated_image": annotated_b64
    }
