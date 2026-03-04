from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from PIL import Image
import io
import os

# Initialize FastAPI app
app = FastAPI(title="Lung Opacity Detection API")

# Allow frontend to communicate with backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load all 3 trained models when server starts
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_level1 = YOLO(f'{BASE_DIR}/models_trained/level1_binary.pt')
model_level2 = YOLO(f'{BASE_DIR}/models_trained/level2_multiclass.pt')
model_level3 = YOLO(f'{BASE_DIR}/models_trained/level3_detection.pt')

def calculate_affected_area(boxes, img_width=640, img_height=640):
    # Calculate percentage of lung affected by opacity
    total_lung_area = img_width * img_height * 0.60
    total_opacity_area = 0
    for box in boxes:
        box_width = box[2] * img_width
        box_height = box[3] * img_height
        total_opacity_area += box_width * box_height
    percentage = (total_opacity_area / total_lung_area) * 100
    return round(min(percentage, 100.0), 1)

def assess_severity(affected_percentage, is_bilateral=False):
    # Determine severity based on affected area
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
    # Generate clinical recommendation
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
    # Check if server is running
    return {"status": "System operational"}

@app.post("/api/analyze")
async def analyze_xray(file: UploadFile = File(...)):
    # Read uploaded image
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data)).convert("RGB")

    # Save temporarily for model prediction
    temp_path = "/tmp/temp_xray.png"
    image.save(temp_path)

    # Level 1: Normal vs Abnormal
    result1 = model_level1.predict(temp_path, verbose=False)
    level1_class = result1[0].names[result1[0].probs.top1]
    level1_confidence = round(result1[0].probs.top1conf.item() * 100, 1)

    # If normal stop here
    if level1_class == "normal":
        return {
            "level1": {"result": "NORMAL", "confidence": level1_confidence},
            "level2": None,
            "level3": None,
            "level4": None,
            "level5": None,
            "level6": None
        }

    # Level 2: Disease Classification
    result2 = model_level2.predict(temp_path, verbose=False)
    level2_class = result2[0].names[result2[0].probs.top1]
    level2_confidence = round(result2[0].probs.top1conf.item() * 100, 1)
    probs = {result2[0].names[i]: round(p * 100, 1) for i, p in enumerate(result2[0].probs.data.tolist())}

    # Level 3: Location Detection
    result3 = model_level3.predict(temp_path, verbose=False, conf=0.25)
    boxes = result3[0].boxes
    left_lung = False
    right_lung = False
    box_list = []

    for box in boxes:
        coords = box.xywhn[0].tolist()
        box_list.append(coords)
        if coords[0] < 0.5:
            left_lung = True
        else:
            right_lung = True

    is_bilateral = left_lung and right_lung
    if is_bilateral:
        location = "Both lungs"
    elif left_lung:
        location = "Left lung"
    elif right_lung:
        location = "Right lung"
    else:
        location = "Not detected"

    # Level 4: Affected Area
    affected_percentage = calculate_affected_area(box_list) if box_list else 0

    # Level 5: Severity
    severity = assess_severity(affected_percentage, is_bilateral)

    # Level 6: Recommendation
    recommendation = generate_recommendation(severity, affected_percentage, level2_class)

    # Return complete analysis
    return {
        "level1": {"result": "ABNORMAL", "confidence": level1_confidence},
        "level2": {"disease": level2_class, "confidence": level2_confidence, "probabilities": probs},
        "level3": {"boxes_detected": len(box_list), "location": location, "is_bilateral": is_bilateral},
        "level4": {"affected_percentage": affected_percentage},
        "level5": {"severity": severity},
        "level6": recommendation
    }