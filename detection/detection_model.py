import torch
import cv2
import numpy as np

# Charger YOLOv5s (ou yolov5n si plus léger)
model = torch.hub.load('yolov5', 'yolov5n', source='local')
model.eval()

def detect_and_estimate_distance(img):
    # Convertir image BGR → RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Inference avec YOLO
    results = model(img_rgb)
    predictions = results.xyxy[0]  # [x1, y1, x2, y2, conf, class]

    detected = []

    for *box, conf, cls in predictions.tolist():
        x1, y1, x2, y2 = map(int, box)
        label = model.names[int(cls)]
        width = x2 - x1
        distance = round(500 / (width + 1), 2)  # estimation empirique

        detected.append({
            "label": label,
            "confidence": round(conf, 2),
            "distance": distance
        })

    return {
        "objects": detected,
        "message": generate_message(detected)
    }

def generate_message(objs):
    if not objs:
        return "Aucun obstacle détecté"
    closest = min(objs, key=lambda x: x["distance"])
    return f"{closest['label']} détecté à {closest['distance']} mètre"
