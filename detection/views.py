from django.shortcuts import render

# Create your views here.
import numpy as np
import cv2
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from .detection_model import detect_and_estimate_distance

import os
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser
from ultralytics import YOLO
from PIL import Image
import tempfile

@csrf_exempt
def analyze_image(request):
    if request.method == 'POST' and request.FILES.get('image'):
        image_file = request.FILES['image']
        image_data = np.frombuffer(image_file.read(), np.uint8)
        img = cv2.imdecode(image_data, cv2.IMREAD_COLOR)

        result = detect_and_estimate_distance(img)
        return JsonResponse(result)

    return JsonResponse({'error': 'Image non reçue'}, status=400)



model = YOLO("yolov8n.pt") 

class YoloDetectionView(APIView):
    parser_classes = [MultiPartParser]

    def post(self, request, *args, **kwargs):
        image_file = request.FILES.get("image")
        if not image_file:
            return Response({"error": "Aucune image envoyée"}, status=400)

        # Sauvegarde temporaire
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_image:
            for chunk in image_file.chunks():
                temp_image.write(chunk)
            image_path = temp_image.name

        # Détection avec YOLO
        results = model(image_path)[0] 

        # Formatage des résultats
        predictions = []
        for box in results.boxes:
            class_id = int(box.cls[0])
            label = model.names[class_id]
            confidence = float(box.conf[0])
            coords = box.xyxy[0].tolist()
            predictions.append({
                "label": label,
                "confidence": round(confidence, 2),
                "bbox": [round(c, 2) for c in coords]
            })
            
        os.remove(image_path)

        return Response({"detections": predictions})

