from typing import List

from ultralytics import YOLO
from ultralytics.engine.results import Results

from settings import YOLO_MODEL_PATH

model = YOLO(YOLO_MODEL_PATH, verbose=True)  # 加载预训练模型


def detect_faces_bbox(image: str):
    results = model(image)
    faces = []
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        faces.append((x1, y1, x2, y2))
    return faces


def detect_faces_results(image) -> List[Results]:
    """
    :param image:
    :return: [
      {
        "name": "FACE",
        "class": 0,
        "confidence": 0.89534,
        "box": {
          "x1": 461.7702,
          "y1": 131.69777,
          "x2": 755.21881,
          "y2": 544.22852
        }
      }
    ]
    """
    return model.predict(image)
