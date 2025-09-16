from ultralytics import YOLO
import cv2
from PIL import Image
import numpy as np

class MoneyCounter:
    def __init__(self, model_path, device="cpu"):
        self.model = YOLO(model_path)
        self.device = device

    def _parse_classname(self, class_name: str) -> int:
        try:
            return int(class_name.split("_")[1])
        except Exception:
            return 0

    def process(self, image: Image.Image):
        results = self.model.predict(image, device=self.device, verbose=False)
        total_sum = 0

        for box in results[0].boxes:
            cls_id = int(box.cls.item())
            class_name = self.model.names[cls_id]
            value = self._parse_classname(class_name)
            total_sum += value
            
            x1, y1, x2, y2 = box.xyxy[0].int().tolist()

            label = f"{value} RUB"
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                img,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

        return img, total_sum
