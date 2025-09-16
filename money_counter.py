from ultralytics import YOLO
from PIL import Image
import numpy as np
import pandas as pd
import re

class MoneyCounter:
    def __init__(self, model_path: str, device="cpu", conf=0.35, custom_map=None):
        self.model = YOLO(model_path)
        self.device = device
        self.conf = conf
        self.model_names = {int(k): str(v) for k, v in self.model.names.items()}
        self.custom_map = custom_map or {}

    def _parse_value(self, name: str) -> int:
        if name in self.custom_map:
            return self.custom_map[name]

        m = re.search(r'(\d+)', name.replace(',', '').replace(' ', ''))
        if m:
            return int(m.group(1))
        return 0

    def process(self, image: Image.Image):
        """Запускает YOLO на картинке. Возвращает размеченную картинку, DataFrame, сумму."""
        img_np = np.array(image.convert("RGB"))
        results = self.model(img_np, imgsz=640, conf=self.conf, device=self.device)

        r = results[0]
        annotated = r.plot()
        annotated_img = Image.fromarray(annotated)

        rows = []
        total = 0

        boxes = getattr(r, "boxes", None)
        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().numpy()
            cls_ids = boxes.cls.cpu().numpy().astype(int)
            confs = boxes.conf.cpu().numpy()

            for xy, cid, conf in zip(xyxy, cls_ids, confs):
                name = self.model_names.get(int(cid), str(cid))
                value = self._parse_value(name)
                total += value
                xmin, ymin, xmax, ymax = xy
                rows.append({
                    "class_id": int(cid),
                    "class_name": name,
                    "value": value,
                    "confidence": float(conf),
                    "xmin": float(xmin),
                    "ymin": float(ymin),
                    "xmax": float(xmax),
                    "ymax": float(ymax)
                })

        df = pd.DataFrame(rows)
        return annotated_img, df, total
