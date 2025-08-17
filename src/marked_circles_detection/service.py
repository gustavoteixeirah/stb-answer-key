from typing import List, Union
import numpy as np
import cv2
from ultralytics import YOLO
from ultralytics.engine.results import Results

from ..shared.log_utils import draw_circles_on_image, log_image, log_json
from ..shared.geometry_utils import filter_spatial_outliers 


from ..shared.types import Circle
from . import debug

class MarkedCirclesDetector:
    """
    Encapsula o modelo de detecção e sua lógica para ser carregado uma única vez.
    """
    def __init__(self, model_path: str):
        self.model = YOLO(model_path)

    def _convert_yolo_results_to_circles(self, yolo_results: List[Results]) -> List[Circle]:
        detected_circles: List[Circle] = []

        if not yolo_results or not yolo_results[0].boxes:
            return detected_circles

        boxes = yolo_results[0].boxes.xyxy.cpu().numpy()
        print(f"DEBUG: Detectado '{len(boxes)}' circulos.")

        for box in boxes:
            x1, y1, x2, y2 = box
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            radius = ((x2 - x1) + (y2 - y1)) / 4

            circle = Circle(
                center_x=center_x,
                center_y=center_y,
                radius=radius,
                filled=True,
            )
            detected_circles.append(circle)
        
        return detected_circles

    def detect(self, image: Union[str, np.ndarray], trace_id: str) -> List[Circle]:
        if isinstance(image, str):
            image_np = cv2.imread(image)
            if image_np is None:
                raise FileNotFoundError(f"Não foi possível carregar a imagem em: {image}")
        else:
            image_np = image
        log_image(trace_id, image, "1_original")     

        results = self.model(image_np)

        raw_circles = self._convert_yolo_results_to_circles(results)
        print(f"DEBUG: Detectado '{len(raw_circles)}' circulos.")
        log_json(trace_id, raw_circles, "todos circulos marcados")
        annotated_image = draw_circles_on_image(image.copy(), raw_circles)
        log_image(trace_id, annotated_image, "2_anotada")     

        inliers, outliers, log_data = filter_spatial_outliers(raw_circles)
        print(f"DEBUG: Filtrados circulos. Restam '{len(inliers)}' circulos.")
        log_json(trace_id, log_data, "circulos marcados filtrados")
        annotated_image = draw_circles_on_image(image.copy(), inliers)
        log_image(trace_id, annotated_image, "3_anotada_filtrada")     

        return inliers