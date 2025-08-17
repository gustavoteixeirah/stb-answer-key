# Arquivo: src/marked_circles_detection/debug.py

import numpy as np
from typing import List
from dataclasses import asdict

from ..shared import log_utils
from ..shared.types import Circle
# Note que não precisamos mais do db_handler aqui!

MODULE_NAME = "marked_circles_detection"

def log_circle_detection_step(
    handler,  # Recebe um handler genérico (pode ser o de Markdown ou de DB)
    step_name: str,
    image: np.ndarray,
    circles: List[Circle],
):
    """
    Prepara os dados de um passo de detecção e os envia para o handler de log.
    """
    debug_image = log_utils.draw_circles_on_image(image, circles)
    
    circles_metadata = {
        "circles": [asdict(c) for c in circles],
        "count": len(circles)
    }
    
    # Usa o método do handler para registrar o passo
    handler.log_step(
        step_name=step_name,
        module_name=MODULE_NAME,
        image_np=debug_image,
        metadata=circles_metadata
    )
    print(f"DEBUG: Etapa '{step_name}' registrada via handler.")



def log_image_step(
    handler,  # Recebe um handler genérico (pode ser o de Markdown ou de DB)
    step_name: str,
    image: np.ndarray,
):
    """
    Prepara os dados de um passo de detecção e os envia para o handler de log.
    """
    # Usa o método do handler para registrar o passo
    handler.log_step(
        step_name=step_name,
        module_name=MODULE_NAME,
        image_np=image
    )
    print(f"DEBUG: Etapa '{step_name}' registrada via handler.")