import numpy as np
from itertools import combinations
import math
import cv2
from typing import List, Tuple, Dict, Any, Union

from ..shared.log_utils import log_json
from ..shared.types import Circle

from ..shared.geometry_utils import merge_close_circles
from ..shared.log_utils import draw_circles_on_image, log_image, log_json
from ..shared.geometry_utils import filter_spatial_outliers 


def detect_circles_hough(image: np.ndarray, hough_params: Dict[str, Any], trace_id:str) -> List[Circle]:
    """
    Detecta círculos em uma imagem usando a Transformada de Hough e retorna uma lista de objetos Circle.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
    circles = cv2.HoughCircles(
        binary,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=hough_params['minDist'],
        param1=hough_params['param1'],
        param2=hough_params['param2'],
        minRadius=hough_params['minRadius'],
        maxRadius=hough_params['maxRadius']
    )
    
    detected_circles: List[Circle] = []
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            # Converte a saída do Hough para a nossa dataclass Circle
            detected_circles.append(Circle(
                center_x=float(i[0]),
                center_y=float(i[1]),
                radius=float(i[2]),
                filled=False # Hough não sabe se o círculo é preenchido, então o padrão é False
            ))
    log_image(trace_id, binary, "hough_blurred")     
            
    return detected_circles

def detect_circles_multi_operation(image: np.ndarray, hough_params: Dict[str, Any], trace_id:str) -> List[Circle]:
    """
    Executa a detecção de Hough em múltiplas versões da imagem (após operações morfológicas).
    """
    if hough_params is None:
        print("Parâmetros de Hough não fornecidos. Nenhuma detecção realizada.")
        return []

    images_to_process = []
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    
    images_to_process.append(image)

    # Aplica uma sequência de operações morfológicas
    op1 = cv2.erode(image, kernel, iterations=1)
    op2 = cv2.dilate(op1, kernel, iterations=1)
    op3 = cv2.erode(op2, kernel, iterations=1)
    op4 = cv2.dilate(op3, kernel, iterations=1)
    op5 = cv2.erode(op4, kernel, iterations=1)
    op6 = cv2.dilate(op5, kernel, iterations=1)
    images_to_process.extend([op1, op2, op3, op4, op5, op6])

    all_detected_circles: List[Circle] = []
    print("Iniciando detecção com múltiplas operações morfológicas...")

    for i, processed_img in enumerate(images_to_process):
        circles_from_op = detect_circles_hough(processed_img, hough_params, trace_id)

        circles_from_op_annotated_image = draw_circles_on_image(processed_img.copy(), circles_from_op)
        log_image(trace_id, circles_from_op_annotated_image, f"processed_img_from_operation_{i}")

        print(f"   - Operação {i}: Encontrados {len(circles_from_op)} círculos.")
        all_detected_circles.extend(circles_from_op)

    print("Detecção com múltiplas operações finalizada.")
    return all_detected_circles


def _criar_parametros_hough_otimizados(lista_circulos_confiaveis: List[Circle], raio_tolerancia: int = 5) -> Dict[str, Any]:
    """
    Cria um dicionário de parâmetros otimizados para a Transformada de Hough
    com base em uma lista de círculos já detectados e confiáveis.
    """
    if not lista_circulos_confiaveis:
        return None 
    raios = [c.radius for c in lista_circulos_confiaveis]
    raio_medio = np.mean(raios)

    dist_minima = raio_medio * 2

    parametros = {
        'minDist': int(dist_minima),
        'minRadius': max(0, int(raio_medio - raio_tolerancia)),
        'maxRadius': int(raio_medio + raio_tolerancia),
        'param1': 80, # Valor inicial padrão
        'param2': 60  # Valor inicial padrão
    }

    return parametros

def detect_todos_circulos_possiveis(image: np.ndarray, lista_circulos_marcados: List[Circle], trace_id: str) -> List[Circle]:
    """
    Orquestra a detecção de todos os círculos (marcados e não marcados) em uma imagem.
    """
    log_json(trace_id, lista_circulos_marcados, "lista_circulos_marcados")

    parametros_otimos = _criar_parametros_hough_otimizados(lista_circulos_marcados, raio_tolerancia=5)
    
    # Executa a detecção de Hough de várias maneiras para maximizar a chance de encontrar todos os círculos
    circulos_encontrados_hough = detect_circles_hough(image.copy(), parametros_otimos, trace_id)
    circulos_encontrados_hough.extend(detect_circles_multi_operation(image.copy(), parametros_otimos, trace_id))

    print(f"\nForam encontrados {len(circulos_encontrados_hough)} círculos com o método de Hough otimizado.")
    
    # Unifica os círculos detectados pelo YOLO (marcados) com os detectados pelo Hough
    lista_unificada_com_duplicatas = circulos_encontrados_hough + lista_circulos_marcados
    print(f"Total de círculos ANTES da de-duplicação: {len(lista_unificada_com_duplicatas)}")
    
    # Calcula o raio médio para definir um limiar de fusão dinâmico
    if not lista_unificada_com_duplicatas:
        return []
        
    # --- CORREÇÃO AQUI ---
    # Acessa o atributo .radius em vez da chave ['radius']
    raio_medio = sum(c.radius for c in lista_unificada_com_duplicatas) / len(lista_unificada_com_duplicatas)
    dist_threshold = raio_medio 
    
    # Funde círculos que estão muito próximos (duplicatas)
    lista_final_unica = merge_close_circles(
        lista_unificada_com_duplicatas, 
        dist_threshold=dist_threshold
    )
    
    print(f"Total de círculos DEPOIS da de-duplicação: {len(lista_final_unica)}")

    
    annotated_image = draw_circles_on_image(image.copy(), lista_final_unica)
    log_image(trace_id, annotated_image, "todos_detectados")     
    return lista_final_unica
