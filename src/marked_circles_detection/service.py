# Importações necessárias para anotações de tipo
from typing import List, Union
import numpy as np

# Importações das bibliotecas
from ultralytics import YOLO
from ultralytics.engine.results import Results  # Tipo específico para os resultados do YOLO

# Importação do nosso tipo customizado
from shared.types import Circle


def _convert_yolo_results_to_circles(yolo_results: List[Results]) -> List[Circle]:
    """Converte os resultados brutos de uma detecção do YOLO para uma lista de Círculos.

    Esta é uma função auxiliar que extrai as caixas delimitadoras (bounding boxes)
    do objeto de resultado do YOLO, calcula as propriedades de um círculo
    (centro e raio) a partir de cada caixa e as instancia como objetos Circle.

    Args:
        yolo_results (List[Results]): A lista de resultados retornada por uma
            chamada ao modelo YOLO. Normalmente contém um único elemento para
            uma única imagem.

    Returns:
        List[Circle]: Uma lista de objetos Circle, cada um representando um
            círculo detectado. A lista estará vazia se nenhuma detecção for feita.
    """
    detected_circles: List[Circle] = []
    
    # yolo_results é uma lista, normalmente com um resultado por imagem
    if not yolo_results or not yolo_results[0].boxes:
        print("\nNenhuma caixa delimitadora foi encontrada nos resultados.")
        return detected_circles

    # Converte os tensores para arrays NumPy na CPU para facilitar a manipulação
    boxes = yolo_results[0].boxes.xyxy.cpu().numpy()
    # confidences = yolo_results[0].boxes.conf.cpu().numpy() # Descomente se precisar usar a confiança

    for i, box in enumerate(boxes):
        # Extrai as coordenadas da caixa (x-inicial, y-inicial, x-final, y-final)
        x1, y1, x2, y2 = box
        print(f"Caixa {i}: {box}.")

        # Calcula o centro (cx, cy) da caixa
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        # Calcula a largura e a altura da caixa
        width = x2 - x1
        height = y2 - y1

        # Estima o raio como a média da metade da largura e da metade da altura
        radius = (width + height) / 4

        # Cria a instância do nosso tipo Circle
        circle = Circle(
            center_x=center_x,
            center_y=center_y,
            radius=radius,
            filled=True  # Assumimos que este modelo só detecta círculos preenchidos
        )
        detected_circles.append(circle)
        
    print(f"\nForam encontrados {len(detected_circles)} círculos marcados.")
    return detected_circles


def detect_filled_circles(image: Union[str, np.ndarray]) -> List[Circle]:
    """Detecta círculos preenchidos em uma imagem usando um modelo YOLOv8 pré-treinado.

    Esta função carrega um modelo YOLO, executa a inferência na imagem fornecida e, em seguida,
    converte os resultados da detecção em uma lista estruturada de objetos Circle.

    Args:
        image (Union[str, np.ndarray]): A imagem a ser processada. Pode ser
            o caminho para o arquivo de imagem (str) ou a imagem já carregada
            como um array NumPy (por exemplo, usando OpenCV `cv2.imread`).

    Returns:
        List[Circle]: Uma lista de objetos Circle, onde cada objeto representa
            um círculo preenchido detectado na imagem.
    """
    # Carrega o modelo YOLO a partir do arquivo de pesos.
    # OBS: Veja o ponto de melhoria abaixo.
    model = YOLO("detect_marks/detect_filled_circles.pt")
    
    # Executa a predição na imagem
    results = model(image)
    
    # Converte os resultados brutos para a nossa estrutura de dados Circle
    return _convert_yolo_results_to_circles(results)