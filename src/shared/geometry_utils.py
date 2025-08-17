# Arquivo: src/shared/geometry_utils.py

from typing import List, Tuple, Dict, Any
import numpy as np
from .types import Circle

def filter_spatial_outliers(circles: List[Circle], k: int = 5, std_dev_multiplier: float = 2.0) -> Tuple[List[Circle], List[Circle], Dict[str, Any]]:
    """
    Filtra círculos que são outliers espaciais e retorna dados detalhados para log.

    Args:
        circles (List[Circle]): A lista de círculos detectados.
        k (int): O número de vizinhos mais próximos a considerar.
        std_dev_multiplier (float): O fator de multiplicação do desvio padrão.

    Returns:
        Tuple[List[Circle], List[Circle], Dict[str, Any]]: Uma tupla contendo:
            - inliers (List[Circle]): A lista de círculos mantidos.
            - outliers (List[Circle]): A lista de círculos removidos.
            - log_data (Dict[str, Any]): Um dicionário com informações detalhadas do processo.
    """
    num_circles = len(circles)
    
    # Se houver poucos círculos, a filtragem não faz sentido.
    if num_circles <= k:
        # Retorna os círculos originais e um log vazio
        return circles, [], {}

    # 1. Extrai as coordenadas dos centros
    centers = np.array([[c.center_x, c.center_y] for c in circles])

    # 2. Calcula a matriz de distância
    dist_matrix = np.sqrt(((centers[:, np.newaxis, :] - centers[np.newaxis, :, :]) ** 2).sum(axis=2))

    # 3. Calcula a distância média para os k vizinhos de cada círculo
    avg_distances_to_neighbors = []
    for i in range(num_circles):
        distances = dist_matrix[i]
        nearest_distances = np.sort(distances)[1:k+1]
        avg_distances_to_neighbors.append(np.mean(nearest_distances))

    # 4. Calcula as estatísticas para o limiar de corte
    avg_distances_np = np.array(avg_distances_to_neighbors)
    mean_of_avgs = np.mean(avg_distances_np)
    std_dev_of_avgs = np.std(avg_distances_np)
    threshold = mean_of_avgs + std_dev_multiplier * std_dev_of_avgs

    # 5. Separa os círculos em inliers e outliers
    inliers = []
    outliers = []
    
    # Prepara uma lista detalhada com a "pontuação" de cada círculo para o log
    circle_scores = []
    for i, avg_dist in enumerate(avg_distances_np):
        # --- CORREÇÃO AQUI ---
        # Converte o resultado de numpy.bool_ para um bool padrão do Python
        is_outlier = bool(avg_dist > threshold)
        
        circle_scores.append({
            "circle_index": i,
            "center_x": circles[i].center_x,
            "center_y": circles[i].center_y,
            "avg_neighbor_distance": avg_dist,
            "is_outlier": is_outlier
        })
        if is_outlier:
            outliers.append(circles[i])
        else:
            inliers.append(circles[i])

    num_outliers = len(outliers)
    if num_outliers > 0:
        print(f"DEBUG: Filtro de outlier removeu {num_outliers} círculo(s).")

    # 6. Monta o dicionário de log com todos os dados úteis
    log_data = {
        "parameters": {
            "k_neighbors": k,
            "std_dev_multiplier": std_dev_multiplier
        },
        "statistics": {
            "mean_of_avg_distances": mean_of_avgs,
            "std_dev_of_avg_distances": std_dev_of_avgs,
            "outlier_threshold": threshold
        },
        "summary": {
            "circles_before": num_circles,
            "inliers_found": len(inliers),
            "outliers_removed": num_outliers
        },
        "circle_scores": circle_scores
    }

    return inliers, outliers, log_data
