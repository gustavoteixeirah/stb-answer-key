# Arquivo: src/shared/geometry_utils.py

from typing import List, Tuple, Dict, Any
import numpy as np
from .types import Circle
import math

def _calculate_distance(c1: Circle, c2: Circle) -> float:
    """Função auxiliar para calcular a distância euclidiana entre os centros de dois círculos."""
    return math.sqrt((c1.center_x - c2.center_x)**2 + (c1.center_y - c2.center_y)**2)

def merge_close_circles(circles: List[Circle], dist_threshold: float) -> List[Circle]:
    """
    Remove círculos duplicados de uma lista, fundindo os que estão muito próximos.

    Esta função itera sobre uma lista de círculos e agrupa aqueles cujos centros
    estão mais próximos do que um determinado limiar. Os círculos agrupados são
    substituídos por um único círculo cuja posição e raio são a média dos
    círculos originais.

    Args:
        circles (List[Circle]): A lista de objetos Circle para processar.
        dist_threshold (float): A distância máxima entre os centros para que dois
                                círculos sejam considerados uma duplicata a ser fundida.

    Returns:
        List[Circle]: Uma nova lista contendo os círculos únicos e fundidos.
    """
    unique_circles: List[Circle] = []
    
    for candidate_circle in circles:
        match_found_idx = -1  # Índice do círculo correspondente na lista de únicos
        
        # Procura por um círculo próximo na nossa lista de círculos já validados
        for i, existing_circle in enumerate(unique_circles):
            dist = _calculate_distance(candidate_circle, existing_circle)
            
            if dist < dist_threshold:
                match_found_idx = i
                break  # Encontrou uma correspondência, pode parar de procurar
        
        if match_found_idx != -1:
            # --- LÓGICA DE FUSÃO (MERGE) ---
            # Se encontrou um duplicado, funde o candidato com o existente
            
            circle_to_merge_with = unique_circles[match_found_idx]
            
            # Calcula a média simples dos centros e raios
            new_center_x = (candidate_circle.center_x + circle_to_merge_with.center_x) / 2
            new_center_y = (candidate_circle.center_y + circle_to_merge_with.center_y) / 2
            new_radius = (candidate_circle.radius + circle_to_merge_with.radius) / 2
            
            # O status 'filled' será True se qualquer um dos círculos fundidos for True
            new_filled_status = candidate_circle.filled or circle_to_merge_with.filled

            # Atualiza o círculo na lista de únicos com uma nova instância
            unique_circles[match_found_idx] = Circle(
                center_x=new_center_x,
                center_y=new_center_y,
                radius=new_radius,
                filled=new_filled_status
            )
        else:
            # Se nenhum círculo correspondente foi encontrado, adiciona o candidato
            # à lista de únicos.
            unique_circles.append(candidate_circle)
            
    return unique_circles


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
