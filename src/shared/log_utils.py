# Arquivo: src/shared/log_utils.py
# Um módulo com funções "estáticas" para logging em Markdown.


from typing import List
from .types import Circle  # Supondo que 'types.py' está no mesmo diretório 'shared'
import os
import json
import datetime
import cv2
import numpy as np
import re
from dataclasses import is_dataclass, asdict

OUTPUT_DIR = "debug_output"

def _get_paths(trace_id: str) -> tuple[str, str, str]:
    """Função auxiliar interna para gerar os caminhos necessários."""
    run_dir = os.path.join(OUTPUT_DIR, trace_id)
    assets_dir = os.path.join(run_dir, "assets")
    md_filepath = os.path.join(run_dir, "trace_report.md")
    return run_dir, assets_dir, md_filepath

def initialize_trace(trace_id: str):
    """
    Prepara os diretórios e o arquivo Markdown para um novo trace.
    Deve ser chamada no início de cada execução que será logada.
    """
    _, assets_dir, md_filepath = _get_paths(trace_id)
    os.makedirs(assets_dir, exist_ok=True)
    
    with open(md_filepath, "w", encoding="utf-8") as f:
        f.write(f"# Relatório de Debug - Trace ID: {trace_id}\n\n")
        f.write(f"Gerado em: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

def log_image(trace_id: str, image_np: np.ndarray, description: str):
    """
    Salva uma imagem de debug com um nome customizado e a adiciona ao relatório.
    """
    _, assets_dir, md_filepath = _get_paths(trace_id)
    
    safe_filename = re.sub(r'[^a-z0-9_]', '', description.lower().replace(' ', '_'))
    image_filename = f"{safe_filename}.png"
    image_path = os.path.join(assets_dir, image_filename)

    cv2.imwrite(image_path, image_np)
    print(f"DEBUG: Imagem '{image_filename}' salva.")

    with open(md_filepath, "a", encoding="utf-8") as f:
        relative_image_path = os.path.join("assets", image_filename)
        f.write(f"\n### Imagem Adicional: {description}\n")
        f.write(f"![{description}]({relative_image_path})\n")

def log_json(trace_id: str, data_object: any, description: str):
    """
    Salva um objeto Python como um arquivo JSON e o adiciona ao relatório.
    """
    _, assets_dir, md_filepath = _get_paths(trace_id)
    
    safe_filename = re.sub(r'[^a-z0-9_]', '', description.lower().replace(' ', '_'))
    json_filename = f"{safe_filename}.json"
    json_path = os.path.join(assets_dir, json_filename)

    def json_converter(o):
        """
        Converte objetos não serializáveis, como dataclasses e tipos do NumPy.
        """
        # --- MUDANÇA PRINCIPAL AQUI ---
        # Se for um tipo de ponto flutuante do NumPy, converte para float
        if isinstance(o, np.floating):
            return float(o)
        # Se for um tipo de inteiro do NumPy, converte para int
        if isinstance(o, np.integer):
            return int(o)
        # Se for um array do NumPy, converte para lista
        if isinstance(o, np.ndarray):
            return o.tolist()
        # -------------------------------

        if is_dataclass(o):
            return asdict(o)
        if isinstance(o, (datetime.datetime, datetime.date)):
            return o.isoformat()
            
        raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")

    try:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data_object, f, indent=4, ensure_ascii=False, default=json_converter)
        print(f"DEBUG: Arquivo JSON '{json_filename}' salvo.")

        with open(md_filepath, "a", encoding="utf-8") as f:
            relative_json_path = os.path.join("assets", json_filename)
            f.write(f"\n### Dados Adicionais: {description}\n")
            f.write(f"Os dados completos foram salvos no arquivo: [{json_filename}]({relative_json_path})\n")

    except TypeError as e:
        print(f"ERRO ao serializar o objeto para JSON: {e}")
    except Exception as e:
        print(f"ERRO ao salvar o arquivo JSON: {e}")


def draw_circles_on_image(image: np.ndarray, circles: List[Circle], color_filled: tuple = (0, 255, 0), color_unfilled: tuple = (0, 0, 255), thickness: int = 2) -> np.ndarray:
    """
    Desenha uma lista de círculos em uma cópia da imagem.

    Args:
        image (np.ndarray): A imagem original (formato OpenCV BGR).
        circles (List[Circle]): Uma lista de objetos Circle para desenhar.
        color_filled (tuple): A cor (BGR) para círculos marcados como 'filled'. Padrão é verde.
        color_unfilled (tuple): A cor (BGR) para círculos não marcados. Padrão é vermelho.
        thickness (int): A espessura da borda do círculo.

    Returns:
        np.ndarray: Uma nova imagem com os círculos desenhados.
    """
    # Cria uma cópia da imagem para não modificar o array original.
    # Isso é importante para evitar efeitos colaterais inesperados.
    output_image = image.copy()
    
    for circle in circles:
        # Converte as coordenadas do centro para inteiros, que é o que o OpenCV espera.
        center_coordinates = (int(circle.center_x), int(circle.center_y))
        radius = int(circle.radius)
        
        # Escolhe a cor baseada no atributo 'filled' do círculo.
        color = color_filled if circle.filled else color_unfilled
        
        # Desenha o círculo na imagem de saída.
        cv2.circle(
            img=output_image,
            center=center_coordinates,
            radius=radius,
            color=color,
            thickness=thickness
        )
        
    return output_image
