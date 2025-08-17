from dataclasses import dataclass

@dataclass
class Circle:
    """
    Representa um círculo com raio, centro e um estado opcional de preenchimento.
    
    Attributes:
        radius (float): O raio do círculo.
        center_x (float): A coordenada X do centro.
        center_y (float): A coordenada Y do centro.
        filled (bool, optional): True se o círculo está preenchido. 
                                 O padrão é False.
    """
    radius: float
    center_x: float
    center_y: float
    filled: bool = False
