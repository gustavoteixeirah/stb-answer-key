from dataclasses import dataclass

@dataclass
class Circle:
    """
    Representa um círculo com raio, centro e um estado opcional de preenchimento.
    Garante que os atributos numéricos sejam sempre do tipo float padrão do Python.
    """
    radius: float
    center_x: float
    center_y: float
    filled: bool = False

    def __post_init__(self):
        """
        Executado automaticamente após a criação de uma instância de Circle.
        Este método converte os atributos numéricos para o tipo float padrão,
        garantindo a compatibilidade com serializadores como a biblioteca json.
        """
        self.radius = self.radius
        self.center_x = self.center_x
        self.center_y = self.center_y