import torch
import torch.nn as nn
# Definición de la Red Neuronal Profunda DQN
class DQN_Network(nn.Module):
    """
    Red neuronal para el algoritmo Deep Q-Network (DQN).
    Se compone de capas totalmente conectadas (FC) con activaciones ReLU.
    """
    def __init__(self, num_actions, input_dim):
        """
        Inicializa la red neuronal.
        Parámetros:
        num_actions (int): Número total de acciones posibles en el entorno.
        input_dim (int): Dimensión del espacio de estados de entrada.
        """
        super(DQN_Network, self).__init__()
        # Definición de las capas de la red neuronal
        self.FC = nn.Sequential(
            nn.Linear(input_dim, 12), # Capa de entrada con 12 neuronas
        nn.ReLU(inplace=True), # Función de activación ReLU
        nn.Linear(12, 8), # Capa oculta con 8 neuronas
        nn.ReLU(inplace=True), # Otra activación ReLU
        nn.Linear(8, num_actions) # Capa de salida con 'num_actions' neuronas
        )
        # Inicialización de pesos usando He initialization (Kaiming Uniform)
        for layer in [self.FC]:
            for module in layer:
                if isinstance(module, nn.Linear):
                    nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')

def forward(self, x):
    """
    Propagación hacia adelante de la red para calcular los valores Q.
    Parámetros:
    x (torch.Tensor): Estado de entrada representado como tensor.
    Retorna:
    Q (torch.Tensor): Valores Q para cada acción posible.
    """
    Q = self.FC(x) # Pasa el estado a través de la red neuronal
    return Q
