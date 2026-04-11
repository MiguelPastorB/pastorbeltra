import torch
import torch.nn as nn

class QNetwork(nn.Module):
    """
    Red neuronal para aproximar la función Q en espacios continuos.
    Utiliza capas totalmente conectadas (FC) y activaciones ReLU.
    """
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 64),  # Capa de entrada
            nn.ReLU(),                 # Activación ReLU
            nn.Linear(64, 64),         # Capa oculta
            nn.ReLU(),
            nn.Linear(64, action_dim)  # Capa de salida (valores Q por acción)
        )
        
        # Inicialización de pesos sugerida (Kaiming)
        for m in self.fc:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')

    def forward(self, x):
        """Propagación hacia adelante para calcular valores Q."""
        return self.fc(x)