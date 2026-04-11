import torch
import torch.optim as optim
import numpy as np
import random
from .networks import QNetwork

class SARSASemiGradientAgent:
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99, epsilon=0.1):
        """
        Agente SARSA con aproximación de funciones (Semi-gradiente).
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        
        # Red Neuronal para aproximar Q(s,a)
        self.policy_net = QNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)

    def get_action(self, state, train=True):
        """Selección de acción epsilon-greedy."""
        if train and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        
        state_t = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.policy_net(state_t)
        return torch.argmax(q_values).item()

    def update(self, state, action, reward, next_state, next_action, done):
        """
        Actualización semi-gradiente de SARSA.
        """
        state_t = torch.FloatTensor(state).unsqueeze(0)
        next_state_t = torch.FloatTensor(next_state).unsqueeze(0)
        
        # Valor Q actual estimado: Q(s, a; w)
        current_q = self.policy_net(state_t)[0][action]
        
        # Calculamos el objetivo usando la acción siguiente real (On-policy)
        with torch.no_grad():
            if done:
                target = torch.tensor(reward, dtype=torch.float32)
            else:
                next_q = self.policy_net(next_state_t)[0][next_action]
                target = reward + self.gamma * next_q
        
        # Error cuadrático medio y paso de gradiente 
        loss = torch.nn.functional.mse_loss(current_q, target)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()