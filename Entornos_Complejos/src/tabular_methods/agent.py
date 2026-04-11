import numpy as np
import gymnasium as gym

class TabularAgent:
    def __init__(self, env, alpha=0.1, gamma=0.9, epsilon=0.1):
        """
        Clase base para agentes con representación tabular (Tablas Q).
        :param env: Entorno de Gymnasium.
        """
        self.env = env
        self.alpha = alpha   # Tasa de aprendizaje
        self.gamma = gamma   # Factor de descuento
        self.epsilon = epsilon # Parámetro de exploración
        
        # Inicializamos la tabla Q con ceros (n_estados x n_acciones)
        self.q_table = np.zeros([env.observation_space.n, env.action_space.n])

    def get_action(self, state, train=True):
        """
        Selecciona una acción usando la política epsilon-greedy.
        """
        if train and np.random.random() < self.epsilon:
            return self.env.action_space.sample() # Exploración
        return np.argmax(self.q_table[state])     # Explotación

    def update(self):
        """Método abstracto a implementar por cada algoritmo específico."""
        pass