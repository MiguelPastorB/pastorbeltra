from .agent import TabularAgent
import numpy as np

class MonteCarloAgent(TabularAgent):
    def __init__(self, env, alpha=0.1, gamma=0.9, epsilon=0.1):
        super().__init__(env, alpha, gamma, epsilon)
        # Diccionario para llevar el conteo de visitas o promedios
        self.episode_buffer = []

    def add_experience(self, state, action, reward):
        """Almacena la transición para procesarla al final del episodio."""
        self.episode_buffer.append((state, action, reward))

    def update(self):
        """
        Actualización de Monte Carlo:
        Calcula el retorno G para cada estado-acción visitado en el episodio.
        """
        G = 0
        # Procesamos el episodio de atrás hacia adelante para calcular G eficientemente
        for state, action, reward in reversed(self.episode_buffer):
            G = reward + self.gamma * G
            
            # Actualización incremental de la tabla Q hacia el retorno G
            # Q(S,A) <- Q(S,A) + alpha * (G - Q(S,A))
            self.q_table[state][action] += self.alpha * (G - self.q_table[state][action])
        
        # Limpiamos el buffer para el siguiente episodio
        self.episode_buffer = []