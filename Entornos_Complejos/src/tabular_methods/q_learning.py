from .agent import TabularAgent
import numpy as np

class QLearningAgent(TabularAgent):
    def update(self, state, action, reward, next_state, terminated):
        """
        Actualización de Q-Learning según la ecuación de Bellman.
        Q(s,a) = Q(s,a) + alpha * [reward + gamma * max(Q(s',a')) - Q(s,a)]
        """
        # Valor máximo esperado en el siguiente estado (Off-policy)
        best_next_q = np.max(self.q_table[next_state])
        
        # Si el estado es terminal, no hay valor futuro
        td_target = reward + (0 if terminated else self.gamma * best_next_q)
        
        # Actualización de la tabla
        self.q_table[state][action] += self.alpha * (td_target - self.q_table[state][action])