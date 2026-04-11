from .agent import TabularAgent
import numpy as np

class SARSAAgent(TabularAgent):
    def update(self, state, action, reward, next_state, terminated, next_action=None):
        """
        Actualización de SARSA (On-policy).
        """
        # Si no se proporciona next_action (por el bucle del trainer), la seleccionamos siguiendo la política actual
        if next_action is None:
            next_action = self.get_action(next_state)

        td_target = reward + (0 if terminated else self.gamma * self.q_table[next_state][next_action])
        td_error = td_target - self.q_table[state][action]
        
        self.q_table[state][action] += self.alpha * td_error