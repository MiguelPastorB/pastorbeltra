import numpy as np

from algorithms.algorithm import Algorithm

class EpsilonGreedy(Algorithm):

    def __init__(self, k: int, epsilon: float = 0.1):
        """
        Inicializa el algoritmo epsilon-greedy.

        :param k: Número de brazos.
        :param epsilon: Probabilidad de exploración (seleccionar un brazo al azar).
        :raises ValueError: Si epsilon no está en [0, 1].
        """
        assert 0 <= epsilon <= 1, "El parámetro epsilon debe estar entre 0 y 1."

        super().__init__(k)
        self.epsilon = epsilon

    def select_arm(self) -> int:
        """
        Selecciona un brazo basado en la política epsilon-greedy.

        :return: índice del brazo seleccionado.
        """

        # Observa que para para epsilon=0 solo selecciona un brazo y no hace un primer recorrido por todos ellos.
        # ¿Podrías modificar el código para que funcione correctamente para epsilon=0?

        if np.random.random() < self.epsilon:
            # Selecciona un brazo al azar
            chosen_arm = np.random.choice(self.k)
        else:
            # Selecciona el brazo con la recompensa promedio estimada más alta
            chosen_arm = np.argmax(self.values)

        return chosen_arm

class DecayingEpsilonGreedy(EpsilonGreedy):
    def __init__(self, k, initial_epsilon=1.0, decay_rate=0.01):
        super().__init__(k, initial_epsilon)
        self.initial_epsilon = initial_epsilon
        self.decay_rate = decay_rate
        self.t = 0

    def reset(self):
        # Reiniciamos el contador de tiempo y el epsilon para que el algoritmo no sea demasiado agresivo
        super().reset()
        self.t = 0
        self.epsilon = self.initial_epsilon

    def select_arm(self):
        self.t += 1
        # Usamos un decaimiento más suave para dar tiempo a explorar
        self.epsilon = self.initial_epsilon / (1 + self.decay_rate * self.t)
        
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.k)
        else:
            return np.argmax(self.values)