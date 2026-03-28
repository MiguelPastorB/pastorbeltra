import numpy as np

from arms import Arm

class ArmBinomial(Arm):
    def __init__(self, n: int, p: float):
        assert n > 0, "N debe ser un entero positivo"
        assert 0 <= p <= 1, "La probabilidad debe estar entre 0 y 1"

        self.n = n
        self.p = p

    def pull(self):
        reward = np.random.binomial(self.n, self.p)
        return reward

    def get_expected_value(self) -> float:
        return self.n * self.p

    def __str__(self):
        return f"ArmBinomial(n={self.n}, p={self.p})"

    @classmethod
    def generate_arms(cls, k: int, n: int = 10, p_min: float = 0.1, p_max: float = 0.9):
        assert k > 0, "El número de brazos k debe ser mayor que 0."
        assert 0 <= p_min < p_max <= 1, "Rango de p inválido. Debe estar entre 0 y 1"

        p_vals = set()
        while len(p_vals) < k:
            p = np.random.uniform(p_min, p_max) # Generar k valores únicos de p
            p = round(p, 2)
            p_vals.add(p)

        p_vals = list(p_vals)

        arms = [ArmBinomial(n, p) for p in p_vals]

        return arms