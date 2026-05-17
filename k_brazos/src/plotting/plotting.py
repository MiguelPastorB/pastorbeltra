from typing import List

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from ..algorithms.algorithm import Algorithm
from ..algorithms.epsilon_greedy import EpsilonGreedy


def get_algorithm_label(algo: Algorithm) -> str:
    label = type(algo).__name__
    # Manejo de EpsilonGreedy y sus variantes
    if hasattr(algo, 'epsilon'):
        label += f" ($epsilon$={algo.epsilon})"
    # Manejo de UCB1 (suele usar un parámetro 'c')
    if hasattr(algo, 'c'):
        label += f" ($c$={algo.c})"
    # Manejo de Softmax (usa temperatura)
    if hasattr(algo, 'temperature'):
        label += f" (temp={algo.temperature})"
    
    return label


def plot_average_rewards(steps: int, rewards: np.ndarray, algorithms: List[Algorithm]):
    """
    Genera la gráfica de Recompensa Promedio vs Pasos de Tiempo.

    :param steps: Número de pasos de tiempo.
    :param rewards: Matriz de recompensas promedio.
    :param algorithms: Lista de instancias de algoritmos comparados.
    """
    sns.set_theme(style="whitegrid", palette="muted", font_scale=1.2)

    plt.figure(figsize=(14, 7))
    for idx, algo in enumerate(algorithms):
        label = get_algorithm_label(algo)
        plt.plot(range(steps), rewards[idx], label=label, linewidth=2)

    plt.xlabel('Pasos de Tiempo', fontsize=14)
    plt.ylabel('Recompensa Promedio', fontsize=14)
    plt.title('Recompensa Promedio vs Pasos de Tiempo', fontsize=16)
    plt.legend(title='Algoritmos')
    plt.tight_layout()
    plt.show()


def plot_optimal_selections(steps: int, optimal_selections: np.ndarray, algorithms: List[Algorithm]):
    sns.set_theme(style="whitegrid", palette="muted")
    plt.figure(figsize=(12, 6))

    for idx, algo in enumerate(algorithms):
        plt.plot(range(steps), optimal_selections[idx], label=get_algorithm_label(algo))

    plt.xlabel('Pasos de Tiempo')
    plt.ylabel('% Selección Brazo Óptimo')
    plt.title('Rendimiento: Selección del Brazo Óptimo')
    plt.legend()
    plt.show()

def plot_regret(steps: int, regret_accumulated: np.ndarray, algorithms: List[Algorithm], *args):
    """
    Genera la gráfica de Regret Acumulado vs Pasos de Tiempo.
    """
    sns.set_theme(style="whitegrid", palette="dark")
    plt.figure(figsize=(12, 6))

    for idx, algo in enumerate(algorithms):
        plt.plot(range(steps), regret_accumulated[idx], label=get_algorithm_label(algo))

    plt.xlabel('Pasos de Tiempo')
    plt.ylabel('Regret Acumulado')
    plt.title('Evolución del Rechazo (Regret) Acumulado')
    plt.legend()
    plt.show()