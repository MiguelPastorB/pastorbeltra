import matplotlib.pyplot as plt
import numpy as np

def plot_learning_curves(rewards_per_algorithm, window=50):
    """
    Genera la gráfica de recompensa media suavizada por episodio.
    """
    plt.figure(figsize=(10, 6))
    for label, rewards in rewards_per_algorithm.items():
        # Suavizado mediante media móvil para identificar tendencias 
        smoothed_rewards = np.convolve(rewards, np.ones(window)/window, mode='valid')
        plt.plot(smoothed_rewards, label=label)
    
    plt.title("Evolución de la Recompensa Media (Suavizada)")
    plt.xlabel("Episodios")
    plt.ylabel(f"Recompensa (Media móvil {window})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def plot_steps_per_episode(steps_per_algorithm, window=50):
    """
    Muestra la longitud media del episodio hasta la convergencia.
    Un descenso en los pasos indica que el agente encuentra el objetivo más rápido.
    """
    plt.figure(figsize=(10, 6))
    for label, steps in steps_per_algorithm.items():
        smoothed_steps = np.convolve(steps, np.ones(window)/window, mode='valid')
        plt.plot(smoothed_steps, label=label)
    
    plt.title("Pasos por Episodio (Eficiencia del Aprendizaje)")
    plt.xlabel("Episodios")
    plt.ylabel("Número de pasos")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()