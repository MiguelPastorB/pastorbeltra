from tqdm import tqdm

def train_tabular_agent(agent, n_episodes, seed=None):
    """
    Orquesta el aprendizaje de un agente tabular en un entorno de Gymnasium.
    
    Args:
        agent: Instancia de QLearningAgent, SARSAAgent o MonteCarloAgent.
        n_episodes: Número de episodios de entrenamiento.
        seed: Semilla para reproducibilidad del entorno.
        
    Returns:
        rewards_history: Lista con la recompensa total de cada episodio.
        steps_history: Lista con el número de pasos por episodio.
    """
    rewards_history = []
    steps_history = []
    
    # Determinamos el tipo de algoritmo para aplicar la lógica correcta
    algo_type = agent.__class__.__name__

    for episode in tqdm(range(n_episodes), desc=f"Entrenando {algo_type}"):
        # Reiniciamos entorno con semilla para consistencia 
        state, info = agent.env.reset(seed=seed + episode if seed else None)
        done = False
        total_reward = 0
        steps = 0
        
        # Para SARSA necesitamos la acción inicial (On-policy)
        if algo_type == "SARSAAgent":
            action = agent.get_action(state)

        while not done:
            if algo_type != "SARSAAgent":
                action = agent.get_action(state)
            
            # Interacción con el entorno
            next_state, reward, terminated, truncated, info = agent.env.step(action)
            done = terminated or truncated
            
            # Actualizaciones según el algoritmo
            if algo_type == "QLearningAgent":
                agent.update(state, action, reward, next_state, terminated)
            
            elif algo_type == "SARSAAgent":
                next_action = agent.get_action(next_state)
                agent.update(state, action, reward, next_state, terminated, next_action)
                action = next_action # Siguiente acción para el próximo paso
            
            elif algo_type == "MonteCarloAgent":
                agent.add_experience(state, action, reward)
            
            state = next_state
            total_reward += reward
            steps += 1
        
        # Monte Carlo actualiza solo al finalizar el episodio
        if algo_type == "MonteCarloAgent":
            agent.update()
            
        rewards_history.append(total_reward)
        steps_history.append(steps)
        
    return rewards_history, steps_history