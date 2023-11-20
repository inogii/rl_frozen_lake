import gymnasium as gym
import numpy as np
import argparse

'''
FUNCIONES DE NUESTRO CÓDIGO
- value iteration: calcula con el algoritmo de iteración de valor la política óptima dado el entorno
- policy iteration: calcula con el algoritmo de iteración de valor la política óptima dado el entorno
    - policy_evaluation: evalúa la política
    - policy_improvement: mejora la política
- extract_environment_dynamics: extrae los parámetros de un entorno env
'''

def value_iteration(states, actions, transition_probabilities, rewards, gamma=0.99, threshold=0.001):
    V = np.zeros(len(states))
    policy = np.zeros(len(states), dtype=int)
    
    while True:
        V_prev = np.copy(V)
        for s in states:
            Q = [sum([p * (reward + gamma * V_prev[s_next]) for p, s_next, reward, _ in transition_probabilities[s][a]]) for a in actions]
            V[s] = max(Q)
            policy[s] = np.argmax(Q)
        
        if np.max(np.abs(V - V_prev)) < threshold:
            break
            
    return policy, V

def extract_environment_dynamics(env):
    states = list(range(env.observation_space.n))
    actions = list(range(env.action_space.n))
    transition_probabilities = {s: {a: [] for a in actions} for s in states}
    rewards = np.zeros((len(states), len(actions)))

    for s in states:
        for a in actions:
            for probability, next_state, reward, terminated in env.P[s][a]:
                transition_probabilities[s][a].append((probability, next_state, reward, terminated))
                rewards[s][a] += probability * reward  # Average reward for this state-action pair
    
    return states, actions, transition_probabilities, rewards

def policy_evaluation(policy, states, actions, transition_probabilities, rewards, gamma=0.99, threshold=0.001):
    V = np.zeros(len(states))
    
    while True:
        V_prev = np.copy(V)
        for s in states:
            a = policy[s]
            V[s] = sum([p * (reward + gamma * V_prev[s_next]) for p, s_next, reward, _ in transition_probabilities[s][a]])
        
        if np.max(np.abs(V - V_prev)) < threshold:
            break

    return V

def policy_improvement(V, states, actions, transition_probabilities, rewards, gamma=0.99):
    policy = np.zeros(len(states), dtype=int)
    
    for s in states:
        Q = [sum([p * (reward + gamma * V[s_next]) for p, s_next, reward, _ in transition_probabilities[s][a]]) for a in actions]
        policy[s] = np.argmax(Q)
    
    return policy

def policy_iteration(states, actions, transition_probabilities, rewards, gamma=0.99, threshold=0.001):
    policy = np.zeros(len(states), dtype=int)
    stable_policy = False

    while not stable_policy:
        V = policy_evaluation(policy, states, actions, transition_probabilities, rewards, gamma, threshold)
        new_policy = policy_improvement(V, states, actions, transition_probabilities, rewards, gamma)

        if np.array_equal(new_policy, policy):
            stable_policy = True
        else:
            policy = new_policy

    return policy, V


def main(args):
    # VARIABLES GLOBALES
    ## mapa con el lago congelado
    desc = ["SFFF", "FHFH", "FFFH", "HFFG"]
    ## creamos el entorno frozen lake
    env = gym.make('FrozenLake-v1', desc=desc, is_slippery=True, render_mode="human")
    states, actions, transition_probabilities, rewards = extract_environment_dynamics(env)
    
    # Choose algorithm based on args
    if args.algorithm == 'value':
        policy, _ = value_iteration(states, actions, transition_probabilities, rewards, gamma=0.99)
    elif args.algorithm == 'policy':
        policy, _ = policy_iteration(states, actions, transition_probabilities, rewards, gamma=0.99)
    else:
        raise ValueError("Invalid algorithm choice. Use 'value' or 'policy'.")
    
    # Use the learned policy in the environment
    observation, info = env.reset(seed=42)
    for _ in range(100000):
        action = policy[observation]
        observation, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            observation, info = env.reset(seed=42)
    env.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run value or policy iteration on the Frozen Lake environment.")
    parser.add_argument('algorithm', choices=['value', 'policy'], help="The algorithm to use: 'value' for value iteration, 'policy' for policy iteration")
    args = parser.parse_args()

    main(args)