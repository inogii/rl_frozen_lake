import gymnasium as gym
import numpy as np
import random

# Extract environment dynamics function (assumed to be in the original file)
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

# Q-learning algorithm
def q_learning(env, states, actions, episodes=5000, alpha=0.1, gamma=0.99, epsilon=0.1):
    Q = np.zeros((len(states), len(actions)))

    for episode in range(episodes):
        state = env.reset()
        state = np.asscalar(np.array(state))  # Convert state to integer if necessary
        done = False

        while not done:
            # Epsilon-greedy action selection
            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()  # Explore action space
            else:
                action = np.argmax(Q[state])  # Exploit learned values

            next_state, reward, done, _ = env.step(action)
            next_state = np.asscalar(np.array(next_state))  # Convert next_state to integer

            # Q-learning update
            Q[state, action] = Q[state, action] + alpha * (reward + gamma * np.max(Q[next_state]) - Q[state, action])
            state = next_state

    return Q


# Main function
def main():
    # Setup environment
    desc = ["SFFF", "FHFH", "FFFH", "HFFG"]
    env = gym.make('FrozenLake-v1', desc=desc, is_slippery=True, render_mode="human")
    states = list(range(env.observation_space.n))
    actions = list(range(env.action_space.n))

    # Train Q-learning model
    Q = q_learning(env, states, actions)

    # Test the learned policy
    state = env.reset()
    env.render()
    done = False
    while not done:
        action = np.argmax(Q[state])
        state, _, done, _ = env.step(action)
        env.render()

    env.close()

if __name__ == '__main__':
    main()