import gymnasium as gym

## mapa con el lago congelado
desc = ["SFFF", "FHFH", "FFFH", "HFFG"]

## creamos el entorno frozen lake
env = gym.make('FrozenLake-v1', desc=desc, is_slippery=True, render_mode="human")

observation, info = env.reset(seed=42)

for _ in range(100000):
    #Escoge una accion al azar. Aqui deberia estar la funcion del agente que escoge la accion a ejecutar
    action = env.action_space.sample()
    #Step para realizar los cambios en el entorno
    observation, reward, terminated, truncated, info = env.step(action)
    #Reset del entorno si se acaba el episodio
    if terminated or truncated:
        observation, info = env.reset()
    #Cerrar la instancia de pygame
env.close()
