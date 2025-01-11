import gymnasium
from sarsa import SemiGradientSarsa
from utils import mc_tile_encoding

env = gymnasium.make("MountainCar-v0")
agent = SemiGradientSarsa(env, encode_fct=mc_tile_encoding)

# Entraînement
agent.train(nb_episodes=1000, alpha=0.1, gamma=0.99, epsilon=0.001)

# # Sauvegarder les paramètres
# agent.save_params("params/weights.npy")
