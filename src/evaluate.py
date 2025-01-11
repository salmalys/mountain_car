import gym
from sarsa import SemiGradientSarsa
from utils import mc_tile_encoding

env = gym.make("MountainCar-v0", render_mode="human")
agent = SemiGradientSarsa(env, encode_fct=mc_tile_encoding)

# Charger les paramètres et évaluer
agent.load_params("params/weights.npy")
agent.evaluate_policy(num_episodes=10, verbose=True, render_mode="human")
