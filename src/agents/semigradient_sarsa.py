import numpy as np
import gymnasium as gym

from agents.td_agent import TDAgent


class SemiGradientSarsa(TDAgent):
    def __init__(self, encode_fct: callable, nb_actions: int, weights_dim: tuple):
        """
        - nb_actions (int): Number of possible actions.
        - encode_fct (callable): Function to encode the state into features (e.g., tile coding).
            The function must return an array integers representing the state. It it possible to return an array of just
            on integer that represent the state.
        - weights_dim (tuple): The dimension of the array of the weights
        """
        self.nb_actions = nb_actions
        self.encode_fct = encode_fct

        # Define weights (one per (state, action))
        # self.weights = np.random.uniform(-0.1, 0.1, size=(env.action_space.n, iht_size))
        self.weights_dim = weights_dim
        self.weights = np.random.uniform(-0.1, 0.1, size=weights_dim)

    def reset(self):
        # self.weights = np.random.uniform(-0.1, 0.1, size=(env.action_space.n, iht_size))
        self.weights = np.random.uniform(-0.1, 0.1, size=self.weights_dim)

    def q_value(self, state, action):

        encoded_state = self.encode_fct(state)
        return np.sum(self.weights[action][encoded_state])

    def choose_action(self, state, epsilon=None, exploration=True):
        if (exploration) and (epsilon is None):
            raise ValueError("Epsilon must be specified when exploration is True.")

        if (exploration) and (np.random.rand() < epsilon):
            return np.random.choice(self.nb_actions)
        return np.argmax([self.q_value(state, a) for a in range(self.nb_actions)])

    def update_parameters(
        self, state, action, reward, next_state, next_action, alpha, gamma
    ):
        enocoded_state = self.encode_fct(state)
        next_q_value = self.q_value(next_state, next_action)
        target = reward + gamma * next_q_value
        td_error = target - self.q_value(state, action)

        # Update weights for the selected action and
        for encoded_state in enocoded_state:
            self.weights[action][encoded_state] += alpha * td_error
