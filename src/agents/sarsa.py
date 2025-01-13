import numpy as np
import gymnasium as gym

from agents.td_agent import TDAgent


class Sarsa(TDAgent):
    """
    Theoretical guarantees: https://sites.ualberta.ca/~szepesva/papers/sarsa98.ps.pdf
    """

    def __init__(
        self,
        encode_fct,
        nb_actions=None,
    ):
        """
        - encode_fct (callable): Function to encode the state into features (e.g., tile coding).
            The function must return an array integers representing the state. It it possible to return an array of just
            on integer that represent the state.
        - nb_actions (int): Number of possible actions.
        """
        self.nb_actions = nb_actions
        self.encode_fct = encode_fct
        self.q = {}

    def reset(self):
        self.q = {}

    def q_value(self, state, action):
        # Sum zero if no key,value found in the dict
        return sum(self.q.get((tile, action), 0) for tile in self.encode_fct(state))

    def choose_action(self, state, epsilon=None, exploration=True):
        if (exploration) and (epsilon is None):
            raise ValueError("Epsilon must be specified when exploration is True.")

        if (exploration) and (np.random.rand() < epsilon):
            return np.random.choice(self.nb_actions)  # Exploration
        return np.argmax(
            [self.q_value(state, action) for action in range(self.nb_actions)]
        )

    def update_parameters(
        self, state, action, reward, next_state, next_action, alpha, gamma
    ):
        """
        Perform the update of the weights.

        Args:
            - state (hashable): The current state
            - action (int): The chosen action
            - reward (float): The reward for the next (state, action)
            - next_state (hashable): The t+1 state
            - next_action (int): The t+1 action
            - alpha (float): The learning rate
            - gamma (float): The discount factor
        """
        # Compute the SARSA update
        q_current = self.q_value(state, action)
        q_next = self.q_value(next_state, next_action)
        target = reward + gamma * q_next
        error = target - q_current

        # Update Q-values for all tiles
        for tile in self.encode_fct(state):
            self.q[(tile, action)] = self.q.get((tile, action), 0) + alpha * error
