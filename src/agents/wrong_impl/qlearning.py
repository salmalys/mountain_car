import numpy as np
import gymnasium as gym

from agents.td_agent import TDAgent


class QLearning(TDAgent):
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
            - Args:
                - state (tuple): a gymnasium state
                    * For example, what is returned by gym.Env.step()[0].
                    * See https://gymnasium.farama.org/api/env/ for more information
            - Returns:
                - (list(int)): a list of integers representing a state. Can be a list of just one integer.
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

    def choose_action(self, state, time_step, epsilon=None):
        if time_step not in ["t", "t+1"]:
            raise ValueError("Time step must be 't' or 't+1'")

        if (epsilon is None) and (time_step == "t"):
            raise ValueError("Epsilon must be given if time_step = 1")

        # Choose the action following an epsilon greedy policy for first action
        if time_step == "t":
            return self.epsilon_greedy_policy(state, epsilon)

        # Choose the argmax for qlearning agent
        if time_step == "t+1":
            return np.argmax(
                [self.q_value(state, action) for action in range(self.nb_actions)]
            )

    def update_parameters(
        self, state, action, reward, next_state, alpha, gamma, epsilon=None
    ):
        # Compute the SARSA update
        q_current = self.q_value(state, action)
        q_next = self.q_value(
            next_state, self.choose_action(next_state, time_step="t+1")
        )
        target = reward + gamma * q_next
        error = target - q_current

        # Update Q-values of the state
        for encoded_state in self.encode_fct(state):
            self.q[(encoded_state, action)] = (
                self.q.get((encoded_state, action), 0) + alpha * error
            )
