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

        # Choose the action following a softmax policy i.e. choosing the action by i
        if time_step == "t+1":
            return self.softmax_policy(state)

    def softmax(self, x: list):
        """substracting max_x to handle numerical overflow for exp. Doesn't change mathematically the result"""
        x = np.array(x)
        max_x = np.max(x)
        e_x = np.exp(x - max_x)
        return e_x / sum(e_x)
        # return np.exp(x) / sum(np.exp(x))

    def softmax_policy(self, state):
        """
        Chose an action with a probability relative to its q-value

        Args:
            - state (hashable): the state for which we choose an action

        Returns:
            - (int): an action.
        """
        # A list of the values for each action for the given state
        state_values = [
            self.q_value(state, action) for action in range(self.nb_actions)
        ]
        return np.random.choice(
            a=[action for action in range(self.nb_actions)],  # Array of possible action
            p=self.softmax(x=state_values),  # Array of probabilities for each action
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

        # Update Q-value of the state
        for encoded_state in self.encode_fct(state):
            self.q[(encoded_state, action)] = (
                self.q.get((encoded_state, action), 0) + alpha * error
            )
