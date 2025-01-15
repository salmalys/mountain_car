import numpy as np
import gymnasium as gym

from agents.td_agent import TDAgent


class SemiGradientSarsa(TDAgent):
    def __init__(self, encode_fct: callable, nb_actions: int, weights_dim: tuple):
        """
        - encode_fct (callable): Function to encode the state into features (e.g., tile coding).
            - Args:
                - state (tuple): a gymnasium state
                    * For example, what is returned by gym.Env.step()[0].
                    * See https://gymnasium.farama.org/api/env/ for more information
            - Returns:
                - (list(int)): a list of integers representing a state. Can be a list of just one integer.
        - nb_actions (int): Number of possible actions.
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

    def softmax(self, x: list):
        """substracting max_x to handle numerical overflow for exp"""
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
            return self.softmax_policy(state)

    def update_parameters(self, state, action, reward, next_state, alpha, gamma):
        enocoded_state = self.encode_fct(state)
        q_current = self.q_value(state, action)
        next_q_value = self.q_value(
            next_state, self.choose_action(next_state, time_step="t+1")
        )
        target = reward + gamma * next_q_value
        td_error = target - q_current

        # Update weights for the selected action and
        for encoded_state in enocoded_state:
            self.weights[action][encoded_state] += alpha * td_error
