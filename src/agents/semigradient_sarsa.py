import numpy as np
import gymnasium as gym

from agents.td_agent import TDAgent


class SemiGradientSarsa(TDAgent):
    def __init__(
        self,
        encode_fct: callable,
        policy: callable,
        nb_actions: int,
        weights_dim: tuple,
    ):
        """
        - encode_fct (callable): Function to encode the state into features (e.g., tile coding).
            - Args:
                - state (tuple): a gymnasium state
                    * For example, what is returned by gym.Env.step()[0].
                    * See https://gymnasium.farama.org/api/env/ for more information
            - Returns:
                - (list(int)): a list of integers representing a state. Can be a list of just one integer.
        - policy (callable): Function to choose an action for a state.
            - Args:
                - q_values (list): The list of q-values for a state
                - *kwargs: parameters of the policy
            -Returns
                - (int): an action
        - nb_actions (int): Number of possible actions.
        - weights_dim (tuple): The dimension of the array of the weights
        """
        self.nb_actions = nb_actions
        self.encode_fct = encode_fct
        self.policy = policy

        # Define weights (one per (state, action))
        self.weights_dim = weights_dim
        self.weights = np.random.uniform(-0.1, 0.1, size=weights_dim)

    def reset(self):
        # self.weights = np.random.uniform(-0.1, 0.1, size=(env.action_space.n, iht_size))
        self.weights = np.random.uniform(-0.1, 0.1, size=self.weights_dim)

    def q_value(self, state, action):

        encoded_state = self.encode_fct(state)
        return np.sum(self.weights[action][encoded_state])

    def choose_action(self, state, **kwargs):
        return self.policy(
            [
                self.q_value(state, action) for action in range(self.nb_actions)
            ],  # q-values
            **kwargs
        )

    def update_parameters(
        self,
        state,
        action,
        reward,
        next_state,
        next_action,
        alpha,
        gamma,
        is_final=False,
    ):
        # Compute the t and t+1 q-values
        q_current = self.q_value(state, action)
        q_next = self.q_value(next_state, next_action)  # On policy update

        # Compute the difference between t+1 and t
        target = reward if is_final else reward + gamma * q_next
        td_error = target - q_current

        # Update the weights of the current state
        for encoded_state in self.encode_fct(state):
            self.weights[action][encoded_state] += alpha * td_error
