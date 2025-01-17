import numpy as np
import gymnasium as gym

from agents.td_agent import TDAgent
from policies.policy import Policy


class QLearning(TDAgent):
    """
    Theoretical guarantees: https://sites.ualberta.ca/~szepesva/papers/sarsa98.ps.pdf
    """

    def __init__(
        self,
        encode_fct: callable,
        policy: Policy,
        nb_actions: int,
    ):
        """
        - encode_fct (callable): Function to encode the state into features (e.g., tile coding).
            - Args:
                - state: a gymnasium state
                    * For example, what is returned by gym.Env.step()[0].
                    * See https://gymnasium.farama.org/api/env/ for more information
            - Returns:
                - (list(int)): a list of integers representing a state. Can be a list of just one integer.

        - policy (callable): Function to choose an action for a state.
            - Args:
                - q_values (list): The list of q-values for a state
                - **kwargs: parameters of the policy
            -Returns
                - (int): an action

        - nb_actions (int): Number of possible actions.
        """
        self.encode_fct = encode_fct
        self.policy = policy
        self.nb_actions = nb_actions
        self.q = {}

    def reset(self):
        self.q = {}

    def q_value(self, state, action):
        # Sum zero if no key,value found in the dict
        return sum(self.q.get((tile, action), 0) for tile in self.encode_fct(state))

    def choose_action(self, state, **kwargs):
        return self.policy.choose_action(
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
        q_next = max(
            [self.q_value(next_state, a) for a in range(self.nb_actions)]
        )  # Off policy update

        # Compute the difference between t+1 and t
        target = reward if is_final else reward + gamma * q_next
        error = target - q_current

        # Update Q-value of the current state
        for encoded_state in self.encode_fct(state):
            self.q[(encoded_state, action)] = (
                self.q.get((encoded_state, action), 0) + alpha * error
            )
