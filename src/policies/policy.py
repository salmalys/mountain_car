from abc import ABC, abstractmethod
import numpy as np
import gymnasium as gym


class Policy(ABC):
    """
    Base Class for a policy in the context of reinforcement learning in gymnasium environment with discrete action.
    """

    def choose_action(self, q_values: np.array, hard_policy=False, **kwargs):
        """
        Method that choose an action according to specific parameters

        Args:
            - q_values (np.array): A 1D array of the q values of a state.
            - hard_policy (bool): If True, choose the argmax of q-values. If False, uses the policy
            - **kwargs (any): any parameters needed by the policy

        Returns:
            -(int): an action for the corresponding state
        """
        if hard_policy:
            # return np.random.choice(np.where(q_values == np.max(q_values))[0])
            return np.argmax(q_values)
        else:
            return self.soft_policy(q_values=q_values, **kwargs)

    @abstractmethod
    def soft_policy(self, q_values: np.array, **kwargs):
        """Using the policy to choose the action"""
        pass

    @abstractmethod
    def set_parameters(self, **kwargs):
        """Set the parameters of the policy"""
        pass

    @abstractmethod
    def update(self, **kwargs):
        """
        Update the parameters of the policy.
        """
        pass
