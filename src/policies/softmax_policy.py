from policies.policy import Policy
import numpy as np


class Softmax(Policy):

    def __init__(self, temperature):
        # Temperature add noise to the output distribution of softmax.
        # Can be interpreted as a level of confidence of the agent
        # The higher the temperature, the less the output looks like a distribution
        #       low temperature softmax probs : [0.01,0.01,0.98]
        #       high temperature softmax probs : [0.2,0.2,0.6]
        self.temperature = temperature

    def softmax(self, x: list):
        # Substracting max_x to handle numerical overflow for exp.
        # Doesn't change mathematically the result.
        x = np.array(x)
        max_x = np.max(x)
        e_x = np.exp(x - max_x)
        return e_x / sum(e_x)

    def soft_policy(self, q_values: np.array):
        """
        Chose an action for a state with a probability relative to its q-values

        Args:
            - q_values (list): The list of q-values for a state

        Returns:
            - (int): an action.
        """
        return np.random.choice(
            a=[action for action in range(len(q_values))],  # Array of possible action
            p=self.softmax(x=q_values),  # Array of probabilities for each action
        )

    def update(
        self,
        max_step: int,
        curr_step: int,
        to_decay: bool = False,
        min_temperature: float = 0.001,
        verbose: int = 0,
    ):
        """
        Decay epsilon through training.

        Args:
            - to_decay (bool): True if we cant to decay temperature during training
            - max_step (int): the maximum number of action or episode during the training.
            - curr_step (int): the current step or episode
            - min_temperature (float): the minimum possible epsilon
            - verbose (int): Verbosity level for debugging (0: silent, 1: general informations, 2: Precise informations).
        """
        if (to_decay) and (curr_step % (max_step // 5) == 0) and (curr_step > 0):
            if not (self.temperature / 2 < min_temperature):
                self.temperature /= 2
                if verbose > 1:
                    print(f"\nself. updated to: {self.temperature}\n")
        else:
            pass

    def set_parameters(self, temperature):
        self.temperature = temperature
