from policies.policy import Policy
import numpy as np


class EpsGreedy(Policy):

    def __init__(self, epsilon: float = None):
        self.epsilon = epsilon

    def soft_policy(self, q_values: np.array, epsilon: float = None):
        """
        Let a* = argmax_a[q(s, a)]. If a* is not unique, we randomly chose one
        -  If a = a*, policy(s, a) = 1 - ϵ + (ϵ / nb_actions).
        -  If not, policy(s, a) = ϵ / nb_actions.

        Args:
            - q_values (numpy.array)): any parameters needed by the policy
            - epsilon (float): If not None: temporary epsilon for the action Else: self.epsilon.
                Used in grid search for different combinations of parameters
        Returns:
            -(int): an action
        """
        if (self.epsilon is None) and (epsilon is None):
            raise ValueError("Epsilon is required for epsilon greedy policy")

        treshold = self.epsilon if epsilon is None else epsilon
        if np.random.rand() < treshold:
            return np.random.choice(len(q_values))  # Exploration
        return np.random.choice(np.where(q_values == np.max(q_values))[0])

    def update(
        self,
        max_step: int,
        curr_step: int,
        update_frequency: int = 5,
        verbose: int = 0,
        # kwargs:
        decay_type: str = "by_stage",
        use_glei: bool = False,
        min_epsilon: float = 0.001,
    ):
        """
        Decay epsilon through training.

        Args:
            - to_decay (bool): True if we cant to decay epsilon during training
            - max_step (int): the maximum number of action or episode during the training.
            - curr_step (int): the current step or episode
            - min_epsilon (float): the minimum possible epsilon
            - verbose (int): Verbosity level for debugging (0: silent, 1: general informations, 2: Precise informations).
        """
        if decay_type == "by_stage":
            if (
                (use_glei)
                and (curr_step % (max_step // update_frequency) == 0)
                and (curr_step > 0)
            ):
                if not (self.epsilon / 2 < min_epsilon):
                    self.epsilon /= 2
                    if verbose > 1:
                        print(f"\nEpsilon updated to: {self.epsilon}\n")
        elif decay_type == "gradually":
            epsilon = max(0.01, 0.995 * epsilon)
        else:
            pass

    def set_parameters(self, epsilon):
        self.epsilon = epsilon
