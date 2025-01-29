import numpy as np
from gymnasium.envs.toy_text import CliffWalkingEnv

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3


class CustomCliffWalkingEnv(CliffWalkingEnv):
    """
    A custom Cliff Waling environment from gymnasium.

    This environment is used to study the impact of different probabilities of slippery.
    The easier way to proceed to a custom probability is to set slippery of the initial environment to False. Then,
    overwritte the step method and take a perpendicular action with the given probability (self.p)
    """

    def __init__(self, render_mode=None, is_slippery=False, slippery_prob=0.33):
        super().__init__(render_mode=render_mode, is_slippery=is_slippery)
        self.slippery_prob = slippery_prob  # Default probability for random actions

        # Proba to actually take the selected action.
        no_slippery_prob = 1 - self.slippery_prob

        # Probability of perpendicular actions to be taken.
        side_slippery_prob = self.slippery_prob / 2

        # Keys: action, values: array of probability of slippery to action each other action (actions are the index of array)
        self.transition_prob = {
            UP: [no_slippery_prob, side_slippery_prob, 0, side_slippery_prob],
            RIGHT: [side_slippery_prob, no_slippery_prob, side_slippery_prob, 0],
            DOWN: [0, side_slippery_prob, no_slippery_prob, side_slippery_prob],
            LEFT: [side_slippery_prob, 0, side_slippery_prob, no_slippery_prob],
        }

    def reset(self, seed=None):
        state, _ = super().reset(seed=seed)

        return state, {"slippey_prob": self.slippery_prob}

    def step(self, action):
        # Call the parent step method
        state, reward, done, truncated, _ = super().step(
            np.random.choice(
                a=[UP, RIGHT, DOWN, LEFT], p=self.transition_prob[action]
            )  # Choosing the action according to the probability of slippery
        )

        return state, reward, done, truncated, {"slippey_prob": self.slippery_prob}
