import numpy as np


def epsilon_greedy_policy(q_values, epsilon=None):
    """
    Choose an action for a state following an epsilon greedy policy.

    Args:
        - q_values (list): The list of q-values for a state
        - epsilon (float): the exploration probability

    Returns:
        - (int): an action
    """
    if epsilon is None:
        raise ValueError("Epsilon is required for epsilon greedy policy")

    if np.random.rand() < epsilon:
        return np.random.choice(len(q_values))  # Exploration
    return np.argmax(q_values)


def softmax(x: list):
    # Substracting max_x to handle numerical overflow for exp.
    # Doesn't change mathematically the result.
    x = np.array(x)
    max_x = np.max(x)
    e_x = np.exp(x - max_x)
    return e_x / sum(e_x)
    # return np.exp(x) / sum(np.exp(x))


def softmax_policy(q_values):
    """
    Chose an action for a state with a probability relative to its q-values

    Args:
        - q_values (list): The list of q-values for a state

    Returns:
        - (int): an action.
    """
    return np.random.choice(
        a=[action for action in range(len(q_values))],  # Array of possible action
        p=softmax(x=q_values),  # Array of probabilities for each action
    )
