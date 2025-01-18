from abc import ABC, abstractmethod
from itertools import product
import numpy as np
import gymnasium as gym


class TDAgent(ABC):
    """
    Base Class for TD(0) Agent (i.e. First order Time Difference Agent).
    This class is built to handle gymnasium environments with discrete or continuous states
    and discrete actions.
    """

    @abstractmethod
    def reset(self):
        """Reset the weights/q-values/network of the agent"""
        pass

    @abstractmethod
    def q_value(self, state, action: int):
        """
        Compute the Q-value for a given state and action.

        Args:
            - state (hashable): A state encoded
            - action (int): An action

        Returns:
            float: the value of the (state, action) tuple
        """
        pass

    @abstractmethod
    def update_parameters(
        self,
        state,
        action: int,
        reward: float,
        next_state,
        next_action,
        alpha: float,
        gamma: float,
        is_final=False,
    ):
        """
        Update the weights/q-values/network of the agent with the t+1 q(s,) value

        Args:
            - state (hashable): The current state
            - action (int): The chosen action
            - reward (float): The reward for next (state, action)
            - next_state (hashable): The t+1 state
            - next_acton (int): the t+1 action
            - alpha (float): The learning rate
            - gamma (float): The discount factor
            - is_final (bool): If the episode is over at t+1
        """

    @abstractmethod
    def choose_action(self, state, *args):
        """
        Choose an action for a given state by following the agent policy (self.policy).

        Args:
            - state (hashable): The current state.
            - *args (float): parameters for the agent policy. For example, epsilon for epsilon greedy policy

        Returns:
            - int: The selected action by the agent policy.

        Raises:
            - ValueError: If time_step is not in ["t", "t+1"]
            - ValueError: If epsilon is None and time step is "t"
        """
        pass

    def train(
        self,
        env: gym.Env,
        policy_action_params: dict,
        policy_update_params: dict,
        alpha: float = 0.1,
        gamma: float = 0.99,
        seed: int = None,
        nb_episodes: int = 1000,
        max_step: int = None,
        to_evaluate: bool = False,
        evaluation_frequency: int = 3,
        verbose: int = 0,
    ):
        """
        Training algorithm of the agent

        Args:
            - env (gymnasium.Env): the gymnasium environment to train on
            - policy_action_params (dict): arguments for the policy choose_action() method
                ex: {"epsilon": 0.05}
            - policy_update_params (dict): arguments for the policy update() method
                ex: {"to_decay": False}
            - alpha (float): Learning rate for updating Q-values.
            - gamma (float): Discount factor for future rewards.
            - nb_episodes (int): Number of episodes to train for.
            - seed (int): seed for the starting state randomness. Stochastic if None, Determinist if not
            - max_step (int): The maximum number of steps for environments with no episode,
            - to_evaluate (bool): If True, proceed to agent evaluation through training.
            - evaluation_frequency (int): evaluate every nb_episodes // evaluation frequency.
            - verbose (int): Verbosity level for debugging (0: silent, 1: general informations, 2: Precise informations).

        Returns:
            - rewards_historic (list): History of rewards across episodes.
        """
        if nb_episodes > 1 and max_step is not None:
            raise ValueError("You can not give both nb_episodes and max_step")

        # Initializations
        self.reset()
        rewards_per_episode = []
        evaluations = {"x": [], "data": []}
        step = 0

        for episode in range(nb_episodes):
            # Potentially update the policy parameters when using episodes as a limit
            # Potentially evaluate the agent at this point.
            self.policy.update(
                max_step=nb_episodes,
                curr_step=episode,
                verbose=verbose,
                **policy_update_params,
            )
            if to_evaluate:
                evaluations = self.evaluate_through_training(
                    env=env,
                    with_step=False,
                    max_step=nb_episodes,
                    curr_step=episode,
                    evaluation_frequency=evaluation_frequency,
                    evaluations=evaluations,
                    verbose=verbose,
                )

            # Initialize a new episode
            state, _ = env.reset(seed=seed)
            action = self.choose_action(state, **policy_action_params)
            task_completed, episode_over, episode_reward = False, False, 0

            # Simulate an episode
            while not (task_completed or episode_over):
                # Compute next state informations
                next_state, reward, task_completed, episode_over, _ = env.step(action)
                next_action = self.choose_action(next_state, **policy_action_params)
                step += 1

                # Update agent with t and t+1 values if task is not over at t+1
                self.update_parameters(
                    state,
                    action,
                    reward,
                    next_state,
                    next_action,
                    alpha,
                    gamma,
                    is_final=(task_completed or episode_over),
                )

                # Potentially update the policy parameters when using steps as a limit.
                # Potentially evaluate the agent at this point.
                if max_step is not None:
                    self.policy.update(
                        max_step=max_step,
                        curr_step=step,
                        verbose=verbose,
                        **policy_update_params,
                    )
                    if to_evaluate:
                        evaluations = self.evaluate_through_training(
                            env=env,
                            with_step=True,
                            max_step=max_step,
                            curr_step=step,
                            evaluation_frequency=evaluation_frequency,
                            evaluations=evaluations,
                            verbose=verbose,
                        )

                # Move to the next state and action
                state = next_state
                action = next_action
                episode_reward += reward

                # End training if max_step reached
                if (max_step is not None) and (step > max_step):
                    return rewards_per_episode + [episode_reward]

            rewards_per_episode.append(episode_reward)
            if verbose > 1:
                print(f"Episode {episode + 1}: Total Reward = {episode_reward}")

        if verbose == 1:
            print(f"Last reward of training {rewards_per_episode[-1]}")

        return rewards_per_episode, evaluations

    def evaluate_through_training(
        self,
        env,
        with_step,
        max_step,
        curr_step,
        evaluation_frequency,
        evaluations,
        verbose,
    ):
        """
        Handle evaluations through training
        """
        if (curr_step % (max_step // evaluation_frequency) == 0) and (curr_step > 0):
            evaluations["x"].append(curr_step),
            evaluations["data"].append(
                self.evaluate_policy(
                    env=env,
                    policy_action_params={"hard_policy": True},
                    nb_episodes=10 if with_step == False else 1,
                    max_step=None if with_step == False else max_step,
                    verbose=verbose,
                )
            )
        return evaluations

    def evaluate_policy(
        self,
        env: gym.Env,
        policy_action_params: dict,
        nb_episodes: int = 10,
        max_step: int = None,
        verbose: int = 0,
    ):
        """
        Evaluate the current Q-values of the agent and print the average reward.

        Args:
            - env (gymnasium.Env): the gymnasium environment evaluate the agent on.
            - policy_action_params (dict): arguments for the policy choose_action() method
            - nb_episodes (int): Number of episodes to evaluate the policy.
            - max_step (int): maximum step for evaluation.
            - verbose (bool): Verbosity level for debugging (0: silent, 1: general informations, 2: Precise informations).

        Returns:
            float: The average reward obtained over the evaluated episodes.
            OR
            list: When no episodes, the list of rewards for each step

        """
        # Initializations
        rewards_per_episode, step = [], 0

        for episode in range(nb_episodes):
            # Initialize episode
            state, _ = env.reset()
            task_completed, episode_over, episode_reward = False, False, 0

            # Simulate an episo
            while not (task_completed or episode_over):
                action = self.choose_action(state, **policy_action_params)
                state, reward, task_completed, episode_over, _ = env.step(action)
                step += 1
                episode_reward += reward
                if (max_step is not None) and (step > max_step):
                    return rewards_per_episode + [episode_reward]

            rewards_per_episode.append(episode_reward)
            if verbose > 1:
                print(f"Episode {episode + 1}: Total Reward = {episode_reward}")

        if verbose == 1:
            print(
                f"Average Total Reward over {nb_episodes} episodes: {np.mean(rewards_per_episode)}"
            )

        return np.mean(rewards_per_episode)

    def grid_search(
        self,
        env: gym.Env,
        alpha: list,
        gamma: list,
        policy_action_params=dict,
        policy_update_params=dict,
        nb_episodes=1000,
        nb_iter=10,
        verbose=0,
    ):
        """
        Perform a grid search over hyperparameters.

        Args:
            - alpha (list): List of values for the alpha hyperparameter.
            - gamma (list): List of values for the gamma hyperparameter.
            - nb_episodes (int): Number of episodes per training iteration.
            - nb_iter (int): Number of iterations for averaging results.
            - use_glei (bool): Whether to use the glei method during training.
            - verbose (int): Verbosity level for debugging (0: silent, 1: general informations, 2: Precise informations).
            - **kwargs (dict): Lists of other hyperparameter values to test (e.g., epsilon).
                Most likely to be parameters of the policy of the agent.
                Example: {epsilon: [0.005, 0.01]} for eps-greedy
                Example: {}temperature: [0.5, 1, 1.5]} for temp-softmax

        Returns:
            dict: A ranking of hyperparameter sets based on the metrics.
        """
        tune_historic = {}

        # Extract params names
        param_names = (
            ["alpha", "gamma"]
            + list(policy_action_params.keys())
            + list(policy_update_params.keys())
        )
        # Combine  values of alpha, gamma and policy params
        param_combinations = list(
            product(
                alpha,
                gamma,
                *(policy_action_params[param] for param in policy_action_params),
                *(policy_update_params[parma] for parma in policy_update_params),
            )
        )

        for combination in param_combinations:

            # Initialize variables for the given set of hyperparameters
            # Array where to stock training data
            data = np.empty((nb_iter, nb_episodes))

            # Current combination of parameters into a dict: {"param1": value1, "param2": value2, etc}
            params = dict(zip(param_names, combination))

            # Extracting from params the parameters for the train method
            train_policy_action_params = {
                k: v for k, v in params.items() if k in policy_action_params.keys()
            }
            train_policy_update_params = {
                k: v for k, v in params.items() if k in policy_update_params.keys()
            }

            # String to store the hyperparameters used for the data
            param_key = "_".join(f"{key}={value}" for key, value in params.items())

            if verbose == 1:
                print(f"\nProcessing:\n{param_key}\n")

            # Proceed to multiple training with the current set of hyperparameters
            for iteration in range(nb_iter):
                if verbose > 1:
                    print(f"Iteration {iteration + 1}/{nb_iter} for {param_key}")

                # Prepare training
                self.reset()

                # Training number i
                rewards, _ = self.train(
                    env=env,
                    alpha=params["alpha"],
                    gamma=params["gamma"],
                    policy_action_params=train_policy_action_params,
                    policy_update_params=train_policy_update_params,
                    nb_episodes=nb_episodes,
                    verbose=verbose,
                )
                data[iteration] = rewards

            # Store data for one set of hyperparameters
            tune_historic[param_key] = {
                "avg": list(np.mean(data, axis=0)),
                "std": list(np.std(data, axis=0)),
            }

        return tune_historic
