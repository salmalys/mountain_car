from abc import ABC, abstractmethod
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

    def update_epsilon(
        self,
        epsilon: float,
        nb_episodes: int,
        episode: int,
        min_epsilon: float = 0.001,
        verbose: int = 0,
    ):
        """
        Update epsilon following a GLEI (Greedy to the Limit to Exploration Infinite)

        Args:
            - epsilon (float): exploration rate to update
            - nb_episodes (int): total number of episodes
            - episode (int): the current episode
            - min_epsilon (float): the minimum decayed epsilon
            - verbose (int): 0 or 1 if we want to print informations

        Returns:
            - (float): the updated epsilon
        """
        if episode % (nb_episodes // 5) == 0 and episode > 0:
            if not (epsilon / 2 < min_epsilon):
                epsilon /= 2
                if verbose == 1:
                    print(f"\nEpsilon updated to: {epsilon}\n")
        return epsilon

    def train(
        self,
        env: gym.Env,
        nb_episodes: int = 1000,
        max_step: int = None,
        alpha: float = 0.1,
        gamma: float = 0.99,
        use_glei: bool = False,
        min_epsilon: float = 0.001,
        verbose: bool = False,
        **kwargs,
    ):
        """
        Training algorithm of the agent

        Args:
            - env (gymnasium.Env): the gymnasium environment to train on
            - nb_episodes (int): Number of episodes to train for.
            - max_step (int): The maximum number of steps for environments with no episode,
            - alpha (float): Learning rate for updating Q-values.
            - gamma (float): Discount factor for future rewards.
            - use_glei (bool): Whether to use a decaying epsilon (GLEI policy). Devide epislon by 2 every (nb_episodes // 5) episodes
            - min_epsilon (float): Minimum epsilon value in GLEI policy.
            - verbose (boolean): Print or not informations about training
            - **kwargs: optional parameters for policy. For example, epsilon for epsilong-greedy policy

        Returns:
            - rewards_historic (list): History of rewards across episodes.
        """
        # Initializations
        epsilon = kwargs.get("epsilon", None)  # Get epsilon if given as parameter
        self.reset()
        rewards_per_episode = []
        step = 0

        for episode in range(nb_episodes):
            # Decay epsilon if we use a glei learning
            if use_glei:
                epsilon = self.update_epsilon(
                    epsilon, nb_episodes, episode, min_epsilon, verbose
                )

            # Initialize a new episode
            state, _ = env.reset()
            action = self.choose_action(state, **kwargs)
            episode_reward = 0
            task_completed, episode_over = False, False

            # Simulate an episode
            while not (task_completed or episode_over):
                # Compute next state informations
                next_state, reward, task_completed, episode_over, _ = env.step(action)
                next_action = self.choose_action(next_state, **kwargs)
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

                # Case of environment with no episodes
                if max_step is not None:
                    if step > max_step:
                        return rewards_per_episode + [episode_reward]
                    epsilon = self.update_epsilon(
                        epsilon, max_step, step, min_epsilon, verbose
                    )

                # Move to the next state and action
                state = next_state
                action = next_action
                episode_reward += reward

            rewards_per_episode.append(episode_reward)
            if verbose == 1:
                print(f"Episode {episode + 1}: Total Reward = {episode_reward}")

        return rewards_per_episode

    def evaluate_policy(
        self,
        env: gym.Env,
        nb_episodes: int = 10,
        max_step: int = None,
        verbose: int = 0,
        **kwargs,
    ):
        """
        Evaluate the current Q-values of the agent and print the average reward.

        Args:
            - env (gymnasium.Env): the gymnasium environment evaluate the agent on.
            - nb_episodes (int): Number of episodes to evaluate the policy.
            - max_step (int): maximum step for training.
            - verbose (bool): If True, prints detailed information about the evaluation process,
                            including training parameters and rewards for each episode.

        Returns:
            float: The average reward obtained over the evaluated episodes.
            OR
            list: When no episodes, the list of rewards for each step

        """
        # Initializations
        rewards_over_episodes, step = [], 0

        for episode in range(nb_episodes):
            state, _ = env.reset()
            task_completed, episode_over, episode_reward = False, False, 0

            while not (task_completed or episode_over):
                state, reward, task_completed, episode_over, _ = env.step(
                    self.choose_action(state, **kwargs)
                )
                step += 1
                episode_reward += reward

                if (max_step is not None) and (step > max_step):
                    if episode == 0:
                        return episode_reward
                    else:
                        return np.mean(rewards_over_episodes + [episode_reward])

            rewards_over_episodes.append(episode_reward)
            if verbose == 1:
                print(f"Episode {episode + 1}: Total Reward = {episode_reward}")

        if verbose == 1:
            print(
                f"Average Total Reward over {nb_episodes} episodes: {np.mean(rewards_over_episodes)}"
            )

        return np.mean(rewards_over_episodes)

    def grid_search(
        self,
        env: gym.Env,
        alpha_values=[0.001, 0.1, 0.2],
        gamma_values=[0.99],
        epsilon_values=None,
        nb_episodes=1000,
        nb_iter=10,
        use_glei=False,
        verbose=0,
    ):
        """
        Perform a grid search over hyperparameters.

        Args:
            - alpha_values (list): Learning rates to test.
            - gamma_values (list): Discount factors to test.
            - epsilon_values (list): Exploration rates to test.
            - nb_episodes (int): Number of episodes per training iteration.
            - nb_iter (int): Number of iterations for averaging results.
            - moving_avg_size (int): Window size for the moving average in plotting.
            - verbose (bool): If True, print details of the training process.
            - **kwargs (list):
        Returns:
            dict: A ranking of hyperparameter sets based on the metrics.
        """
        tune_historic = {}

        # Iterate over all combinations of parameters
        for epsilon in epsilon_values:
            for alpha in alpha_values:
                for gamma in gamma_values:
                    param_key = (
                        f"glei={use_glei}_epsilon={epsilon}_alpha={alpha}_gamma={gamma}"
                    )
                    if verbose == 1:
                        print(f"Processing:\n{param_key}\n")
                    data = np.empty((nb_iter, nb_episodes))
                    for iteration in range(nb_iter):
                        if verbose == 1:
                            print(
                                f"Iteration {iteration + 1}/{nb_iter} for {param_key}"
                            )
                        self.reset()
                        rewards = self.train(
                            env=env,
                            nb_episodes=nb_episodes,
                            alpha=alpha,
                            gamma=gamma,
                            use_glei=use_glei,
                            verbose=verbose,
                            epsilon=epsilon,
                        )
                        data[iteration] = rewards

                    tune_historic[param_key] = {
                        "avg": list(np.mean(data, axis=0)),
                        "std": list(np.std(data, axis=0)),
                    }
        return tune_historic
