import numpy as np
import gymnasium as gym

class SarsaAgent:
    """
    Theoretical guarantees: https://sites.ualberta.ca/~szepesva/papers/sarsa98.ps.pdf
    """

    def __init__(
        self,
        env: gym.Env,
        encode_fct,
        nb_actions=None,
    ):
        """
        - env_name (gymnasium.Env): The environment to train on.
        - nb_actions (int): Number of possible actions.
        - encode_fct (callable): Function to encode the state into features (e.g., tile coding).
        """
        self.env = env
        self.nb_actions = env.action_space.n if nb_actions == None else nb_actions
        self.encode_fct = encode_fct
        self.q = {}

    def init(self):
        """Initialize agent and environment"""
        self.env.reset()
        self.q = {}

    def get_q_value(self, state, action):
        """Compute Q-value for a state-action pair by summing over the tiles."""
        return sum(self.q.get((tile, action), 0) for tile in self.encode_fct(state))

    def choose_action(self, state, epsilon=None, soft_policy=True):
        """
        Choose an action using an epsilon-greedy policy.

        Args:
        - q (dict): The Q-value dictionary.
        - state (hashable): The current state.
        - epsilon (float, optional): The probability of exploration (for soft policy).
        - soft_policy (bool, optional): If True, use epsilon-greedy; otherwise, choose the action with the highest Q-value.

        Returns:
        - int: The selected action.

        Raises:
        - ValueError: If soft_policy is True and epsilon is None.
        """
        if (soft_policy) and (epsilon is None):
            raise ValueError("Epsilon must be specified when soft_policy is True.")

        if (soft_policy) and (np.random.rand() < epsilon):
            return np.random.choice(self.nb_actions)  # Exploration
        return np.argmax(
            [self.get_q_value(state, action) for action in range(self.nb_actions)]
        )

    def update_epsilon(
        self, epsilon, nb_episodes, episode, min_epsilon=0.001, verbose=False
    ):
        """
        Update epsilon following a GLEI (Greedy to the Limit to Exploration Infinite)

        Args:
            - epsilon (float): exploration rate to update
            - nb_episodes (int): total number of episodes
            - nb_
        """
        if episode % (nb_episodes // 5) == 0 and episode > 0:
            if not (epsilon / 2 < min_epsilon):
                epsilon /= 2
                if verbose:
                    print(f"\nEpsilon updated to: {epsilon}\n")
        return epsilon

    def train(
        self,
        nb_episodes,
        alpha=0.1,
        gamma=0.99,
        epsilon=0.1,
        use_glei=False,
        min_epsilon=0.001,
        verbose=False,
    ):
        """
        SARSA algorithm for on-policy reinforcement learning.

        Args:
            - nb_episodes (int): Number of episodes to train for.
            - alpha (float): Learning rate for updating Q-values.
            - gamma (float): Discount factor for future rewards.
            - epsilon (float): Initial exploration rate for epsilon-greedy policy.
            - use_glei (bool): Whether to use a decaying epsilon (GLEI policy). Devide epislon by 2 every (nb_episodes // 5) episodes
            - min_epsilon (float): Minimum epsilon value in GLEI policy.
            - verbose (boolean): Print or not informations about training
        Returns:
            - rewards_historic (list): History of rewards across episodes.
        """
        # Initializations
        self.init()
        self.q["Params"] = {
            "Nb training period": nb_episodes,
            "alpha": alpha,
            "epsilon": epsilon,
            "GLEI": use_glei,
        }
        rewads_historic = []

        for episode in range(nb_episodes):
            # Decay epsilon if we use a glei learning
            if use_glei:
                epsilon = self.update_epsilon(
                    epsilon, nb_episodes, episode, min_epsilon, verbose
                )

            state, _ = self.env.reset()
            action = self.choose_action(state, epsilon=epsilon)

            task_completed, episode_over = False, False
            total_reward = 0

            while not (task_completed or episode_over):
                next_state, reward, task_completed, episode_over, _ = self.env.step(
                    action
                )
                next_action = self.choose_action(next_state, epsilon=epsilon)

                # Compute the SARSA update
                q_current = self.get_q_value(state, action)
                q_next = (
                    self.get_q_value(next_state, next_action)
                    if not (task_completed or episode_over)
                    else 0
                )
                target = reward + gamma * q_next
                error = target - q_current

                # Update Q-values for all tiles
                for tile in self.encode_fct(state):
                    self.q[(tile, action)] = (
                        self.q.get((tile, action), 0) + alpha * error
                    )

                # Move to the next state and action
                state = next_state
                action = next_action
                total_reward += reward

            rewads_historic.append(total_reward)
            if verbose:
                None
            # print(f"Episode {episode + 1}: Total Reward = {total_reward}")

        return rewads_historic

    def evaluate_policy(
        self, num_episodes=10, soft_policy=False, render_mode=None, verbose=False
    ):
        """
        Evaluate the current Q-values of the agent and print the average reward.

        Args:
            num_episodes (int): Number of episodes to evaluate the policy.
            soft_policy (bool): If True, uses an epsilon-greedy policy for action selection.
                                If False, selects the action with the highest Q-value (greedy policy).
            render_mode (str, optional): Rendering mode for the environment. For example, "human" to visualize the environment.
            verbose (bool): If True, prints detailed information about the evaluation process,
                            including training parameters and rewards for each episode.

        Returns:
            float: The average reward obtained over the evaluated episodes.
        """
        evaluate_historic = {}

        if verbose:
            print(f"Training parameters:\n", f"{self.q['Params']}")
            print(f"\nEvaluation:")
        env = gym.make("MountainCar-v0", render_mode=render_mode)
        total_rewards = []

        for episode in range(num_episodes):


            state, _ = env.reset()
            time_over = False
            done = False
            total_reward = 0

            while not (done or time_over):
                # Exécuter l'action choisie
                state, reward, done, time_over, _ = env.step(
                    self.choose_action(state, soft_policy=soft_policy)
                )
                total_reward += reward

            total_rewards.append(total_reward)
            if verbose:
                print(f"Episode {episode + 1}: Total Reward = {total_reward}")

        env.close()
        if verbose:
            print(
                f"Average Total Reward over {num_episodes} episodes: {np.mean(total_rewards)}"
            )
        return np.mean(total_rewards)

    def grid_search(
        self,
        alpha_values=[0.001, 0.1, 0.2],
        gamma_values=[0.99],
        epsilon_values=[0.01, 0.1, 0.2],
        nb_episodes=1000,
        nb_iter=10,
        use_glei=False,
        verbose=False,
    ):
        """
        Perform a grid search over hyperparameters for the SARSA agent.

        Metrics used:
            - Optimality: the best average reward on the last 10% training episodes.
            - Convergence speed: the smallest number of training episodes to reach the optimality ± 0.1%.

        Args:
            - alpha_values (list): Learning rates to test.
            - gamma_values (list): Discount factors to test.
            - epsilon_values (list): Exploration rates to test.
            - nb_episodes (int): Number of episodes per training iteration.
            - nb_iter (int): Number of iterations for averaging results.
            - moving_avg_size (int): Window size for the moving average in plotting.
            - verbose (bool): If True, print details of the training process.

        Returns:
            dict: A ranking of hyperparameter sets based on the metrics.
        """
        tune_historic = {}
        rankings = []

        # Iterate over all combinations of parameters
        for epsilon in epsilon_values:
            for alpha in alpha_values:
                for gamma in gamma_values:
                    if verbose:
                        print(
                            f"Processing:\nglei={use_glei}\nepsilon={epsilon}\nalpha={alpha}\ngamma={gamma}\n"
                        )
                    param_key = (
                        f"glei={use_glei}_epsilon={epsilon}_alpha={alpha}_gamma={gamma}"
                    )
                    data = np.empty((nb_iter, nb_episodes))
                    for iteration in range(nb_iter):
                        if verbose:
                            print(
                                f"Iteration {iteration + 1}/{nb_iter} for {param_key}"
                            )

                        rewards = self.train(
                            nb_episodes=nb_episodes,
                            alpha=alpha,
                            gamma=gamma,
                            epsilon=epsilon,
                            use_glei=use_glei,
                            verbose=verbose,
                        )
                        data[iteration] = rewards

                    tune_historic[param_key] = {
                        "avg": list(np.mean(data, axis=0)),
                        "std": list(np.std(data, axis=0)),
                    }

        return tune_historic