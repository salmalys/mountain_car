import gym 
import numpy as np


iht_size = 2024
num_tilings = 8
num_tiles = 8

class SemiGradientSarsa:
    """
    Theoretical guarantees : ? (à compléter)
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
        self.w = np.random.uniform(low=-0.05, high=0.05, size=(iht_size,))

    def init(self):
        """Initialize agent and environment"""
        self.env.reset()
        self.w = np.random.uniform(low=-0.05, high=0.05, size=(iht_size,))

    def get_q_value(self, state, action):
        """Compute Q-value for a state-action pair by summing over the tiles."""
        # Encode the state and action into features
        tiles = self.encode_fct(state)

        # Compute the Q-value by summing over the tiles
        q_value = sum(self.w[tile] for tile in tiles)

        # print(f"Q-value for state {state}, action {action}, Encoded Tiles: {tiles}: {q_value}")
        return q_value
    
    def choose_action(self, state, epsilon=None, soft_policy=True):
        """
        Choose an action using an epsilon-greedy policy.

        Args:
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
            # Randomly choose an action with probability epsilon
            action = np.random.choice(self.nb_actions)

            # print(f"Random action chosen: {action} for state {state}")
            return action
        
        # Choose the action with the highest Q-value
        q_values = [self.get_q_value(state, action) for action in range(self.nb_actions)]
        best_action = np.argmax(q_values)

        # print(f"Best action chosen: {best_action} for state {state}")        
        return best_action
    
    
    def train(
        self,
        nb_episodes,
        alpha=0.1,
        gamma=0.99,
        epsilon=0.1,
        use_glei=False,
        min_epsilon=0.005,
        verbose=False,
    ):
        """
        Semi-gradient SARSA algorithm for on-policy reinforcement learning.

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
        rewards_historic = []
        epsilon

        for episode in range(nb_episodes):
            
            # Decay epsilon if we use a glei learning
            if (use_glei) and (episode % (nb_episodes // 5) == 0 and episode > 0):
                if not (epsilon / 2 < min_epsilon):
                    epsilon /= 2
                    if verbose:
                        print(f"\nEpsilon updated to: {epsilon}\n")

            state, _ = self.env.reset()
            action = self.choose_action(state, epsilon=epsilon)

            task_completed, episode_over = False, False
            total_reward = 0

            while not (task_completed or episode_over):
                next_state, reward, task_completed, episode_over, _ = self.env.step(action)
                next_action = self.choose_action(next_state, epsilon=epsilon)

                # Compute the Semi-Gradient SARSA update
                q_current = self.get_q_value(state, action)
                q_next = self.get_q_value(next_state, next_action)

                terminal = task_completed or episode_over

                if terminal:
                    error = reward - q_current
                else : 
                    error = reward + gamma * q_next - q_current

                # print(f"State: {state}, Action: {action}, Reward: {reward}")
                # print(f"Q_current: {q_current}, Q_next: {q_next}, Error: {error}")

                # Update Q-values for all tiles
                for tile in self.encode_fct(state):
                    self.w[tile] += alpha * error
                    # print(f"Tile: {tile}, Updated Weight: {self.w[tile]}")

                # Move to the next state and action
                state = next_state
                action = next_action
                total_reward += reward

            rewards_historic.append(total_reward)
            if verbose:
                print(f"Episode {episode + 1}: Total Reward = {total_reward}")

        return rewards_historic
    
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
        if verbose:
            print(f"Training parameters:\n", f"{self.w}")
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
        print(
            f"Average Total Reward over {num_episodes} episodes: {np.mean(total_rewards)}"
        )

        return np.mean(total_rewards)

   