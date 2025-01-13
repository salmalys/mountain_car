import numpy as np
import gymnasium as gym

from agents.td_agent import TDAgent


class SemiGradientSarsa(TDAgent):
    def __init__(self, encode_fct: callable, nb_actions: int, weights_dim: tuple):
        """
        - env_name (gymnasium.Env): The environment to train on.
        - nb_actions (int): Number of possible actions.
        - encode_fct (callable): Function to encode the state into features (e.g., tile coding).
        """
        self.nb_actions = nb_actions
        self.encode_fct = encode_fct

        # Define weights (one per (state, action))
        # self.weights = np.random.uniform(-0.1, 0.1, size=(env.action_space.n, iht_size))
        self.weights_dim = weights_dim
        self.weights = np.random.uniform(-0.1, 0.1, size=weights_dim)

    def reset(self):
        # self.weights = np.random.uniform(-0.1, 0.1, size=(env.action_space.n, iht_size))
        self.weights = np.random.uniform(-0.1, 0.1, size=self.weights_dim)

    def q_value(self, state, action):

        encoded_state = self.encode_fct(state)
        return np.sum(self.weights[action][encoded_state])

    def choose_action(self, state, epsilon=None, exploration=True):
        if (exploration) and (epsilon is None):
            raise ValueError("Epsilon must be specified when exploration is True.")

        if (exploration) and (np.random.rand() < epsilon):
            return np.random.choice(self.nb_actions)
        return np.argmax([self.q_value(state, a) for a in range(self.nb_actions)])

    def update_weights(
        self, state, action, reward, next_state, next_action, alpha, gamma
    ):
        """
        Perform the update of the weights.

        Args:
            - state (hashable): The current state
            - action (int): The chosen action
            - reward (float): The reward for The given (state, action)
            - next_state (hashable): The t+1 state
            - next_action (int): The t+1 action
            - alpha (float): The learning rate
            - gamma (float): The discount factor
        """
        enocoded_state = self.encode_fct(state)
        next_q_value = self.q_value(next_state, next_action)
        target = reward + gamma * next_q_value
        td_error = target - self.q_value(state, action)

        # Update weights for the selected action and
        for encoded_state in enocoded_state:
            self.weights[action][encoded_state] += alpha * td_error

    def train(
        self,
        env: gym.Env,
        nb_episodes,
        max_step=None,
        alpha=0.1,
        gamma=0.99,
        epsilon=0.1,
        use_glei=False,
        min_epsilon=0.001,
        verbose=0,
    ):
        rewards_per_episode = []

        step = 0
        for episode in range(nb_episodes):
            # Decay epsilon if we use a glei learning
            if use_glei:
                epsilon = super().update_epsilon(
                    epsilon, nb_episodes, episode, min_epsilon, verbose
                )

            state, _ = env.reset()
            action = self.choose_action(state=state, epsilon=epsilon)
            step += 1
            total_reward = 0

            done = False
            time_over = False
            while not (done or time_over):
                next_state, reward, done, time_over, _ = env.step(action)
                next_action = self.choose_action(state=next_state, epsilon=epsilon)
                step += 1
                self.update_weights(
                    state, action, reward, next_state, next_action, alpha, gamma
                )

                state = next_state
                action = next_action
                total_reward += reward

                # Case of environment with no episodes
                if max_step is not None:
                    if step > max_step:
                        return total_reward
                    epsilon = super().update_epsilon(
                        epsilon, max_step, step, min_epsilon, verbose
                    )

            rewards_per_episode.append(total_reward)
            if verbose == 1:
                print(
                    f"Episode {episode + 1}/{nb_episodes}, Total Reward: {total_reward}"
                )

        return rewards_per_episode
