import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym

from agents.td_agent import TDAgent
from policies.policy import Policy


class SimpleDenseModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1):
        super().__init__()
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.hidden_layer = nn.Linear(hidden_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.input_layer(x)
        x = self.relu(x)
        x = self.hidden_layer(x)
        x = self.relu(x)
        return self.output_layer(x)


class NeuralNetworkSarsa(TDAgent):
    def __init__(
        self,
        nb_actions: int,
        encode_fct: callable,
        policy: Policy,
        model: nn.Module,
        optimizer: optim,
        loss: nn.Module = nn.MSELoss(),
    ):
        """
        - encode_fct (callable): Function to encode the state into features (e.g., tile coding).
            - Args:
                - state: a gymnasium state
                    * For example, what is returned by gym.Env.step()[0].
                    * See https://gymnasium.farama.org/api/env/ for more information
            - Returns:
                - (list(int)): a list of integers representing a state. Can be a list of just one integer.

        - policy (callable): Function to choose an action for a state.
            - Args:
                - q_values (list): The list of q-values for a state
                - **kwargs: parameters of the policy
            -Returns
                - (int): an action

        - nb_actions (int): Number of possible actions.
        """
        self.nb_actions = nb_actions
        self.encode_fct = encode_fct
        self.policy = policy
        self.model = model
        self.optimizer = optimizer(model.parameters())
        self.loss = loss
        self.qvalues_batch = list()
        self.target_batch = list()
        self.reward_batch = list()
        self.gamma_batch = list()

    def reset(self):
        self.q = {}

    def q_value(self, state, action, to_tensor=False):
        q_value = self.model(torch.tensor([*state, action]))
        return q_value.item() if not to_tensor else q_value

    def choose_action(self, state, **kwargs):
        # Building a batch of input for each possible action in the environment
        # The input has a dimension of (self.nb_actions, nb_states_features + nb_actions_features)
        q_values = [self.q_value(state, action) for action in range(self.nb_actions)]
        return self.policy.choose_action(q_values, **kwargs)

    def collect_data(
        self,
        state,
        action,
        reward,
        next_state,
        next_action,
        gamma,
        is_final,
    ):
        """
        Add current transition q-values to the current batch of data for the next update of the network
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
        """"""
        # # Compute q-values for the given transition
        # q_current = self.q_value(state, action)
        # q_next = self.q_value(next_state, next_action)
        # y_i = reward + gamma * q_next if not is_final else reward

        # # Stacking training tensors and target tensors
        # self.qvalues_batch.append([*state, action])
        # self.target_batch.append([*next_state, next_action])
        # self.gamma_batch.append([gamma])
        # self.reward_batch.append([reward if not is_final else 0])

        # Compute q-values for the given transition
        q_current = self.q_value(state, action, to_tensor=True)
        q_next = self.q_value(next_state, next_action, to_tensor=True)
        y_i = reward + gamma * q_next if not is_final else reward + 0 * q_next

        # Stacking training tensors and target tensors
        self.qvalues_batch.append(q_current)
        self.target_batch.append(y_i)

    def update_parameters(
        self,
        alpha,
    ):
        # Initializations
        self.optimizer.zero_grad()
        self.optimizer.lr = alpha
        predicted_batch = torch.stack(self.qvalues_batch)
        target_batch = torch.stack(self.target_batch)

        # Forward pass
        # predicted_qvalues = self.model(input_batch)

        # Backward pass
        loss = self.loss(predicted_batch, target_batch)  # Compute the loss
        loss.backward()  # Compute the gradients
        self.optimizer.step()  # Update parameters

        # Clear batches
        self.qvalues_batch = list()
        self.target_batch = list()

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
        network_update_frequency: int = None,
        to_evaluate: bool = False,
        evaluation_params: dict = None,
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
            - network_update_frequency (int): Number of action before updating the network weights. If None, update every end of episode
            - to_evaluate (bool): If True, proceed to agent evaluation through training.
            - evaluation_params (dict): If to_evaluate = True, is the policy_action_params for evaluation
            - evaluation_frequency (int): evaluate every nb_episodes // evaluation frequency.
            - verbose (int): Verbosity level for debugging (0: silent, 1: general informations, 2: Precise informations).

        Returns:
            - rewards_historic (list): History of rewards across episodes.
        """
        # if nb_episodes > 1 and max_step is not None:
        #     raise Warning("You gave both nb_episodes and max step")

        # Initializations
        self.reset()
        rewards_per_episode = []
        evaluations = {"x": [], "data": []}
        step = 0

        for episode in range(nb_episodes):
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

                # Build batch of qvalues
                self.collect_data(
                    state=state,
                    action=action,
                    reward=reward,
                    next_state=next_state,
                    next_action=next_action,
                    gamma=gamma,
                    is_final=(task_completed or episode_over),
                )

                # Update network weights
                if (
                    network_update_frequency is not None
                    and step % network_update_frequency == 0
                    and step > 0
                ) or (task_completed or episode_over):
                    self.update_parameters(alpha)

                # Potentially update the policy parameters
                if (max_step is not None and nb_episodes == 1) or (
                    task_completed or episode_over
                ):
                    max_stage = max_step if max_step is not None else nb_episodes
                    curr_stage = step if max_step is not None else episode
                    self.policy.update(
                        max_step=max_stage,
                        curr_step=curr_stage,
                        verbose=verbose,
                        **policy_update_params,
                    )
                    # Potentially evaluate the agent at this point.
                    if to_evaluate:
                        evaluations = self.evaluate_through_training(
                            env=env,
                            with_step=True,
                            max_step=max_stage,
                            curr_step=curr_stage,
                            evaluation_params=evaluation_params,
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
