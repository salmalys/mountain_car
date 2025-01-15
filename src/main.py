from src.agents.sarsa import Sarsa
from agents.qlearning import QLearning
from utils.encoding import mc_tile_encoding
import gymnasium as gym
import argparse


def execute_agent(agent, use_glei, render_mode):
    """
    Main function to train and evaluate a reinforcement learning agent on MountainCar.

    Args:
        - agent (str): name of the agent to use.
        - use_glei (bool):, flag to use GLEI strategy during training.
        - render_mode (str):, render mode for evaluation.

    Returns:
        - None
    """
    # Initialize the environment
    env = gym.make("MountainCar-v0")

    # Set up the agent
    if agent == "Sarsa":
        agent = Sarsa(env=env, encode_fct=mc_tile_encoding)

    elif agent == "QLearning":
        agent = QLearning(env=env, encode_fct=mc_tile_encoding)
    else:
        raise ValueError(f"Unsupported agent: {agent}")

    # Train the agent
    _ = agent.train(
        nb_episodes=1000, alpha=0.1, gamma=0.99, epsilon=0.001, use_glei=use_glei
    )

    # Evaluate the q values
    agent.evaluate_policy(verbose=True, render_mode=render_mode)


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Train and evaluate a reinforcement learning agent on MountainCar."
    )

    parser.add_argument(
        "--agent",
        type=str,
        default="SarsaAgent",
        help="Name of the agent to use (default: SarsaAgent).",
    )
    parser.add_argument(
        "--use_glei",
        action="store_true",
        help="Flag to use GLEI strategy during training.",
    )
    parser.add_argument(
        "--render_mode",
        type=str,
        default=None,
        choices=["human", None],
        help="Render mode for evaluation (default: None).",
    )

    args = parser.parse_args()

    # Run main with provided arguments
    execute_agent(
        agent=args.agent, use_glei=args.use_glei, render_mode=args.render_mode
    )
