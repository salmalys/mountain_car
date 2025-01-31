import matplotlib.pyplot as plt
import numpy as np
import json


def line_bar_plot(
    lineplot_window_size,
    bar_chart_width=400,
    traning_data=None,
    evaluation_data=None,
    plot_name="Moving average of training rewards with evaluations during training",
):
    """
    Plots moving averages of line plots and a bar chart with two bars at specific x-axis values.

    Args:
        lineplot_window_size (int): Window size for the moving average.
        traning_data (dict): Dictionary of data for line plots, where keys are labels and values are arrays.
        evaluation_data (dict): Dictionary of data for bar charts, where keys are labels and values are dictionaries with 'x' and 'data'.
    """
    # Plot the data
    plt.subplots(figsize=(12, 8))

    # Plot line plots with moving averages
    moving_avgs = []  # Store moving averages to define axis limit later
    if traning_data is not None:

        for name, data in traning_data.items():
            moving_avg = np.convolve(
                data, np.ones(lineplot_window_size) / lineplot_window_size, mode="valid"
            )
            moving_avgs.append(moving_avg)
            plt.plot(moving_avg, label=f"Moving Average of {name}", linewidth=2)

    # Plot the bar chart
    bar_values = []
    if evaluation_data is not None:
        n_bars = len(evaluation_data)  # Number of bars at each x position
        bar_width = (
            bar_chart_width / n_bars
        )  # Adjust the bar width to make them more visible
        for i, (name, data) in enumerate(evaluation_data.items()):
            bar_x = np.array(data["x"])
            bar_y = np.array(data["data"])  # Keep the original y-values
            bar_values.append(bar_y)
            offset = (i - (n_bars - 1) / 2) * bar_width  # Center bars around bar_x
            plt.bar(
                bar_x + offset,
                bar_y - np.min(moving_avgs) if len(moving_avg != 0) else bar_y,
                width=bar_width,
                alpha=0.5,
                label=f"{name}",
                bottom=np.min(moving_avgs),  # Set the bottom of the bars to 0
            )

    min_y = np.min(moving_avg)

    max_y = (
        max(np.max(bar_values), np.max(moving_avg)) + 2
        if len(bar_values) != 0
        else np.max(moving_avg) + 5
    )
    plt.ylim(min_y, max_y)

    # Add title
    plt.title(plot_name)
    plt.xlabel("Episodes")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_moving_averages(rewards_dict, window_size):
    """
    Plots the moving averages and standard deviation of rewards for multiple reward histories
    on the same graph, with named histories provided in a dictionary.

    Args:
        - rewards_dict (dict): A dictionary where keys are parameter combinations
          and values are {"avg": list, "std": list}
        - window_size (int): The size of the moving average window.

    Returns:
        None
    """

    plt.figure(figsize=(15, 10))
    moving_avgs = []  # Store moving averages to define axis limit later
    for param_key, data in rewards_dict.items():
        # Calculate the moving average using np.convolve
        moving_avg_rewards = np.convolve(
            data["avg"], np.ones(window_size) / window_size, mode="valid"
        )
        moving_avgs.append(moving_avg_rewards)

        # Calculate the standard deviation for the same window
        std_rewards = [
            np.std(data["avg"][i : i + window_size])
            for i in range(len(data["avg"]) - window_size + 1)
        ]

        x_range = range(len(moving_avg_rewards))

        # Plot the moving average
        plt.plot(
            x_range,
            moving_avg_rewards,
            label=param_key,
        )

        # Plot the shaded standard deviation area
        plt.fill_between(
            x_range,
            moving_avg_rewards - std_rewards,
            moving_avg_rewards + std_rewards,
            alpha=0.2,  # Transparency for the shaded area
        )

    plt.ylim(np.min(moving_avgs) - 50, np.max(moving_avgs) + 50)

    plt.xlabel("Number of Episodes")
    plt.ylabel("Moving Average of Rewards with Standard Deviation")
    plt.title(
        f"Comparison of {window_size}-Episode Moving Averages with Standard Deviations"
    )
    plt.legend()
    plt.grid()
    plt.show()


def process_json(file_name, mode, data_dict=None):
    """
    Handles saving a dictionary to a JSON file or loading a JSON file into a dictionary.

    Args:
        file_name (str): The name of the JSON file.
        mode (str): The mode of operation, either "w" or "r".
        data_dict (dict, optional): The dictionary to save when in "save" mode.

    Returns:
        dict: The loaded dictionary if mode is "load".
        None: If mode is "save".
    """
    if mode == "w":
        if data_dict is None:
            raise ValueError("data_dict must be provided in 'save' mode.")
        with open(f"{file_name}.json", "w") as json_file:
            json.dump(data_dict, json_file, indent=4)
        print(f"Dictionary saved to {file_name}")
    elif mode == "r":
        with open(f"{file_name}.json", "r") as json_file:
            return json.load(json_file)
    else:
        raise ValueError("Invalid mode. Use 'w' or 'r'.")


def compare_with_moving_averages(
    agent1_dict, agent2_dict, num_episodes=1000, window_size=100, verbose=False
):
    """
    Compare two agents using moving averages and standard deviation.

    Args:
        agent1_dict (dict): The first agent instance.
        agent2_dict (dict): The second agent instance.
        env (gymanisum.Env): The Gym environment.
        num_episodes (int): Number of episodes to evaluate each agent.
        window_size (int): The size of the moving average window for the plot.

    Returns:
        None
    """
    agent1_name, agent1 = agent1_dict["name"], agent1_dict["agent"]
    agent2_name, agent2 = agent2_dict["name"], agent2_dict["agent"]

    if verbose:
        print(f"Evaluating {agent1_name} agent...")
    agent1_rewards = [
        agent1.evaluate_policy(
            num_episodes=1, soft_policy=False, render_mode=None, verbose=False
        )
        for _ in range(num_episodes)
    ]

    if verbose:
        print(f"Evaluating {agent2_name} agent...")
    agent2_rewards = [
        agent2.evaluate_policy(
            num_episodes=1, soft_policy=False, render_mode=None, verbose=False
        )
        for _ in range(num_episodes)
    ]

    # Prepare data for plot_moving_averages
    rewards_dict = {
        agent1_name: {"avg": agent1_rewards},
        agent2_name: {"avg": agent2_rewards},
    }

    # Use the plot_moving_averages function to visualize the results
    plot_moving_averages(
        rewards_dict, nb_episodes=num_episodes, window_size=window_size
    )


def compare_agents(agent1_dict, agent2_dict, num_episodes=1000, verbose=False):
    """
    Compare two agents

    Args:
        agent1_dict (dict): The first agent instance.
        agent2_dict (dict): The second agent instance.
        env (gymanisum.Env): The Gym environment.
        num_episodes (int): Number of episodes to evaluate each agent.

    Returns:
        None
    """
    agent1_name, agent1 = agent1_dict["name"], agent1_dict["agent"]
    agent2_name, agent2 = agent2_dict["name"], agent2_dict["agent"]

    if verbose:
        print(f"Evaluation for {agent1_name} agent ...")
    agent1_rewards = [
        agent1.evaluate_policy(
            num_episodes=10, soft_policy=False, render_mode=None, verbose=False
        )
        for _ in range(num_episodes)
    ]

    if verbose:
        print(f"Evaluating {agent2_name} agent ...\n")
    agent2_rewards = [
        agent2.evaluate_policy(
            num_episodes=1, soft_policy=False, render_mode=None, verbose=False
        )
        for _ in range(num_episodes)
    ]

    # Print results
    means = [np.mean(agent1_rewards), np.mean(agent2_rewards)]
    stds = [np.std(agent1_rewards), np.std(agent2_rewards)]

    if verbose:
        print(f"Average reward for {agent1_name}: {means[0]}")
        print(f"Average reward for {agent2_name}: {means[1]}\n")

        print(f"Standard deviation for {agent1_name}: {stds[0]}")
        print(f"Standard deviation for {agent2_name}: {stds[1]}\n")

    return means, stds
