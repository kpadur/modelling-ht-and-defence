# %% [markdown]
# ## Modelling Hybrid Threats and Defensice Countermeasures
# Chapter 3\
# Experiment 1

# %% [markdown]
# Import libraries
from environment_exp1 import Environment
from a2c_agent import A2CRegAgent
from data_analysis import process_regagent_rewards
from save_data import read_csv_to_dict
import numpy as np
import pandas as pd
import torch
import random
import matplotlib.pyplot as plt
from IPython.display import clear_output
import os
import datetime

# %% [markdown]
# Setup device, date, chapter, and experiment
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # cpu
date = datetime.datetime.now().strftime("%Y-%m-%d")
chapter = 3
experiment = 1

# %% [markdown]
# Specify output directory
save_path = os.path.join("results", "exp1-results")
nn_path = os.path.join("regagent-parameters")

# %% [markdown]
# Specify number of agents in the environment
nProviders = 3
nRegAgents = 110
nMalAgents = 0

# %% [markdown]
# Initialise (tuned) hyperparameters
hyperparameters = read_csv_to_dict("parameters/hyperparameters.csv")
alpha_rnn1 = hyperparameters['alpha_1']
alpha_rnn2 = hyperparameters['alpha_2']
gamma_rnn1 = hyperparameters['gamma_1']
gamma_rnn2 = hyperparameters['gamma_2']
beta_start = 1
beta_end = hyperparameters['beta_1']
beta_decay = int(hyperparameters['n_1'])

# %% [markdown]
# Initialise social network, cyber-physical system, and agent parameters
parameters = read_csv_to_dict("parameters/parameters.csv")

# Social network parameters
kappa = int(parameters['kappa'])
rho = parameters['rho']

# Cyber-physical system parameters
center_up_to_down = [parameters['center_up_to_down']] * nProviders  # Psi: prob (1 to -1)
center_down_to_up = [parameters['center_down_to_up']] * nProviders  # psi: prob (-1 to 1)
end_up_to_down = [parameters['end_up_to_down']] * nProviders        # Lambda: prob (1 to -1)
end_down_to_up = [parameters['end_down_to_up']] * nProviders        # lambda: prob (-1 to 1)
cost = [parameters['cost']] * nProviders

# Agents' attributes (parameters)
direct_exp_weight = parameters['direct_exp_weight']
feedback_adj_rate = parameters['feedback_adj_rate']
forgetting_factor = parameters['forgetting_factor']

# %% [markdown]
# Define training time, visualisation and saving frequency
n_steps = 500 # number of steps per episode
number_of_episodes = 500
vis_freq = 10
saving_freq = 10
save_fig = False
save_nns = True

# %% [markdown]
# Initialise seed for reproducibility
seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# %% [markdown]
# Create environment
env = Environment(nRegAgents, nProviders, 
                  kappa, rho, center_up_to_down, center_down_to_up, end_up_to_down, end_down_to_up, cost,
                  direct_exp_weight, feedback_adj_rate, forgetting_factor)

# Create lists of agents, form social network of agents, and create attributes
regagents, providers = env.regagents, env.providers

# Pick one agent randomly from each type of agents
regagent_example = np.random.choice(regagents)

# Define regular agents' state space, actions, and opinions
observation_space, action_space = env.observation_spaces[f"regagent{regagent_example}"], env.action_spaces[f"regagent{regagent_example}"]

state_shape, n_actions, n_opinions = env.observation_spaces[f"regagent{regagent_example}"].shape[0], \
    env.action_spaces[f"regagent{regagent_example}"][0].n, env.action_spaces[f"regagent{regagent_example}"][1].n

# Reset environment state
envstate, info = env.reset(seed=seed)

print("Regular agents' state shape is", state_shape, ", number of actions is", n_actions, " and number of opinions is", n_opinions)

# Visualise graph
fig = env.render(graph_type='actions')
plt.show()

# %% [markdown]
# Create regular agents
regular_agents = {f"regagent{agent}": A2CRegAgent(state_shape, n_actions, n_opinions, alpha_rnn1, alpha_rnn2, device) 
                  for agent in env.regagents}
# %% [markdown]
# Collect data
total_rewards = np.zeros(number_of_episodes + 1)
action_rewards, opinion_rewards = np.zeros(number_of_episodes + 1), np.zeros(number_of_episodes + 1)

loss1_agents, loss2_agents = np.zeros((number_of_episodes + 1, len(env.regagents))), np.zeros((number_of_episodes + 1, len(env.regagents)))
loss1_history, loss2_history = np.zeros(number_of_episodes + 1), np.zeros(number_of_episodes + 1)

# %% [markdown]
# Train, evaluate, and visualise the agent's behaviour
for episode in range(1, number_of_episodes + 1):
    print(f"Episode {episode} of {number_of_episodes} episodes")

    # Decay entropy coefficient for new episode
    entropy_coef = beta_start + (beta_end - beta_start) * min(episode, beta_decay) / beta_decay

    # Restart environment
    observations, _ = env.reset(seed=seed)

    # Initialise dictionaries to store rewards, observations, and actions in one episode
    all_observations = {agent_name: [observations[agent_name]] for agent_name in regular_agents.keys()}
    all_actions = {agent_name: [] for agent_name in regular_agents.keys()}
    episode_rewards = {agent_name: [] for agent_name in regular_agents.keys()}

    # Collect data for one episode (n_steps)
    for timestep in range(1, n_steps+1): # one episode

        actions = {}  # Dictionary to store the current timestep's actions
        
        for agent_name, agent in regular_agents.items():
            state = observations[agent_name]
            # Sample action and opinion
            action, opinion = agent.sample_actions(state)
            actions[agent_name] = (action, opinion) # save action and opinion

        # Perform actions, determine next state, reward, and termination
        observations, rewards, terminations, truncations, infos = env.step(actions)

        # Render environment
        if episode in [1,100,500] and timestep == 500:
            clear_output(wait=True)  # Clear the previous output
            fig = env.render(graph_type='both')  # Render the graph for the current timestep
            fig.text(0.01, 0.90, f'Episode: {episode}', ha='left', fontsize=14, color='black')
            fig.text(0.01, 0.86, f'Timestep: {timestep}', ha='left', fontsize=14, color='black')
            plt.show()
            if save_fig:
                fig.savefig(os.path.join(save_path, f'ch{chapter}-exp{experiment}-{date}-{seed}-{episode}-environment.png'))
                plt.clf()

        # Store the rewards, observations, and actions for each agent for the current timestep
        for agent_name in episode_rewards.keys():
            episode_rewards[agent_name].append(rewards[agent_name])
            all_observations[agent_name].append(observations[agent_name])
            all_actions[agent_name].append(actions[agent_name])

    # Compute A2C loss (and backpropagate)
    for agent_name, agent in regular_agents.items():
        loss1, loss2 = agent.compute_a2c_loss(all_observations[agent_name][:-1], all_actions[agent_name], episode_rewards[agent_name], gamma_rnn1, gamma_rnn2, entropy_coef)
        agent_id = int(agent_name.replace("regagent", ""))
        loss1_agents[episode][agent_id] = loss1.data.cpu().item()
        loss2_agents[episode][agent_id] = loss2.data.cpu().item()

    # Process regagent rewards
    sum_regagent_rewards, service, feedback = process_regagent_rewards(episode_rewards)
    total_rewards[episode] = sum_regagent_rewards
    action_rewards[episode] = service
    opinion_rewards[episode] = feedback

    # Process regagents loss
    loss1_history[episode] = np.mean(loss1_agents[episode])
    loss2_history[episode] = np.mean(loss2_agents[episode])
        
    # Visualise data
    if episode != 1 and episode % vis_freq == 0:
        clear_output(True)
        # Plot 1: Visualise cumulative reward
        plt.figure()
        plt.plot(total_rewards[1:episode], linewidth=0.9, color = 'mediumvioletred', label = "Cumulative reward")
        plt.plot(action_rewards[1:episode], linewidth=0.9, color = 'red', label = "Service reward")
        plt.plot(opinion_rewards[1:episode], linewidth=0.9, color = 'orange', label = "Feedback reward")
        plt.xlabel("Episode")
        plt.ylabel("Cumulative reward\nfor regular agents per episode")
        plt.grid()
        plt.legend(loc=(0.01,0.50), fontsize='x-small')
        if save_fig:
            plt.savefig(os.path.join(save_path, f'ch{chapter}-exp{experiment}-{date}-{seed}-score.png'))
            plt.clf()

        # Plot 2.1/2.2: Visualise loss/objective value per episode
        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(15, 4))
        axs[0].plot(loss1_history[1:episode], linewidth=0.9, color = 'red')
        axs[0].set_xlabel("Episode")
        axs[0].set_ylabel("Loss (actions)")
        axs[0].grid()
        axs[1].plot(loss2_history[1:episode], linewidth=0.9, color = 'orange')
        axs[1].set_xlabel("Episode")
        axs[1].set_ylabel("Loss (opinions)")
        axs[1].grid()
        if save_fig:
            plt.savefig(os.path.join(save_path, f'ch{chapter}-exp{experiment}-{date}-{seed}-loss.png'))
            plt.clf()
        plt.show()

# %% [markdown]
# Save data
regular_agent_reward_loss_df = pd.DataFrame({'total_rewards': total_rewards, 
                                             'action_rewards': action_rewards, 
                                             'opinion_rewards': opinion_rewards,
                                             'mean_loss1': loss1_history, 
                                             'mean_loss2': loss2_history})
# Save dataframes to csv
regular_agent_reward_loss_df.to_csv(os.path.join(output_dir, f'ch{chapter}-exp{experiment}-{date}-{seed}-regagents-data.csv'), index = False)

# %% [markdown]
# Save models
if save_nns:
    os.makedirs(nn_path, exist_ok=True)  # Ensure the folder exists

    for agent_name, agent in regular_agents.items():
        torch.save({
            'actions_state_dict': agent.action_nn.state_dict(),
            'actions_opt_state_dict': agent.action_opt.state_dict(),
        }, os.path.join(nn_path, f'{agent_name}_checkpoint_actions.pth'))

        torch.save({
            'opinions_state_dict': agent.opinion_nn.state_dict(),
            'opinions_opt_state_dict': agent.opinion_opt.state_dict(),
        }, os.path.join(nn_path, f'{agent_name}_checkpoint_opinions.pth'))
