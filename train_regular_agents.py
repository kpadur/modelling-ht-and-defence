# %% [markdown]
# ## Modelling Hybrid Threats and Defensice Countermeasures
# Chapter 3\
# Experiment 1

# %%
from environment import Environment
from a2c_agent import A2CRegAgent
from data_analysis import process_regagent_rewards
from other_functions import scheduler
from other_functions import smoothen
import numpy as np
import pandas as pd
import torch
import random
import matplotlib.pyplot as plt
from IPython.display import clear_output
import os
import datetime

# %%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # should be cpu
date = datetime.datetime.now().strftime("%Y-%m-%d")
timestamp = datetime.datetime.now().strftime("%H-%M-%S")
chapter = 3
experiment = 1

# %%
# Specify output directory
output_dir = os.path.join("results")
save_path = os.path.join("results", "model_output")

# %%
# Initialise (tuned) hyperparameters from file
df = pd.read_csv("hyperparameters.csv")
hyperparameters = dict(zip(df['hyperparameter'], df['value']))

alpha_rnn1 = hyperparameters['alpha_1']
alpha_rnn2 = hyperparameters['alpha_2']
gamma_rnn1 = hyperparameters['gamma_1']
gamma_rnn2 = hyperparameters['gamma_2']
beta_start = 1
beta_end = hyperparameters['beta_1']
beta_decay = int(hyperparameters['n_1'])

# %%
# Define training time
n_steps = 500 # number of steps per episode
number_of_episodes = 500
vis_freq = 10

# %%
# CPS setup values
nProviders = 3
center_up_to_down = [0.01,0.01,0.01] # Psi: prob (1 to -1)
center_down_to_up = [0.25,0.25,0.25] # psi: prob (-1 to 1)
end_up_to_down = [0.03,0.03,0.03] # Lambda: prob (1 to -1)
end_down_to_up = [0.30,0.30,0.30] # lambda: prob (-1 to 1)
cost = [0.5,0.5,0.5]

# SN setup values
nRegAgents = 110
nMalAgents = 0
kappa = 6; rho = 0.05

# Agents' attributes (parameters)
direct_exp_weight = 0.9
satisfaction_threshold = 0.5
forgetting_factor = 0.9

# %% 
# Initialise seed for reproducibility
seed = random.randint(0,10000)
#seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# %%
# Create environment
env = Environment(experiment, nRegAgents, nMalAgents, nProviders, 
                  kappa, rho, center_up_to_down, center_down_to_up, end_up_to_down, end_down_to_up, cost,
                  direct_exp_weight, satisfaction_threshold, forgetting_factor)

# Create lists of agents, form social network of agents, and create attributes
regagents, malagents, providers = env.regagents, env.malagents, env.providers

# Pick one agent randomly from each type of agents
regagent_example = np.random.choice(regagents)

# Define regular agents' state space, actions, and opinions
observation_space, action_space = env.observation_spaces[f"regagent{regagent_example}"], env.action_spaces[f"regagent{regagent_example}"]

state_shape, n_actions, n_opinions = \
    env.observation_spaces[f"regagent{regagent_example}"].shape[0], env.action_spaces[f"regagent{regagent_example}"][0].n, env.action_spaces[f"regagent{regagent_example}"][1].n

# Reset environment state
envstate, info = env.reset(seed=seed)

print("Regular agents' state shape is", state_shape, ", number of actions is", n_actions, " and number of opinions is", n_opinions)

# %%
# Create regular agents
regular_agents = {f"regagent{agent}": A2CRegAgent(state_shape, n_actions, n_opinions, alpha_rnn1, alpha_rnn2, device) for agent in env.regagents}

# %%
# Collect data
total_rewards, action_rewards, opinion_rewards = np.zeros(number_of_episodes + 1), np.zeros(number_of_episodes + 1), np.zeros(number_of_episodes + 1)

loss1_agents, loss2_agents = np.zeros((number_of_episodes + 1, len(regagents))), np.zeros((number_of_episodes + 1, len(regagents)))
mean_loss1, mean_loss2 = np.zeros(number_of_episodes + 1), np.zeros(number_of_episodes + 1)

# %%
# Train and evaluate the agent

for episode in range(1, number_of_episodes + 1):
    print(f"Episode {episode} of {number_of_episodes} episodes")
    # Decay entropy coefficient for new episode
    entropy_coef = scheduler(beta_start, beta_end, episode, beta_decay)

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

    # Calculate episode mean loss over all agents
    mean_loss1[episode] = np.mean(loss1_agents[episode]) # mean loss for actions
    mean_loss2[episode] = np.mean(loss2_agents[episode]) # mean loss for opinions
        
    # Visualise data
    if episode != 1 and episode % vis_freq == 0:
        clear_output(True)
        # Plot 1: Visualise average cumulative reward
        plt.figure()
        plt.plot(total_rewards[:episode], linewidth=0.9, color = 'mediumvioletred', label = "Mean score")
        plt.plot(action_rewards[:episode], linewidth=0.9, color = 'red', label = "Mean service score")
        plt.plot(opinion_rewards[:episode], linewidth=0.9, color = 'orange', label = "Mean feedback score")
        plt.xlabel("Episodes")
        plt.ylabel("Average cumulative reward\nfor regular agents per episode")
        plt.grid()
        plt.legend(loc=(0.01,0.50), fontsize='x-small')
        # plt.savefig(os.path.join(output_dir, f'regagent-{date}-{seed}-{episode}-regagent-score.png'))
        # plt.clf()

        # Plot 3.1/3.2: Visualise loss/objective value per episode
        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(15, 4))
        axs[0].plot(smoothen(mean_loss1[:episode]), linewidth=0.9, color = 'red')
        axs[0].set_xlabel("Episodes")
        axs[0].set_ylabel("Loss (actions)")
        axs[0].grid()
        axs[1].plot(smoothen(mean_loss2[:episode]), linewidth=0.9, color = 'orange')
        axs[1].set_xlabel("Episodes")
        axs[1].set_ylabel("Loss (opinions)")
        axs[1].grid()
        # plt.savefig(os.path.join(output_dir, f'marl-{date}-{seed}-{episode}-regagent-actions-opinions.png'))
        # plt.clf()
        plt.show()

# Save data
regular_agent_reward_loss_df = pd.DataFrame({'total_rewards': total_rewards, 'action_rewards': action_rewards, 'opinion_rewards': opinion_rewards,
                             'mean_loss1': mean_loss1, 'mean_loss2': mean_loss2})
# Save dataframes to csv
regular_agent_reward_loss_df.to_csv(os.path.join(output_dir, f'marl-{chapter}-exp{experiment}-{date}-{timestamp}-{seed}-regagents-data.csv'), index = False)

# %%
# Save models
# for agent_name, agent in regular_agents.items():
#     torch.save({
#         'actions_state_dict': agent.action_nn.state_dict(),
#         'actions_opt_state_dict': agent.action_opt.state_dict(),
#     }, f'{save_path}{agent_name}-checkpoint-actions-{date}.pth')

#     torch.save({
#         'opinions_state_dict': agent.opinion_nn.state_dict(),
#         'opinions_opt_state_dict': agent.opinion_opt.state_dict(),
#     }, f'{save_path}{agent_name}-checkpoint-opinions{date}.pth')