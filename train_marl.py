# %% [markdown]
# ## Modelling Hybrid Threats and Defensice Countermeasures
# Chapter 3\
# Experiment 2

# %%
# Import libraries
from environment_exp2 import Environment
from a2c_agent import A2CRegAgent
from a2c_def_agent import A2CServiceProvider
from a2c_mal_agent import A2CMalAgent
from data_analysis import process_regagent_states_average, process_regagent_actions, process_regagent_rewards, \
    process_attacker_rewards, process_attacker_actions, process_defender_rewards, process_service_provider_availability
from other_functions import moving_average 
import numpy as np
import pandas as pd
import torch
import random
import matplotlib.pyplot as plt
from IPython.display import clear_output
import os
import re
import datetime

# %%
# Setup device, date, chapter, and experiment
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # currently cpu
date = datetime.datetime.now().strftime("%Y-%m-%d")
chapter = 3
experiment = 2
# Specify output directory
output_dir = os.path.join("results")
save_path = os.path.join("results", "model_output")

# %%
# Specify number of agents in the environment
nProviders = 3
nRegAgents = 100
nMalAgents = 10

# %%
# Initialise (tuned) hyperparameters from file
df = pd.read_csv("hyperparameters.csv")
hyperparameters = dict(zip(df['hyperparameter'], df['value']))

alpha_rnn1 = hyperparameters['alpha_1']
alpha_rnn2 = hyperparameters['alpha_2']
alpha_ann1 = hyperparameters['alpha_3']
alpha_ann2 = hyperparameters['alpha_4']
alpha_ann3 = hyperparameters['alpha_5']
alpha_dnn1 = hyperparameters['alpha_6']
alpha_dnn2 = hyperparameters['alpha_7']
gamma_ann1 = hyperparameters['gamma_3']
gamma_ann2 = hyperparameters['gamma_4']
gamma_ann3 = hyperparameters['gamma_5']
gamma_dnn1 = hyperparameters['gamma_6']
gamma_dnn2 = hyperparameters['gamma_7']
beta_start = 1
beta_end = hyperparameters['beta_2']
beta_decay = int(hyperparameters['n_2'])

# %%
# Initialise parameters from file
df = pd.read_csv("parameters.csv")
parameters = dict(zip(df['parameter'], df['value']))

# Social network parameters
kappa = int(parameters['kappa'])
rho = parameters['rho']

# Cyber-physical system parameters
center_up_to_down = [parameters['center_up_to_down']] * nProviders # Psi: prob (1 to -1)
center_down_to_up = [parameters['center_down_to_up']] * nProviders # psi: prob (-1 to 1)
end_up_to_down = [parameters['end_up_to_down']] * nProviders # Lambda: prob (1 to -1)
end_down_to_up = [parameters['end_down_to_up']] * nProviders # lambda: prob (-1 to 1)
cost = [parameters['cost']] * nProviders

# Agents' attributes (parameters)
direct_exp_weight = parameters['direct_exp_weight']
satisfaction_threshold = parameters['satisfaction_threshold']
forgetting_factor = parameters['forgetting_factor']

# %%
# Define training time
n_steps = 100 # number of steps per episode
number_of_episodes = 20000
eval_freq = 100
vis_freq = 1000
saving_freq = 1000

# %%
# Initialise seed for reproducibility
seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# %%
# Create environment
env = Environment(nRegAgents, nMalAgents, nProviders, 
                  kappa, rho, center_up_to_down, center_down_to_up, end_up_to_down, end_down_to_up, cost,
                  direct_exp_weight, satisfaction_threshold, forgetting_factor)

# Create lists of agents, form social network of agents, and create attributes
agent_ids, agents, regagents, malagents, providers, social_network, neighbours=\
    env.agents, env.sn_agents, env.regagents, env.malagents, env.providers, env.social_network, env.neighbours
# Pick one agent randomly from each type of agents
regagent_example = np.random.choice(regagents)
defender_example = np.random.choice(providers)
malagent_example = np.random.choice(malagents)

# Define regular agents' state space, actions, and opinions
state_shape, n_actions, n_opinions = env.observation_spaces[f"regagent{regagent_example}"].shape[0], env.action_spaces[f"regagent{regagent_example}"][0].n, env.action_spaces[f"regagent{regagent_example}"][1].n

# Define defenders' (service providers') state space, and actions
state_shape_defenders, n_filter, n_answer = (len(env.observation_spaces[f"defagent{defender_example}"]), nRegAgents + nMalAgents), env.action_spaces[f"defagent{defender_example}"][0].n, env.action_spaces[f"defagent{defender_example}"][1].n

# Define attackers' state space, and actions
state_size_attackers, n_stage_actions, n_cyber_actions, n_misinfo_actions, stage_actions_dict, mal_contacts_dict, malagents_dict = \
    len(env.observation_spaces[f"malagent"]), env.action_spaces[f"malagent"][0].n, env.action_spaces[f"malagent"][1].n, env.action_spaces[f"malagent"][2][f"malagent{malagent_example}"].n,\
    env.stage_actions_dict , env.mcontacts_dict, env.malagents_dict

# Reset environment state
envstate, info = env.reset(seed=seed)

# Visualise graph
env.render()
# plt.savefig("sn.png", dpi=300)

print("Regular agents' state shape is", state_shape, ", number of actions is", n_actions, " and number of opinions is", n_opinions)
print("Defenders' state shape is", state_shape_defenders, ", number of filter actions is", n_filter, " and number of answer actions is", n_answer)
print("Attackers' state shape is", state_size_attackers, ", number of stage actions is", n_stage_actions, ", number of cyber actions is", n_cyber_actions, " and number of misinformation actions is", n_misinfo_actions)

# %%
# Load previously trained regular agents
regular_agents = {f"regagent{agent}": A2CRegAgent(state_shape, n_actions, n_opinions, alpha_rnn1, alpha_rnn2, device) 
                for agent in env.regagents}

for agent_name, agent in regular_agents.items():
    # Load Action NN and its optimiser
    action_checkpoint = torch.load(f'/Users/kartpadur/Documents/GitHub/pytorch_project/ch3-model/regagent-parameters/{agent_name}_checkpoint_actions_2024-06-24_5281.pth')
    agent.action_nn.load_state_dict(action_checkpoint['actions_state_dict'])
    agent.action_opt.load_state_dict(action_checkpoint['actions_opt_state_dict'])
    
    # Load Opinion NN and its optimiser
    opinion_checkpoint = torch.load(f'/Users/kartpadur/Documents/GitHub/pytorch_project/ch3-model/regagent-parameters/{agent_name}_checkpoint_opinions_2024-06-24_5281.pth')
    agent.opinion_nn.load_state_dict(opinion_checkpoint['opinions_state_dict'])
    agent.opinion_opt.load_state_dict(opinion_checkpoint['opinions_opt_state_dict'])

# %%
# Initialise defenders
service_providers = {f"defagent{agent}": A2CServiceProvider(state_shape_defenders, n_filter, n_answer, alpha_dnn1, alpha_dnn2, device) 
                    for agent in env.providers}
# %%
# Initialise attackers
attack_target = 1 # initialised with environment
misinfo_opinion = 2 # initialised with environment

malicious_agent = {f"malagent": A2CMalAgent(state_size_attackers, n_stage_actions, n_cyber_actions, n_misinfo_actions, 
                 stage_actions_dict, mal_contacts_dict, malagents_dict, malagents, neighbours, attack_target,
                 alpha_ann1, alpha_ann2, alpha_ann3, device)}

# %%
# Collect data
# Regular agents
regagent_total_rewards, regagent_action_rewards, regagent_opinion_rewards = np.zeros(number_of_episodes + 1), np.zeros(number_of_episodes + 1), np.zeros(number_of_episodes + 1)
regagent_actions_history = np.zeros((number_of_episodes + 1, len(providers)))
regagent_opinions_history = np.zeros((number_of_episodes + 1, len(providers)*2))
social_trust_history = np.zeros((number_of_episodes + 1, len(providers)))
# Service providers
sp_availability_history = np.zeros((number_of_episodes + 1, len(providers)))
sum_sp_availability = [0, 0, 0]
# Defenders
total_defender_rewards, defender_filter_rewards, defender_answer_rewards =\
        np.zeros((number_of_episodes + 1, len(providers))), np.zeros((number_of_episodes + 1, len(providers))), np.zeros((number_of_episodes + 1, len(providers)))
loss4_history, loss5_history = np.zeros((number_of_episodes + 1, len(providers))), np.zeros((number_of_episodes + 1, len(providers)))
# Attackers
total_attacker_rewards, reconnaissance_rewards, cyber_attack_rewards, disinfo_rewards, termination_rewards =\
        np.zeros(number_of_episodes + 1), np.zeros(number_of_episodes + 1), np.zeros(number_of_episodes + 1), np.zeros(number_of_episodes + 1), np.zeros(number_of_episodes + 1)
loss1_history, loss2_history, loss3_history = np.zeros(number_of_episodes + 1), np.zeros(number_of_episodes + 1), np.zeros(number_of_episodes + 1)

cyber_actions_history = np.zeros((number_of_episodes + 1, n_cyber_actions+1))
contacts_history =  np.zeros((number_of_episodes + 1, n_misinfo_actions+1))
attack_order = np.zeros((number_of_episodes + 1, 6))
count_timesteps_per_stage = np.zeros((number_of_episodes + 1, 6))

# %%
for episode in range(1, number_of_episodes + 1):
    # Decay entropy coefficient
    entropy_coef = beta_start + (beta_end - beta_start) * min(episode, beta_decay) / beta_decay

    # Restart environment
    observations, _ = env.reset()

    # Initialise dictionaries to store rewards, observations, and actions
    all_observations = {agent_name: [observations[agent_name]] for agent_name in agent_ids if "malagent" not in agent_name}
    all_observations["malagent"] = [observations["malagent"]]
    all_actions = {agent_name: [] for agent_name in agent_ids}
    all_actions["malagent"] = []
    episode_rewards = {agent_name: [] for agent_name in agent_ids}
    episode_rewards["malagent"] = []

    # Generate trajectory over n_steps (episode length)
    for timestep in range(1, n_steps+1): # one episode

        actions = {}  # Dictionary to store the current timestep's actions

        for agent_name, agent in regular_agents.items():
                s = observations[agent_name]
                # Sample action and opinion
                action, opinion = agent.sample_actions(s)
                actions[agent_name] = (action, opinion) # save action and opinion
        
        for attacker_name, attacker in malicious_agent.items(): # Attackers make decisions together
            states = all_observations[attacker_name]
            # Sample actions for malicious agents
            attack_stage_action, bot_action, sn_actions_dict, botnet = attacker.sample_actions(states)
            for agent_name, provider in botnet.items():
                actions[agent_name] = (provider, -1)

        defenders_obs = env._monitor_requests_and_sn(actions) # which agents requested service, which opinions they expressed
        observations.update(defenders_obs)

        # Update actions with attacker information
        actions["malagent"] = (attack_stage_action, bot_action, sn_actions_dict)

        for defender_name, defender in service_providers.items():
            state = observations[defender_name]
            # Sample action and opinion
            filters, answers = defender.sample_actions(state)
            actions[defender_name] = (filters, answers)

        # Perform actions, determine next state, reward, and termination
        observations, rewards, _, _, _ = env.step(actions)

        for provider in providers:
            for agent_name, agent in regular_agents.items():
                if env.service_received[agent_name][provider] !=[] and env.service_received[agent_name][provider][-1][1] == 1:
                    sum_sp_availability[provider] += 1

        # Store the rewards, observations, and actions for each agent for the current timestep
        for agent_name in agent_ids:
            if agent_name != "malagent":
                agent = int(re.findall(r'\d+', agent_name)[0])
                if agent in providers or agent in env.regagents: # regular agents and defenders
                    all_observations[agent_name].append(observations[agent_name])
                    all_actions[agent_name].append(actions[agent_name])
                    episode_rewards[agent_name].append(rewards[agent_name])
        # attackers
        all_observations["malagent"].append(observations["malagent"])
        all_actions["malagent"].append(actions["malagent"])
        episode_rewards["malagent"].append(rewards["malagent"])
    
    # Store agents information, including states, actions, and rewards
    # Process regagent rewards
    sum_regagent_rewards, service, feedback = process_regagent_rewards(episode_rewards)
    regagent_total_rewards[episode] = sum_regagent_rewards
    regagent_action_rewards[episode] = service
    regagent_opinion_rewards[episode] = feedback
    # Process social trust
    episode_states_pd = process_regagent_states_average(all_observations, providers, episode, n_steps)
    social_trust_history[episode][0] = episode_states_pd["trust_in_sp0"].mean()
    social_trust_history[episode][1] = episode_states_pd["trust_in_sp1"].mean()
    social_trust_history[episode][2] = episode_states_pd["trust_in_sp2"].mean()
    # Processing regagent actions and opinions
    _, regagent_action_counts, regagent_opinion_counts = process_regagent_actions(all_actions, regagents, episode, n_steps)
    regagent_actions_history[episode] = regagent_action_counts # add occurrences of each action (as %)
    regagent_opinions_history[episode] = regagent_opinion_counts # add occurrences of each opinion (as %)
    # Process service provider availability
    sp_availability_episode = process_service_provider_availability(nRegAgents, n_steps, sum_sp_availability)
    for provider in providers:
        sp_availability_history[episode][provider] = sp_availability_episode[provider]
    sum_sp_availability = [0, 0, 0]
    
    # Attackers compute A2C loss (and backpropagate)
    for agent_name, agent in malicious_agent.items():
        loss1, loss2, loss3 = agent.compute_a2c_loss(all_observations[agent_name][1:], all_actions[agent_name], episode_rewards[agent_name], gamma_ann1, gamma_ann2, gamma_ann3, entropy_coef)
        loss1_history[episode] = loss1.data.cpu().item()
        loss2_history[episode] = loss2.data.cpu().item()
        loss3_history[episode] = loss3.data.cpu().item()
    
    # Store attackers information, including actions and rewards
    # Process attacker rewards
    attack_stage_reward, recon_reward, term_reward, bot_reward, contact_reward = process_attacker_rewards(all_actions, episode_rewards)
    total_attacker_rewards[episode] = attack_stage_reward
    reconnaissance_rewards[episode] = recon_reward
    termination_rewards[episode] = term_reward
    cyber_attack_rewards[episode] = bot_reward
    disinfo_rewards[episode] = contact_reward

    # Process attacker actions
    bot_counts, disinfo_counts, episode_attack_order, episode_count_actions_per_stage = process_attacker_actions(all_actions, n_cyber_actions, n_misinfo_actions)
    cyber_actions_history[episode] = bot_counts # [0 0 0 ..] for every potential bot size
    contacts_history[episode] = disinfo_counts # [0 0 1 0 ...] for every contacted agent
    attack_order[episode] = episode_attack_order # store attack order
    count_timesteps_per_stage[episode] = episode_count_actions_per_stage # store time in each stage

    # Defenders compute A2C loss (and backpropagate)
    for agent_name, agent in service_providers.items():
        loss4, loss5 = agent.compute_a2c_loss(all_observations[agent_name][1:], all_actions[agent_name], episode_rewards[agent_name], gamma_dnn1, gamma_dnn2, entropy_coef)
        agent_id = int(agent_name.replace("defagent", ""))
        loss4_history[episode][agent_id] = loss4.data.cpu().item()
        loss5_history[episode][agent_id] = loss5.data.cpu().item()
    
    # Store defenders information, including rewards
    # Process defender rewards
    filter_rewards, answer_rewards = process_defender_rewards(episode_rewards, providers)
    for defender in providers:
        defender_filter_rewards[episode][defender] = filter_rewards[defender]
        defender_answer_rewards[episode][defender] = answer_rewards[defender]
        total_defender_rewards[episode][defender] = filter_rewards[defender] + answer_rewards[defender]
    
    if episode != 1 and episode % saving_freq == 0:
        # Save regular agents' data
        regagents_df = pd.DataFrame({'total_rewards': regagent_total_rewards[1:episode+1], 'action_rewards': regagent_action_rewards[1:episode+1], 'opinion_rewards': regagent_opinion_rewards[1:episode+1],
                            'social_trust_sp0': social_trust_history[1:episode+1,0], 'social_trust_sp1': social_trust_history[1:episode+1,1], 'social_trust_sp2': social_trust_history[1:episode+1,2],
                            'request_rate_sp0': regagent_actions_history[1:episode+1,0], 'request_rate_sp1': regagent_actions_history[1:episode+1,1], 'request_rate_sp2': regagent_actions_history[1:episode+1,2],
                            'op0_expression_rate': regagent_opinions_history[1:episode+1,0], 'op1_expression_rate': regagent_opinions_history[1:episode+1,1],
                            'op2_expression_rate': regagent_opinions_history[1:episode+1,2], 'op3_expression_rate': regagent_opinions_history[1:episode+1,3],
                            'op4_expression_rate': regagent_opinions_history[1:episode+1,4], 'op5_expression_rate': regagent_opinions_history[1:episode+1,5],
                            'availability_sp0': sp_availability_history[1:episode+1,0], 'availability_sp1': sp_availability_history[1:episode+1,1], 'availability_sp2': sp_availability_history[1:episode+1,2]})
        regagents_df.to_csv(os.path.join(output_dir, f'ch{chapter}-exp{experiment}-{date}-{seed}-regagents-data.csv'), index = False)

        # Save attackers' data
        attacker_reward_loss_df = pd.DataFrame({'total_rewards': total_attacker_rewards[1:episode+1], 'recon_rewards': reconnaissance_rewards[1:episode+1], 
                            'cyber_attack_rewards': cyber_attack_rewards[1:episode+1], 'disinfo_rewards': disinfo_rewards[1:episode+1],' term_rewards': termination_rewards[1:episode+1],
                            'loss1_history': loss1_history[1:episode+1], 'loss2_history': loss1_history[1:episode+1], 'loss3_history': loss1_history[1:episode+1]})
        # Save dataframes to csv
        attacker_reward_loss_df.to_csv(os.path.join(output_dir, f'ch{chapter}-exp{experiment}-{date}-{seed}-attacker-data.csv'), index = False)
        attack_order_df = pd.DataFrame(attack_order, columns = ['recon', 'cyber', 'disinfo', 'comb', 'cyber2', 'disinfo2'])
        attack_order_df.to_csv(os.path.join(output_dir, f'ch{chapter}-exp{experiment}-{date}-{seed}-attack-order.csv'), index=False)
        count_attack_actions_df = pd.DataFrame(count_timesteps_per_stage, columns = ['recon', 'cyber', 'disinfo', 'comb', 'cyber2', 'disinfo2'])
        count_attack_actions_df.to_csv(os.path.join(output_dir, f'ch{chapter}-exp{experiment}-{date}-{seed}-attack-actions-count.csv'), index=False)

        # Collect data
        defender_reward_loss_df = pd.DataFrame({'total_rewards_sp0': total_defender_rewards[1:episode+1,0], 'total_rewards_sp1': total_defender_rewards[1:episode+1,1], 'total_rewards_sp2': total_defender_rewards[1:episode+1,2],
                                                'filter_rewards_sp0': defender_filter_rewards[1:episode+1,0], 'filter_rewards_sp1': defender_filter_rewards[1:episode+1,1], 'filter_rewards_sp2': defender_filter_rewards[1:episode+1,2], 
                                                'answer_rewards_sp0': defender_answer_rewards[1:episode+1,0], 'answer_rewards_sp1': defender_answer_rewards[1:episode+1,1], 'answer_rewards_sp2': defender_answer_rewards[1:episode+1,2],
                                                'loss4_sp0': loss4_history[1:episode+1,0],'loss4_sp1': loss4_history[1:episode+1,1], 'loss4_sp2': loss4_history[1:episode+1,2],
                                                'loss5_sp0': loss5_history[1:episode+1,0],'loss5_sp1': loss5_history[1:episode+1,1], 'loss5_sp2': loss5_history[1:episode+1,2]})
        # Save dataframes to csv
        defender_reward_loss_df.to_csv(os.path.join(output_dir, f'ch{chapter}-exp{experiment}-{date}-{seed}-defender-data.csv'), index = False)

    if episode != 1 and episode % vis_freq == 0:
        clear_output(True)
        print("Episode", episode, ": mean attacker reward: %.3f" % (np.mean(total_attacker_rewards[:episode])),
              " and mean defender reward: %.3f" % (np.mean(total_defender_rewards[:episode])))

        # Figure 1. Defender rewards
        plt.figure()
        for sp in providers:
            plt.plot(moving_average(total_defender_rewards[1:episode+1,sp]), linewidth=0.9, label=f'Provider {sp+1}')
        plt.xlabel('Episodes')
        plt.ylabel("Cumulative reward for defenders")
        plt.grid()
        plt.legend(loc=(0.01,0.50), fontsize='x-small')
        # plt.savefig(os.path.join(output_dir, f'ch{chapter}-exp{experiment}-{date}-{seed}-defender-score.png'))
        # plt.clf()

        # Figure 2.1/2.2: Defender filtering and answering rewards
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 4))
        for sp in providers:
            axes[0].plot(moving_average(defender_filter_rewards[1:episode+1,sp]), linewidth=0.9, label=f'Provider {sp+1}')
        axes[0].set_xlabel('Episodes')
        axes[0].set_ylabel("Cumulative reward\nfor filtering")
        axes[0].grid()
        axes[0].legend(loc=(0.01,0.50), fontsize='x-small')
        for sp in providers:
            axes[1].plot(moving_average(defender_answer_rewards[1:episode+1,sp]), linewidth=0.9, label=f'Provider {sp+1}')
        axes[1].set_xlabel('Episodes')
        axes[1].set_ylabel("Cumulative reward\nfor information spreading")
        axes[1].grid()
        axes[1].legend(loc=(0.01,0.50), fontsize='x-small')
        # plt.savefig(os.path.join(output_dir, f'ch{chapter}-exp{experiment}-{date}-{seed}-defender-score2.png'))
        # plt.clf()

        # Figure 3.1/3.2: Defender loss
        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(15, 4))
        for sp in providers:
            axs[0].plot(moving_average(loss4_history[1:episode+1,sp]), linewidth=0.9, label=f'Provider {sp+1}')
        axs[0].set_xlabel("Episodes")
        axs[0].set_ylabel("Loss (filtering)")
        axs[0].grid()
        axs[0].legend(loc=(0.01,0.50), fontsize='x-small')
        for sp in providers:
            axs[1].plot(moving_average(loss5_history[1:episode+1,sp]), linewidth=0.9, label=f'Provider {sp+1}')
        axs[1].set_xlabel("Episodes")
        axs[1].set_ylabel("Loss (answering)")
        axs[1].grid()
        axs[0].legend(loc=(0.01,0.50), fontsize='x-small')
        # plt.savefig(os.path.join(output_dir, f'ch{chapter}-exp{experiment}-{date}-{seed}-defender-loss.png'))
        # plt.clf()

        # Attackers
        # Figure 4: Attacker rewards 
        plt.figure()
        plt.plot(moving_average(total_attacker_rewards[1:episode+1]), linewidth=0.9, color='mediumvioletred', label="Score")
        plt.plot(moving_average(reconnaissance_rewards[1:episode+1]), linewidth=0.9, color='blue', label='Reconnaissance score')
        plt.plot(moving_average(cyber_attack_rewards[1:episode+1]), linewidth=0.9, color='orange', label="Cyberattack score")
        plt.plot(moving_average(disinfo_rewards[1:episode+1]), linewidth=0.9, color='green', label="Disinformation score")
        plt.plot(moving_average(termination_rewards[1:episode+1]), linewidth=0.9, color='gray', label='Termination score')
        plt.xlabel("Episodes")
        plt.ylabel("Cumulative reward\nfor attackers")
        plt.grid()
        plt.legend(loc=(0.01,0.50), fontsize='x-small')
        # plt.savefig(os.path.join(output_dir, f'ch{chapter}-exp{experiment}-{date}-{seed}-attacker-score.png'))
        # plt.clf()

        # Figure 5.1/5.2/5.3: Attacker loss
        fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(23, 4))
        ax[0].plot(moving_average(loss1_history[1:episode+1]), linewidth=0.9, color = 'mediumvioletred')
        ax[0].set_xlabel("Episodes")
        ax[0].set_ylabel("Loss (attack stage)")
        ax[0].grid()
        ax[1].plot(moving_average(loss2_history[1:episode+1]), linewidth=0.9, color = 'orange')
        ax[1].set_xlabel("Episodes")
        ax[1].set_ylabel("Loss (cyberattack)")
        ax[1].grid()
        ax[2].plot(moving_average(loss3_history[1:episode+1]), linewidth=0.9, color = 'green')
        ax[2].set_xlabel("Episodes")
        ax[2].set_ylabel("Loss (disinfo)")
        ax[2].grid()
        # plt.savefig(os.path.join(output_dir, f'ch{chapter}-exp{experiment}-{date}-{seed}-attacker-loss.png'))
        # plt.clf()

        # Figure 6: Regular agents rewards
        plt.figure()
        plt.plot(moving_average(regagent_total_rewards[1:episode+1]), linewidth=0.9, color = 'mediumvioletred', label = "Mean score")
        plt.plot(moving_average(regagent_action_rewards[1:episode+1]), linewidth=0.9, color = 'red', label = "Mean service score")
        plt.plot(moving_average(regagent_opinion_rewards[1:episode+1]), linewidth=0.9, color = 'orange', label = "Mean feedback score")
        plt.xlabel("Episodes")
        plt.ylabel("Cumulative reward\nfor regular agents")
        plt.grid()
        plt.legend(loc=(0.01,0.50), fontsize='x-small')
        # plt.savefig(os.path.join(output_dir, f'ch{chapter}-exp{experiment}-{date}-{seed}-regagent-score.png'))
        # plt.clf()

        # Figure 7: Visualise social trust in service providers
        plt.figure()
        for sp in range(nProviders):
            plt.plot(moving_average(social_trust_history[1:episode+1, sp]), linewidth=0.9, label = "Provider  {}".format(sp+1))
        plt.xlabel("Episode")
        plt.ylabel("Average social trust\nin service providers per episode")
        plt.grid()
        plt.legend(loc=(0.01,0.50), fontsize='x-small')
        # plt.savefig(os.path.join(output_dir, f'ch{chapter}-exp{experiment}-{date}-{seed}-regagent-trust.png'))
        # plt.clf()

        # Actions and opinions
        fig, axe = plt.subplots(nrows=1, ncols=2, figsize=(15, 4))
        for sp in range(nProviders):
            axe[0].plot(regagent_actions_history[1:episode+1,sp], linewidth=0.9, label = "Provider  {}".format(sp+1))
        axe[0].set_xlabel("Episodes")
        axe[0].set_ylabel("Average service request rate\nper episode")
        axe[0].grid()
        axe[0].legend(loc=(0.01,0.50), fontsize='x-small')
        for o in range(nProviders*2):
            if o in [0]:
                axe[1].plot(regagent_opinions_history[1:episode+1,o], "--", color = 'tab:blue', linewidth=0.9, label = "Provider  {} -".format(o+1))
            elif o in [1]:
                axe[1].plot(regagent_opinions_history[1:episode+1,o], "-", color = 'tab:blue', linewidth=0.9, label = "Provider  {} +".format(o))
            elif o in [2]:
                axe[1].plot(regagent_opinions_history[1:episode+1,o], "--", color = 'tab:orange', linewidth=0.9, label = "Provider  {} -".format(o))
            elif o in [3]:
                axe[1].plot(regagent_opinions_history[1:episode+1,o], "-", color = 'tab:orange', linewidth=0.9, label = "Provider  {} +".format(o-1))
            elif o in [4]:
                axe[1].plot(regagent_opinions_history[1:episode+1,o], "--", color = 'tab:green', linewidth=0.9, label = "Provider  {} -".format(o-1))
            elif o in [5]:
                axe[1].plot(regagent_opinions_history[1:episode+1,o], "-", color = 'tab:green', linewidth=0.9, label = "Provider  {} +".format(o-2))
            elif o in [6]:
                axe[1].plot(regagent_opinions_history[1:episode+1,o], "--", color = 'tab:red', linewidth=0.9, label = "Provider  {} -".format(o-2))
            else:
                axe[1].plot(regagent_opinions_history[1:episode+1,o], "-", color = 'tab:red', linewidth=0.9, label = "Provider  {} +".format(o-3))
        axe[1].set_xlabel("Episodes")
        axe[1].set_ylabel("Average opinion expression rate\nper episode")
        axe[1].grid()
        axe[1].legend(loc=(0.01,0.50), fontsize='x-small')
        # plt.savefig(os.path.join(output_dir, f'ch{chapter}-exp{experiment}-{date}-{seed}-regagent-actions-opinions.png'))
        # plt.clf()

        plt.show()