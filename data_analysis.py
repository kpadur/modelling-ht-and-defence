import pandas as pd
import numpy as np

def process_regagent_states_average(all_observations, providers, step, n_steps):
    """
    Calculate the average social trust (per episde) for every service provider.
    """
    # Only process regagent states
    all_regagent_observations = {agent: values for agent, values in all_observations.items() if agent.startswith('regagent')}

    # Initialise a matrix to hold the average trust values for each service provider at each time step
    average_trust = np.zeros((n_steps, len(providers)))

    # Initialise column names for the DataFrame
    columns = [f"trust_in_sp{j}" for j in range(len(providers))]
    # Initialise a pandas dataframe for storing data
    average_trust_pd = pd.DataFrame(index=range(n_steps), columns=columns)

    # Calculate the average trust values for each service provider at each time step
    for t in range(n_steps):
        total_trust = np.zeros(len(providers))
        for agent in all_regagent_observations:
            total_trust += all_regagent_observations[agent][t]
        average_trust[t] = total_trust / len(all_regagent_observations)
        # Fill the DataFrame
        for i, trust in enumerate(average_trust[t]):
            column_name = f"trust_in_sp{i}"
            average_trust_pd.at[t, column_name] = trust
    
    # Add 'episode' and 'timestep' columns
    average_trust_pd.insert(0, 'episode', step)
    average_trust_pd.insert(1, 'timestep', range(n_steps))

    return average_trust_pd

def process_regagent_actions(all_actions, regagents, step, n_steps):
    """
    Calculate the average service request rate (per episode) for each service provider and
    the average opinion experssing rate (per episode) for each service provider.
    """
    # Identify all required columns
    columns = []
    for agent in all_actions:
        if agent.startswith('regagent'):
            columns.append(f"{agent}_action")
            columns.append(f"{agent}_opinion")

    # Pre-allocate the DataFrame with identified columns and NaNs
    episode_actions_pd = pd.DataFrame(index=range(1, n_steps + 1), columns=columns)

    # Populate the DataFrame
    for agent, values in all_actions.items():
        if agent.startswith('regagent'):
            for i, actions in enumerate(values):
                for j, act in enumerate(actions):
                    if j == 0:
                        column_name = f"{agent}_action"
                    else:
                        column_name = f"{agent}_opinion"
                    episode_actions_pd.at[i + 1, column_name] = int(act)  # i + 1 to match index range

    # Convert all columns to integer type
    for column in episode_actions_pd.columns:
        episode_actions_pd[column] = episode_actions_pd[column].astype(int)

    # Create 'episode' and 'timestep' columns and concatenate them
    episode_col = pd.DataFrame({'episode': [step] * n_steps}, index=range(1, n_steps + 1))
    timestep_col = pd.DataFrame({'timestep': range(1, n_steps + 1)}, index=range(1, n_steps + 1))
    episode_actions_pd = pd.concat([episode_col, timestep_col, episode_actions_pd], axis=1)

    # Analyse action and opinion counts
    regagent_actions = {agent_name: value for agent_name, value in all_actions.items() if agent_name.startswith('regagent')}
    regagent_action_counts = [0] * 3 # NOTE: n_actions - not scaleble, works for now
    regagent_opinion_counts = [0] * 6 # NOTE: n_opinions - not scalable, works for now
    for actions_list in regagent_actions.values():
        for action, opinion in actions_list:
            regagent_action_counts[action] += 1
            regagent_opinion_counts[opinion] += 1
    regagent_action_counts = np.array(regagent_action_counts) * 100/ (len(regagents)*n_steps)
    regagent_opinion_counts = np.array(regagent_opinion_counts) * 100/ (len(regagents)*n_steps)
    
    return episode_actions_pd, regagent_action_counts, regagent_opinion_counts

def process_regagent_rewards(episode_rewards):
    """
    Calculate the cumulative rewards per episode for the regular agents.
    """
    # Determine total rewards
    regagent_rewards = {agent_name: value for agent_name, value in episode_rewards.items() if agent_name.startswith('regagent')}
    sum_regagent_rewards = sum([sum(r) for rewards in regagent_rewards.values() for r in rewards])
    # Determine service rewards
    service = sum([r[0] for rewards in regagent_rewards.values() for r in rewards])
    # Determine feedback rewards
    feedback = sum([r[1] for rewards in regagent_rewards.values() for r in rewards])
    return sum_regagent_rewards, service, feedback

def process_service_provider_availability(nRegAgents, n_steps, sum_sp_availability):
    """Calculate average service availability rate per episode for each service provider."""
    return [sum_val / (nRegAgents * n_steps) for sum_val in sum_sp_availability]

def process_attacker_rewards(all_actions, episode_rewards):
    # Determine attacker rewards
    attack_actions = {agent_name: value for agent_name, value in all_actions.items() if agent_name.startswith('malagent')}
    attacker_rewards = {agent_name: value for agent_name, value in episode_rewards.items() if agent_name.startswith('malagent')}
    attack_stage_actions = np.array([action[0] for action in attack_actions.values() for action in action])
    attack_stage_rewards = np.array([reward[0] for reward in attacker_rewards.values() for reward in reward])
    # Calculate recon rewards
    recon_rewards = attack_stage_rewards[np.where(attack_stage_actions == 0)]
    # Determine term rewards
    term_rewards = attack_stage_rewards[np.where(attack_stage_actions == 7)]
    # Bot (cyberattack) rewards
    bot_rewards = np.array([reward[1] for reward in attacker_rewards.values() for reward in reward])
    # Contact rewards
    contact_rewards = 0
    for _, (attack_stage_reward, bot_reward, misinfo_reward) in enumerate(attacker_rewards["malagent"]):
        contact_rewards += sum(misinfo_reward.values())
    return np.sum(attack_stage_rewards), np.sum(recon_rewards), np.sum(term_rewards), np.sum(bot_rewards), contact_rewards

def process_attacker_actions(all_actions, n_cyber_actions, n_misinfo_actions):
    # Collect attacker's data at every step for visualisation MISSING
    all_attack_actions = {agent_name: value for agent_name, value in all_actions.items() if agent_name.startswith('malagent')}
    # Cyberattack actions
    attack_actions = np.array([action[1] for actions in all_attack_actions.values() for action in actions]) + 1 # so that -1 would account to 0
    # Misinformation actions
    misinfo_actions = np.array([action[2] for actions in all_attack_actions.values() for action in actions])
    contact_actions = np.array([val for d in misinfo_actions for val in d.values()])

    # Cyberattack
    bot_counts = np.zeros(n_cyber_actions+1, dtype=np.int64)
    if np.any(attack_actions != 0):
        index = np.where(attack_actions != 0)
        attack_actions = attack_actions[index]
        unique_actions1, action_counts1 = np.unique(attack_actions, return_counts=True)
        bot_counts[unique_actions1] = action_counts1
    # Misinfo
    misinfo_counts = np.zeros(n_misinfo_actions+1, dtype = np.int64)
    if np.any(contact_actions != -1):
        index = np.where(contact_actions != -1)
        contact_actions = contact_actions[index]
        unique_actions2, action_counts2 = np.unique(contact_actions, return_counts=True)
        misinfo_counts[unique_actions2] = action_counts2

    stage_actions = np.array([action[0] for actions in all_attack_actions.values() for action in actions])
    episode_attack_order, episode_count_actions_per_stage = determine_attack_order_and_length(stage_actions)
    return bot_counts, misinfo_counts, episode_attack_order, episode_count_actions_per_stage

def determine_attack_order_and_length(stage_actions):
    # Stage actions is an array of actions made by the attacker in each attack stage that determines how attacker 'moves' between stages
    # Actions
    # Attack stage
    values_to_stop = 7 # terminate action
    for index, element in enumerate(stage_actions):
        if element == values_to_stop:
            stage_actions = stage_actions[:index+1]
            break

    episode_attack_order = np.zeros(4+2, dtype = int)
    episode_count_actions_per_stage = np.zeros(4+2, dtype = int)
    for action in stage_actions:
        # Create timeline of actions
        max_value = max(episode_attack_order)
        if action in [0] and episode_attack_order[0] == 0:
            episode_attack_order[0] = max_value + 1
        elif action in [1]:
            # First time cyberattack
            if episode_attack_order[1] == 0: episode_attack_order[1] = max_value + 1
            # Return to cyberattack
            else: episode_attack_order[4] = max_value + 1
        elif action in [2]:
            # First time misinformation
            if episode_attack_order[2] == 0: episode_attack_order[2] = max_value + 1
            # Return to misinformation
            else: episode_attack_order[5] = max_value + 1
        elif action in [3] and episode_attack_order[3] == 0:
            episode_attack_order[3] = max_value + 1
        # Count actions
        index = np.where(episode_attack_order== max(episode_attack_order))[0][0]
        if action not in [7]:
            episode_count_actions_per_stage[index] += 1
    # return stage_actions, attack_order, count_actions_per_stage
    return episode_attack_order, episode_count_actions_per_stage

def process_defender_rewards(episode_rewards, providers):
    defenders_rewards = {agent_name: value for agent_name, value in episode_rewards.items() if agent_name.startswith('defagent')}
    filter_rewards = np.zeros(len(providers))
    answer_rewards = np.zeros(len(providers))
    for defender in providers:
        filter_sum, answer_sum = 0, 0
        for _, (f, a) in enumerate(defenders_rewards[f"defagent{defender}"]):
            filter_sum += np.sum(f)
            answer_sum += np.sum(a)
        filter_rewards[defender] = filter_sum
        answer_rewards[defender] = answer_sum
    return filter_rewards, answer_rewards

def process_attack_strategy(attack_order, count_timesteps_per_stage):
    # Analyse attack order data
    # Initialise an empty list to collect data
    attack_order_data = []

    # Expand each tuple into a list of (step, attack_order) pairs
    for step, orders in attack_order:
        row = {'step': step}
        for i, order in enumerate(orders):
            row[f'attack_stage_{i}'] = order
        attack_order_data.append(row)

    # Create a DataFrame from the collected data
    attack_order_df = pd.DataFrame(attack_order_data)

    # Reshaping the DataFrame to have one row per step
    attack_order_df = attack_order_df.groupby('step').first().reset_index()

    # Fill NaN values with 0 and cast to integer
    attack_order_df = attack_order_df.fillna(0).astype(int)

    # Analyse attack actions count data
    attack_count_data = []

    for step, counts in count_timesteps_per_stage:
        row = {'step': step}
        for i, count in enumerate(counts):
            row[f'attack_stage_{i}'] = count
        attack_count_data.append(row)
    
    # Create a DataFrame from the collected data
    count_attack_actions_df = pd.DataFrame(attack_count_data)

    # Reshaping the DataFrame to have one row per step
    count_attack_actions_df = count_attack_actions_df.groupby('step').first().reset_index()

    # Fill NaN values with 0 and cast to integer
    count_attack_actions_df = count_attack_actions_df.fillna(0).astype(int)

    return attack_order_df, count_attack_actions_df

def process_regagent_states(all_observations, step, n_steps):
    # Determine the number of columns based on the structure of 'all_observations'
    columns = [f"{agent}_trust_in_sp{j}" for agent in all_observations if agent.startswith('regagent') for j in range(len(next(iter(all_observations.values()))[0]))]

    # Pre-allocate DataFrame
    episode_states_pd = pd.DataFrame(index=range(n_steps + 1), columns=columns)
    
    # Fill the DataFrame
    for agent, values in all_observations.items():
        if agent.startswith('regagent'):
            for timestep, trust_values in enumerate(values):
                for sp_index, trust in enumerate(trust_values):
                    column_name = f"{agent}_trust_in_sp{sp_index}"
                    episode_states_pd.at[timestep, column_name] = trust

    # Add 'episode' and 'timestep' columns
    episode_states_pd.insert(0, 'episode', step)
    episode_states_pd.insert(1, 'timestep', range(n_steps + 1))

    return episode_states_pd