import pandas as pd
import numpy as np

def moving_average(data): ##
    window_size = 100
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def process_regagent_rewards(episode_rewards): ##
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

def process_attacker_rewards(all_actions, episode_rewards): ##
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

def process_defender_rewards(episode_rewards, providers): ##
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
