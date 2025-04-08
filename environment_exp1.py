from gymnasium.spaces import Discrete, Tuple, Box
import numpy as np
import networkx as nx
import torch
import random
from pettingzoo import AECEnv
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib import colormaps

class Environment(AECEnv):
    """
    This environment represents a cyber-physical-social system (CPSS).
    """
    
    def __init__(self, nRegAgents, nProviders,
                kappa, rho, center_up_to_down, center_down_to_up, end_up_to_down, end_down_to_up, cost,
                direct_exp_weight, feedback_adj_rate, forgetting_factor):
        """
        The init method takes in environment arguments and creates the environment.

        Attributes are not changed after initialisation.
        """
        self.nRegAgents = nRegAgents
        self.nAgents = nRegAgents
        self.nProviders = nProviders
        self.kappa = kappa
        self.rho = rho
        self.center_up_to_down = center_up_to_down
        self.center_down_to_up = center_down_to_up
        self.end_up_to_down = end_up_to_down
        self.end_down_to_up = end_down_to_up
        self.cost = cost
        self.direct_exp_weight = direct_exp_weight
        self.feedback_adj_rate = feedback_adj_rate
        self.forgetting_factor = forgetting_factor

        # Create social network as a graph (G = (V, E)) and neighbours dictionary
        self.social_network = nx.Graph()
        self._form_social_network()

        # Create lists of agent ids (integers)
        self.sn_agents, self.regagents, self.providers = [], [], []
        self._create_agents()

        # Generate separate lists with agent names
        self.agents = [f"regagent{r}" for r in self.regagents]
        self.possible_agents = self.agents.copy() # all agents that may appear in the environment

        # Initialise regular agents information
        self.opinions_dict, self.opinion_pairs, self.positive_opinions = {}, {}, []
        self._create_regular_agents_info()

        # Observation and action spaces for regular agents  
        self.observation_spaces = {**{f"regagent{ragent}": Box(low=0.0, high=1.0, shape=(self.nProviders,), dtype=np.float32) for ragent in self.regagents}} # trust values
        self.action_spaces = {**{f"regagent{ragent}": Tuple((Discrete(self.nProviders), Discrete(self.nProviders*2))) for ragent in self.regagents}} # service providers, opinions

    def reset(self, seed=None, options=None):
        """
        Resets the environment.
        Returns a dictionary of observations and infos (keyed by the agent name)
        """
        # Seed env
        if seed is not None:
            np.random.seed(seed); random.seed(seed); torch.manual_seed(seed)

        # Initialise timestep
        self.timestep = 1

        # Reset CPS state
        # NOTE: (Assumption) all providers central and endpoint states are initially available
        self.center = {f"defagent{provider}": 1 for provider in self.providers} 
        self.endpoint = {f"regagent{agent}": [1 for _ in range(self.nProviders)] for agent in self.sn_agents}

        # Reset agents
        self.agents = self.possible_agents.copy()

        # Reset regular agents' information
        self.action_taken = {**{f"regagent{ragent}": None for ragent in self.regagents}} # no actions taken yet
        self.opinion_expressed = {**{f"regagent{ragent}": None for ragent in self.regagents}} # no actions taken yet
        self.service_received = {**{f"regagent{ragent}": [[] for _ in range(self.nProviders)] for ragent in self.regagents}} # no service received yet
        self.feedback_received = {**{f"regagent{ragent}": [[] for _ in range(self.nProviders*2)] for ragent in self.regagents}} # no feedback values yet

        # Reset agents' observations
        # NOTE: (Assumption) regular agents have no previous information about providers (0.5 trust values)
        self.observations = {**{f"regagent{ragent}": [0.5] * self.nProviders for ragent in self.regagents}}

        # Get dummy infos. Necessary for proper environment testing
        self.terminations = {a: False for a in self.agents}
        self.truncations = {a: False for a in self.agents}
        self.infos = {a: {} for a in self.agents}
        return self.observations, self.infos

    def step(self, actions):
        """
        Receives a dictionary of actions keyed by the agent name. 
        Returns observations, rewards, terminations, truncations, 
        and infos dictionaries keyed by the agent name.
        """
        # Create (new) agent pairs for information spreading
        self.pairs = self._create_pairs_from_graph()

        # Calculate average situational trust
        self.average_situational_trust = self._calculate_average_situational_trust()

        # Give rewards to agents
        rewards = {**{f"regagent{agent}": (self._calculate_service_reward(f"regagent{agent}", actions), 
                                           self._calculate_feedback_reward(f"regagent{agent}", actions)) 
                for agent in self.regagents}}

        # Update observations for agents
        self.observations = {**{f"regagent{agent}": self._calculate_situational_trust(self.service_received[f"regagent{agent}"], 
                                                                                      self.feedback_received[f"regagent{agent}"]) 
               for agent in self.regagents}}
        
        # Update cps state (center, endpoint)
        self._update_cps_state()
        
        # Update timestep
        self.timestep += 1
        
        # Get dummy infos. Necessary for proper parallel_to_aec conversion
        self.terminations = {a: False for a in self.agents}
        self.truncations = {a: False for a in self.agents}
        self.infos = {a: {} for a in self.agents}

        return self.observations, rewards, self.terminations, self.truncations, self.infos

    def render(self, graph_type='both'):
        """
        Displays rendered frames from the environment for actions and/or opinions.
        Parameters:
        - graph_type: 'both', 'actions', or 'opinions' to control which graph(s) to display.
        """
        if graph_type not in ['both', 'actions', 'opinions']:
            raise ValueError("Invalid graph_type. Choose 'both', 'actions', or 'opinions'.")

        # Create subplots conditionally
        if graph_type == 'both':
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        else:
            fig, ax = plt.subplots(1, 1, figsize=(8, 8))

        # Create service provider (action) color mapping
        action_colormap = plt.cm.get_cmap(colormaps['tab10'], len(self.providers)) 
        # Create a new graph for visualisation purposes
        combined_graph = nx.Graph(self.social_network)  # Copy the original social network

        # Define colors for the original nodes
        node_colors_actions = {node: 'gold' if node in self.regagents else 'red' for node in self.social_network.nodes}
        node_colors_opinions = node_colors_actions.copy()
        # Adding n new separate nodes (representing service providers) with no connections
        extra_nodes = [f'd_{provider}' for provider in self.providers]

        # Manually position the extra nodes using a circular layout
        pos = nx.spring_layout(self.social_network, k=0.5, seed=42)
        angle_increment = 1.75 * np.pi / len(extra_nodes)
        radius = 1.25  # Distance from the center of the plot

        for i, node in enumerate(extra_nodes):
            angle = i * angle_increment
            pos[node] = [radius * np.cos(angle), radius * np.sin(angle)]

        # Add extra nodes to the graph and assign colors
        for i, node in enumerate(extra_nodes):
            combined_graph.add_node(node)
            color = action_colormap(i)  # Use 'tab10' color directly
            node_colors_actions[node] = color
            node_colors_opinions[node] = color  # Ensure opinions have the same colors

        action_opinion_colors = {}
        # Assign colors for actions and lighten for negative opinions
        for action in range(self.nProviders):
            action_color = to_rgba(action_colormap(action))
            # Lighten the action color for negative opinion
            lightened_color = tuple((1 - 0.5) * c + 0.5 for c in action_color[:3]) + (action_color[3],)
            action_opinion_colors[action] = {
                'positive': action_color,
                'negative': lightened_color
            }
        
        # Render during simulation
        if self.timestep > 1:
            for agent in self.regagents:
                agent_name = f"regagent{agent}"
                action = self.action_taken.get(agent_name)
                opinion = self.opinion_expressed.get(agent_name)

                if action is not None and opinion is not None:
                    node_colors_actions[agent] = action_opinion_colors[action]['positive']

                    if opinion in self.positive_opinions:
                        node_colors_opinions[agent] = action_opinion_colors[action]['positive']
                    else:
                        node_colors_opinions[agent] = action_opinion_colors[action]['negative']

        # Plot for actions if requested
        if graph_type in ['both', 'actions']:
            ax = ax1 if graph_type == 'both' else ax  # Choose correct axis if 'both' or 'actions'
            edge_colors = [combined_graph[u][v].get('color', 'gray') for u, v in combined_graph.edges]
            edge_styles = [combined_graph[u][v].get('style', 'solid') for u, v in combined_graph.edges]

            nx.draw_networkx_edges(combined_graph, pos=pos, edgelist=combined_graph.edges, edge_color=edge_colors, style=edge_styles, alpha=0.6, ax=ax)
            nx.draw_networkx_nodes(combined_graph, pos=pos, node_color=[node_colors_actions.get(node, 'mediumvioletred') for node in combined_graph.nodes], \
                                   node_size=[300 if node in self.social_network.nodes else 2000 for node in combined_graph.nodes], ax=ax)
            ax.text(0, 1, "A", transform=ax.transAxes, fontsize=20, verticalalignment='top')
            ax.axis('off')

        # Plot for opinions if requested
        if graph_type in ['both', 'opinions']:
            ax = ax2 if graph_type == 'both' else ax  # Choose correct axis if 'both' or 'opinions'
            edge_colors = [combined_graph[u][v].get('color', 'gray') for u, v in combined_graph.edges]
            edge_styles = [combined_graph[u][v].get('style', 'solid') for u, v in combined_graph.edges]

            nx.draw_networkx_edges(combined_graph, pos=pos, edgelist=combined_graph.edges, edge_color=edge_colors, style=edge_styles, alpha=0.6, ax=ax)
            nx.draw_networkx_nodes(combined_graph, pos=pos, node_color=[node_colors_opinions.get(node, 'mediumvioletred') for node in combined_graph.nodes], \
                                   node_size=[300 if node in self.social_network.nodes else 2000 for node in combined_graph.nodes], ax=ax)
            ax.text(0, 1, "B", transform=ax.transAxes, fontsize=20, verticalalignment='top')
            ax.axis('off')

        plt.tight_layout()
        return fig  # Return the figure without showing it

    @property
    def num_agents(self) -> int: # length of the agent list
        return len(self.agents)

    @property
    def max_num_agents(self) -> int: # length of possible agents list
        return len(self.possible_agents)

    def _form_social_network(self):
        """
        Creates social network graph as a small-world network with high clustering and low average path length,
        here, Watts-Strogatz graph.
        """
        self.graph = nx.watts_strogatz_graph(self.nAgents, self.kappa, self.rho)
        self.social_network.add_nodes_from(self.graph)
        self.social_network.add_edges_from(self.graph.edges)

    def _create_agents(self):
        """
        Creates lists of regular agents, malicious agents (attackers), and service providers (defenders)
        """
        # Create list of all agents
        self.sn_agents = [agent for agent in range(self.nAgents)]
        
        # Specify list of regular agents
        self.regagents = self.sn_agents.copy()

        # Create list of service providers
        self.providers = [provider for provider in range(self.nProviders)]

    def _create_regular_agents_info(self):
        """
        Create opinions dictionery, e.g., {0:[-1,0], 1:[1,0], 2:[-1,1], 3:[1,1], 4:[-1,2], 5:[1,2]},
        opinion pairs, e.g., {0:1, 1:0, 2:3, 3:2, 4:5, 5:4}, and positiove opinions, e.g., [1,3,5]
        """
        # Opinions dictionary
        for i in range(self.nProviders*2):
            if i%2==0: self.opinions_dict[i] = [-1, i//2]
            else: self.opinions_dict[i] = [1, i//2]
        # Opinion pairs
        for j in range(0, self.nProviders*2, 2):
            self.opinion_pairs[j] = j+1
            self.opinion_pairs[j+1] = j
        # Generate list of positive opinions
        for k, v in self.opinions_dict.items():
            if v[0] == 1: self.positive_opinions.append(k)

    def _create_pairs_from_graph(self):
        """
        Determine agent pairs from social network graph that can interact with each other
        """
        nodes = list(self.social_network.nodes())
        pairs = []
        # Copy the list of all nodes to remaining nodes to be able to modify it
        remaining_nodes = nodes.copy()

        # Randomise remaining_nodes (instead of the original nodes list)
        random.shuffle(remaining_nodes)
        # While there are nodes left in the remaining_nodes list
        while remaining_nodes:
            node = remaining_nodes.pop()
            neighbors = list(self.social_network.neighbors(node))
            random.shuffle(neighbors)

            for neighbor in neighbors:
                if neighbor in remaining_nodes:
                    pairs.append((node, neighbor))
                    remaining_nodes.remove(neighbor)
                    break
        return pairs

    def _get_interaction(self, agent):
        """
        Determine the agent's interaction partner
        return:     int, agent's interaction partner
        """
        interaction_dict = dict(self.pairs)
        # Check if the agent is a key in the dictionary
        if agent in interaction_dict:
            return interaction_dict[agent]
        # Check if the agent is a value in the dictionary
        for key, value in interaction_dict.items():
            if value == agent:
                return key
        # If not found, return -1
        return -1

    def _update_cps_state(self):
        """
        Updates service providers' behaviour based on Markov model for the next time step;
        center and endpoint as dictionaries 
        """
        for provider in self.providers:
            provider_name = f"defagent{provider}"
            # Determine provider's central transition matrix (from 0 to 0, from 0 to 1, from 1 to 0, from 1 to 1)
            center_matrix = [[1 - self.center_down_to_up[provider], self.center_down_to_up[provider]],\
                             [self.center_up_to_down[provider], 1 - self.center_up_to_down[provider]]]

            # Determine the next state based on the transition probabilities
            center_state = self.center[provider_name]
            next_center = random.choices([0, 1], weights=center_matrix[center_state])[0]
            self.center[provider_name] = next_center

        for provider in self.providers:
            provider_name = f"defagent{provider}"
            # Determine provider's endpoint transition matrix (from 0 to 0, from 0 to 1, from 1 to 0, from 1 to 1)
            end_matrix = [[1 - self.end_down_to_up[provider], self.end_down_to_up[provider]],\
                          [self.end_up_to_down[provider], 1 - self.end_up_to_down[provider]]]

            for agent in self.sn_agents: # Generate service for each agent
                agent_name = f"regagent{agent}"
                # Generate service for each agent
                if self.center[provider_name] == 1: 
                    # Generate endpoint only when center is available
                    endpoint_state = self.endpoint[agent_name][provider]
                    next_endpoint = random.choices([0, 1], weights=end_matrix[endpoint_state])[0]
                    self.endpoint[agent_name][provider] = next_endpoint
                else:
                    self.endpoint[agent_name][provider] = 0

    def _calculate_service_reward(self, agent_name, actions):
        """
        Calculates reward for regular agent based on service and cost
        return:     float
        """
        provider = actions[agent_name][0] # agent's action (sp id) for this time step

        # Determine provider's endpoint state for this specific agent
        endpoint_state = self.endpoint[agent_name][provider] # 1: available, 0: unavailable

        # Check if service request was unavailable
        if endpoint_state == 1:
            service_reward = 1
        else:
            service_reward = -1

        # Determine cost associated with the action
        if self.action_taken[agent_name] is None: # agent has not taken any actions yet
            reward = service_reward
        elif self.action_taken[agent_name] == provider: # agent stayed with the same provider
            reward = service_reward + self.cost[provider]
        else: # agent changed provider
            reward = service_reward - self.cost[provider]

        # Change action to the most recent action
        self.action_taken[agent_name] = provider

        # Save requested service state and time step for this agent
        self.service_received[agent_name][provider].append((self.timestep, service_reward))
        return float(reward)

    def _calculate_feedback_reward(self, agent_name, actions):
        """
        Calculates reward for regular agent based on feedback from neighbour
        return:     float
        """
        # Determine interacting agents
        agent = int(agent_name.replace("regagent", ""))  # agent ID
        neighbour = self._get_interaction(agent)  # neighbour ID
        opinion = actions[agent_name][1]  # Agent's (expressed) opinion

        # Identify the service provider associated with the expressed opinion
        agent_opinion_value = self.opinions_dict[opinion][0]  # Opinion value (-1 or 1)
        agent_opinion_sp = self.opinions_dict[opinion][1]  # Service provider ID
        self.opinion_expressed[agent_name] = opinion  # Save opinion for tracking

        # Determine feedback from the neighbour
        if neighbour == -1:  # Agent has no contact this timestep
            feedback = 0
        else:
            neighbour_name = f"regagent{neighbour}"
            neighbour_trust = self.observations[neighbour_name][agent_opinion_sp]  # neighbour's trust for this provider
            # Determine feedback based on dynamic trust threshold
            if neighbour_trust >= self.average_situational_trust[agent_opinion_sp]:
                feedback = 1 * agent_opinion_value # neighbour has pos opinion
            else:
                feedback = -1 * agent_opinion_value # neighbour has neg opinion
        
        # Give additional support for opinion that agent itself has strong direct experience with
        if actions[agent_name][0] == agent_opinion_sp:
            support = feedback + self.feedback_adj_rate
        else:
            support = feedback - self.feedback_adj_rate

        # Save feedback received
        if feedback != 0:
            self.feedback_received[agent_name][opinion].append((self.timestep, feedback))
        return float(support)
    
    def _calculate_situational_trust(self, service_states, feedback_values):
        """
        Calculates situational trust for regular agent based on service states.
        The objective of a trust assessment mechanism is to allow the best agent selection
        in the light of the uncertainties linked to dynamic agent behaviour.
        return:     float
        """
        current_time = self.timestep

        # Analyse direct experiences
        pos_exps = np.zeros(self.nProviders, dtype = float)
        neg_exps = np.zeros(self.nProviders, dtype = float)

        for provider in range(self.nProviders):
            service_states[provider].sort(key=lambda x: x[0]) # should be sorted already

            # Separate positive experiences from negative experiences
            for i, reward in service_states[provider]:
                if reward > 0:
                    pos_exps[provider] += 1 * self.forgetting_factor**(current_time-i)
                else:
                    neg_exps[provider] += 1 * self.forgetting_factor**(current_time-i)
        # Calculate direct experience score for all service providers
        direct_exp_score = (pos_exps + 1)/(pos_exps + neg_exps + 2)

        # Analyse witness information
        pos_wit = np.zeros(self.nProviders, dtype = float)
        neg_wit = np.zeros(self.nProviders, dtype = float)

        for opinion in self.positive_opinions:
            alt_opinion = self.opinion_pairs[opinion]
            provider = self.opinions_dict[opinion][1]

            # Separate positive feedback from negative feedback
            for i, reward in feedback_values[opinion]:
                if reward > 0:
                    pos_wit[provider] += 1 * self.forgetting_factor**(current_time-i)
                else:
                    neg_wit[provider] += 1 * self.forgetting_factor**(current_time-i)
            # Deal with alternative opinion
            for i, reward in feedback_values[alt_opinion]:
                if reward > 0:
                    neg_wit[provider] += 1 * self.forgetting_factor**(current_time-i)
                else:
                    pos_wit[provider] += 1 * self.forgetting_factor**(current_time-i)
        # Calculate feedback score for all service providers
        feedback_score = (pos_wit + 1)/(pos_wit + neg_wit + 2)

        # Calculate situational trust
        situational_trust = self.direct_exp_weight * direct_exp_score + (1 - self.direct_exp_weight) * feedback_score
        return situational_trust
    
    def _calculate_average_situational_trust(self):
        """
        Calculate average situational trust and dynamic thresholds for each service provider.
        Returns:
            mean_trusts (np.ndarray): Mean trust values for each provider.
            dynamic_thresholds (np.ndarray): Dynamic thresholds for each provider.
        """
        # Initialise arrays to collect trust values for each provider
        situational_trust_values = [[] for _ in range(self.nProviders)]

        # Collect situational trust values for each provider
        for agent_trust in self.observations.values():
            for provider in range(self.nProviders):
                situational_trust_values[provider].append(agent_trust[provider])

        # Compute mean and standard deviation for each provider
        mean_trusts = np.zeros(self.nProviders, dtype=float)
        std_trusts = np.zeros(self.nProviders, dtype=float)
        for provider in range(self.nProviders):
            trust_values = situational_trust_values[provider]
            mean_trusts[provider] = np.mean(trust_values)
            std_trusts[provider] = np.std(trust_values)

        # Compute dynamic thresholds
        dynamic_thresholds = mean_trusts - std_trusts

        return dynamic_thresholds
