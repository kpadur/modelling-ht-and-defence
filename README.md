# Code for "Modelling hybrid threats and defensive countermeasures"
## Introduction
Hybrid attacks are a growing concern due to their potential to undermine trust and disrupt social stability. Modelling attacker and defender behaviour is important for identifying potential attack strategies and determining effective countermeasures. In our approach both attackers and defenders are deep reinforcement learning agents that adapt their strategies against their opponents during training. We demonstrate the feasibility of this MARL approach in deriving stable policies for both agents.

## Project structure
```
modelling-ht-and-defence/
├── train_marl.py        # Experiment for training attackers and defenders (experiment 2)
├── environment.py       # Environment setup
├── a2c_agent.py         # Regular agents' behaviour in the environment
├── a2c_def_agent.py     # Defenders' behaviour in the environment
├── a2c_mal_agent.py     # Attackers' behaviour in the environment
├── nns.py               # Architecture of deep neural networks
├── data_analysis.txt    # Functions to analyse the training results
├── other_functions.txt  # Functions to visualise the training results
└── README.md            # Project documentation
```
## Usage
```
# Run the main file
python train_marl.py
```
