# Code for "Modelling hybrid threats and defensive countermeasures"
## Introduction
Hybrid attacks are a growing concern due to their potential to undermine trust and disrupt social stability. Modelling attacker and defender behaviour is important for identifying potential attack strategies and determining effective countermeasures. In our approach both attackers and defenders are deep reinforcement learning agents that adapt their strategies against their opponents during training. We demonstrate the feasibility of this MARL approach in deriving stable policies for both agents.

## Project structure
```
modelling-ht-and-defence/
├── train_regular_agents.py          # Experiment for training regular agents (experiment 1)
├── train_marl.py                    # Experiment for training attackers and defenders (experiment 2)
├── environment.py                   # Environment setup
├── a2c_agent.py                     # Regular agents' behaviour in the environment
├── a2c_def_agent.py                 # Defenders' behaviour in the environment
├── a2c_mal_agent.py                 # Attackers' behaviour in the environment
├── nns.py                           # Architecture of deep neural networks
├── data_analysis.py                 # Functions to collect relevant training data
├── other_functions.py               # Functions to visualise the training results
├── hyperparameters.csv              # (Tuned) hyperparameter values
├── regagent-parameters              # Folder containing trained regular agents neural networks
├── R-scripts                        # Folder containing R scripts for data analysis and visualisation
    ├── ch3-exp1-analysis.R          # R script for regular agents' behaviour analysis
    ├── ch3-exp2-attack-analysis.R   # R script for attacker behaviour analysis
    ├── ch3-exp2-defence-analysis.R  # R script for defender behaviour analysis
└── README.md                        # Project documentation
```
## Prerequisits
```
Python 3.10 or higher version is required.

The following Python libraries are required:
- numpy (version 1.24.2 or higher)
- pandas (version 1.5.3 or higher)
- torch (version 1.13.1 or higher)
- matplotlib (version 3.7.0 or higher)
- gymnasium (version 0.29.1 or higher)
- networkx (version 3.0 or higher)
- pettingzoo (version 1.24.1 or higher)
```
## Run experiments
```
# Run experiment 1
python train_regular_agents.py
# Run experiment 2
python train_marl.py
```
