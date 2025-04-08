# Code for "Modelling hybrid threats and defensive countermeasures"
## Introduction
Hybrid threats are a growing concern due to their potential to undermine public trust and disrupt social stability. Few public examples exist, but the risks of a successful attack are high, so modelling attacker and defender behaviours is necessary to identify potential attack strategies and determine effective countermeasures. Traditional game-theoretic methods, commonly used for this purpose, can struggle to capture the complexity and dynamic nature of these types of emerging threats. To address this problem, we propose a novel approach in which both attackers and defenders use RL to devise strategies against their opponents.
## Project structure
```
modelling-ht-and-defence/
├── train_regular_agents.py          # Experiment for training regular agents (experiment 1)
├── train_marl.py                    # Experiment for training attackers and defenders (experiment 2)
├── environment_exp1.py              # Environment setup for experiment 1
├── environment_exp2.py              # Environment setup for experiment 2
├── a2c_agent.py                     # Regular agents' behaviour in the environment
├── a2c_def_agent.py                 # Defenders' behaviour in the environment
├── a2c_mal_agent.py                 # Attackers' behaviour in the environment
├── nns.py                           # Architecture of deep neural networks
├── data_analysis.py                 # Functions to analyse training data
├── save_data.py                     # Functions to save data
├── parameters                       # Folder containing data and R scripts for hyperparameter tuning
    ├── hyperparameters.csv          # (Tuned) hyperparameter values
    └── parameters.csv               # Parameter values
├── regagent-parameters              # Folder containing trained regular agents neural networks
├── results                          # Folder containing R scripts for data analysis and visualisation
    ├── exp1-results                 # Data collected from experiment 1
    ├── exp2-results                 # Data collected from experiment 2
    └── plots                        # Figures
├── R-scripts                        # Folder containing R scripts for data analysis and visualisation
    ├── ch3-exp1-analysis.R          # R script for analysing and visualising experiment 1 data
    ├── ch3-exp2-attack-analysis.R   # R script for analysing and visualising attacker behaviour
    └── ch3-exp2-defence-analysis.R  # R script for analysing and visualising defender behaviour
├── LICENSE.md                       # License
└── README.md                        # Project documentation
```
## License
MIT
## Prerequisites
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

R version 4.3.1 or higher version is required for data analysis and visualisation.

The following R libraries are required:
- readr (version 2.1.4 or higher)
- ggplot2 (version 3.4.4 or higher)
- zoo (version 1.8 or higher)
```
## Run experiments
```
# Run experiment 1
python train_regular_agents.py
# Run experiment 2
python train_marl.py
```
## Results
Results will be saved to ```/results``` directory. Experiment 1 results are located in ```/results/exp1-results``` and experiment 2 results are located in ```/results/exp2-results```.

Plots generated with R scripts using data generated during training are located in ```/results/plots```.

Directory ```/regagent-parameters``` contains trained deep neural networks from experiment 1 that are used in experiment 2.

## Contact
For any questions or issues, please feel free to contact [kart.padur.20@ucl.ac.uk] and I will be happy to assist.
