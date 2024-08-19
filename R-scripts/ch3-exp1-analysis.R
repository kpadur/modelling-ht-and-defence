# Chapter 3
# Experiment 1
# Load libraries
library(readr)
library(ggplot2)

# Read all files and combine
path <- "/results/exp1-results/"
all_files <- list.files(path, pattern = "*regagents-data.csv", full.names = TRUE)

# Calculate metrics (from hyperparameters file)
stabilised_episode = 145
number_of_episodes = 500
stabilised_period = number_of_episodes-stabilised_episode

# Calculate cumulative reward over stabilised episodes
extract_stabilised_rw_values <- function(file) {
  data <- read_csv(file, show_col_types = FALSE)
  rw_values <- tail(data$total_rewards, stabilised_period)
  return(rw_values)
}
rw_values <- sapply(all_files, extract_stabilised_rw_values)
rw_result <- list(
  mean = mean(rw_values),
  sd = sd(rw_values), # Convergence
  min = min(rw_values),
  max = max(rw_values)
)

print(rw_result)

# Visualise rewards
# Read 'total rewards' column from each file and combine
total_reward_data <- lapply(all_files, function(file) {
  df <- read_csv(file,show_col_types = FALSE)
  return(df$total_rewards)
})
names(total_reward_data) <- sapply(all_files, function(f) paste0("repetition_", basename(f)))
combined_total_reward_data <- as.data.frame(total_reward_data)
episode_mean_total_rewards <- apply(combined_total_reward_data, 1, mean, na.rm = TRUE)
episode_sd_total_rewards <- apply(combined_total_reward_data, 1, sd, na.rm = TRUE)
episode_variance_total_rewards <- apply(combined_total_reward_data, 1, var, na.rm = TRUE)

summary_total_rewards_data <- data.frame(
  episode = 1:length(episode_mean_total_rewards),
  mean = episode_mean_total_rewards,
  sd = episode_sd_total_rewards,
  variance = episode_variance_total_rewards
)

# Read 'action rewards' column from each file and combine
action_reward_data <- lapply(all_files, function(file) {
  df <- read_csv(file,show_col_types = FALSE)
  return(df$action_rewards)
})
names(action_reward_data) <- sapply(all_files, function(f) paste0("repetition_", basename(f)))
combined_action_reward_data <- as.data.frame(action_reward_data)
episode_mean_action_rewards <- apply(combined_action_reward_data, 1, mean, na.rm = TRUE)
episode_sd_action_rewards <- apply(combined_action_reward_data, 1, sd, na.rm = TRUE)
episode_variance_action_rewards <- apply(combined_action_reward_data, 1, var, na.rm = TRUE)

summary_action_rewards_data <- data.frame(
  episode = 1:length(episode_mean_action_rewards),
  mean = episode_mean_action_rewards,
  sd = episode_sd_action_rewards,
  variance = episode_variance_action_rewards
)

# Read 'opinion rewards' column from each file and combine
opinion_reward_data <- lapply(all_files, function(file) {
  df <- read_csv(file,show_col_types = FALSE)
  return(df$opinion_rewards)
})
names(opinion_reward_data) <- sapply(all_files, function(f) paste0("repetition_", basename(f)))
combined_opinion_reward_data <- as.data.frame(opinion_reward_data)
episode_mean_opinion_rewards <- apply(combined_opinion_reward_data, 1, mean, na.rm = TRUE)
episode_sd_opinion_rewards <- apply(combined_opinion_reward_data, 1, sd, na.rm = TRUE)
episode_variance_opinion_rewards <- apply(combined_opinion_reward_data, 1, var, na.rm = TRUE)

summary_opinion_rewards_data <- data.frame(
  episode = 1:length(episode_mean_opinion_rewards),
  mean = episode_mean_opinion_rewards,
  sd = episode_sd_opinion_rewards,
  variance = episode_variance_opinion_rewards
)

# Visualise rewards
p1 <- ggplot() +
  geom_line(data = summary_total_rewards_data, aes(x = episode, y = mean, color = "Average cumulative reward")) +
  geom_ribbon(data = summary_total_rewards_data, aes(x = episode, ymin = mean - sd, ymax = mean + sd), 
              fill = "#FF99CC", alpha = 0.4) +
  
  geom_line(data = summary_action_rewards_data, aes(x = episode, y = mean, color = "Average cumulative reward for actions")) +
  geom_ribbon(data = summary_action_rewards_data, aes(x = episode, ymin = mean - sd, ymax = mean + sd),
              fill = "#FFCC66", alpha = 0.4) +
  
  geom_line(data = summary_opinion_rewards_data, aes(x = episode, y = mean, color = "Average cumulative reward for opinions")) +
  geom_ribbon(data = summary_opinion_rewards_data, aes(x = episode, ymin = mean - sd, ymax = mean + sd),
              fill = "#CCFFCC", alpha = 0.4) +
  
  scale_color_manual(values = c("Average cumulative reward" = "#CC0033", 
                                "Average cumulative reward for actions" = "#FF9900", 
                                "Average cumulative reward for opinions" = "#006633")) +
  labs(x = "Episode", y = "Average cumulative reward\n(across 50 simulations)", color = NULL) +
  theme_bw() +
  theme(
    legend.position = c(0.70, 0.75),
    plot.title = element_text(size = rel(1)),
    axis.title = element_text(size = rel(1)),
    axis.text = element_text(size = rel(1)),
    legend.title = element_text(size = rel(1)),
    legend.text = element_text(size = rel(0.80)),
    plot.margin = margin(1, 1, 1, 1, "cm"),
    axis.title.x = element_text(vjust = -1),
    axis.title.y = element_text(vjust = 1),
    legend.background = element_blank(),
    legend.key = element_blank(),
    panel.grid.major = element_line(color = "grey", linewidth = 0.5), 
    panel.grid.minor = element_line(color = "lightgrey", linewidth = 0.5) 
  )
p1

# Saved /ch3-model/results/plots/ch3-exp1-rewards.pdf as 5 x 7.50 (landscape)