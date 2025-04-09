# Exp 2 data analysis and visualisation
# Load libraries
library(readr)
library(ggplot2)
library(zoo)
library(dplyr)
library(tidyr)

#####
# Attacker rewards: Analyse the effectiveness of attackers' performance
#####
# Import data
path <- "exp2-results/"
all_files <- list.files(path, pattern = "*attacker-data.csv", full.names = TRUE)

# Calculate metrics
stabilised_episode = 11148
number_of_episodes = 20000
stabilised_period = number_of_episodes-stabilised_episode

# Rewards: Cumulative reward over stabilised episodes
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

# Define the window size for the moving average (0 to get original results)
window_size <- 10

# Apply moving average to each column
smoothed_total_reward_data <- as.data.frame(lapply(combined_total_reward_data, function(column) {
  rollmean(column, window_size, fill = NA, align = "right")
}))

# Calculate mean, sd, and variance of the smoothed data
episode_mean_total_rewards <- apply(smoothed_total_reward_data, 1, mean, na.rm = TRUE)
episode_sd_total_rewards <- apply(smoothed_total_reward_data, 1, sd, na.rm = TRUE)
episode_variance_total_rewards <- apply(smoothed_total_reward_data, 1, var, na.rm = TRUE)

# Create summary data frame
summary_total_rewards_data <- data.frame(
  episode = 1:length(episode_mean_total_rewards),
  mean = episode_mean_total_rewards,
  sd = episode_sd_total_rewards,
  variance = episode_variance_total_rewards
)

# Read 'recon rewards' column from each file and combine
recon_reward_data <- lapply(all_files, function(file) {
  df <- read_csv(file,show_col_types = FALSE)
  return(df$recon_rewards)
})
names(recon_reward_data) <- sapply(all_files, function(f) paste0("repetition_", basename(f)))
combined_recon_reward_data <- as.data.frame(recon_reward_data)

# Apply moving average to each column
smoothed_recon_reward_data <- as.data.frame(lapply(combined_recon_reward_data, function(column) {
  rollmean(column, window_size, fill = NA, align = "right")
}))

# Calculate mean, sd, and variance of the smoothed data
episode_mean_recon_rewards <- apply(smoothed_recon_reward_data, 1, mean, na.rm = TRUE)
episode_sd_recon_rewards <- apply(smoothed_recon_reward_data, 1, sd, na.rm = TRUE)
episode_variance_recon_rewards <- apply(smoothed_recon_reward_data, 1, var, na.rm = TRUE)

summary_recon_rewards_data <- data.frame(
  episode = 1:length(episode_mean_recon_rewards),
  mean = episode_mean_recon_rewards,
  sd = episode_sd_recon_rewards,
  variance = episode_variance_recon_rewards
)

# Read 'cyber attack rewards' column from each file and combine
cyber_reward_data <- lapply(all_files, function(file) {
  df <- read_csv(file,show_col_types = FALSE)
  return(df$cyber_attack_rewards)
})
names(cyber_reward_data) <- sapply(all_files, function(f) paste0("repetition_", basename(f)))
combined_cyber_reward_data <- as.data.frame(cyber_reward_data)

# Apply moving average to each column
smoothed_cyber_reward_data <- as.data.frame(lapply(combined_cyber_reward_data, function(column) {
  rollmean(column, window_size, fill = NA, align = "right")
}))

# Calculate mean, sd, and variance of the smoothed data
episode_mean_cyber_rewards <- apply(smoothed_cyber_reward_data, 1, mean, na.rm = TRUE)
episode_sd_cyber_rewards <- apply(smoothed_cyber_reward_data, 1, sd, na.rm = TRUE)
episode_variance_cyber_rewards <- apply(smoothed_cyber_reward_data, 1, var, na.rm = TRUE)

summary_cyber_rewards_data <- data.frame(
  episode = 1:length(episode_mean_cyber_rewards),
  mean = episode_mean_cyber_rewards,
  sd = episode_sd_cyber_rewards,
  variance = episode_variance_cyber_rewards
)

# Read 'disinfo rewards' column from each file and combine
disinfo_reward_data <- lapply(all_files, function(file) {
  df <- read_csv(file,show_col_types = FALSE)
  return(df$disinfo_rewards)
})
names(disinfo_reward_data) <- sapply(all_files, function(f) paste0("repetition_", basename(f)))
combined_disinfo_reward_data <- as.data.frame(disinfo_reward_data)

# Apply moving average to each column
smoothed_disinfo_reward_data <- as.data.frame(lapply(combined_disinfo_reward_data, function(column) {
  rollmean(column, window_size, fill = NA, align = "right")
}))

# Calculate mean, sd, and variance of the smoothed data
episode_mean_disinfo_rewards <- apply(smoothed_disinfo_reward_data, 1, mean, na.rm = TRUE)
episode_sd_disinfo_rewards <- apply(smoothed_disinfo_reward_data, 1, sd, na.rm = TRUE)
episode_variance_disinfo_rewards <- apply(smoothed_disinfo_reward_data, 1, var, na.rm = TRUE)

summary_disinfo_rewards_data <- data.frame(
  episode = 1:length(episode_mean_disinfo_rewards),
  mean = episode_mean_disinfo_rewards,
  sd = episode_sd_disinfo_rewards,
  variance = episode_variance_disinfo_rewards
)

# Read 'term rewards' column from each file and combine
term_reward_data <- lapply(all_files, function(file) {
  df <- read_csv(file,show_col_types = FALSE)
  return(df$term_rewards)
})
names(term_reward_data) <- sapply(all_files, function(f) paste0("repetition_", basename(f)))
combined_term_reward_data <- as.data.frame(term_reward_data)

# Apply moving average to each column
smoothed_term_reward_data <- as.data.frame(lapply(combined_term_reward_data, function(column) {
  rollmean(column, window_size, fill = NA, align = "right")
}))

# Calculate mean, sd, and variance of the smoothed data
episode_mean_term_rewards <- apply(smoothed_term_reward_data, 1, mean, na.rm = TRUE)
episode_sd_term_rewards <- apply(smoothed_term_reward_data, 1, sd, na.rm = TRUE)
episode_variance_term_rewards <- apply(smoothed_term_reward_data, 1, var, na.rm = TRUE)

summary_term_rewards_data <- data.frame(
  episode = 1:length(episode_mean_term_rewards),
  mean = episode_mean_term_rewards,
  sd = episode_sd_term_rewards,
  variance = episode_variance_term_rewards
)

# Visualise rewards
a1 <- ggplot() +
  geom_line(data = summary_total_rewards_data, aes(x = episode, y = mean, color = "Rewards score")) +
  geom_ribbon(data = summary_total_rewards_data, aes(x = episode, ymin = mean - sd, ymax = mean + sd), 
              fill = "#FF99CC", alpha = 0.2) +
  
  geom_line(data = summary_recon_rewards_data, aes(x = episode, y = mean, color = "Rewards for reconnaissance")) +
  geom_ribbon(data = summary_recon_rewards_data, aes(x = episode, ymin = mean - sd, ymax = mean + sd),
              fill = "#99CCFF", alpha = 0.2) +
  
  geom_line(data = summary_cyber_rewards_data, aes(x = episode, y = mean, color = "Rewards for a cyberattack")) +
  geom_ribbon(data = summary_cyber_rewards_data, aes(x = episode, ymin = mean - sd, ymax = mean + sd),
              fill = "#FFCC66", alpha = 0.2) +
  
  geom_line(data = summary_disinfo_rewards_data, aes(x = episode, y = mean, color = "Rewards for disinformation")) +
  geom_ribbon(data = summary_disinfo_rewards_data, aes(x = episode, ymin = mean - sd, ymax = mean + sd),
              fill = "#CCFFCC", alpha = 0.2) +
  
  geom_line(data = summary_term_rewards_data, aes(x = episode, y = mean, color = "Rewards for termination")) +
  geom_ribbon(data = summary_term_rewards_data, aes(x = episode, ymin = mean - sd, ymax = mean + sd),
              fill = "#CCCCCC", alpha = 0.2) +
  
  scale_color_manual(values = c("Rewards score" = "#CC0033", 
                                "Rewards for reconnaissance" = "#0033CC",
                                "Rewards for a cyberattack" = "#FF9900",
                                "Rewards for disinformation" = "#006633",
                                "Rewards for termination" = "#666666")) +
  labs(x = "Episode", y = "Average cumulative reward for attackers\n(across 100 simulations)", color = NULL) +
  theme_bw() +
  theme(
    legend.position = c(0.20, 0.70),
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
    panel.grid.major = element_line(color = "grey", linewidth = 0.6), 
    panel.grid.minor = element_line(color = "lightgrey", linewidth = 0.5)
  )
a1
