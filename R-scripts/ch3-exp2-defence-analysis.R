# Chapter 3
# Experiment 2

# Load libraries
library(readr)
library(ggplot2)

# Import data
path <- "/results/exp2-results"
all_files <- list.files(path, pattern = "*defender-data.csv", full.names = TRUE)

# Calculate metrics
stabilised_episode = 11148
number_of_episodes = 20000
stabilised_period = number_of_episodes-stabilised_episode

# Calculate cumulative reward over stabilised episodes
# NOTE: Replace with total_rewards_sp0, total_rewards_sp1, or total_rewards_sp2
total_reward_data <- function(file) {
  data <- read_csv(file, show_col_types = FALSE)
  sp_rw_values <- tail(data$total_rewards_sp2, stabilised_period) 
  return(sp_rw_values)
}
sp_rw_values <- sapply(all_files, total_reward_data)
sp_rw_result <- list(
  mean = mean(sp_rw_values),
  sd = sd(sp_rw_values), # Convergence
  min = min(sp_rw_values),
  max = max(sp_rw_values)
)

print(sp_rw_result)

# Visualise rewards
# Read 'total_rewards_sp0' column from each file
total_reward_data_sp1 <- lapply(all_files, function(file) {
  df <- read_csv(file,show_col_types = FALSE)
  return(df$total_rewards_sp0)
})
names(total_reward_data_sp1) <- sapply(all_files, function(f) paste0("repetition_", basename(f)))

# Combine them into one large data frame
combined_total_reward_data_sp1 <- as.data.frame(total_reward_data_sp1)
combined_total_reward_data_sp1[is.na(combined_total_reward_data_sp1)] <- 0

# Calculate sd and variance
episode_mean_total_rewards_sp1 <- apply(combined_total_reward_data_sp1, 1, mean, na.rm = TRUE)
episode_sd_total_rewards_sp1 <- apply(combined_total_reward_data_sp1, 1, sd, na.rm = TRUE)
episode_variance_total_rewards_sp1 <- apply(combined_total_reward_data_sp1, 1, var, na.rm = TRUE)

summary_total_rewards_data_sp1 <- data.frame(
  episode = 1:length(episode_mean_total_rewards_sp1),
  mean = episode_mean_total_rewards_sp1,
  sd = episode_sd_total_rewards_sp1,
  variance = episode_variance_total_rewards_sp1
)

# Read 'total_rewards_sp1' column from each file
total_reward_data_sp2 <- lapply(all_files, function(file) {
  df <- read_csv(file,show_col_types = FALSE)
  return(df$total_rewards_sp1)
})
names(total_reward_data_sp2) <- sapply(all_files, function(f) paste0("repetition_", basename(f)))

# Combine them into one large data frame
combined_total_reward_data_sp2 <- as.data.frame(total_reward_data_sp2)
combined_total_reward_data_sp2[is.na(combined_total_reward_data_sp2)] <- 0

# Calculate sd and variance
episode_mean_total_rewards_sp2 <- apply(combined_total_reward_data_sp2, 1, mean, na.rm = TRUE)
episode_sd_total_rewards_sp2 <- apply(combined_total_reward_data_sp2, 1, sd, na.rm = TRUE)
episode_variance_total_rewards_sp2 <- apply(combined_total_reward_data_sp2, 1, var, na.rm = TRUE)

summary_total_rewards_data_sp2 <- data.frame(
  episode = 1:length(episode_mean_total_rewards_sp2),
  mean = episode_mean_total_rewards_sp2,
  sd = episode_sd_total_rewards_sp2,
  variance = episode_variance_total_rewards_sp2
)

# Read 'total_rewards_sp2' column from each file
total_reward_data_sp3 <- lapply(all_files, function(file) {
  df <- read_csv(file,show_col_types = FALSE)
  return(df$total_rewards_sp2)
})
names(total_reward_data_sp3) <- sapply(all_files, function(f) paste0("repetition_", basename(f)))

# Combine them into one large data frame
combined_total_reward_data_sp3 <- as.data.frame(total_reward_data_sp3)
combined_total_reward_data_sp3[is.na(combined_total_reward_data_sp3)] <- 0

# Calculate sd and variance
episode_mean_total_rewards_sp3 <- apply(combined_total_reward_data_sp3, 1, mean, na.rm = TRUE)
episode_sd_total_rewards_sp3 <- apply(combined_total_reward_data_sp3, 1, sd, na.rm = TRUE)
episode_variance_total_rewards_sp3 <- apply(combined_total_reward_data_sp3, 1, var, na.rm = TRUE)

summary_total_rewards_data_sp3 <- data.frame(
  episode = 1:length(episode_mean_total_rewards_sp3),
  mean = episode_mean_total_rewards_sp3,
  sd = episode_sd_total_rewards_sp3,
  variance = episode_variance_total_rewards_sp3
)

# Visualise defender rewards
d1 <- ggplot() +
  geom_line(data = summary_total_rewards_data_sp1, aes(x = episode, y = mean, color = "Average cumulative reward for provider 1")) +
  geom_ribbon(data = summary_total_rewards_data_sp1, aes(x = episode, ymin = mean - sd, ymax = mean + sd), 
              fill = "#3366FF", alpha = 0.3) +
  geom_line(data = summary_total_rewards_data_sp2, aes(x = episode, y = mean, color = "Average cumulative reward for provider 2")) +
  geom_ribbon(data = summary_total_rewards_data_sp2, aes(x = episode, ymin = mean - sd, ymax = mean + sd), 
              fill = "#FFCC66", alpha = 0.3) +
  geom_line(data = summary_total_rewards_data_sp3, aes(x = episode, y = mean, color = "Average cumulative reward for provider 3")) +
  geom_ribbon(data = summary_total_rewards_data_sp3, aes(x = episode, ymin = mean - sd, ymax = mean + sd), 
              fill = "#339900", alpha = 0.3) +
  scale_color_manual(values = c("Average cumulative reward for provider 1" = "#0033CC", 
                                "Average cumulative reward for provider 2" = "#FF9900",
                                "Average cumulative reward for provider 3" = "#006633")) +
  labs(x = "Episode", y = "Average cumulative reward for defenders\n(across 50 simulations)", color = NULL) +
  theme_bw() +
  theme(
    legend.position = c(0.70, 0.80),
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
d1

# Saved /ch3-model/results/plots/ch3-exp2-defender-rewards.pdf as 5 x 7.50 (landscape)
combined_plot <- cowplot::plot_grid(a1, d1, labels = c("A", "B"), ncol = 1)
combined_plot
# NOTE: Thesis uses a combined plot of attacker and defender rewards (9 x 7.50 (portrait))
