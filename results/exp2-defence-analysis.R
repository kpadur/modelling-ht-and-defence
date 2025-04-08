# Exp 2 data analysis and visualisation

# Load libraries
library(readr)
library(ggplot2)
library(cowplot)

# Import data
path <- "exp2-results/"
all_files <- list.files(path, pattern = "*defender-data.csv", full.names = TRUE)

# Calculate metrics
stabilised_episode = 11148
number_of_episodes = 20000
stabilised_period = number_of_episodes-stabilised_episode

# Rewards: Cumulative reward over stabilised episodes
# Service provider 1
total_reward_data_sp1 <- function(file) {
  data <- read_csv(file, show_col_types = FALSE)
  sp1_rw_values <- tail(data$total_rewards_sp0, stabilised_period)
  return(sp1_rw_values)
}
sp1_rw_values <- sapply(all_files, total_reward_data_sp1)
sp1_rw_result <- list(
  mean = mean(sp1_rw_values),
  sd = sd(sp1_rw_values),
  min = min(sp1_rw_values),
  max = max(sp1_rw_values)
)

print(sp1_rw_result)

# Service provider 2
total_reward_data_sp2 <- function(file) {
  data <- read_csv(file, show_col_types = FALSE)
  sp2_rw_values <- tail(data$total_rewards_sp1, stabilised_period)
  return(sp2_rw_values)
}
sp2_rw_values <- sapply(all_files, total_reward_data_sp2)
sp2_rw_result <- list(
  mean = mean(sp2_rw_values),
  sd = sd(sp2_rw_values),
  min = min(sp2_rw_values),
  max = max(sp2_rw_values)
)

print(sp2_rw_result)

# Service provider 3
total_reward_data_sp3 <- function(file) {
  data <- read_csv(file, show_col_types = FALSE)
  sp3_rw_values <- tail(data$total_rewards_sp2, stabilised_period)
  return(sp3_rw_values)
}
sp3_rw_values <- sapply(all_files, total_reward_data_sp3)
sp3_rw_result <- list(
  mean = mean(sp3_rw_values),
  sd = sd(sp3_rw_values),
  min = min(sp3_rw_values),
  max = max(sp3_rw_values)
)

print(sp3_rw_result)

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
  geom_line(data = summary_total_rewards_data_sp1, aes(x = episode, y = mean, color = "Average cumulative reward for service provider 1")) +
  geom_ribbon(data = summary_total_rewards_data_sp1, aes(x = episode, ymin = mean - sd, ymax = mean + sd), 
              fill = "#3366FF", alpha = 0.2) +
  geom_line(data = summary_total_rewards_data_sp2, aes(x = episode, y = mean, color = "Average cumulative reward for service provider 2")) +
  geom_ribbon(data = summary_total_rewards_data_sp2, aes(x = episode, ymin = mean - sd, ymax = mean + sd), 
              fill = "#FFCC66", alpha = 0.2) +
  geom_line(data = summary_total_rewards_data_sp3, aes(x = episode, y = mean, color = "Average cumulative reward for service provider 3")) +
  geom_ribbon(data = summary_total_rewards_data_sp3, aes(x = episode, ymin = mean - sd, ymax = mean + sd), 
              fill = "#339900", alpha = 0.2) +
  scale_color_manual(values = c("Average cumulative reward for service provider 1" = "#0033CC", 
                                "Average cumulative reward for service provider 2" = "#FF9900",
                                "Average cumulative reward for service provider 3" = "#006633")) +
  labs(x = "Episode", y = "Average cumulative reward for defenders\n(across 100 simulations)", color = NULL) +
  theme_bw() +
  theme(
    legend.position = c(0.70, 0.30),
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
d1
