# ReinforcementLearning Constants:

DISCOUNT = 0.99
REPLAY_MEMORY_SIZE = 3_000   # How many last steps to keep for model training
MIN_REPLAY_MEMORY_SIZE = 1_000   # Minimum number of steps in a memory to start training
UPDATE_TARGET_EVERY = 20      # Terminal states (end of episodes)
MIN_REWARD = 1000    # For model save
SAVE_MODEL_EVERY = 1000    # Episodes
SHOW_EVERY = 20      # Episodes
EPISODES = 100  # Number of episodes
#  Stats settings
AGGREGATE_STATS_EVERY = 20  # episodes
SHOW_PREVIEW = False
