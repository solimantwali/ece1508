# Any complexities of the ppo training process will be stored here. 

# Samples a batch of prompts
# computes rewards using the reward_function(prompts, responses)
# reward_function wraps reward model, KL, transforms

# Then compute advantages
# Then use a clipped Loss from the slides or google one
# Calculate the Value Loss and apply any bonuses

# Make sure to log the metrics

