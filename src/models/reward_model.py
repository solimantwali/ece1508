# This file contains the reward model that we will be training
# The model takes (prompt, user_statement and model_reply) and converts it into a scalar score


"""
Note that user_statement is the part of the prompt that contains the user's belief
The prompt can be the entire multi-turn conversation. But the user_statement will be the part of the entire conversation that contains the user's claim. 
We then evaluate sycophancy by model response's belief contrasted against the belief from the user_statement

In single-turn examples the user_statement will just be the prompt. IF it even contains a belief. Idk what to do in the case where it doesn't contain a belief.
"""


