import os
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import TimeLimit
import random

EPISODES_NUM = 50000
TIMESTEPS_NUM = 200

# For a large TIMESTEPS_NUM, a smaller ALPHA is preferable.
ALPHA = 0.2

# We care more about the end score than immediate reward.
GAMMA = 0.99

EXPLORATION_RATE = 1

# Tweak-able parameters below, set max and min to the same value for a constant exploration rate.
MAX_EXPLORATION_RATE = 1
MIN_EXPLORATION_RATE = 0.01
EXPLORATION_RATE_DECAY = 0.05

TRAINING_DATA_SAVE_FILE_NAME = f"training-data-alpha-{ALPHA}-gamma-{GAMMA}-episode-{EPISODES_NUM}-timesteps-{TIMESTEPS_NUM}-exploration-rate-{EXPLORATION_RATE}-max-exploration-rate-{MAX_EXPLORATION_RATE}-min-exploration-rate-{MIN_EXPLORATION_RATE}-exploration-rate-decay-{EXPLORATION_RATE_DECAY}.npy"
# TRAINING_DATA_SAVE_FILE_NAME = "training-data-alpha-0.2-gamma-0.99-episode-50000-timesteps-200-exploration-rate-1-max-exploration-rate-1-min-exploration-rate-0.01-exploration-rate-decay-0.05.npy"
# TRAINING_DATA_SAVE_FILE_NAME = "training-data-alpha-0.2-gamma-0.99-episode-10000-timesteps-200-exploration-rate-1-max-exploration-rate-1-min-exploration-rate-0.01-exploration-rate-decay-0.05.npy"
# TRAINING_DATA_SAVE_FILE_NAME = "training-data-alpha-0.6-gamma-0.99-episode-10000-timesteps-200-exploration-rate-1-max-exploration-rate-1-min-exploration-rate-0.01-exploration-rate-decay-0.05.npy"

q_table = dict()


def get_hash(state) -> int:
    return hash(tuple(state))


def set_state(env, action, state, reward):
    key = get_hash(state)

    if key not in q_table:
        q_table[key] = np.zeros(env.action_space.n)
        q_table[key][action] = reward
        return

    q_table[key][action] += reward
    return


def get_max_action(env, state):
    key = get_hash(state)

    if key not in q_table:
        q_table[key] = np.zeros(env.action_space.n)
        return env.action_space.sample()

    return np.argmax(q_table[key])


def get_max_state_reward(env, state):
    key = get_hash(state)

    if key not in q_table:
        q_table[key] = np.zeros(env.action_space.n)
        return 0.0

    return np.max(q_table[key])


def get_reward_by_action_and_state(env, state, action):
    key = get_hash(state)

    if key not in q_table:
        q_table[key] = np.zeros(env.action_space.n)
        return 0.0

    return q_table[key][action]


def train(env):
    rewards = []
    mean_rewards_per_episodes = []

    global EXPLORATION_RATE
    for episode in range(EPISODES_NUM):
        state, info = env.reset()
        total_reward = 0

        done = False

        while not done:
            exp_tradeoff = random.uniform(0, 1)

            # ϵ-greedy strategy to learn
            if exp_tradeoff > EXPLORATION_RATE:
                action = get_max_action(env, state)
            else:
                action = env.action_space.sample()
            new_state, reward, terminated, truncated, info = env.step(action)

            # terminated is a natural game state, truncated is when the timelimit we specify is reached.
            done = terminated or truncated

            new_state_max_reward = get_max_state_reward(env, new_state)
            current_state_reward = get_reward_by_action_and_state(env, state, action)
            discounted_reward = ALPHA * (
                reward + GAMMA * new_state_max_reward - current_state_reward
            )

            set_state(env, action, state, discounted_reward)

            total_reward += reward
            state = new_state

        print(f"Episode: {episode}, total reward: {total_reward}")

        # Exploration rate decay, at the end of every episode.
        # It is done after an episode, as such it will take longer to converge
        # compared to after every action but the advantage is it offers more
        # exploration.
        EXPLORATION_RATE = MIN_EXPLORATION_RATE + (
            MAX_EXPLORATION_RATE - MIN_EXPLORATION_RATE
        ) * np.exp(-EXPLORATION_RATE_DECAY * episode)

        # Keep track of the total roward of the episode.
        rewards.append(total_reward)

        # Keep track of the mean reward of the episode, so the mean reward per episode can be plotted.
        mean_rewards_per_episodes.append(sum(rewards) / len(rewards))

    # The mean reward of the training.
    mean = sum(rewards) / len(rewards)

    return [rewards, mean_rewards_per_episodes, mean]


def replay(env, path_to_training_results):
    global q_table
    # When saving a dictionary as a numpy object, the entire dictionary is treated as a scalar, in otherwords the ndarray has 0 dimensions.
    # To access the actual dictionary saved by np.save(), when doing np.load(), call .item() on the returned result.
    q_table = np.load(path_to_training_results, allow_pickle=True).item()

    state, info = env.reset()
    total_reward = 0

    done = False

    while not done:
        action = get_max_action(env, state)
        new_state, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        state = new_state
        done = terminated or truncated

    print(f"Using {path_to_training_results}, a score of {total_reward} was reached.")
    return total_reward


env = None
if not os.path.isfile(TRAINING_DATA_SAVE_FILE_NAME):
    # To view training, pass render_mode="human" to gym.make as well.
    env = gym.make("MsPacman-ram-v4")
    env = TimeLimit(env, max_episode_steps=TIMESTEPS_NUM)
    state, info = env.reset()
    # Save the results of the training, so you can plot them and see how the
    # tweaking the parameters effects the reward.
    rewards, mean_rewards_per_episodes, mean = train(env)

    np.save(TRAINING_DATA_SAVE_FILE_NAME, q_table)
    q_table.clear()
else:
    print(f"Replaying using training data from: {TRAINING_DATA_SAVE_FILE_NAME}")
    # env = gym.make("MsPacman-ram-v4",  render_mode="human")
    env = gym.make("MsPacman-ram-v4")
    score = 0
    for i in range(100):
        score += replay(env, TRAINING_DATA_SAVE_FILE_NAME)
    score = score / 100
    print(f"Average score was {score}")
