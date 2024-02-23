import gymnasium as gym
from agent import Agent
import seaborn as sns
import matplotlib.pyplot as plt

env = gym.make("CartPole-v1")

debug = False # Set to true to log debug features.
num_episodes = 300
max_steps = 300
score = [] # Saves score during each iteration for graphing.
agent = Agent()

answer = input("Would you like to load an agent? [y/n]")
if answer == 'y' or answer == 'Y':
    print("Loading agent")
    agent.load("agent.h5")
else:
    print("Starting with untrained agent")

answer = input("Would you like to train network? [y/n]")
if answer == 'y' or answer == 'Y':
    print("Beginning training process")
    for e in range(num_episodes):
        state, _ = env.reset()
        done = False
        step = 0

        while not done and step < max_steps:
            action = agent.get_action(state)
            new_state, reward, done, _, _ = env.step(action)
            if not done or step >= max_steps - 1: # If the pole falls before max_sets sets to -100
                reward = reward
            else:
                reward = -100

            if debug:
                print(f"Step: {step}, State: {state}, Action: {action}, Reward: {reward}, New State: {new_state}, Done: {done}")

            agent.remember(state, action, reward, new_state, done)  # Saves information in memory for training.
            state = new_state
            step += 1
            agent.learn()  # Implements learning algorithm.

        print(f"Episode: {e}, Number Steps: {step}, Epsilon: {agent.epsilon}")
        score.append(step)

    agent.save("agent.h5")

    #  Graphs the learning progress
    sns.set_style("darkgrid", {"axes.facecolor": ".9"})
    sns.lineplot(data=score)
    plt.xlabel("Training Episode")
    plt.ylabel("Number of Steps Achieved")
    plt.title("Deep Quality Network Learning Score")
    plt.show()

env = gym.make("CartPole-v1", render_mode="human")

for e in range(num_episodes):
    state, _ = env.reset()
    done = False
    step = 0
    agent.epsilon = agent.epsilon_min  # Setting epsilon to min to demonstrate full learning
    while not done and step < max_steps:
        action = agent.get_action(state)
        new_state, reward, done, _, _ = env.step(action)
        if not done or step >= max_steps - 1:
            reward = reward
        else:
            reward = -100

        state = new_state
        step += 1
