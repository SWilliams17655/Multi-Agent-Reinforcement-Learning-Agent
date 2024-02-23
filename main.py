import gymnasium as gym
from agent import Agent
import seaborn as sns
import matplotlib.pyplot as plt

env = gym.make("CartPole-v1")

debug = False
num_episodes = 500
max_steps = 500
score = []
agent = Agent()

for e in range(num_episodes):
    state, _ = env.reset()
    done = False
    step = 0

    while not done and step < max_steps:
        action = agent.get_action(state)
        new_state, reward, done, _, _ = env.step(action)
        if not done or step >= max_steps - 1:
            reward = reward
        else:
            reward = -100

        if debug:
            print(f"Step: {step}, State: {state}, Action: {action}, Reward: {reward}, New State: {new_state}, Done: {done}")

        agent.remember(state, action, reward, new_state, done)
        state = new_state
        step += 1
        agent.learn()

    print(f"Episode: {e}, Number Steps: {step}, Epsilon: {agent.epsilon}")
    score.append(step)

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

    while not done and step < max_steps:
        action = agent.get_action(state)
        new_state, reward, done, _, _ = env.step(action)
        if not done or step >= max_steps - 1:
            reward = reward
        else:
            reward = -100

        if debug:
            print(f"Step: {step}, State: {state}, Action: {action}, Reward: {reward}, New State: {new_state}, Done: {done}")

        agent.remember(state, action, reward, new_state, done)
        state = new_state
        step += 1
    print(f"Episode: {e}, Number Steps: {step}, Epsilon: {agent.epsilon}")