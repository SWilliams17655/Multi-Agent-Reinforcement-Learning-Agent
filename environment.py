import numpy as np
from perlin_noise import PerlinNoise
from agent import Agent
from agent_logic import Agent_Logic
import random
import matplotlib.pyplot as plt
from logistics_hub import Logistics_Hub
from matplotlib.animation import FuncAnimation


class Environment:
    def __init__(self, load_new_terrain):
        self.WIDTH = 3000
        self.HEIGHT = 2000
        self.NUM_AGENTS = 30
        self.NUM_LOG_HUBS = 3

        self.terrain = []  # Terrain map for agents to navigate.

        self.logistics_hubs = []  # Logistics hubs for agents to resupply.
        self.logistics_hub_points = []  # Graphical point to display logistics hubs.

        self.agents = []  # Agents to navigate terrain.
        self.agent_points = []  # Graphical points to display agents.

        self.logic = Agent_Logic()

        # If user selects to create a new terrain map generates map using Perlin Noise.
        if load_new_terrain:
            print("Creating terrain map, this may take a moment.")
            noise = PerlinNoise(octaves=10)
            self.terrain = [[noise([i / self.WIDTH, j / self.HEIGHT]) for j in range(self.WIDTH)] for i in
                            range(self.HEIGHT)]
            np.save('terrain.npy', self.terrain)
        else:
            print("Loading terrain map.")
            self.terrain = np.load('terrain.npy')

        # Creates graphic to display agents, hubs, and environment.
        self.fig = plt.figure(figsize=(16, 10), facecolor='white')
        self.ax = plt.subplot()
        self.ax.imshow(self.terrain)

        # Adds the logistics hubs for agents to resupply.
        for i in range(self.NUM_LOG_HUBS):
            self.logistics_hubs.append(Logistics_Hub(random.randrange(0, self.WIDTH),
                                                     random.randrange(0, self.HEIGHT), 500))
            point, = self.ax.plot(self.logistics_hubs[i].location_x,
                                  self.logistics_hubs[i].location_y,
                                  marker='D', color='white', alpha=.5, markersize=15)
            self.logistics_hub_points.append(point)
        print(f"Added {len(self.logistics_hubs)} logistics hubs to the environment.")

        # Add agents and sets their initial target logistics hub.
        for k in range(self.NUM_AGENTS):
            dest = random.randrange(0, len(self.logistics_hubs))
            self.agents.append(Agent(random.randrange(0, self.WIDTH),
                                     random.randrange(0, self.HEIGHT),
                                     self.logistics_hubs[dest].location_x,
                                     self.logistics_hubs[dest].location_y,
                                     dest))
            point, = self.ax.plot(self.agents[k].location_x,
                                  self.agents[k].location_y,
                                  marker='^', color='white', markersize=5)
            self.agent_points.append(point)
        print(f"Added {len(self.agents)} agents to the environment.")

        self.logic.load("last_save.h5")

    # ***********************************************************************************
    def animate(self, frame):
        for i, individual_agent in enumerate(self.agents):
            state = individual_agent.state(self.WIDTH, self.HEIGHT)
            action = self.logic.get_action(state)
            individual_agent.make_move(action, self.WIDTH, self.HEIGHT)
            self.agent_points[i].set_data(individual_agent.location_x, individual_agent.location_y)
            reward = individual_agent.reward()
            new_state = individual_agent.state(self.WIDTH, self.HEIGHT)
            self.logic.remember(state, action, reward, new_state)
            self.logic.learn()

            if (abs(individual_agent.location_x - individual_agent.destination_x) < individual_agent.velocity*2 and
                    abs(individual_agent.location_y - individual_agent.destination_y) < individual_agent.velocity*2):
                dest = random.randrange(0, len(self.logistics_hubs))
                individual_agent.destination_x = self.logistics_hubs[dest].location_x
                individual_agent.destination_y = self.logistics_hubs[dest].location_y
                individual_agent.destination_hub = dest

        return self.agent_points

    def show_display(self):
        ani = FuncAnimation(self.fig, self.animate, interval=20, blit=True, cache_frame_data=True)

        # Set plot limits
        self.ax.set_xlim(0, self.WIDTH)
        self.ax.set_ylim(0, self.HEIGHT)
        self.ax.set_title("Reinforcement Learning-Deep Quality Neural Network")
        plt.show()
