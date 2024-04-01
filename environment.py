import numpy as np
from perlin_noise import PerlinNoise
from agent_logic import Agent_Logic
import random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import sqlite3
import time
import math

RUN_LENGTH = 100

AGENT_LAST_LOCATION_X = 1
AGENT_LAST_LOCATION_Y = 2
AGENT_LOCATION_X = 3
AGENT_LOCATION_Y = 4
AGENT_VELOCITY = 5
AGENT_DESTINATION = 6
DESTINATION_X = 8
DESTINATION_Y = 9
LOG_HUB_LOCATION_X = 1
LOG_HUB_LOCATION_Y = 2

con = sqlite3.connect("environment.db")
cur = con.cursor()
cur.execute("DELETE FROM agent")
cur.execute("DELETE FROM logistics_hub")
con.commit()


class Environment:
    def __init__(self, load_new_terrain):
        self.WIDTH = 2000
        self.HEIGHT = 1000
        self.NUM_AGENTS = 20
        self.NUM_LOG_HUBS = 3
        self.DELIVERY_RANGE = 10
        self.SENSOR_RANGE = 3

        self.terrain = []  # Terrain map for agents to navigate.
        self.logistics_hub_points = []  # Graphical point to display logistics hubs.
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
            x = random.randrange(0, self.WIDTH)
            y = random.randrange(0, self.HEIGHT)
            cur.execute("INSERT INTO logistics_hub (logistics_hub_id, location_x, location_y) VALUES (?, ?, ?)",
                        (i, x, y))

        # Add agents and sets their initial target logistics hub.
        for j in range(self.NUM_AGENTS):
            dest = random.randrange(0, self.NUM_LOG_HUBS)
            x = random.randrange(0, self.WIDTH)
            y = random.randrange(0, self.HEIGHT)
            cur.execute(
                'INSERT INTO agent (agent_id, last_location_x, last_location_y, location_x, location_y, velocity, destination_hub) VALUES (?, ?, ?, ?, ?, ?, ?)',
                (j, x, y, x, y, 10, dest))
        con.commit()

        # Gets all hubs from db and adds to the graphic.
        cur.execute('SELECT * FROM logistics_hub')
        hubs = cur.fetchall()
        for hub in hubs:
            point, = self.ax.plot(hub[LOG_HUB_LOCATION_X], hub[LOG_HUB_LOCATION_Y], marker='D', color='white', alpha=.5,
                                  markersize=15)
            self.logistics_hub_points.append(point)
        print(f"\nAdded {len(hubs)} logistics hubs to the environment.")

        cur.execute('SELECT * FROM agent INNER JOIN logistics_hub')
        agents = cur.fetchall()

        for agent in agents:
            point, = self.ax.plot(agent[AGENT_LOCATION_X], agent[AGENT_LOCATION_Y], marker='^', color='white',
                                  markersize=5)
            self.agent_points.append(point)
        print(f"Added {len(agents)} agents to the environment.")

        # self.logic.load("last_save.h5")

    def execute(self):
        cur.execute(
            'SELECT * FROM agent INNER JOIN logistics_hub ON agent.destination_hub = logistics_hub.logistics_hub_id')
        agents = cur.fetchall()

        for _, agent in enumerate(agents):

            # Calculating the action to be taken and moving the agent.
            agent_last_location_x = int(agent[AGENT_LOCATION_X])
            agent_last_location_y = int(agent[AGENT_LOCATION_Y])
            agent_location_x = int(agent[AGENT_LOCATION_X])
            agent_location_y = int(agent[AGENT_LOCATION_Y])

            if self.terrain[agent_location_y, agent_location_x] < -.25:
                velocity = 1
            else:
                velocity = (self.terrain[agent_location_y, agent_location_x] + 1.0) * agent[AGENT_VELOCITY]
            # Defines the state of the agent.
            local_terrain = self.terrain[agent_location_y - self.SENSOR_RANGE: agent_location_y + self.SENSOR_RANGE,
                            agent_location_x - self.SENSOR_RANGE: agent_location_x + self.SENSOR_RANGE]
            state = [(agent[AGENT_LOCATION_X] - agent[DESTINATION_X]) / self.WIDTH,
                     (agent[AGENT_LOCATION_Y] - agent[DESTINATION_Y]) / self.HEIGHT]
            state = np.append(state, local_terrain.reshape([1, (self.SENSOR_RANGE * 2)**2]))
            action = self.logic.get_action(state)
            if action == 0 or action == 7 or action == 6:
                agent_location_x = int(agent[AGENT_LOCATION_X]) - velocity
            if action == 2 or action == 3 or action == 4:
                agent_location_x = int(agent[AGENT_LOCATION_X]) + velocity
            if action == 0 or action == 1 or action == 2:
                agent_location_y = int(agent[AGENT_LOCATION_Y]) + velocity
            if action == 6 or action == 5 or action == 4:
                agent_location_y = int(agent[AGENT_LOCATION_Y]) - velocity

            if agent_location_x < 5 or agent_location_x > self.WIDTH - 5:
                agent_location_x = agent_last_location_x
            if agent_location_y < 5 or agent_location_y > self.HEIGHT - 5:
                agent_location_y = agent_last_location_y

            # Calculating the reward for that move.
            previous_distance = math.sqrt((agent_last_location_x - agent[DESTINATION_X]) ** 2 + (
                    agent_last_location_y - agent[DESTINATION_Y]) ** 2)
            new_distance = math.sqrt(
                (agent_location_x - agent[DESTINATION_X]) ** 2 + (agent_location_y - agent[DESTINATION_Y]) ** 2)

            if new_distance < self.DELIVERY_RANGE:
                reward = 75
            elif (previous_distance - new_distance) > 0:
                reward = abs(previous_distance - new_distance) / agent[AGENT_VELOCITY]
            else:
                reward = -100

            local_terrain = self.terrain[int(agent_location_y) - self.SENSOR_RANGE: int(agent_location_y) + self.SENSOR_RANGE,
                            int(agent_location_x) - self.SENSOR_RANGE: int(agent_location_x) + self.SENSOR_RANGE]
            new_state = [(agent[AGENT_LOCATION_X] - agent[DESTINATION_X]) / self.WIDTH,
                         (agent[AGENT_LOCATION_Y] - agent[DESTINATION_Y]) / self.HEIGHT]
            new_state = np.append(new_state, local_terrain.reshape([1, (self.SENSOR_RANGE * 2)**2]))

            # Saving the move for later learning.
            self.logic.remember(state, action, reward, new_state)

            dest = agent[AGENT_DESTINATION]
            if abs(agent_location_x - agent[DESTINATION_X]) < 30 and abs(agent_location_y - agent[DESTINATION_Y]) < 30:
                dest = random.randrange(0, self.NUM_LOG_HUBS)

            # Updating the database after the move
            command = f"UPDATE agent SET " \
                      f"last_location_x = {agent_last_location_x}, " \
                      f"last_Location_y = {agent_last_location_y}," \
                      f"location_x = {agent_location_x}," \
                      f"location_y = {agent_location_y}, " \
                      f"destination_hub = {dest} " \
                      f"WHERE agent_id={agent[0]}"

            cur.execute(command)

        con.commit()

        self.logic.learn()

    # ***********************************************************************************
    def animate(self, frame):
        self.execute()
        cur.execute('SELECT * FROM agent INNER JOIN logistics_hub')
        agents = cur.fetchall()

        for i, agent in enumerate(agents):
            self.agent_points[i].set_data(agent[AGENT_LOCATION_X], agent[AGENT_LOCATION_Y])

        return self.agent_points

    def show_display(self):
        ani = FuncAnimation(self.fig, self.animate, interval=20, blit=True, cache_frame_data=True)

        # Set plot limits
        self.ax.set_xlim(0, self.WIDTH)
        self.ax.set_ylim(0, self.HEIGHT)
        self.ax.set_title("Reinforcement Learning-Deep Quality Neural Network")
        plt.show()
