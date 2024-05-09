import numpy as np
from perlin_noise import PerlinNoise
from agent_logic import Agent_Logic
import random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import sqlite3
import math

AGENT_ID = 0
AGENT_LOCATION_X = 1
AGENT_LOCATION_Y = 2
AGENT_DESTINATION = 3
AGENT_REWARD = 4
AGENT_DESTINATION_X = 6
AGENT_DESTINATION_Y = 7

LOG_HUB_ID = 0
LOG_HUB_LOCATION_X = 1
LOG_HUB_LOCATION_Y = 2

STATE_AGENT_LOCATION_X = 0
STATE_AGENT_LOCATION_Y = 1
STATE_AGENT_DESTINATION_X = 2
STATE_AGENT_DESTINATION_Y = 3

con = sqlite3.connect("environment.db")
cur = con.cursor()
cur.execute("DELETE FROM agent")
cur.execute("DELETE FROM logistics_hub")
con.commit()


class Environment:
    """
    Description: Environment represents a logistics network where agents resupply logistics hubs and is intended to
    simulate an environment for reinforcement learning agents.
    Author: Sean "Wiki" Williams
    Date: 6 April 2024 Version-Beta
    """

    def __init__(self, load_new_terrain, load_prior_agent, epsilon):
        """
        Description: Initializes training environment.
        @:param load_new_terrain: Boolean value representing whether function should generate a new terrain.
        @:param load_prior_agent: Boolean value representing whether function should load prior agent neural network.
        @:param epsilon: A starting epsilon value introducing randomness into each agent's actions.
        """
        self.WIDTH = 2000
        self.HEIGHT = 1000
        self.NUM_AGENTS = 100
        self.MAX_VELOCITY = 20
        self.NUM_LOG_HUBS = 2
        self.DELIVERY_RANGE = self.MAX_VELOCITY * 2
        self.SENSOR_RANGE = 20
        self.EDGE_BUFFER = self.SENSOR_RANGE
        self.terrain = []

        # Initializes the neural network
        self.input_array_size = 6 + (self.SENSOR_RANGE * 2) ** 2
        self.logic = Agent_Logic(self.input_array_size, epsilon)

        # Generates map using Perlin Noise.
        if load_new_terrain:
            print("\nCreating terrain map, this may take a moment...\n")
            noise = PerlinNoise(octaves=8)
            self.terrain = [[noise([i / self.WIDTH, j / self.HEIGHT]) for j in range(self.WIDTH)] for i in
                            range(self.HEIGHT)]
            self.terrain = np.array(self.terrain)
            self.terrain = self.terrain + .5
            self.terrain = np.where(self.terrain < 0.25, 0.0, self.terrain)
            np.save('terrain.npy', self.terrain)
        else:
            print("\nLoading terrain map.\n")
            self.terrain = np.load('terrain.npy')

        # Creates graphic using Matplotlib animatefunc to display agents, hubs, and environment.
        self.logistics_hub_points = []  # Graphical point to display logistics hubs on the map.
        self.agent_points = []  # Graphical points to display agents on the map.
        self.fig, (self.ax1) = plt.subplots(figsize=(20, 16), nrows=1, ncols=1, facecolor='white')
        vel = self.ax1.imshow(self.terrain)
        self.fig.colorbar(vel, ax=self.ax1, location='right', shrink=0.5, label='Maximum Velocity')

        # Adds the logistics hubs for agents to resupply.
        for i in range(self.NUM_LOG_HUBS):
            x = random.randrange(self.EDGE_BUFFER, self.WIDTH - self.EDGE_BUFFER)
            y = random.randrange(self.EDGE_BUFFER, self.HEIGHT - self.EDGE_BUFFER)
            while self.terrain[y, x] <= 0:
                x = random.randrange(self.EDGE_BUFFER, self.WIDTH - self.EDGE_BUFFER)
                y = random.randrange(self.EDGE_BUFFER, self.HEIGHT - self.EDGE_BUFFER)
            cur.execute("INSERT INTO logistics_hub (logistics_hub_id, location_x, location_y) VALUES (?, ?, ?)",
                        (i, x, y))

        # Add agents and sets their initial target logistics hub.
        for j in range(self.NUM_AGENTS):
            dest = random.randrange(0, self.NUM_LOG_HUBS)
            x = random.randrange(self.EDGE_BUFFER, self.WIDTH - self.EDGE_BUFFER)
            y = random.randrange(self.EDGE_BUFFER, self.HEIGHT - self.EDGE_BUFFER)

            #  Ensures agent is not randomly added to a location with un-navigable terrain.
            while self.terrain[y, x] <= 0:
                x = random.randrange(self.EDGE_BUFFER, self.WIDTH - self.EDGE_BUFFER)
                y = random.randrange(self.EDGE_BUFFER, self.HEIGHT - self.EDGE_BUFFER)
            cur.execute(
                'INSERT INTO agent (agent_id, location_x, location_y, destination_hub, reward) '
                'VALUES (?, ?, ?, ?, ?)',
                (j, x, y, dest, 0.0))
        con.commit()

        # Gets all hubs from db and adds to the map.
        cur.execute('SELECT * FROM logistics_hub')
        hubs = cur.fetchall()
        for hub in hubs:
            point, = self.ax1.plot(hub[LOG_HUB_LOCATION_X], hub[LOG_HUB_LOCATION_Y], marker='D', color='white',
                                   alpha=.5,
                                   markersize=self.DELIVERY_RANGE / 3)
            self.logistics_hub_points.append(point)
        print(f"\nAdded {self.NUM_LOG_HUBS} logistics hubs to the environment.")

        # Gets all agents and adds them to the graphic.
        cur.execute('SELECT * FROM agent INNER JOIN logistics_hub')
        agents = cur.fetchall()
        for agent in agents:
            point, = self.ax1.plot(agent[AGENT_LOCATION_X], agent[AGENT_LOCATION_Y], marker='^', color='white',
                                   markersize=5)
            self.agent_points.append(point)
        print(f"\nAdded {self.NUM_AGENTS} agents to the environment.")

        #  If option was selected by user, loads prior agent training.
        if load_prior_agent:
            self.logic.load("last_save.keras")
            print("\nLoaded prior trained agent from last_save.keras")

    def execute(self):
        """
        Description: Loops through all agents and executes a single move of all agents. Implements a reinforcement
        learning algorithm.
        """

        #  Pulls all agent data from database.
        cur.execute(
            'SELECT * FROM agent INNER JOIN logistics_hub ON agent.destination_hub = logistics_hub.logistics_hub_id')
        agents = cur.fetchall()
        states = np.zeros((self.NUM_AGENTS, self.input_array_size))
        new_states = np.zeros((self.NUM_AGENTS, self.input_array_size))
        rewards = np.zeros(self.NUM_AGENTS)

        # Creates a numpy array of states for all agents then feed that into neural network to calc q-values.
        for i, agent in enumerate(agents):
            states[i] = self.state(agent_location_x=agent[AGENT_LOCATION_X],
                                   agent_location_y=agent[AGENT_LOCATION_Y],
                                   agent_destination_x=agent[AGENT_DESTINATION_X],
                                   agent_destination_y=agent[AGENT_DESTINATION_Y])
        q_values = self.logic.get_q_values(states)  # Simultaneously calculates q-values for all agents.
        actions = np.argmax(q_values, axis=1)  # Simultaneously calculates actions for all agents.

        #  Loops through agents and move agents based on calculated action.
        for i, state in enumerate(states):

            # If random action is chosen based on epsilon.
            if np.random.rand() <= self.logic.epsilon:
                actions[i] = np.random.randint(0, self.logic.action_space)

            new_states[i], rewards[i] = self.execute_action(state=state, action=actions[i])
            if rewards[i] == 100:
                destination = random.randint(0, self.NUM_LOG_HUBS-1)
                command = f"UPDATE agent SET " \
                          f"location_x = {new_states[i][STATE_AGENT_LOCATION_X]*self.WIDTH}," \
                          f"location_y = {new_states[i][STATE_AGENT_LOCATION_Y]*self.HEIGHT}, " \
                          f"reward = {rewards[i]}, " \
                          f"destination_hub = {destination} " \
                          f"WHERE agent_id={i}"
            else:
                command = f"UPDATE agent SET " \
                          f"location_x = {new_states[i][STATE_AGENT_LOCATION_X] * self.WIDTH}," \
                          f"location_y = {new_states[i][STATE_AGENT_LOCATION_Y] * self.HEIGHT}, " \
                          f"reward = {rewards[i]} " \
                          f"WHERE agent_id={i}"

            cur.execute(command)

        for i, state in enumerate(states):
            self.logic.remember(state, actions[i], rewards[i], new_states[i])

        #  Initiates learning from prior actions.
        if self.logic.epsilon >= self.logic.epsilon_min:
            self.logic.learn()

    def execute_action(self, state, action):
        """
        Description: Moves agent to a new location based on input action returning agent location and reward.
        @:param state: The initial state for the agent being moved.
        @:param action: The action to be taken.
        @:return new_state: The final state the agent ends up at.
        @:return reward: The reward received for the previous move.
        """

        agent_location_x = int(state[STATE_AGENT_LOCATION_X]*self.WIDTH)
        agent_location_y = int(state[STATE_AGENT_LOCATION_Y]*self.HEIGHT)
        agent_last_location_x = int(state[STATE_AGENT_LOCATION_X]*self.WIDTH)
        agent_last_location_y = int(state[STATE_AGENT_LOCATION_Y]*self.HEIGHT)
        agent_destination_x = int(state[STATE_AGENT_DESTINATION_X]*self.WIDTH)
        agent_destination_y = int(state[STATE_AGENT_DESTINATION_Y]*self.HEIGHT)

        velocity = self.terrain[agent_location_y, agent_location_x] * self.MAX_VELOCITY

        if action == 0 or action == 7 or action == 6:
            agent_location_x = int(agent_location_x - velocity)
        if action == 2 or action == 3 or action == 4:
            agent_location_x = int(agent_location_x + velocity)
        if action == 0 or action == 1 or action == 2:
            agent_location_y = int(agent_location_y + velocity)
        if action == 6 or action == 5 or action == 4:
            agent_location_y = int(agent_location_y - velocity)

        # Prevents agent from going off the map.
        if agent_location_x < self.EDGE_BUFFER or agent_location_x > self.WIDTH - self.EDGE_BUFFER or \
                agent_location_y < self.EDGE_BUFFER or agent_location_y > self.HEIGHT - self.EDGE_BUFFER:
            agent_location_x = agent_last_location_x
            agent_location_y = agent_last_location_y

        # Prevents agent from going into unnavigable terrain.
        if self.terrain[agent_location_y, agent_location_x] <= 0:
            agent_location_x = agent_last_location_x
            agent_location_y = agent_last_location_y

        # Calculating the reward for that move.
        previous_distance = math.sqrt((agent_last_location_x - agent_destination_x) ** 2 +
                                      (agent_last_location_y - agent_destination_y) ** 2)

        new_distance = math.sqrt((agent_location_x - agent_destination_x) ** 2 +
                                 (agent_location_y - agent_destination_y) ** 2)

        reward = int(previous_distance - new_distance)

        if new_distance < self.DELIVERY_RANGE:
            reward = 100

        new_state = self.state(agent_location_x=agent_location_x,
                               agent_location_y=agent_location_y,
                               agent_destination_x=agent_destination_x,
                               agent_destination_y=agent_destination_y)

        return new_state, reward

    def state(self, agent_location_x, agent_location_y, agent_destination_x, agent_destination_y):
        """
        Description: Takes agent array as input and creates an array representing the agent's state.
        @:param agent_location_x: X location of the agent state is being generated for.
        @:param agent_location_y: Y location of the agent state is being generated for.
        @:param agent_destination_x: X location of the agent's destination hub.
        @:param agent_destination_y: Y location of the agent's destination hub.
        @:param agents: An array representing all agents in the environment.
        @:return An array representing the state of that agent.
        """

        #  Translates local terrain observed by the agent into the state array.
        local_terrain = self.terrain[agent_location_y - self.SENSOR_RANGE: agent_location_y + self.SENSOR_RANGE,
                                     agent_location_x - self.SENSOR_RANGE: agent_location_x + self.SENSOR_RANGE]

        state = [agent_location_x/self.WIDTH, agent_location_y/self.HEIGHT,
                 agent_destination_x/self.WIDTH, agent_destination_y/self.HEIGHT,
                 (agent_location_x - agent_destination_x) / self.WIDTH,
                 (agent_location_y - agent_destination_y) / self.HEIGHT]
        state = np.append(state, local_terrain.reshape([1, (self.SENSOR_RANGE * 2) ** 2]))

        return state

    def animate(self, frame):
        """
        Description: Loops through agents and environment providing a graphical representation of the environment.
        """

        self.execute()
        cur.execute('SELECT * FROM agent INNER JOIN logistics_hub')
        agents = cur.fetchall()

        for i, agent in enumerate(agents):
            self.agent_points[i].set_data(agent[AGENT_LOCATION_X], agent[AGENT_LOCATION_Y])
            if agent[AGENT_REWARD] == 100:
                self.agent_points[i].set_color("red")
            else:
                self.agent_points[i].set_color("white")
        return self.agent_points

    def show_display(self):
        ani = FuncAnimation(self.fig, self.animate, interval=20, blit=True, cache_frame_data=False)

        # Set plot limits
        self.ax1.set_xlim(0, self.WIDTH)
        self.ax1.set_ylim(0, self.HEIGHT)
        self.ax1.set_title("Reinforcement Learning-Deep Quality Neural Network")
        plt.show()

