import random
import tensorflow as tf
import numpy as np
from keras.models import load_model
from collections import deque


class Agent_Logic:
    def __init__(self, input_array_size, epsilon):
        print(f"\nThe following GPU processor is being used: {tf.config.list_physical_devices('GPU')}\n")

        self.learning_rate = 0.001
        self.learning_rate_decay = .98
        self.state_size = input_array_size
        self.action_space = 8
        self.gamma = 0.9
        self.epsilon = epsilon
        self.epsilon_decay = .95
        self.epsilon_min = 0.1
        self.train_set_size = 10000
        self.memory = deque(maxlen=self.train_set_size)
        self.model = self.build_model()
        self.batch_size = 64
        self.num_epoch = 30

    def build_model(self):
        """
        Description: Builds a neural network allowing agents to calculate an optimum q-value.
        @:return Model used for agents to calculate their q-value.
        """

        model = tf.keras.Sequential()
        model.add(tf.keras.Input(shape=(self.state_size,)))
        model.add(tf.keras.layers.Dense(128, activation='relu', kernel_initializer='he_uniform'))
        model.add(tf.keras.layers.Dense(128, activation='relu', kernel_initializer='he_uniform'))
        model.add(tf.keras.layers.Dense(self.action_space, activation='linear', kernel_initializer='he_uniform'))

        model.compile(loss="mse",
                      optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
                      metrics=["mean_absolute_error"])

        model.summary()

        return model

    def remember(self, state, action, reward, next_state):
        """
        Description: Takes in agent data to build training set for later learning.
        """

        self.memory.append((state, action, reward, next_state))

    def get_q_values(self, states):
        """
        Description: Use the model to predict the Q-values for all agents using their current states.
        @:param: states: a 2D array representing the states of all agents in the environment.
        @:return: An array of all agent q_values.
        """

        q_values = self.model.predict(states, verbose=0)
        return q_values

    def learn(self):
        """
        Description: Uses all states saved using remember function to train neural network to calculate q-values.
        """

        if len(self.memory) < self.train_set_size:  # If there are not enough training cases returns.
            return

        training_set = random.sample(self.memory, int(len(self.memory) * 0.75))  # Randomly samples training set.

        state = np.zeros((len(training_set), self.state_size))  # Creates a 2d numpy array to load training set.
        next_state = np.zeros((len(training_set), self.state_size))
        action, reward = [], []

        for i in range(len(training_set)):  # Splits up memory to create training set
            state[i] = training_set[i][0]
            action.append(training_set[i][1])
            reward.append(training_set[i][2])
            next_state[i] = training_set[i][3]

        target = self.model.predict(state)  # Makes prediction for current state
        target_next = self.model.predict(next_state)  # Makes prediction for next state

        for i in range(len(training_set)):
            target[i][action[i]] = reward[i] + self.gamma * (np.argmax(target_next[i]))

        # Decreases epsilon to ensure less random decisions are made after each learning iteration
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            print(f"\nUpdated Epsilon is {self.epsilon} & training set size is {self.train_set_size}\n")

        # Resets learning memory to load new training cases.
        self.memory = deque(maxlen=self.train_set_size)

        self.model.fit(state, target, batch_size=self.batch_size, epochs=self.num_epoch, verbose=0)
        self.save("last_save.keras")

    def load(self, name):
        """
        Description: Loads a previously saved model for use.
        @:param: name: The string name of the .keras model to load.
        """

        self.model = load_model(name)

    def save(self, name):
        """
        Description: Saves model for use in later iterations.
        @:param: name: The string name of the .keras model to save.
        """
        self.model.save(name)
