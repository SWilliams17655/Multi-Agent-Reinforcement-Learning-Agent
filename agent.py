import random
import tensorflow as tf
import numpy as np
from keras.optimizers import RMSprop
from keras.models import load_model
from collections import deque

class Agent:
    def __init__(self):
        self.state_size = 4
        self.action_space = 2
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_decay = .8
        self.epsilon_min = 0.001
        self.train_set_size = 1000
        self.memory = deque(maxlen=self.train_set_size)
        self.model = self.build_model()
        self.batch_size = 124
        self.num_epoch = 30

    # *********************************************************************************************************************
    def build_model(self):
        num_hidden_layer_1 = 512
        num_hidden_layer_2 = 256
        num_hidden_layer_3 = 64

        # Create a sequential model
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(num_hidden_layer_1, input_shape=(self.state_size,), kernel_initializer='he_uniform',
                                  activation='relu'),
            tf.keras.layers.Dense(num_hidden_layer_2, kernel_initializer='he_uniform', activation='relu'),
            tf.keras.layers.Dense(num_hidden_layer_3, kernel_initializer='he_uniform', activation='relu'),
            tf.keras.layers.Dense(self.action_space, activation='linear', kernel_initializer='he_uniform')
        ])

        model.compile(loss="mse", optimizer=RMSprop(lr=0.00025, rho=0.95, epsilon=0.01), metrics=["accuracy"])
        model.summary()

        return model

    # *********************************************************************************************************************

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    # *********************************************************************************************************************

    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            action = np.random.randint(0, self.action_space)

        else:
            # Add an extra dimension to the state to create a batch with one instance.
            state = np.expand_dims(state, axis=0)

            # Use the model to predict the Q-values for the given state
            q_values = self.model.predict(state, verbose=0)

            action = np.argmax(q_values[0])  # Take the action from the first (and only) entry

        return int(action)

    # *********************************************************************************************************************

    def learn(self):
        if len(self.memory) < self.train_set_size:  # If there are not enough training cases returns.
            return

        training_set = random.sample(self.memory, int(len(self.memory) * 0.5)) # Randomly samples half

        state = np.zeros((len(training_set), self.state_size))
        next_state = np.zeros((len(training_set), self.state_size))
        action, reward, done = [], [], []

        for i in range(len(training_set)):  # Splits up memory to create training set
            state[i] = training_set[i][0]
            action.append(training_set[i][1])
            reward.append(training_set[i][2])
            next_state[i] = training_set[i][3]
            done.append(training_set[i][4])

        target = self.model.predict(state) # Makes prediction for current state
        target_next = self.model.predict(next_state) # Makes prediction for next state

        for i in range(len(training_set)):
            if done[i]:
                target[i][action[i]] = reward[i] # If done then updates Q-Values with reward
            else:
                # Back propagates reward then reduces value by Gamma
                target[i][action[i]] = reward[i] + self.gamma * (np.amax(target_next[i]))

        # Decreases epsilon to ensure less random decisons are made after each learning iteration
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # Resets learning memory to load new training cases.
        self.memory = deque(maxlen=self.train_set_size)

        # Fits model based on
        self.model.fit(state, target, batch_size=self.batch_size, epochs=self.num_epoch, verbose=2)

    # *********************************************************************************************************************

    def load(self, name):
        self.model = load_model(name)

    # *********************************************************************************************************************

    def save(self, name):
        self.model.save(name)
