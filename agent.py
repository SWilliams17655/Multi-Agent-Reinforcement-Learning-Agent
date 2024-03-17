import math


class Agent:
    def __init__(self, start_x, start_y, destination_x, destination_y, destination_hub):
        self.last_location_x = start_x
        self.last_location_y = start_y
        self.location_x = start_x
        self.location_y = start_y
        self.destination_x = destination_x
        self.destination_y = destination_y
        self.destination_hub = destination_hub
        self.velocity = 10
        self.delivery_range = 20

    # *********************************************************************************************************************
    def make_move(self, action, width, height):
        self.last_location_x = self.location_x
        self.last_location_y = self.location_y

        if action == 0 or action == 7 or action == 6:
            self.location_x -= self.velocity

        if action == 2 or action == 3 or action == 4:
            self.location_x += self.velocity

        if action == 0 or action == 1 or action == 2:
            self.location_y += self.velocity

        if action == 6 or action == 5 or action == 4:
            self.location_y -= self.velocity

        if self.location_x < 0 or self.last_location_x > width:
            self.location_x = self.last_location_x
        if self.location_y < 0 or self.last_location_y > height:
            self.location_y = self.last_location_y

    def reward(self):
        previous_distance = math.sqrt((self.last_location_x - self.destination_x) ** 2 + (self.last_location_y - self.destination_y) ** 2)
        new_distance = math.sqrt((self.location_x - self.destination_x) ** 2 + (self.location_y - self.destination_y) ** 2)
        if new_distance < self.delivery_range:
            reward = 75
        elif (previous_distance-new_distance) > 0:
            reward = abs(previous_distance - new_distance) / self.velocity
        else:
            reward = -100

        return reward

    def state(self, width, height):
        return [(self.location_x-self.destination_x)/width,
                (self.location_y-self.destination_y)/height]
