import random
class Logistics_Hub:
    def __init__(self, x, y, initial_supplies):
        self.location_x = x
        self.location_y = y
        self.supplies = initial_supplies
        self.max_supplies = 1000