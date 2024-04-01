from environment import Environment
from multiprocessing import Process
import os
# ********************************
# Loading the environment
# ********************************
valid_option = False
env = []

# while not valid_option:
#     option_selected = input("Select an option for your environment:\n"
#                             "1: Create a new terrain map.\n"
#                             "2: Load the previous terrain map.\n")
#
#     if option_selected == '1':
#         valid_option = True
#
#     elif option_selected == '2':
#         valid_option = True
#
#     else:
#         print("Invalid selection")

if __name__ == "__main__":
    env = Environment(load_new_terrain=False)
    env.show_display()
