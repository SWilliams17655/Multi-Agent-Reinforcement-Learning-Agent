from agent_logic import Agent_Logic
from environment import Environment

# ********************************
# Loading the environment
# ********************************
valid_option = False
env = []
while not valid_option:
    option_selected = input("Select an option for your environment:\n"
                            "1: Create a new terrain map.\n"
                            "2: Load the previous terrain map.\n")

    if option_selected == '1':
        env = Environment(load_new_terrain=True)
        valid_option = True

    elif option_selected == '2':
        env = Environment(load_new_terrain=False)
        valid_option = True
    else:
        print("Invalid selection")

valid_option = False

env.show_display()

