from environment import Environment

# ********************************
# Loading the environment
# ********************************
valid_option = False
load_new_terrain = False
load_prior_agent = False
env = []

while not valid_option:
    response = input("\nCreate a new terrain map [y/n]: \n")
    if response == 'y':
        load_new_terrain = True
        valid_option = True
    elif response == 'n':
        load_new_terrain = False
        valid_option = True
    else:
        print("Invalid selection")

valid_option = False

while not valid_option:
    response = input("\nLoad prior trained agent [y/n]: \n")
    if response == 'y':
        load_prior_agent = True
        valid_option = True
    elif response == 'n':
        load_new_agent = False
        valid_option = True
    else:
        print("Invalid selection")

epsilon = input("\nWhat Epsilon would you like to start with [0.0 : 1.0]: \n")

if __name__ == "__main__":
    env = Environment(load_new_terrain=load_new_terrain, load_prior_agent=load_prior_agent, epsilon=float(epsilon))
    env.show_display()