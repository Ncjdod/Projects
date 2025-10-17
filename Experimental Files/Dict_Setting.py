import numpy as np
import random

quit_probability = 0.1
random_value = random.random() 

if random_value < quit_probability:
    command_choice = 'quit'
else:
    command_choice = 'push'

if command_choice == 'push':
    force_1 = (random.random() * 2) - 1 
    force_2 = (random.random() * 2) - 1
    
    command_dict = {
        'command': 'push',
        'force_vector': [force_1, force_2]
    }
else: 
    command_dict = {'command': 'quit'}

