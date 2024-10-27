from State_enumerator import *
import state_space_parameters as ssp
import random


class agent:

    def __init__(self,input_state:State):

        self.state_sequence = [input_state]
        self.action_sequence = []

    
    def sample_network(self):

        while self.action_sequence[-1].layer_type != 'SM':
            alpha = random.uniform(0,1)

            all_actions = transitions(self.state_sequence[-1]).possible_transitions()  
            # TODO get q_values of actions and then select what action to take
            if alpha > ssp.epsilon:
                actions = max(all_actions)
                index = [i for i in range(len(all_actions)) if all_actions[i] == actions]
                action = actions[random.randrange(len(index))]
                new_state = action
            else:
                action = actions[random.randrange(len(all_actions))]
                new_state = action
            self.action_sequence.append(action)

            if action.terminate != 1:
                self.state_sequence.append(new_state)


            
                





