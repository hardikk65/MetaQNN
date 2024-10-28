from State_enumerator import *
import state_space_parameters as ssp
import random
from helper import *


class agent:

    def __init__(self,input_state:State):

        self.state_sequence = [input_state]
        self.action_sequence = []

    
    def sample_network(self):

        while self.state_sequence[-1].layer_type != 'SM':
            alpha = random.uniform(0,1)

            all_actions,q_values = transitions(self.state_sequence[-1]).possible_transitions()  
            # TODO resolve no transition error in state_enumerator
            if alpha > ssp.epsilon:
                actions = max(q_values)
                index = [i for i in range(len(q_values)) if q_values[i] == actions]
                action = actions[random.randrange(len(index))]
                new_state = action
            else:
                action = all_actions[random.randrange(len(all_actions))]
                new_state = action

        
            self.action_sequence.append(new_state)
            new_state.image_size = calculate_image_size(self.state_sequence[-1].image_size,self.state_sequence[-1].kernel_size,self.state_sequence[-1].strides)
            self.state_sequence.append(new_state)

        return self.state_sequence
    



if __name__ == "__main__":
    input_state = State(layer_type='Conv',kernel_size=3,channels=64,strides = 1,image_size=32,layer_depth=1,neurons=0,fc_layers=0)
    network = agent(input_state=input_state).sample_network()

    print(network)

            


            
                





