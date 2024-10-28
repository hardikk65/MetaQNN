from State_enumerator import *
from state_space_parameters import * 
import pandas as pd


# State (layer_type,kernel_size,channels,strides,image_size,layer_depth,neurons,fc_layers)
# Pool image size in >=8, >= 4 && < 8, <4

# Start-State ----> end-State , Utility
# 1. Pick start state
# 2. generate possible transitions
# 3. Append in db with init utility 0.5

def create_state_transition_db():

    state_transition_db = {
        "Start-State":[],
        "End-State":[],
        "Utility":[]
    }

    for layer_type in layer_types[:3]:

        # Conv and Pool Transitions
        if layer_type != 'fc':
            for size in kernel_size:
                for channels in Channels:
                    for n in repr_size:
                        for depth in range(1,11):

                            if layer_type == 'Conv':  
                                start_state = State(layer_type=layer_type,kernel_size=size,channels=channels,strides = 1,image_size=n,layer_depth=depth,neurons=0,fc_layers=0)
                            else:
                                start_state = State(layer_type=layer_type,kernel_size=size,channels=channels,strides = 2,image_size=n,layer_depth=depth,neurons=0,fc_layers=0)

                            end_states,_ = transitions(start_state).possible_transitions()

                            state_transition_db['Start-State'].extend([start_state.__repr__()]*len(end_states))
                            state_transition_db['End-State'].extend(end_states)
                            state_transition_db['Utility'].extend([0.5]*len(end_states))


        else:
            for neuron in neurons:
                for fc_layer in range(2):
                    for depth in range(1,12):
                        start_state = State(layer_type=layer_type,kernel_size=-1,channels=-1,strides =-1,image_size=-1,layer_depth=depth,neurons=neuron,fc_layers=fc_layer)

                        end_states,_ = transitions(start_state).possible_transitions()
                        state_transition_db['Start-State'].extend([start_state.__repr__()]*len(end_states))
                        state_transition_db['End-State'].extend(end_states)
                        state_transition_db['Utility'].extend([0.5]*len(end_states))



    return state_transition_db



if __name__ == "__main__":

    state_db = create_state_transition_db()
    db = pd.DataFrame(data = state_db,columns=list(state_db.keys()))
    db.to_csv("q_val.csv",index=False)






    

    