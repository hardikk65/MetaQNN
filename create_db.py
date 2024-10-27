from State_enumerator import *
from state_space_parameters import * 
import pandas as pd


# State (layer_type,kernel_size,channels,strides,image_size,layer_depth,neurons,fc_layers)
# Pool image size in >=8, >= 4 && < 8, <4

# Start-State ----> end-State , Utility

'''
    Types of Layers:
    1. Conv (Conv) 
    2. Pooling (Pool)
    3. Fully Connected (fc)
    4. Global Average Pooling (GAP)
    5. Softmax (SM)


    Attributes:
    layer_depth < 12
    Representation_Size -----> {n >= 8, 4 <= n < 8 ,n < 4}

    kernel_size ---> {1,3,5}
    strides ----> {1 (Conv), 2(Pool)}
    No. of Channels ----> {64,128,256,512} (Conv), {Prev_layer channels} (Pooling)


    Consecutive_layers -----> {n < 3}
    Neurons -------> {512,256,128}


    s ---->previous_state
    Global_average_pooling/Softmax -----> Termination_State

'''


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

                            end_states = transitions(start_state).possible_transitions()

                            state_transition_db['Start-State'].extend([start_state.__repr__()]*len(end_states))
                            state_transition_db['End-State'].extend(end_states)
                            state_transition_db['Utility'].extend([0.5]*len(end_states))

    return state_transition_db



state_db = create_state_transition_db()

db = pd.DataFrame(data = state_db,columns=list(state_db.keys()))

db.to_csv("q_val.csv",index=False)






    

    