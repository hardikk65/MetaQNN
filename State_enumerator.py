
import state_space_parameters as ssp
from helper import *

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


# TODO Calculate image_size while transitioning from state A to B.



class State:
    
    def __init__(self,
            layer_type:str,
            kernel_size:int,
            channels:int,
            strides:int,
            image_size:int,
            layer_depth:int,
            neurons:int,
            fc_layers:int):

        
        self.layer_type = layer_type
        self.kernel_size = kernel_size
        self.channels = channels
        self.strides = strides
        self.image_size = image_size
        self.layer_depth = layer_depth
        self.neurons = neurons
        self.fc_layers = fc_layers

    def __repr__(self) -> str:
        return f"layer_type: {self.layer_type},kernel_size: {self.kernel_size},channels: {self.channels},strides: {self.strides},image_size: {self.image_size},layer_depth: {self.layer_depth},neurons: {self.neurons},fc_layers: {self.fc_layers}"



'''

    Possible Transitions:

    Conv ----> Conv,Pool,Fc
    Pool ----> Conv,Fc
    Fc   ----> Fc,SM


'''

class transitions:

    def __init__(self,A:State):
        self.layer_type = A.layer_type
        self.kernel_size = A.kernel_size
        self.channels = A.channels
        self.strides = A.strides
        self.image_size = A.image_size
        self.layer_depth = A.layer_depth
        self.neurons = A.neurons
        self.fc_layers = A.fc_layers

    
    def possible_transitions(self):
        
        all_transitions = []
        if self.layer_depth == 11 or self.fc_layers == 2:
            all_transitions.append(
                State("SM",kernel_size=-1,channels=-1,strides=-1,image_size=-1,layer_depth=self.layer_depth + 1,neurons=0,fc_layers=self.fc_layers)
                )


        else:

            if self.layer_type == 'Conv':

                # Conv to Conv Transition
                if pool_size(self.image_size) >= 8 and self.layer_depth < 10:

                    for kernels in ssp.kernel_size:
                        for channels in ssp.Channels:
                            all_transitions.append(
                                State(layer_type="Conv",kernel_size=kernels,channels=channels,strides=1,image_size=pool_size(self.image_size),layer_depth=self.layer_depth + 1,neurons=self.neurons,fc_layers=self.fc_layers)
                            )
                
                    # Conv to Pool Transition
                    for kernels in ssp.kernel_size:
                        all_transitions.append(
                            State(layer_type="Pool",kernel_size=kernels,channels=self.channels,strides = 2,image_size=pool_size(self.image_size),layer_depth=self.layer_depth + 1,neurons=self.neurons,fc_layers = self.fc_layers)
                        )

                # Conv to fc Transition

                if (self.fc_layers < 2) and (pool_size(self.image_size) < 8 and pool_size(self.image_size) >= 4):
                    for neurons in ssp.neurons:

                        if self.neurons <= neurons:
                            all_transitions.append(
                                State(layer_type="fc",kernel_size=-1,channels = -1,strides = -1,image_size=self.image_size,layer_depth=self.layer_depth + 1,neurons = neurons,fc_layers=self.fc_layers + 1)
                            )
                    

            elif self.layer_type == "Pool":
                
                # Pool to Conv Transition

                if self.layer_depth < 10:
                    for kernels in ssp.kernel_size:
                        for channels in ssp.Channels:
                            all_transitions.append(
                                State(layer_type="Conv",kernel_size=kernels,channels=channels,strides=1,image_size=pool_size(self.image_size),layer_depth=self.layer_depth + 1,neurons=self.neurons,fc_layers=self.fc_layers)
                            )


                # Pool to fc Transition
                if (self.fc_layers < 2) and (pool_size(self.image_size) < 8 and pool_size(self.image_size) >= 4):
                    for neurons in ssp.neurons:
                        if self.neurons <= neurons:
                            all_transitions.append(
                                State(layer_type="fc",kernel_size=-1,channels = -1,strides = -1,image_size=self.image_size,layer_depth=self.layer_depth + 1,neurons = neurons,fc_layers=self.fc_layers + 1,)
                            )


            elif self.layer_type == 'fc':

                if self.fc_layers < 2:
                    for neurons in ssp.neurons:
                        if self.neurons <= neurons:
                            all_transitions.append(
                                State(layer_type="fc",kernel_size=-1,channels = -1,strides = -1,image_size=self.image_size,layer_depth=self.layer_depth + 1,neurons = neurons,fc_layers=self.fc_layers + 1,)
                            )
                
                all_transitions.append(
                State("SM",kernel_size=-1,channels=-1,strides=-1,image_size=-1,layer_depth=self.layer_depth + 1,neurons=0,fc_layers=0)
                )
        
        return all_transitions
    


# input_state = State(layer_type = "fc",kernel_size=1,channels = 64,strides = 1,image_size=8,layer_depth=1,neurons=512,fc_layers=0)


# transition = transitions(input_state)

# actions = transition.possible_transitions()


# for states in actions:
#     print(states)
                    


            




