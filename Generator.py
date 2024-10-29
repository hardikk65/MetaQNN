
from parser import Model
from agent import *
input_state = State(layer_type='Conv',kernel_size=3,channels=64,strides = 1,image_size=32,layer_depth=1,neurons=0,fc_layers=0)
network = agent(input_state=input_state).sample_network()
model = Model(network=network)
print(model)
