import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import state_space_parameters as ssp




class Model(nn.Module):

    def __init__(self,network):
        super(Model,self).__init__()
        all_layers = []
        input_channels = ssp.input_channels
        input_features = None
        for layers in network:

            if layers.layer_type == 'Conv':
                all_layers.append(nn.Conv2d(input_channels,layers.channels,layers.kernel_size))
                input_channels = layers.channels

            elif layers.layer_type == 'Pool':
                all_layers.append(nn.MaxPool2d(layers.kernel_size))

            elif layers.layer_type == 'fc':
                if input_features is None:
                    input_features = layers.image_size*input_channels*input_channels
                all_layers.append(nn.Linear(input_features,layers.neurons))
                input_features = layers.neurons
            
            elif layers.layer_type == 'SM':
                all_layers.append(nn.Linear(input_features,ssp.classes))
        

        self.forward_prop = nn.Sequential(*all_layers)

    
    def forward(self,x):

        return self.forward_prop(x)
    


