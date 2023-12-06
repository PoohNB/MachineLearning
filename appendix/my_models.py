
from torch import nn
import torch.nn.functional as F

class POOH_CNN(nn.Module):

    """
    everything square because it easy
    conv_lc : [input,filter, ..]  (chanel,kernal size) 
    linear_lc : [node1,node2,...,output]
    

    example:

    images shape (224,224,3)
    output 2 class

    conv_lc = [(3,224),(6,5),(16,3)]
    linear_lc = [120,84,2]

    
    """

    def __init__(self,conv_lc,linear_lc):
        super().__init__()


        self.conv_list = nn.ModuleList([
                                        nn.Conv2d(conv_lc[i][0],conv_lc[i+1][0],kernel_size=conv_lc[i+1][1] ,stride=1,padding=1) 
                                        for i in range(len(conv_lc[:-1])) 
                                        ])
        
        curr_shape = conv_lc[0][1]
        for c in conv_lc[1:]:
            curr_shape = self.cal_convshape(I = curr_shape,F=c[1])

        self.conv_out = int(conv_lc[-1][0]*curr_shape**2)

        linear_lc = [self.conv_out]+linear_lc

        self.linear_list = nn.ModuleList([
                                        nn.Linear(linear_lc[i],linear_lc[i+1])
                                          for i in range(len(linear_lc[:-1]))
                                          ])

    def cal_convshape(self,I,F,P=1,S=1):
        return ((I-F+2*P)/S)+1
    
    def cal_pad(self,F,S):
        pass

    def forward(self,images):
        
        output = images
        for layer in self.conv_list:
            output = F.relu(layer(output))

        output = output.reshape(-1,self.conv_out)
        for layer in self.linear_list[:-1]:
            output = F.relu(layer(output))

        output = self.linear_list[-1](output)

        return output





