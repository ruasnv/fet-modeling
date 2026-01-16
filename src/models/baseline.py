# src/simple_nn.py
import torch.nn as nn

class SimpleNN(nn.Module):
    def __init__(self, input_dim):
        super(SimpleNN, self).__init__()
        self.layer1 = nn.Linear(input_dim, 128)
        self.layer2 = nn.Linear(128, 256)
        self.layer3 = nn.Linear(256, 128)
        self.output_layer = nn.Linear(128, 1)
        self.gelu = nn.GELU()
        # GELU activation
        # Avoid ReLU, dying ReLU issue on values close to zero

        #TODO: This model can be improved with optimizations, dropout etc. Due to resource limits this is the most optimal for my case

    def forward(self, x):
        #To add dropout, use:
        #x = self.dropout(self.gelu(self.layer1(x)))
        x = self.gelu(self.layer1(x))
        x = self.gelu(self.layer2(x))
        x = self.gelu(self.layer3(x))
        return self.output_layer(x)