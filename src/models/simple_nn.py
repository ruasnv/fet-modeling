# src/simple_nn.py
import torch.nn as nn

class SimpleNN(nn.Module):
    def __init__(self, input_dim):
        super(SimpleNN, self).__init__()
        self.layer1 = nn.Linear(input_dim, 128)
        self.layer2 = nn.Linear(128, 256)
        self.layer3 = nn.Linear(256, 128)
        self.output_layer = nn.Linear(128, 1)
        self.gelu = nn.GELU()  # GELU activation, non-linear, smooth
        #Avoid ReLU, dying ReLU issue on values close to zero

        #TODO: If overfitting occurs add dropout between layers
        # self.dropout = nn.Dropout(0.1)
        # Dropout(0.1) between layers for regularization

    def forward(self, x):
        #If you added dropout change to this:
        #x = self.dropout(self.gelu(self.layer1(x)))
        x = self.gelu(self.layer1(x))
        x = self.gelu(self.layer2(x))
        x = self.gelu(self.layer3(x))
        return self.output_layer(x)