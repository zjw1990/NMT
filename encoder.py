import torch.nn as nn
from torch import optim
import torch.nn.functional as F 
import torch


class Encoder(nn.Module):

    def __init__(self, input_size, hidden_size, device):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.device = device
    
    def forward(self, input, hidden):
        embeddings = self.embedding(input).view(1, 1, -1)
        output = embeddings
        output, hidden = self.gru(output, hidden)

        return output, hidden
    
    def init_hidden(self):
        return torch.zeros(1,1, self.hidden_size, device=self.device)
