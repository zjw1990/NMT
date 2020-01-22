import torch.nn as nn
from torch import optim
import torch.nn.functional as F 
import torch


class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size, device):
        super(Decoder, self).__init__()

        self.device = device

        self.hidden_size = hidden_size

        self.output_size = output_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out_layer = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)


    def forward(self, input, hidden):
        output = self.embedding(input).view(1,1,-1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out_layer(output[0]))

        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size, device= self.device)



device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class AttentionDecoder(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p = 0.1, max_length = 10):
        super(AttentionDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length


        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.W1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.W2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.V = nn.Linear(self.hidden_size, 1)
        self.tanh = nn.Tanh()
        self.attention_combine = nn.Linear(self.hidden_size*2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)
        self.softmax = nn.Softmax(dim = 0)

    def forward(self, input, hidden, values):
       
        query = self.embedding(input).view(1, -1)
        query = self.dropout(query) #[1, 1, 256]

        queries = query.expand(self.max_length, self.hidden_size)


        score = self.V(self.tanh(self.W1(queries)+self.W2(values)))
        
        
        # calculate attention
        attention_weights =  self.softmax(score) # [1, 10]
        attention_weights = attention_weights.T
        # b, n, m   --  b, m, p -----> b, n, p
        context_vector = torch.bmm(attention_weights.unsqueeze(0), values.unsqueeze(0)) # [1, 1, 256]
        
        
        # decoder
        output = torch.cat((query, context_vector[0]), 1)  # [1, 512]
        output = self.attention_combine(output).unsqueeze(0) # [1, 1, 256]

        output = F.relu(output)# [1, 1, 256]
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attention_weights


    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=self.device)



