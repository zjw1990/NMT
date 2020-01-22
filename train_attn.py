import torch
from load_data import load_words, pairs2tensor
import random
from torch import optim
from encoder import Encoder
from decoder import AttentionDecoder
import torch.nn as nn
import math
import time
from tools import asMinutes, timeSince
from evaluate_attn import evaluateRandomly

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
SOS_TOKEN = 0
EOS_TOKEN = 1
def train_attn():
    n_iters = 75000
    hidden_size = 256
    lr = 0.01
    max_length = 10
    teacher_forcing_ratio = 0.5
    print_every = 5000
    print_loss_total = 0
    src_language, trg_language, pairs = load_words('eng', 'fra')
    train_pairs = [pairs2tensor(src_language, trg_language, random.choice(pairs)) for i in range(n_iters)]
    
    encoder = Encoder(src_language.num_words, hidden_size, device).to(device)
    attn_decoder = AttentionDecoder(hidden_size, trg_language.num_words).to(device)

    encoder_optimizer = optim.SGD(encoder.parameters(), lr = lr)
    decoder_optimizer = optim.SGD(attn_decoder.parameters(), lr = lr)

    loss_func = nn.NLLLoss()
    start = time.time()
    for i in range(1, n_iters+1):
        train_pair = train_pairs[i-1]
        input_tensor = train_pair[0]
        target_tensor = train_pair[1]

        ###################
        loss = 0
        encoder_hidden = encoder.init_hidden() # [1,1,256]
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        input_length = input_tensor.size(0)
        target_length = target_tensor.size(0)

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device) # [10, 256]
        
        loss = 0

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
            encoder_outputs[ei] = encoder_output[0][0]
        
        decoder_input = torch.tensor([[SOS_TOKEN]], device=device) # [1,1]
        decoder_hidden = encoder_hidden  # [1,1, 256]
        
        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

        if use_teacher_forcing:

            for di in range(target_length):
                decoder_output, decoder_hidden, decoder_attn = attn_decoder(decoder_input, decoder_hidden, encoder_outputs)
                loss += loss_func(decoder_output, target_tensor[di])
                decoder_input = target_tensor[di]
        
        else:

            for di in range(target_length):
                decoder_output, decoder_hidden, decoder_attn = attn_decoder(decoder_input, decoder_hidden, encoder_outputs)
                topv, topi = decoder_output.topk(1)

                decoder_input = topi.squeeze().detach()

                loss += loss_func(decoder_output, target_tensor[di])

                if decoder_input.item() == EOS_TOKEN:
                    break
        
        loss.backward()

        encoder_optimizer.step()
        decoder_optimizer.step()


        loss_avg = loss.item() / target_length
        
        #############
        print_loss_total += loss_avg
        
        if i % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, i / n_iters),
                                         i, i / n_iters * 100, print_loss_avg))
       

    
    

    evaluateRandomly(pairs, encoder, attn_decoder, src_language, trg_language)
    











