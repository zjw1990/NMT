import torch
import torch.nn as nn
from load_data import sentence2tensor
import random
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
SOS_TOKEN = 0
EOS_TOKEN = 1

def evaluate(encoder, decoder, src_language, trg_language, sentence, max_length):
    with torch.no_grad():
        input_tensor = sentence2tensor(src_language, sentence)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.init_hidden()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_TOKEN]], device=device)  # SOS

        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            
            if topi.item() == EOS_TOKEN:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(trg_language.idx2word[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words


def evaluateRandomly(pairs, encoder, decoder, src_language, trg_language, n=10):
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words = evaluate(encoder, decoder, src_language, trg_language, pair[0], n)
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')