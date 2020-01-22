import random
import re
from tools import unicodeToAscii, normalizeString, filter_pairs
from io import open
import torch
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

SOS_TOKEN = 0
EOS_TOKEN = 1

class Language:
    def __init__(self, name):

        self.name = name
        self.word2idx = {}
        self.word2count = {}
        self.idx2word = {0:'SOS', 1:'EOS'}
        self.num_words = 2 # EOS & SOS added first

    def add_sentence(self, sentence):
        for word in sentence.split(' '):
            self.add_word(word)
    

    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.num_words
            self.word2count[word] = 1
            self.idx2word[self.num_words] = word
            self.num_words += 1
        
        else:
            self.word2count[word] += 1



def read_language(input_language1, input_language2, reverse = False):
    with open("./data/eng-fra.txt", encoding='utf-8') as f:
        lines = f.readlines()

    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        src_lanugage = Language(input_language2)
        trg_language = Language(input_language1)
    else:
        src_lanugage = Language(input_language1)
        trg_language = Language(input_language2)
    
    return src_lanugage, trg_language, pairs


def load_words(input_language1, input_language2, reverse = False):
    src_lanugage, trg_language, pairs = read_language(input_language1, input_language2, reverse=reverse)
    
    pairs = filter_pairs(pairs)

    for pair in pairs:
        src_lanugage.add_sentence(pair[0])
        trg_language.add_sentence(pair[1])
    return src_lanugage, trg_language, pairs


def sentence2idx(language, sentence):
    return [language.word2idx[word] for word in sentence.split(' ')]


def sentence2tensor(language, sentence):
    idxes = sentence2idx(language, sentence)
    idxes.append(EOS_TOKEN)
    return torch.tensor(idxes, dtype=torch.long, device=device).view(-1, 1)


def pairs2tensor(src_language, trg_language, pair):
    input_tensor = sentence2tensor(src_language, pair[0])
    target_tensor = sentence2tensor(trg_language, pair[1])
    return(input_tensor, target_tensor)

