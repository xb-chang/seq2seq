'''
1. Packed padded sequences: Packed padded sequences allow us to only process the
   non-padded elements of our input sentence with our RNN;

2. Masking: Masking is used to force the model to ignore certain elements we do
   not want it to look at, such as attention over padded elements;
(1 & 2 are somewhat similar function on controling the padded elements;)

3. Inference: view attention values over the source sequence;

4. calculate the BLEU metric from our translations.

Packed Padded Sequence and Masking are two techniques, they are different:
Packed padded sequences are used to tell our RNN to skip over padding tokens in our encoder;
Masking explicitly forces the model to ignore certain values, such as attention over padded elements.
Both common in NLP;
'''

from typing import Text
import torch
import torch.nn as nn
import torch.optim as optim

from torchtext.legacy.datasets import Multi30k
from torchtext.legacy.data import Field, BucketIterator

from PPSM_Models import Encoder,Attention,Decoder,Seq2Seq
from PPSM_Models import train, evaluate

import spacy
import numpy as np

import random
import math
import time

import pdb

SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

spacy_de = spacy.load('de_core_news_sm')
spacy_en = spacy.load('en_core_web_sm')

def tokenize_de(text):
    """
    Tokenizes German text from a string into a list of strings
    """
    return [tok.text for tok in spacy_de.tokenizer(text)]

def tokenize_en(text):
    """
    Tokenizes English text from a string into a list of strings
    """
    return [tok.text for tok in spacy_en.tokenizer(text)]

# TRICK (torchtext related): When using packed padded sequences, we need to tell
# PyTorch how long the actual (non-padded) sequences are. Luckily for us,
# TorchText's Field objects allow us to use the "include_lengths" argument, this
# will cause our "batch.src" to be a tuple.
# The first element of the tuple is the same as before, a batch of numericalized
# source sentence as a tensor, and the second element is the non-padded lengths
# of each source sentence within the batch.
SRC = Field(tokenize = tokenize_de, 
            init_token = '<sos>', 
            eos_token = '<eos>', 
            lower = True, 
            include_lengths = True)

TRG = Field(tokenize = tokenize_en, 
            init_token = '<sos>', 
            eos_token = '<eos>', 
            lower = True)

train_data, valid_data, test_data = Multi30k.splits(exts = ('.de', '.en'), 
                                                    fields = (SRC, TRG))

SRC.build_vocab(train_data, min_freq = 2)
TRG.build_vocab(train_data, min_freq = 2)

BATCH_SIZE = 128

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Trick: One quirk about packed padded sequences is that all elements in the
# batch need to be sorted by their non-padded lengths in descending order, i.e.
# the first sentence in the batch needs to be the longest. 
#
# We use two arguments of the iterator to handle this:
# 'sort_within_batch' which tells the iterator that the contents of the batch need to be sorted;
# 'sort_key' is a function which tells the iterator how to sort the elements in the batch. Here, we sort by the length of the src sentence.
train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
    (train_data, valid_data, test_data), 
     batch_size = BATCH_SIZE,
     sort_within_batch = True,
     sort_key = lambda x : len(x.src),
     device = device)

