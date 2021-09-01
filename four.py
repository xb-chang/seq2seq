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
from PPSM_Models import translate_sentence, display_attention
from metric import calculate_bleu

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

# BATCH_SIZE = 128
BATCH_SIZE = 32

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

INPUT_DIM = len(SRC.vocab)
OUTPUT_DIM = len(TRG.vocab)
ENC_EMB_DIM = 256
DEC_EMB_DIM = 256
ENC_HID_DIM = 512
DEC_HID_DIM = 512
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5
SRC_PAD_IDX = SRC.vocab.stoi[SRC.pad_token]

attn = Attention(ENC_HID_DIM, DEC_HID_DIM)
enc = Encoder(INPUT_DIM, ENC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, ENC_DROPOUT)
dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, DEC_DROPOUT, attn)

model = Seq2Seq(enc, dec, SRC_PAD_IDX, device).to(device)

def init_weights(m):
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.normal_(param.data, mean=0, std=0.01)
        else:
            nn.init.constant_(param.data, 0)
            
model.apply(init_weights)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'The model has {count_parameters(model):,} trainable parameters')

optimizer = optim.Adam(model.parameters())

TRG_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]

# Trick: cross-entropy loss by ignoring the padded token.
criterion = nn.CrossEntropyLoss(ignore_index = TRG_PAD_IDX)

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

# N_EPOCHS = 10
N_EPOCHS = 3
CLIP = 1

best_valid_loss = float('inf')

for epoch in range(N_EPOCHS):
    
    start_time = time.time()
    
    train_loss = train(model, train_iterator, optimizer, criterion, CLIP)
    valid_loss = evaluate(model, valid_iterator, criterion)
    
    end_time = time.time()
    
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'models/tut4-model.pt')
    
    print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')


model.load_state_dict(torch.load('models/tut4-model.pt'))
test_loss = evaluate(model, test_iterator, criterion)
print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')

# Inference and display attention.
def infer_disp(example_idx):
    src = vars(train_data.examples[example_idx])['src']
    trg = vars(train_data.examples[example_idx])['trg']

    print(f'src = {src}')
    print(f'trg = {trg}')

    # inference
    translation, attention = translate_sentence(src, SRC, TRG, model, device)
    print(f'predicted trg = {translation}')

    display_attention(src, translation, attention)

example_idx = 12
infer_disp(example_idx)

example_idx = 14
infer_disp(example_idx)

example_idx = 18
infer_disp(example_idx)

bleu_score = calculate_bleu(test_data, SRC, TRG, model, device)

print(f'BLEU score = {bleu_score*100:.2f}')