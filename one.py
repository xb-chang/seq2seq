'''
This is an re-implementation of "Sequence to Sequence Learning with Neural
Networks" A few key points:
1. reverse input order / Bidirectional LSTM/RNN;
2. beam search; (Auto-regressive Learning)
3. Encoder-Decoder; (Auto-regressive / Teacher Forcing / Curriculum Learning)
4. BLEU score (4 gram) / Word error rate (WER);

Follows "1 - Sequence to Sequence Learning with Neural Networks.ipynb"

ENV: PyTorch 1.8, torchtext 0.9 and spaCy 3.0, using Python 3.8.

conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0
cudatoolkit=10.2 -c pytorch

pip install torchtext==0.9.0

python -m spacy download en_core_web_sm 
python -m spacy download de_core_news_sm
may need to be download manually


An embedding layer to map each word to its feature vector. (How exactly?)

append a start of sequence (<sos>) and 
end of sequence (<eos>) token 
to the start and end of sentence, respectively.

When training/testing our model, we always know how many words are in our target
sentence, so we stop generating words once we hit that many. During inference it
is common to keep generating words until the model outputs an <eos> token or
after a certain amount of words have been generated.

Once we have our predicted target sentence, $\hat{Y} = \{ \hat{y}_1, \hat{y}_2, ..., \hat{y}_T \}$, 
we compare it against our actual target sentence, $Y = \{ y_1, y_2, ..., y_T \}$, to calculate our loss. 
We then use this loss to update all of the parameters in our model.
'''

from typing import Text
import torch
import torch.nn as nn
import torch.optim as optim

from torchtext.legacy.datasets import Multi30k
from torchtext.legacy.data import Field, BucketIterator

from LSTM_Models import Encoder,Decoder,Seq2Seq
from LSTM_Models import train, evaluate
from utils import init_weights, count_parameters, epoch_time

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

# A tokenizer is used to turn a string containing a sentence into a list of
# individual tokens that make up that string, e.g. "good morning!" becomes
# ["good", "morning", "!"]. We'll start talking about the sentences being a
# sequence of tokens from now, instead of saying they're a sequence of words.

# spaCy has model for each language ("de_core_news_sm" for German and
# "en_core_web_sm" for English) which need to be loaded so we can access the
# tokenizer of each model.
# German tokenizer
spacy_de = spacy.load('de_core_news_sm')
# English tokenizer
spacy_en = spacy.load('en_core_web_sm')

# German is input sequence, we reverse it.
def tokenize_de(text):
    """
    Tokenizes German text from a string into a list of strings (tokens) and reverses it
    """
    return [tok.text for tok in spacy_de.tokenizer(text)][::-1]

def tokenize_en(text):
    """
    Tokenizes English text from a string into a list of strings (tokens)
    """
    return [tok.text for tok in spacy_en.tokenizer(text)]

# # TODO: the fields in object 'tok' is needed. 
# def tokenize_fileds(text):
#     for tok in spacy_en.tokenizer(text):
#         # vars() 函数返回对象object的属性和属性值的字典对象。TypeError: vars()
#         # argument must have __dict__ attribute
#         print(vars(tok))
#         pdb.set_trace()

# tokenize_fileds(en_text)

en_text = 'I am Fool!'
print(tokenize_en(en_text))


SRC = Field(tokenize=tokenize_de,
            init_token = '<sos>',
            eos_token = '<eos>',
            lower = True)
TRG = Field(tokenize=tokenize_en,
            init_token = '<sos>',
            eos_token = '<eos>',
            lower = True)

# download and load the train, validation and test data. Multi30k is used. This
# is a dataset with ~30,000 parallel English, German and French sentences, each
# with ~12 words per sentence.
# extract the German and English Pair with the fields as SRC and TRG;
train_data, valid_data, test_data = Multi30k.splits(exts = ('.de', '.en'), 
                                                    fields = (SRC, TRG))
print(f"Number of training examples: {len(train_data.examples)}")
print(f"Number of validation examples: {len(valid_data.examples)}")
print(f"Number of testing examples: {len(test_data.examples)}")

# making sure the source sentence is reversed;
# vars() 函数返回对象object的属性和属性值的字典对象。
print(vars(train_data.examples[0]))
# whether <sos> <eos> should be counted. Nope

# Build the vocabulary of both language from train data. 

# Using the min_freq argument, we only allow tokens that appear at least 2 times to appear in our
# vocabulary. Tokens that appear only once are converted into an <unk> (unknown) token.
SRC.build_vocab(train_data, min_freq = 2)
TRG.build_vocab(train_data, min_freq = 2)
print(f"Unique tokens in source (de) vocabulary: {len(SRC.vocab)}")
print(f"Unique tokens in target (en) vocabulary: {len(TRG.vocab)}")
# pdb.set_trace()


# Final step of data prepare: create the data iterators. These can be iterated on to return a
# batch of data which will have a 'src' attribute (the PyTorch tensors containing
# a batch of numericalized 'source' sentences) and a 'trg' attribute (the PyTorch
# tensors containing a batch of numericalized 'target' sentences). ##Numericalized##
# is just a fancy way of saying they have been converted from a sequence of
# readable tokens to a sequence of corresponding ##indexes##, using the vocabulary.
# one-hot vector as word label?
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 128

# BucketIterator instead of the standard Iterator as it creates batches in such
# a way that it minimizes the amount of padding in both the source and target
# sentences. (NICE Property!!!!!)
train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
    (train_data, valid_data, test_data), 
    batch_size = BATCH_SIZE, 
    device = device)
# The usage of 'torchtext' here is nice but too gigh level. Reverse the input
# sequence order is considered in data preprocessing, as in 'tokenize_de'.


# Building the Seq2Seq Model（LSTM Based）

# Training the Seq2Seq Model. 
# The embedding (vocabulary) sizes and dropout rates of
# encoder and decoder can be different.
INPUT_DIM = len(SRC.vocab)
OUTPUT_DIM = len(TRG.vocab)

# Input feature dimension
ENC_EMB_DIM = 256
DEC_EMB_DIM = 256
HID_DIM = 512
N_LAYERS = 2
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5

enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)
model = Seq2Seq(enc, dec, device).to(device)

model.apply(init_weights)
print(f'The model has {count_parameters(model):,} trainable parameters')

optimizer = optim.Adam(model.parameters())

# CrossEntropyLoss 
# Our loss function calculates the average loss per token,
# however by passing the index of the <pad> token as the ignore_index argument
# we ignore the loss whenever the target token is a padding token.
# nice feature, how to build it?
TRG_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]
criterion = nn.CrossEntropyLoss(ignore_index = TRG_PAD_IDX)

# SRC_PAD_IDX = SRC.vocab.stoi[SRC.pad_token]
# pdb.set_trace()

N_EPOCHS = 10
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
        torch.save(model.state_dict(), './models/tut1-model.pt')
    
    print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')
    # pdb.set_trace()

model.load_state_dict(torch.load('./models/tut1-model.pt'))
test_loss = evaluate(model, test_iterator, criterion)
print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')

# pdb.set_trace()