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

import torch
import torch.nn as nn
import torch.optim as optim

from torchtext.legacy.datasets import Multi30k
from torchtext.legacy.data import Field, BucketIterator

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
# TODO: the fields in object 'tok' is needed. 

en_text = 'I am Fool!'
print(tokenize_en(en_text))

# print('breaked.')
pdb.set_trace()

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
print(vars(train_data.examples[0]))
# whether <sos> <eos> should be counted.


# Build the vocabulary of both language from train data. 

# Using the min_freq argument, we only allow tokens that appear at least 2 times to appear in our
# vocabulary. Tokens that appear only once are converted into an <unk> (unknown) token.
SRC.build_vocab(train_data, min_freq = 2)
TRG.build_vocab(train_data, min_freq = 2)
print(f"Unique tokens in source (de) vocabulary: {len(SRC.vocab)}")
print(f"Unique tokens in target (en) vocabulary: {len(TRG.vocab)}")


# Final step of data prepare: create the data iterators. These can be iterated on to return a
# batch of data which will have a 'src' attribute (the PyTorch tensors containing
# a batch of numericalized 'source' sentences) and a 'trg' attribute (the PyTorch
# tensors containing a batch of numericalized 'target' sentences). ##Numericalized##
# is just a fancy way of saying they have been converted from a sequence of
# readable tokens to a sequence of corresponding ##indexes##, using the vocabulary.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 128

train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
    (train_data, valid_data, test_data), 
    batch_size = BATCH_SIZE, 
    device = device)