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

python -m spacy download en_core_web_sm python -m spacy download de_core_news_sm


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

en_text = 'I am Fool!'
print(tokenize_en(en_text))

# print('breaked.')
pdb.set_trace()