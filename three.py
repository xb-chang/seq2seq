'''
In two.py, the context vector is used as direct input for different components
in decoder, to avoid too much data compression.

In this implementation (three.py), the context vector compression can be avoid
by alowing the decoder to look back at the entire source sentence when decoding,
with attention mechnism used. The weighted sum source vector is used as inputs for
our RNN decoder and linear layer to make a prediction.

KEY: Compute the weighted sum source vector when decoding.
'''

from typing import Text
import torch
import torch.nn as nn
import torch.optim as optim

from torchtext.legacy.datasets import Multi30k
from torchtext.legacy.data import Field, BucketIterator

from GRU_Models import Encoder,Decoder,Seq2Seq
from GRU_Models import train, evaluate

import spacy
import numpy as np

import random
import math
import time

import pdb