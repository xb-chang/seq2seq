from __future__ import unicode_literals, print_function, division
import string
import random

from helper import prepareData

import torch
import torch.nn as nn
from torch import optim

import pdb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# pairs are the sentences with different language (source: french; target: english)
input_lang, output_lang, pairs = prepareData('eng', 'fra', True)
print(random.choice(pairs))

pdb.set_trace()