'''
Attention Decoder If only the contect vector is passed encoder and decoder, that
single vector carries the burden of encoding the entire sentence.

Attention allows the decoder network to "focus" on a different part of the
encoder's outputs for every step of the decoder's own outputs.

First we calculate a set of attention weights. THese will be multiplied by the
encoder output vectors to create a weighted combination. The result (called
'attn_applied' in the code) should contain information about the specific part
of the input sequence, and thus help the decoder choose the right output words.

Calculating the attention weight is done with another feed forward layer 'attn',
using the decoder's input and hidden state as inputs. Because there are
sentences of all sizes in the training data, to acrually create and train this
layer we have to choose a maximum sentence length (input length, for encoder
outputs) that it can apply to. Sentences of the maximum length will use all the
attention weights, while shorter sentences will only use the first few.

Q:
attention input/output temporal relation;
where are the inputs and pre_hidden from?
Adcanced, combined with prototype sequences?
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAX_LENGTH = 10

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        # different ways of attention? 
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)

        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)
    
    def forward(self, input, hidden, encoder_outputs):
        # the batch size is 1 in this tutorial
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights
    
    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)