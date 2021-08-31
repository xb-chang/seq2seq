'''
Packed padded sequences & Masking. 
'''
import torch
import torch.nn as nn

import pdb

'''
Encoder 
The changes here all within the forward method. It now accepts: 
1. the lengths of the source sentences;
2. the sentences themselves.

Packing: After the source sentence (padded automatically within the iterator)
has been embedded, we can then use 'pack_padded_sequence' on it with the lengths
of the sentences. Note that the tensor containing the lengths of the sequences
must be a CPU tensor as of the latest version of PyTorch, which we explicitly do
so with "to('cpu')". packed_embedded will then be our packed padded sequence.

This can be then fed to our RNN as normal which will return packed_outputs, a
packed tensor containing all of the hidden states from the sequence, hidden
which is simply the final hidden state from our sequence. hidden is a standard
tensor and not packed in any way, the only difference is that as the input was a
packed sequence, this tensor is from the final non-padded element in the
sequence.

Unpack: 
unpack our 'packed_outputs' using 'pad_packed_sequence' which returns
the 'outputs' and the lengths of each, which we don't need.
The padded values in 'outputs' are all zeros.
'''

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        
        self.rnn = nn.GRU(emb_dim, enc_hid_dim, bidirectional = True)
        
        self.fc = nn.Linear(enc_hid_dim * 2, dec_hid_dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, src, src_len):
        # 'src, src_len' are paired and returned from the data iterator,
        # as the 'include_lengths = True' is set in the source field.
        # src = [src len, batch size]
        # src_len = [batch size]
        
        embedded = self.dropout(self.embedding(src))
        #embedded = [src len, batch size, emb dim]

        #need to explicitly put lengths on cpu!
        # TRICK: important utils;
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, src_len.to('cpu'))
        # packed_embedded & embedded on GPU or not?
        # check the tensor shape after packed
        pdb.set_trace()

        packed_outputs, hidden = self.rnn(packed_embedded)
        #packed_outputs is a packed sequence containing all hidden states
        #hidden is now from the final non-padded element in the batch
            
        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs) 
        #outputs is now a non-packed sequence, all hidden states obtained
        #  when the input is a pad token are all zeros
            
        #outputs = [src len, batch size, hid dim * num directions]
        #hidden = [n layers * num directions, batch size, hid dim]
        pdb.set_trace()
        
        #hidden is stacked [forward_1, backward_1, forward_2, backward_2, ...]
        #outputs are always from the last layer
        
        #hidden [-2, :, : ] is the last of the forwards RNN 
        #hidden [-1, :, : ] is the last of the backwards RNN

        #initial decoder hidden is final hidden state of the forwards and backwards 
        #  encoder RNNs fed through a linear layer
        # Trick: 'torch.tanh' is the activation function to make an output becomes LSTM hidden state;
        hidden = torch.tanh(self.fc(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)))
        
        #outputs = [src len, batch size, enc hid dim * 2]
        #hidden = [batch size, dec hid dim]
        
        return outputs, hidden