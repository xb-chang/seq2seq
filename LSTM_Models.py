'''
Building the Seq2Seq Model: 1.Encoder; 2.Decoder; 3. seq2seq model.
LSTM Based
'''

import torch
import torch.nn as nn

import random

import pdb

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, bid=False, dropout=0.):
        super().__init__()
        
        # input_dim: input vocab/embed size
        self.input_dim = input_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers

        # The embedding layer is created using nn.Embedding, which is converted
        # words into dense vectors; 
        # A simple lookup table that stores embeddings
        # of a fixed dictionary and size. This module is often used to store
        # word embeddings and retrieve them using indices. The input to the
        # module is a list of indices, and the output is the corresponding word
        # embeddings. 
        # Embedding.weight (Tensor) – the learnable weights of the
        # module of shape (num_embeddings, embedding_dim) initialized from
        # \mathcal{N}(0, 1)N(0,1)
        self.embedding = nn.Embedding(input_dim, emb_dim)

        # Flag: batch_first 
        # False (Default): input shape (seq, batch, feature)
        # True: (batch, seq, feature)
        # Flag: bidirectional; False (Default).
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, bidirectional=bid, dropout=dropout)

        self.dropout = nn.Dropout(dropout)
    
    def forward(self, src):

        #src = [src len, batch size], ||...|
        embedded = self.dropout(self.embedding(src))
        #embedded = [src len, batch size, emb dim]

        # Outputs, Hidden_states and Cell_states;
        outputs, (hidden, cell) = self.rnn(embedded)
        #outputs = [src len, batch size, hid dim * n directions]
        #hidden = [n layers * n directions, batch size, hid dim]
        #cell = [n layers * n directions, batch size, hid dim]

        # outputs are not necessary for decoder.
        return hidden, cell


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, bid=False, dropout=0.):
        super().__init__()

        #output_dim: output vocab/embed size
        self.output_dim = output_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers

        self.embedding = nn.Embedding(output_dim, emb_dim)

        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, bidirectional=bid, dropout=dropout)

        # predicting the output class
        if not bid:
            self.fc_out = nn.Linear(hid_dim, output_dim)
        else:
            # bidirectional, output features are 2*hid_dim
            self.fc_out = nn.Linear(hid_dim*2, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell):
        '''
        Within the forward method, we accept a batch of input tokens, previous
        hidden states and previous cell states. As we are only decoding one
        token at a time, the input tokens will always have a sequence length of
        1. We unsqueeze the input tokens to add a sentence length dimension of
        1. Then, similar to the encoder, we pass through an embedding layer and
        apply dropout. This batch of embedded tokens is then passed into the RNN
        with the previous hidden and cell states. This produces an output
        (hidden state from the top layer of the RNN), a new hidden state (one
        for each layer, stacked on top of each other) and a new cell state (also
        one per layer, stacked on top of each other). We then pass the output
        (after getting rid of the sentence length dimension) through the linear
        layer to receive our prediction. We then return the prediction, the new
        hidden state and the new cell state.

        Note:as we always have a sequence length of 1, we could use
        nn.LSTMCell, instead of nn.LSTM, as it is designed to handle a batch
        of inputs that aren't necessarily in a sequence.
        '''
        # input: a list of <sos> tokens?
        #input = [batch size]
        # previous hidden state
        #hidden = [n layers * n directions, batch size, hid dim]
        # previous cell state
        #cell = [n layers * n directions, batch size, hid dim]

        # seq len is 1
        input = input.unsqueeze(0)
        #input = [1, batch size]
        
        embedded = self.dropout(self.embedding(input))
        #embedded = [1, batch size, emb dim]

        # one step prediction, output seq len is 1;
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        #output = [1, batch size, hid dim * n directions]
        #hidden = [n layers * n directions, batch size, hid dim]
        #cell = [n layers * n directions, batch size, hid dim]

        # one step prediction
        prediction = self.fc_out(output.squeeze(0))
        # prediction = [batch size, output dim] (can make it as prob)

        return prediction, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

        assert encoder.hid_dim == decoder.hid_dim, \
            "Hidden dimensions of encoder and decoder must be equal!"
        # In theory, encoder.n_layers >= decoder.n_layers, with more control added.
        assert encoder.n_layers == decoder.n_layers, \
            "Encoder and decoder must have equal number of layers!"
    
    def forward(self, src, trg, teacher_forcing_ratio = 0.5):
        '''
        receiving the input/source sentence;
        using the encoder to produce the context vectors;
        using the decoder to produce the predicted output/target sentence;
        '''

        #src = [src len, batch size]
        #trg = [trg len, batch size]
        #teacher_forcing_ratio is probability to use teacher forcing
        #e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time

        batch_size = trg.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim

        #tensor to store decoder outputs (a buff to save each prediction)
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)

        # Init states and inputs for decoder
        #last hidden state of the encoder is used as the initial hidden state of the decoder
        hidden, cell = self.encoder(src)
        #first input to the decoder is the <sos> tokens
        input = trg[0,:]
        # double check <sos> token if used
        # why not input = trg[0] ?
        pdb.set_trace()

        # Decoder: Auto-Regressive Way
        for t in range(1, trg_len):
            # insert input and states
            # get next outputs and states
            # output = [batch size, vocab size] (predict vectors)
            output, hidden, cell = self.decoder(input, hidden, cell)

            #place predictions in a tensor holding predict scores for each time step
            outputs[t] = output
            #get the highest predicted token from our predictions
            top1 = output.argmax(1) 

            #decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio

            #if teacher forcing, use actual next token as next input
            #if not, use predicted token
            input = trg[t] if teacher_force else top1
        
        return outputs
