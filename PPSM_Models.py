'''
Packed padded sequences & Masking. 
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

import spacy

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import random

import pdb

'''
Encoder: Packed padded sequences applied here.
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
        # packed_embedded & embedded on GPU or not? BOTH on GPU;
        # check the tensor shape after packed
        # pdb.set_trace()

        packed_outputs, hidden = self.rnn(packed_embedded)
        #packed_outputs is a packed sequence containing all hidden states
        #hidden is now from the final non-padded element in the batch
            
        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs) 
        #outputs is now a non-packed sequence, all hidden states obtained
        #  when the input is a pad token are all zeros
            
        #outputs = [src len, batch size, hid dim * num directions]
        #hidden = [n layers * num directions, batch size, hid dim]
        # pdb.set_trace()
        
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

'''
Attention module: Masking is applied here.

Using MASKING to force the attention on non-padding elements ONLY;
The 'forward' method now takes a 'mask', which is a  [batch size, source sentence length] tensor:
1: non padding element; 0: padding element; e.g.,
["hello", "how", "are", "you", "?", <pad>, <pad>], then the mask would be [1, 1, 1, 1, 1, 0, 0].

The mask is applied when the attention has been calculated, but before the attention softmax.
It is applied using 'masked_fill'. Specifically, the padding attention elements will be set to 
a relatively small number (-1e10) before softmax. So, the corresponding SOFTMAXed values will be very close
to zeros.
'''
class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()
        
        self.attn = nn.Linear((enc_hid_dim * 2) + dec_hid_dim, dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1, bias = False)
        
    def forward(self, hidden, encoder_outputs, mask):
        # hidden = [batch size, dec hid dim]
        # encoder_outputs = [src len, batch size, enc hid dim * 2]
        # mask = [batch size, src len]

        batch_size = encoder_outputs.shape[1]
        src_len = encoder_outputs.shape[0]

        #repeat decoder hidden state src_len times
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)

        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        #hidden = [batch size, src len, dec hid dim]
        #encoder_outputs = [batch size, src len, enc hid dim * 2]

        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim = 2))) 
        #energy = [batch size, src len, dec hid dim]

        attention = self.v(energy).squeeze(2)
        #attention = [batch size, src len]
        
        # TRICK:
        # MASKING: Fill before softmax
        attention = attention.masked_fill(mask == 0, -1e10)
        
        return F.softmax(attention, dim = 1)


'''
Decoder: a few small changes
It needs to accept a mask over the source sentence and pass it to the attention module.
The attention tensor is returned for illustration.
'''
class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout, attention):
        super().__init__()

        self.output_dim = output_dim
        self.attention = attention
        
        self.embedding = nn.Embedding(output_dim, emb_dim)
        
        self.rnn = nn.GRU((enc_hid_dim * 2) + emb_dim, dec_hid_dim)
        
        self.fc_out = nn.Linear((enc_hid_dim * 2) + dec_hid_dim + emb_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, input, hidden, encoder_outputs, mask):
        #input = [batch size]
        #hidden = [batch size, dec hid dim]
        #encoder_outputs = [src len, batch size, enc hid dim * 2] (zero padded back to 'src len')
        #mask = [batch size, src len]

        input = input.unsqueeze(0)
        #input = [1, batch size]
        
        embedded = self.dropout(self.embedding(input))
        #embedded = [1, batch size, emb dim]

        # masking on attention
        a = self.attention(hidden, encoder_outputs, mask)
        #a = [batch size, src len]
        
        a = a.unsqueeze(1)
        #a = [batch size, 1, src len]
        
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        #encoder_outputs = [batch size, src len, enc hid dim * 2]

        weighted = torch.bmm(a, encoder_outputs)
        #weighted = [batch size, 1, enc hid dim * 2]
        
        weighted = weighted.permute(1, 0, 2)
        #weighted = [1, batch size, enc hid dim * 2]

        rnn_input = torch.cat((embedded, weighted), dim = 2)
        #rnn_input = [1, batch size, (enc hid dim * 2) + emb dim]
            
        output, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))
        #output = [seq len, batch size, dec hid dim * n directions]
        #hidden = [n layers * n directions, batch size, dec hid dim]
        
        #seq len, n layers and n directions will always be 1 in this decoder, therefore:
        #output = [1, batch size, dec hid dim]
        #hidden = [1, batch size, dec hid dim]
        #this also means that output == hidden
        assert (output == hidden).all()
        
        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted = weighted.squeeze(0)
        
        prediction = self.fc_out(torch.cat((output, weighted, embedded), dim = 1))
        #prediction = [batch size, output dim]
        
        # masked attention is returned.
        return prediction, hidden.squeeze(0), a.squeeze(1)


'''
Seq2seq: full model;
with a few changes for packed padded sequences, masking and inference.
Trick: got the the source sentence length & 'create_mask' function is here.
The attention at each time-step is stored in the 'attentions'.
'''
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, src_pad_idx, device):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx
        # Trick: define device here.
        self.device = device
    
    def create_mask(self, src):
        # mask out the padded tokens.
        mask = (src != self.src_pad_idx).permute(1, 0)
        return mask

    def forward(self, src, src_len, trg, teacher_forcing_ratio = 0.5):
        #src = [src len, batch size] (indexes)
        #src_len = [batch size]
        #trg = [trg len, batch size]
        #teacher_forcing_ratio is probability to use teacher forcing
        #e.g. if teacher_forcing_ratio is 0.75 we use teacher forcing 75% of the time

        batch_size = src.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        
        #tensor to store decoder outputs
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)

        #encoder_outputs is all hidden states of the input sequence, back and forwards
        #hidden is the final forward and backward hidden states, passed through a linear layer (change dimension)
        encoder_outputs, hidden = self.encoder(src, src_len)
                
        #first input to the decoder is the <sos> tokens
        input = trg[0,:]

        # mask of source sequence
        mask = self.create_mask(src)
        #mask = [batch size, src len]

        for t in range(1, trg_len):
            
            #insert input token embedding, previous hidden state, all encoder hidden states 
            #  and mask
            #receive output tensor (predictions) and new hidden state
            output, hidden, _ = self.decoder(input, hidden, encoder_outputs, mask)
            
            #place predictions in a tensor holding predictions for each token
            outputs[t] = output
            
            #decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio
            
            #get the highest predicted token from our predictions
            top1 = output.argmax(1) 
            
            #if teacher forcing, use actual next token as next input
            #if not, use predicted token
            input = trg[t] if teacher_force else top1
            
        return outputs

def train(model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0

    for i, batch in enumerate(iterator):
        
        # Trick: include_lengths = True for our source field, 
        # batch.src is now a tuple:
        # 1. the numericalized tensor;
        # 2. the lengths of each sentence within the batch.
        src, src_len = batch.src

        trg = batch.trg
        
        optimizer.zero_grad()
        
        output = model(src, src_len, trg)
        #trg = [trg len, batch size]
        #output = [trg len, batch size, output dim]
        
        output_dim = output.shape[-1]
        
        output = output[1:].view(-1, output_dim)
        trg = trg[1:].view(-1)
        #trg = [(trg len - 1) * batch size]
        #output = [(trg len - 1) * batch size, output dim]
        
        # with pad_idx ignored in the corss-entropy
        loss = criterion(output, trg)
        
        loss.backward()
        
        # Trick: backward then clip grad norm;
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        optimizer.step()
        
        epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)

def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    
    with torch.no_grad():
    
        for i, batch in enumerate(iterator):

            src, src_len = batch.src
            trg = batch.trg

            output = model(src, src_len, trg, 0) #turn off teacher forcing
            
            #trg = [trg len, batch size]
            #output = [trg len, batch size, output dim]

            output_dim = output.shape[-1]
            
            output = output[1:].view(-1, output_dim)
            trg = trg[1:].view(-1)

            #trg = [(trg len - 1) * batch size]
            #output = [(trg len - 1) * batch size, output dim]

            loss = criterion(output, trg)

            epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)


'''
Inference: Use the trained model to generate translations.
'translate_sentence' is the main inference function.
the model here is 'seq2seq' class.
'''
def translate_sentence(sentence, src_field, trg_field, model, device, max_len = 50):
    # evaluation mode
    model.eval()

    # tokenize: (dividing) the string into words.
    if isinstance(sentence, str):
        nlp = spacy.load('de')
        tokens = [token.text.lower() for token in nlp(sentence)]
    else:
        tokens = [token.lower() for token in sentence]
    
    # add <sos> & <eos> tokens
    tokens = [src_field.init_token] + tokens + [src_field.eos_token]
    # tokens 2 indexes
    src_indexes = [src_field.vocab.stoi[token] for token in tokens]
    # idx tensor
    src_tensor = torch.LongTensor(src_indexes).unsqueeze(1).to(device)
    # input word length
    src_len = torch.LongTensor([len(src_indexes)])

    # encoding the src features and hidden states
    with torch.no_grad():
        encoder_outputs, hidden = model.encoder(src_tensor, src_len)
    
    # mask out the padded tokens.
    mask = model.create_mask(src_tensor)

    # result buffers
    trg_indexes = [trg_field.vocab.stoi[trg_field.init_token]]
    attentions = torch.zeros(max_len, 1, len(src_indexes)).to(device)

    for i in range(max_len):
        # Auto-regressive
        trg_tensor = torch.LongTensor([trg_indexes[-1]]).to(device)
                
        with torch.no_grad():
            output, hidden, attention = model.decoder(trg_tensor, hidden, encoder_outputs, mask)

        attentions[i] = attention
        
        # index
        pred_token = output.argmax(1).item()
        trg_indexes.append(pred_token)

        # reach <eos>, the generation process ends.
        # .stoi: word to index; 
        if pred_token == trg_field.vocab.stoi[trg_field.eos_token]:
            break
    
    # .itos: index to word;
    trg_tokens = [trg_field.vocab.itos[i] for i in trg_indexes]
    
    # returns the target words list & attentions list
    return trg_tokens[1:], attentions[:len(trg_tokens)-1]


# show the attention matrix across source and target
def display_attention(sentence, translation, attention):
    
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111)
    
    attention = attention.squeeze(1).cpu().detach().numpy()
    
    cax = ax.matshow(attention, cmap='bone')
    ax.tick_params(labelsize=15)
    
    x_ticks = [''] + ['<sos>'] + [t.lower() for t in sentence] + ['<eos>']
    y_ticks = [''] + translation

    ax.set_xticklabels(x_ticks, rotation=45)
    ax.set_yticklabels(y_ticks)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()
    plt.close()