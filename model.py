import nltk
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class DynamicLSTM(nn.Module):
    """
    Dynamic LSTM module, which can handle variable length input sequence.

    Parameters
    ----------
    input_size : input size
    hidden_size : hidden size
    num_layers : number of hidden layers. Default: 1
    dropout : dropout rate. Default: 0.5
    bidirectional : If True, becomes a bidirectional RNN. Default: False.

    Inputs
    ------
    input: tensor, shaped [batch, max_step, input_size]
    seq_lens: tensor, shaped [batch], sequence lengths of batch

    Outputs
    -------
    output: tensor, shaped [batch, max_step, num_directions * hidden_size],
         tensor containing the output features (h_t) from the last layer
         of the LSTM, for each t.
    """

    def __init__(self, input_size, hidden_size=100,
                 num_layers=1, dropout=0., bidirectional=False):
        super(DynamicLSTM, self).__init__()

        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers, bias=True,
            batch_first=True, dropout=dropout, bidirectional=bidirectional)

    def forward(self, x, seq_lens):
        # sort input by descending length
        _, idx_sort = torch.sort(seq_lens, dim=0, descending=True)
        _, idx_unsort = torch.sort(idx_sort, dim=0)
        x_sort = torch.index_select(x, dim=0, index=idx_sort)
        seq_lens_sort = torch.index_select(seq_lens, dim=0, index=idx_sort)

        # pack input
        x_packed = pack_padded_sequence(
            x_sort, seq_lens_sort, batch_first=True)

        # pass through rnn
        y_packed, (h_s,c_s) = self.lstm(x_packed)
        
        # unpack output
        y_sort, length = pad_packed_sequence(y_packed, batch_first=True)

        # unsort output to original order
        y = torch.index_select(y_sort, dim=0, index=idx_unsort)
        h_s = h_s.transpose(0,1).contiguous().view(len(seq_lens), -1)
        h_s = torch.index_select(h_s, dim=0, index=idx_unsort)

        return y,h_s

class MalayalamModel(nn.Module):
    
    def __init__(self,pretrained_embed,padding_idx,pretrained_embed_load=True):
        super(MalayalamModel, self).__init__()
        embed_dim = 200
        num_classes = 6
        num_layers = 1
        hidden_dim = 100
        dropout = 0.5
        if pretrained_embed_load:
            self.embed = nn.Embedding.from_pretrained(pretrained_embed, freeze=False)
        else:
            self.embed = nn.Embedding(96055,embed_dim)
        self.embed.padding_idx = padding_idx
        self.lstm = DynamicLSTM(embed_dim, hidden_dim, num_layers=num_layers,dropout=dropout, bidirectional=True)
        self.out = nn.Linear(hidden_dim*2, num_classes)
        self.loss = nn.CrossEntropyLoss(reduce='mean')

    def forward(self,word_seq,seq_len):
        # Embedding
        e = self.embed(word_seq)
        # LSTM
        r,h_s = self.lstm(e, seq_len)
        # Dense
        logits = self.out(h_s)
        
        return logits,F.log_softmax(logits,dim=1)
