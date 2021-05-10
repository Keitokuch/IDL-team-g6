import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn
import torch.nn.functional as F
from constant import LETTER_LIST

from __main__ import device


class LockedDropoutCell(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
        self.mask = None

    def forward(self, x):
        if not self.training or not self.p:
            return x
        if self.mask is None:
            x = x.clone()
            mask = x.new_empty(x.size(0), x.size(1), requires_grad=False).bernoulli_(1 - self.p)
            mask = mask.div_(1 - self.p)
            self.mask = mask
        return x * self.mask

class LockedDropout(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self.training or not self.p:
            return x
        x = x.clone()
        mask = x.new_empty(x.size(0), 1, x.size(2), requires_grad=False).bernoulli_(1 - self.p)
        mask = mask.div_(1 - self.p)
        mask = mask.expand_as(x)
        self.mask = mask
        return x * self.mask


class pBLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.blstm = nn.LSTM(input_size=input_dim*2, hidden_size=hidden_dim, 
                             num_layers=2, dropout=dropout, bidirectional=True, batch_first=True)
        self.dropout = LockedDropout(dropout)
        
    def forward(self, x):
        x, x_lens = rnn.pad_packed_sequence(x, batch_first=True)
        B, L, d = x.shape
        L = (L//2 * 2)
        x = x[:, :L, :]
        x_lens = x_lens / 2
        x = torch.reshape(x, (B, L//2, d*2))
        x = self.dropout(x)
        packed_x = rnn.pack_padded_sequence(x, x_lens.cpu(), batch_first=True, enforce_sorted=False)
        out, hidden = self.blstm(packed_x)
        return out

class Encoder(nn.Module):
    def __init__(self, input_dim, encoder_dim, num_layers=3, dropout=0.0, key_value_size=128, simple=False):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=encoder_dim, num_layers=2, dropout=dropout,
                            bidirectional=True, batch_first=True)
        self.pblstms = nn.Sequential(*[pBLSTM(encoder_dim*2, encoder_dim, dropout) for _ in range(num_layers)])
        self.key_network = nn.Linear(encoder_dim * 2, key_value_size, bias=False)
        self.value_network = nn.Linear(encoder_dim * 2, key_value_size, bias=False)
        nn.init.xavier_normal_(self.key_network.weight)
        nn.init.xavier_normal_(self.value_network.weight)
        self.key_bn = nn.BatchNorm2d(1)
        self.value_bn = nn.BatchNorm2d(1)
        self.simple = simple

    def forward(self, x, x_lens):
        packed_x = rnn.pack_padded_sequence(x, x_lens.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.lstm(packed_x)
        if not self.simple:
            packed_out = self.pblstms(packed_out)
        out, out_lens = rnn.pad_packed_sequence(packed_out, batch_first=True)
        key = F.relu(self.key_bn(self.key_network(out)[:, None, :]).squeeze(1))
        value = F.relu(self.value_bn(self.value_network(out)[:, None, :]).squeeze(1))
        return key, value, out_lens

class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()

    def forward(self, query, key, value, mask, forcing_index=[]):
        energy = torch.bmm(key, query.unsqueeze(2)).squeeze(2)
        if forcing_index and forcing_index[1] < energy.shape[1]:
            energy = torch.ones_like(energy)
            energy[:, forcing_index[0]:forcing_index[1]] = 10.0
        energy.masked_fill_(mask, -float('inf'))
        attention = F.softmax(energy, dim=1)
        context = torch.bmm(attention.unsqueeze(1), value).squeeze(1)
        return context, attention

class Decoder(nn.Module):
    def __init__(self, decoder_hidden_dim, embed_dim, dropout=0.0,
                 key_value_size=128, vocab_size=len(LETTER_LIST), simple=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=letter2index['<eos>'])
        self.dropout_p = dropout
        self.dropout = nn.Dropout(p=dropout)
        if not simple:
            self.lstm1 = nn.LSTMCell(input_size=embed_dim+key_value_size, hidden_size=decoder_hidden_dim)
        else:
            self.lstm1 = nn.LSTMCell(input_size=embed_dim+key_value_size, hidden_size=key_value_size)
        self.lstm2 = nn.LSTMCell(input_size=decoder_hidden_dim, hidden_size=key_value_size)
        self.attention = Attention()
        self.character_prob = nn.Linear(2 * key_value_size, vocab_size)
        nn.init.xavier_normal_(self.character_prob.weight)
        self.vocab_size = vocab_size
        self.key_value_size = key_value_size
        self.simple = simple

    def forward(self, key, value, encoder_lens, true_y=None, teacher_forcing=0.0, attention_forcing=0.0):
        B, key_seq_max_len, key_val_size = key.shape

        if true_y is not None:
            max_len = true_y.shape[1]
            true_embeddings = self.embedding(true_y)
        else:
            max_len = 600

        mask = torch.arange(key_seq_max_len).unsqueeze(0) >= encoder_lens.unsqueeze(1)
        mask = mask.to(device)

        Y = []
        y = torch.zeros(B, 1).fill_(letter2index['<sos>']).to(device)
        s = [None, None]

        context = torch.zeros(B, key_val_size).to(device)
        attentions = []

        dropout_p = self.dropout_p if self.training else 0
        locked_dropout = LockedDropoutCell(dropout_p)

        for i in range(max_len):
            y_embedding = self.embedding(y.argmax(dim=-1))
            attention_index = []
            if self.training:
                rand = np.random.random()
                if rand < teacher_forcing and i > 0:
                    y_embedding = true_embeddings[:, i-1]
                if rand < attention_forcing:
                    width = (key_seq_max_len//max_len) //2
                    atti = (int)((i/max_len)*key_seq_max_len)
                    attention_index = [atti-width, atti+width+1]
            
            y_with_context = torch.cat([y_embedding, context], dim=1)
            s[0] = self.lstm1(y_with_context, s[0])
            lstm1_h = s[0][0]
            if not self.simple:
                lstm1_h = locked_dropout(lstm1_h)
                s[1] = self.lstm2(lstm1_h, s[1])
                output = s[1][0]
            else:
                output = s[0][0]
            context, attention = self.attention(output, key, value, mask, attention_index)
            output_with_context = torch.cat([output, context], dim=1)
            output_with_context = self.dropout(output_with_context)
            y = self.character_prob(output_with_context)
            Y.append(y.unsqueeze(1))
            attentions.append(attention)
        attentions = torch.stack(attentions, dim=1)
        return torch.cat(Y, dim=1), attentions

class LAS(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, embed_dim, input_dim=40,
                 dropout=0.0, key_value_size=128, vocab_size=len(LETTER_LIST), simple=False):
        super().__init__()
        self.encoder = Encoder(input_dim, encoder_dim, 3, dropout, 
                               key_value_size=key_value_size, simple=simple)
        self.decoder = Decoder(decoder_dim, embed_dim, dropout, key_value_size, vocab_size, simple)
        self.attention = None
        self.simple = simple
    
    def forward(self, x, x_lens, y=None, teacher_forcing=0.0, attention_forcing=0.0):
        key, value, encoder_lens = self.encoder(x, x_lens)
        predictions, attentions = self.decoder(key, value, encoder_lens, y, teacher_forcing, attention_forcing)
        del self.attention
        # self.attention = attentions.mean(0)
        self.attention = attentions[0]
        return predictions

    def get_attention(self):
        if self.attention is not None:
            return self.attention.cpu().detach()
