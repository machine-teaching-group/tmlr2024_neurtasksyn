import math

import torch
from torch import nn


class LSTMNetwork(nn.Module):
    def __init__(self,
                 dict_size,
                 embedding_size,
                 lstm_hidden_size,
                 nb_layers,
                 output_size,
                 add_latent=False,
                 latent_size=64,
                 max_number=None):
        super().__init__()
        self.dict_size = dict_size
        self.embedding_size = embedding_size
        self.output_size = output_size
        self.lstm_hidden_size = lstm_hidden_size
        self.nb_layers = nb_layers
        self.add_latent = add_latent
        self.max_number = max_number

        if max_number is not None:
            # get log of max number
            emb_size = math.log(max_number, 2)
            emb_size = 2 ** emb_size
            self.num_embedding = nn.Embedding(max_number + 1, int(emb_size))
            lstm_in_size = self.embedding_size + int(emb_size)
        else:
            lstm_in_size = self.embedding_size

        self.embedding = nn.Embedding(self.dict_size + 1, self.embedding_size, padding_idx=0)
        self.rnn = nn.LSTM(
            lstm_in_size if not self.add_latent else lstm_in_size + latent_size,
            self.lstm_hidden_size,
            self.nb_layers,
            batch_first=True
        )

        self.out2tok = nn.Linear(self.lstm_hidden_size, self.output_size)

        self.initial_h = torch.zeros(self.nb_layers, 1, self.lstm_hidden_size)
        self.initial_c = torch.zeros(self.nb_layers, 1, self.lstm_hidden_size)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.initial_h.data.uniform_(-initrange, initrange)
        self.initial_c.data.uniform_(-initrange, initrange)

    def init_state(self):
        return self.initial_h, self.initial_c

    def forward(self, x, state, latent=None, num=None):
        x = self.embedding(x)
        if self.add_latent:
            x = torch.cat([x, latent], dim=2)
        if self.max_number is not None:
            num = self.num_embedding(torch.tensor(min(num, self.max_number)))
            num = num.unsqueeze(0).unsqueeze(0)
            x = torch.cat([x, num], dim=2)
        x, (h, c) = self.rnn(x, state)
        x = self.out2tok(x)
        return x, (h, c)
