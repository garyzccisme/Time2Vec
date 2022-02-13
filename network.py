from typing import Tuple

import torch
import torch.nn as nn

from time2vec import T2V

"""
This file stores sequential one-step regression/forecast networks. 
If you want to conduct classification or multi-step modeling, you need to define your own customized network.
"""


class VanillaLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.dense = nn.Linear(hidden_size, 1)

    def forward(self, x):
        _, (x, _) = self.lstm(x)
        x = self.dense(x[0])
        return x.reshape(-1)


class SeqT2v(nn.Module):
    """
    Network combining LSTM with T2V.
    LSTM takes time-series value data; T2V takes time related data (stamp, datetime).

    Input: (x1, x2)
        x1 (sample_size, window_size, feature_number): time-series value data.
        x2 (sample_size, time_feature_number): time related data.

    """
    def __init__(self, linear_channel, period_channel, input_size, hidden_size,
                 time_input_channel=1, period_activation=torch.sin):
        super().__init__()
        self.t2v = T2V(linear_channel, period_channel, time_input_channel, period_activation)
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.dense = nn.Linear(hidden_size + linear_channel + period_channel, 1)

    def forward(self, x: Tuple):
        x1, x2 = x
        _, (x1, _) = self.lstm(x1)
        x2 = self.t2v(x2)
        x = self.dense(torch.cat([x1[0], x2], dim=-1))
        return x.reshape(-1)


class GreedySeqT2v(nn.Module):
    """
    Network combining LSTM with T2V using greedy(iterative) training strategy.

    Given multiple time features, instead of directly feed all of them to network, we iterate through them and train the
    network. After each training, we freeze and save the T2V layer of corresponding time feature, then start next loop.

    References: [Boosted Embeddings for Time Series Forecasting](https://arxiv.org/abs/2104.04781).

    Input: (x1, x2)
        x1 (sample_size, window_size, feature_number): time-series value data.
        x2 (sample_size, time_feature_number): time related data.

    """
    def __init__(self, linear_channel, period_channel, input_size, hidden_size,
                 frozen_t2v=None, time_input_channel=1, period_activation=torch.sin):
        super().__init__()
        # frozen_t2v is a list of frozen T2V layers.
        # After the embedding training, len(frozen_t2v) = time_feature_number - 1.
        if frozen_t2v is None:
            frozen_t2v = []
        self.t2v = T2V(linear_channel, period_channel, time_input_channel, period_activation)
        self.frozen_t2v = frozen_t2v
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.t2v_dense = nn.Linear((linear_channel + period_channel) * (1 + len(self.frozen_t2v)), hidden_size)
        self.dense = nn.Linear(hidden_size * 2, 1)

    def freeze(self):
        for param in self.t2v.parameters():
            param.requires_grad = False
        self.frozen_t2v.append(self.t2v)
        return self.frozen_t2v

    def frozen_forward(self, x):
        """
        Feed x[:, :-1] to frozen t2v layers, feed x[:, [-1]] to trainable t2v.
        """
        frozen_output = [layer(x[:, [i]]) for i, layer in enumerate(self.frozen_t2v)]
        x = torch.cat(frozen_output + [self.t2v(x[:, [-1]])], dim=-1)
        return x

    def forward(self, x: Tuple):
        x1, x2 = x
        _, (x1, _) = self.lstm(x1)
        if self.frozen_t2v:
            x2 = self.frozen_forward(x2)
        else:
            x2 = self.t2v(x2)
        x2 = self.t2v_dense(x2)
        x = self.dense(torch.cat([x1[0], x2], dim=-1))
        return x.reshape(-1)
