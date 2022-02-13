import torch
import torch.nn as nn


class T2V(nn.Module):
    """
    General Time2Vec Embedding/Encoding Layer. The input data should be related with timestamp/datetime.

    Input shape: (*, feature_number), where * means any number of dimensions.
    Output shape: (*, linear_channel + period_channel).
    """
    def __init__(self, linear_channel: int, period_channel: int, input_channel: int = 1, period_activation=torch.sin):
        """
        Args:
            linear_channel: The number of linear transformation elements.
            period_channel: The number of cyclical/periodical transformation elements.
            input_channel: The feature number of input data. Default = 1
            period_activation: The activation function for periodical transformation. Default is sine function.
        """
        super().__init__()
        self.linear_channel = linear_channel
        self.period_channel = period_channel
        self.linear_fc = nn.Linear(input_channel, linear_channel)
        self.period_fc = nn.Linear(input_channel, period_channel)
        self.period_activation = period_activation

    def forward(self, x):
        linear_vec = self.linear_fc(x)
        period_vec = self.period_activation(self.period_fc(x))
        return torch.cat([linear_vec, period_vec], dim=-1)
