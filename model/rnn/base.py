"""
基础网络组件：RNN, RNNWithInit, SubPoser
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class RNN(nn.Module):
    """基础RNN模块，带有输入/输出线性层"""
    def __init__(self, n_input, n_output, n_hidden, n_rnn_layer=2, bidirectional=False, dropout=0.2):
        super().__init__()
        self.n_hidden = n_hidden
        self.n_rnn_layer = n_rnn_layer
        self.num_directions = 2 if bidirectional else 1
        self.rnn = nn.LSTM(
            input_size=n_hidden,
            hidden_size=n_hidden,
            num_layers=n_rnn_layer,
            bidirectional=bidirectional,
            batch_first=True,
            dropout=dropout if n_rnn_layer > 1 else 0.0,
        )
        self.linear1 = nn.Linear(n_input, n_hidden)
        self.linear2 = nn.Linear(n_hidden * self.num_directions, n_output)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, h=None):
        x = self.dropout(F.relu(self.linear1(x)))
        output, _ = self.rnn(x, h)
        output = self.linear2(output)
        return output


class RNNWithInit(RNN):
    """带有初始状态网络的RNN模块"""
    def __init__(self, n_input, n_output, n_hidden, n_init, n_rnn_layer, bidirectional=False, dropout=0.2):
        super().__init__(n_input, n_output, n_hidden, n_rnn_layer, bidirectional, dropout)
        num_directions = 2 if bidirectional else 1
        self.init_net = nn.Sequential(
            nn.Linear(n_init, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden * n_rnn_layer),
            nn.ReLU(),
            nn.Linear(n_hidden * n_rnn_layer, 2 * num_directions * n_rnn_layer * n_hidden),
        )

    def forward(self, inputs, _=None):
        x, x_init = inputs
        batch_size = x.shape[0]
        num_directions = self.num_directions
        nd = self.n_rnn_layer * num_directions
        nh = self.n_hidden

        init = self.init_net(x_init).view(batch_size, 2, nd, nh)
        h0 = init[:, 0].permute(1, 0, 2).contiguous()
        c0 = init[:, 1].permute(1, 0, 2).contiguous()
        return super().forward(x, (h0, c0))


class SubPoser(nn.Module):
    """子姿态预测器，用于预测特定身体部位的速度和姿态"""
    def __init__(self, n_input, v_output, p_output, n_hidden, num_layer, dropout, extra_dim=0):
        super().__init__()
        self.extra_dim = extra_dim
        self.v_output = v_output
        self.p_output = p_output
        self.rnn1 = RNNWithInit(
            n_input=n_input - extra_dim,
            n_output=v_output,
            n_hidden=n_hidden,
            n_init=v_output,
            n_rnn_layer=num_layer,
            bidirectional=False,
            dropout=dropout,
        )
        self.rnn2 = RNNWithInit(
            n_input=n_input + v_output,
            n_output=p_output,
            n_hidden=n_hidden,
            n_init=p_output,
            n_rnn_layer=num_layer,
            bidirectional=False,
            dropout=dropout,
        )

    def forward(self, x, v_init, p_init):
        if self.extra_dim:
            x_v = x[..., :-self.extra_dim]
        else:
            x_v = x
        v = self.rnn1((x_v, v_init))
        p_input = torch.cat((x, v), dim=-1)
        p = self.rnn2((p_input, p_init))
        return v, p

