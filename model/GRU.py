import torch as th
from torch.nn.utils.rnn import pad_packed_sequence, PackedSequence

class RNN_MASK(th.nn.Module):
    def __init__(self,
    num_bins = 256, # num of feature channels
    freq_bins = 257,
    rnn = "gru",
    num_mask = 4,
    num_layer = 2,
    hidden_size = 500,
    dropout = 0.0,
    non_linear = "relu",
    bidirectional = False
    ):
        super(RNN_MASK,self).__init__()
        if non_linear not in ["relu","sigmoid","tanh"]:
            raise ValueError(
                "Unsupported non-linear type:{}".format(non_linear)
            )
        self.num_mask = num_mask
        rnn = rnn.upper()
        if rnn not in ["RNN", "LSTM", "GRU"]:
            raise ValueError(
                "Unsupported rnn type:{}".format(rnn)
            )
        self.rnn = getattr(th.nn, rnn)(
            num_bins,
            hidden_size,
            num_layer,
            batch_first = True,
            dropout = dropout,
            bidirectional = bidirectional
        )
        self.filter_size = 3
        self.drops = th.nn.Dropout(p=dropout)
        self.linear = th.nn.ModuleList([
            th.nn.Linear(hidden_size * 2 if bidirectional else hidden_size, 2 * freq_bins * self.filter_size * self.filter_size)
            for _ in range(self.num_mask)
        ])

        # self.conv1d = th.nn.Conv1d(hidden_size * 2 if bidirectional else hidden_size, self.num_mask * 2 * freq_bins * self.filter_size * self.filter_size, kernel_size=1)

        self.non_linear = {
            "relu": th.nn.functional.relu,
            "sigmoid": th.nn.functional.sigmoid,
            "tanh": th.nn.functional.tanh
        }[non_linear]
        self.num_bins = num_bins
    
    def forward(self,x):
        is_packed = isinstance(x, PackedSequence)
        if not is_packed and x.dim()!=3:
            x = th.unsqueeze(x,0)
        x, _ = self.rnn(x) # [B, C, T]

        if is_packed:
            x, _ = pad_packed_sequence(x,batch_first = True)
        x = self.drops(x)
        m = []
        for linear in self.linear:
            y = linear(x)
            y = self.non_linear(y)
            m.append(y)
        return m