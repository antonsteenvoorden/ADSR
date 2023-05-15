import torch
import torch.nn as nn
import torch.jit as jit


class PSRUCell(jit.ScriptModule):
    __constants__ = ['threshold']

    def __init__(self, input_size=128, hidden_size=128, bias=True, threshold=0.01):
        super(PSRUCell, self).__init__()

        self.threshold = threshold

        self.input_size = input_size
        self.hidden_size = hidden_size
        # Wr and Wz can be merged
        self.Wih = nn.Parameter(torch.randn(input_size, 3 * hidden_size), requires_grad=True)

        if bias == True:
            self.Bih = nn.Parameter(torch.randn(3 * hidden_size), requires_grad=True)
        else:
            self.Bih = torch.tensor(0)

        self.Whh = nn.Parameter(torch.randn(hidden_size, 3 * hidden_size), requires_grad=True)
        if bias == True:
            self.Bhh = nn.Parameter(torch.randn(3 * hidden_size), requires_grad=True)
        else:
            self.Bhh = torch.tensor(0)

    @jit.script_method
    def forward(self, input, hidden, concentration):
        # type: (Tensor, Tensor, Tensor) -> Tensor
        gi = torch.mm(input, self.Wih) + self.Bih
        gh = torch.mm(hidden, self.Whh) + self.Bhh
        i_r, i_i, i_n = gi.chunk(3, 1)
        h_r, h_i, h_n = gh.chunk(3, 1)

        reset_gate = torch.sigmoid(i_r + h_r)
        update_gate = torch.sigmoid(i_i + h_i)
        candidate_gate = torch.tanh(i_n + reset_gate * h_n)

        # concentration per channel
        concentration_gate = torch.einsum('b, bi -> bi', (concentration > self.threshold) * concentration, update_gate)

        new_hidden = (1-concentration_gate) * hidden + (concentration_gate * candidate_gate)
        return new_hidden
