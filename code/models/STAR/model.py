from utils import split_input, convert_to_one_hot, get_indices_to_reduce, Output

import torch
import torch.nn as nn
from torch.nn.functional import dropout
import torch.nn.functional as F


class STAR(nn.Module):
    def __init__(self, item_num, context_num, user_num,
                 num_item_units=256, num_context_units=256, num_user_units=256, num_gru_units=100,
                 dropout_rate=0.1, recommendation_len=10, lambda_score=0.5, device='cpu'):
        super(STAR, self).__init__()
        self.item_num = item_num
        self.context_num = context_num  # is a tuple
        self.user_num = user_num

        self.num_gru_units = num_gru_units
        self.hidden_size = num_gru_units
        self.dropout_rate = dropout_rate
        self.recommendation_len = recommendation_len
        self.lambda_score = lambda_score

        # Note that 0 is reserved for the initial decoder input embedding
        self.item_embedding = nn.Embedding(self.item_num, num_gru_units, padding_idx=0)

        self.LinearH3 = nn.Linear(self.hidden_size,self.hidden_size)

        # Weight and bias for interval context
        self.WIntervalX = torch.nn.Parameter(torch.randn(self.hidden_size,1),requires_grad=True)
        self.BIntervalX = torch.nn.Parameter(torch.randn(self.hidden_size),requires_grad=True)
        self.Linear2ndIntervalLayer = nn.Linear(self.hidden_size,self.hidden_size*self.hidden_size)
        self.BIntervalContext = torch.nn.Parameter(torch.randn(self.hidden_size),requires_grad=True)

        # Weight and bias parameter for x which is the input
        self.Wx = torch.nn.Parameter(torch.randn(self.hidden_size,self.hidden_size),requires_grad=True)
        self.Bx = torch.nn.Parameter(torch.randn(self.hidden_size),requires_grad=True)

        self.prediction_layer = nn.Linear(self.num_gru_units, self.item_num)

        self.device = device
        self.to(device)
        return

    # assuming data has been prepended with zeroes
    def forward(self, inputs):
        item_x, context_x, user_id = split_input(inputs, requires_grad=False,
                                                 dtype=[torch.long, torch.float, torch.long], device=self.device)
        batch_size, sequence_length = item_x.shape
        hidden = torch.zeros(batch_size, self.num_gru_units, requires_grad=True, device=self.device)
        hidden_2 = torch.zeros(batch_size, self.num_gru_units, requires_grad=True, device=self.device)

        # go through RNN
        for j in range(sequence_length):
            hidden, hidden_2 = self.rnn_step(item_x[:, j], context_x[:, j], hidden, hidden_2)

        # calculate p(v | sequence) based on the global preference
        item_logits = self.prediction_layer(hidden)
        if not self.training:
            item_probs = item_logits.softmax(dim=-1)
            _, recommended_list = torch.topk(item_probs, self.recommendation_len, sorted=True, )
        else:
            recommended_list = torch.zeros(1, device=self.device)

        # return as a tuple for the trainer. This will be split up in the loss function.
        recommended_list = recommended_list.requires_grad_(False)

        return Output(preds=recommended_list, logits=item_logits,
                      attention=None, context_logits=None, context_preds=None)

    def rnn_step(self, item_x, context_x, hidden=None, hidden_2=None):
        item_x_embedded = self.item_embedding(item_x)
        item_x_embedded = dropout(item_x_embedded, p=self.dropout_rate, training=self.training)

        h3 = torch.sigmoid(F.linear(context_x.unsqueeze(1),self.WIntervalX,self.BIntervalX)
                     +self.LinearH3(hidden_2))

        context_rnn_output = torch.sigmoid(self.Linear2ndIntervalLayer(h3))
        intervalContext = context_rnn_output.reshape(-1, self.hidden_size,self.hidden_size)
        part1 = F.linear(item_x_embedded, self.Wx, self.Bx)
        part2 = torch.einsum('bi,bij->bj', hidden, intervalContext) + self.BIntervalContext

        hidden = torch.sigmoid(part1 + part2)
        return hidden, hidden_2
