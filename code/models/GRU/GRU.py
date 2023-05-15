from utils import split_input, convert_to_one_hot, get_indices_to_reduce, Output

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.functional import dropout


class GRU(nn.Module):
    def __init__(self, item_num, user_num, num_item_units=256, num_user_units=100, num_gru_units=100,
                 dropout_rate=0.1, recommendation_len=10, device='cpu'):
        super(GRU, self).__init__()
        self.item_num = item_num
        self.user_num = user_num
        self.num_gru_units = num_gru_units
        self.num_user_units = num_user_units

        self.dropout_rate = dropout_rate
        self.recommendation_len = recommendation_len

        # Note that 0 is reserved for the initial decoder input embedding
        self.item_embedding = nn.Embedding(self.item_num, num_item_units, padding_idx=0)
        # self.user_embedding = nn.Embedding(self.user_num, num_user_units, padding_idx=0)
        self.hidden_size = num_gru_units #int(num_gru_units/2)
        # GRU and prediction networks
        self.item_encoder_GRU = nn.GRU(input_size=num_item_units,
                                       hidden_size=self.hidden_size,
                                       batch_first=True)
        self.item_decoder_GRU = nn.GRU(input_size=num_item_units,
                                       hidden_size=num_gru_units,
                                       batch_first=True)
        print("hidden size", self.hidden_size)
        # decoder
        # self.hidden_to_item_tfm = nn.Linear(in_features=num_gru_units + num_user_units,
        self.hidden_to_item_tfm = nn.Linear(in_features=self.hidden_size,
                                            out_features=num_item_units)

        # init weights with Xavier initializer
        nn.init.xavier_uniform_(self.hidden_to_item_tfm.weight)
        self.device = device
        self.to(device)
        return

    def forward(self, inputs):
        item_x, user_id, _ = inputs
        lengths = (item_x != 0).sum(dim=-1)

        # Look up embeddings
        item_x_embedded = self.item_embedding(item_x)
        # Apply dropout (as this is done on the inputs in the DSR model)
        item_x_embedded = dropout(item_x_embedded, p=self.dropout_rate, training=self.training)

        # Pack for GRU, read more about it:
        # https://pytorch.org/docs/stable/nn.html?highlight=nn%20gru#torch.nn.utils.rnn.pack_padded_sequence
        item_x = pack_padded_sequence(item_x_embedded, lengths, batch_first=True, enforce_sorted=False)

        # Do forward passes GRU stuff
        # h_n of shape (num_layers * num_directions, batch, hidden_size), only 1 layer so we squash it
        _, item_enc_hidden = self.item_encoder_GRU(item_x)
        item_enc_hidden = item_enc_hidden.view((-1, self.hidden_size))

        # user_embeddings = self.user_embedding(user_id)  # (batch x num_user_units)
        # global_preference = torch.cat((item_enc_hidden, user_embeddings), dim=-1)
        # global_preference = self.hidden_to_item_tfm(global_preference)
        global_preference = self.hidden_to_item_tfm(item_enc_hidden)

        # calculate p(v | sequence) based on the global preference
        # (batch x num_item_units) times (|V| x num_item_units) -> (batch x |V|)
        item_logits = torch.einsum('bd,vd->bv', global_preference, self.item_embedding.weight)
        item_probs = item_logits.softmax(dim=-1)
        _, recommended_list = torch.topk(item_probs, self.recommendation_len, sorted=True)
        recommended_list = recommended_list.requires_grad_(False)
        return Output(preds=recommended_list, logits=item_logits,
                      attention=None, context_logits=None, context_preds=None)
