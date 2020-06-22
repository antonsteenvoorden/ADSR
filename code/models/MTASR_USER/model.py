from utils import split_input, convert_to_one_hot, get_indices_to_reduce, Output

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn.functional import dropout
from ..BilinearAttention import BilinearAttention


class MTASR_USER(nn.Module):
    def __init__(self, item_num, context_num, user_num,
                 num_item_units=256, num_context_units=256, num_user_units=256, num_gru_units=100,
                 dropout_rate=0.1, recommendation_len=10, lambda_score=0.5, device='cpu'):
        super(MTASR_USER, self).__init__()
        self.item_num = item_num
        self.context_num = context_num  # is a tuple
        self.user_num = user_num

        self.num_gru_units = num_gru_units
        self.num_item_units = num_item_units
        self.num_user_units = num_user_units

        after_encoding_dim = 2*num_gru_units
        self.num_context_units = num_context_units

        self.dropout_rate = dropout_rate
        self.recommendation_len = recommendation_len
        self.lambda_score = lambda_score

        # Note that 0 is reserved for the initial decoder input embedding
        self.item_embedding = nn.Embedding(self.item_num, num_item_units, padding_idx=0)
        self.user_embedding = nn.Embedding(self.user_num, num_user_units, padding_idx=0)

        # add context embeddings and prediction networks
        self.context_embedding = nn.Embedding(self.context_num, num_context_units).weight
        self.context_MLP = nn.Sequential(nn.Linear(in_features=2*after_encoding_dim,
                                                   out_features=num_gru_units + num_gru_units),
                                         nn.BatchNorm1d(num_features=num_gru_units + num_gru_units),
                                         nn.ReLU(),
                                         # final logits
                                         nn.Linear(in_features=num_gru_units + num_gru_units,
                                                   out_features=self.context_num)
                                         )

        # GRU and prediction networks
        self.item_encoder_GRU = nn.GRU(
            input_size=after_encoding_dim + num_item_units,
            hidden_size=num_gru_units,
            batch_first=True,
            bidirectional=True)

        self.context_encoder_GRU = nn.GRU(input_size=num_context_units + num_item_units + num_user_units,
                                          hidden_size=num_gru_units,
                                          batch_first=True,
                                          bidirectional=True)

        # encoder transformations
        self.bilinear_mapping = nn.Bilinear(
            in1_features=after_encoding_dim,
            in2_features=num_context_units,
            out_features=num_item_units,
            bias=True)

        # decoder
        self.hidden_to_item_tfm = nn.Linear(in_features=after_encoding_dim,
                                            out_features=num_item_units)

        self.attention = BilinearAttention(after_encoding_dim, after_encoding_dim, after_encoding_dim)
        self.device = device
        self.to(device)
        return

    def forward(self, inputs):
        item_x, context_x, user_id = split_input(inputs, requires_grad=False,
                                                 dtype=[torch.long, torch.float, torch.long], device=self.device)
        batch_size = item_x.shape[0]
        (item_enc_hidden, ctxt_enc_hidden, norm_attn) = self.encode(item_x, context_x, user_id)

        # user_embeddings = self.user_embedding(user_id)  # (batch x num_user_units)
        context_predict_input = torch.cat((item_enc_hidden, ctxt_enc_hidden), dim=-1)
        context_logits = self.context_MLP(context_predict_input)
        context_probs = context_logits.softmax(dim=-1)
        context_preds = context_logits.argmax(dim=-1)

        context_y_embedded = torch.einsum('bc, cd -> bd', context_probs, self.context_embedding)
        context_y_embedded = dropout(context_y_embedded, p=self.dropout_rate, training=self.training)

        global_preference = self.bilinear_mapping(item_enc_hidden, context_y_embedded)

        # calculate p(v | sequence) based on the global preference
        item_logits = torch.einsum('bd,vd->bv', global_preference, self.item_embedding.weight)
        if not self.training:
            item_probs = item_logits.softmax(dim=-1)
            _, recommended_list = torch.topk(item_probs, self.recommendation_len, sorted=True, )
        else:
            recommended_list = torch.zeros(1, device=self.device)

        # return as a tuple for the trainer. This will be split up in the loss function.
        recommended_list = recommended_list.requires_grad_(False)
        context_preds = context_preds.requires_grad_(False)
        return Output(preds=recommended_list, logits=item_logits, attention=norm_attn,
                      context_logits=context_logits, context_preds=context_preds)

    def encode(self, item_x, context_x, user_id):
        lengths = (item_x != 0).sum(dim=-1)
        total_length = item_x.shape[1]  # (batch x sequence length)
        mask = item_x.ne(0)
        batch_size = item_x.shape[0]

        # Look up embeddings
        item_x_embedded = self.item_embedding(item_x)
        item_x_embedded = dropout(item_x_embedded, p=self.dropout_rate, training=self.training)

        context_x_embedded = torch.einsum('blc, cd -> bld', context_x, self.context_embedding)
        context_x_embedded = dropout(context_x_embedded, p=self.dropout_rate, training=self.training)

        user_embedded = self.user_embedding(user_id)
        user_embedded = dropout(user_embedded, p=self.dropout_rate, training=self.training)
        user_embedded = user_embedded.unsqueeze(1).repeat(1, total_length, 1)

        context_input = torch.cat([item_x_embedded, context_x_embedded, user_embedded], dim=-1)

        # Pack for GRU, read more about it:
        context_x = pack_padded_sequence(context_input, lengths, batch_first=True, enforce_sorted=False)

        # Do forward passes GRU stuff
        ctxt_all_hidden, ctxt_enc_hidden = self.context_encoder_GRU(context_x)
        ctxt_all_hidden = pad_packed_sequence(ctxt_all_hidden, total_length=total_length, batch_first=True)
        ctxt_all_hidden = ctxt_all_hidden[0]  # 0 contains the padded sequences

        ctxt_all_hidden = dropout(ctxt_all_hidden, p=self.dropout_rate, training=self.training)

        item_input = torch.cat((item_x_embedded, ctxt_all_hidden), dim=-1)
        item_x = pack_padded_sequence(item_input, lengths, batch_first=True, enforce_sorted=False)

        item_all_hidden, item_enc_hidden = self.item_encoder_GRU(item_x)
        item_enc_hidden = item_enc_hidden.view((batch_size, 1, -1))
        item_all_hidden = pad_packed_sequence(item_all_hidden, total_length=total_length, batch_first=True)
        item_all_hidden = item_all_hidden[0]

        item_enc_hidden = dropout(item_enc_hidden, p=self.dropout_rate, training=self.training)
        item_all_hidden = dropout(item_all_hidden, p=self.dropout_rate, training=self.training)

        # Attention
        item_enc_hidden, _, norm_attn = self.attention(item_enc_hidden,
                                                       item_all_hidden,
                                                       item_all_hidden,
                                                       mask=mask.unsqueeze(1))
        # Attention
        ctxt_enc_hidden = torch.einsum('bl, bld -> bd', norm_attn.view(batch_size, -1), ctxt_all_hidden)
        item_enc_hidden = item_enc_hidden.view(batch_size, -1)
        ctxt_enc_hidden = ctxt_enc_hidden.view(batch_size, -1)
        return (item_enc_hidden, ctxt_enc_hidden, norm_attn.view(batch_size, -1))
