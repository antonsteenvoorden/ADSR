import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn.functional import dropout
from ..BilinearAttention import BilinearAttention
from utils import Output

class BSR(nn.Module):
    def __init__(self, item_num, user_num, num_item_units=256, num_user_units=100, num_gru_units=100,
                 dropout_rate=0.1, recommendation_len=10, device='cpu'):
        super(BSR, self).__init__()
        self.item_num = item_num
        self.user_num = user_num
        self.num_gru_units = num_gru_units
        after_encoding_dim = 2 * num_gru_units
        self.num_user_units = num_user_units
        self.num_item_units = num_item_units

        self.dropout_rate = dropout_rate
        self.recommendation_len = recommendation_len

        # Note that 0 is reserved for the initial decoder input embedding
        self.item_embedding = nn.Embedding(self.item_num, num_item_units, padding_idx=0)

        # GRU and prediction networks
        self.item_encoder_GRU = nn.GRU(input_size=num_item_units,
                                       hidden_size=num_gru_units,
                                       batch_first=True,
                                       num_layers=1,
                                       bidirectional=True)
        print("hidden size", num_gru_units)
        # decoder
        self.hidden_to_item_tfm = nn.Linear(in_features=after_encoding_dim,
                                            out_features=num_item_units)

        self.attention = BilinearAttention(after_encoding_dim, after_encoding_dim, after_encoding_dim)

        self.device = device
        self.to(device)
        return

    def forward(self, inputs):
        item_x, _, _ = inputs
        lengths = (item_x != 0).sum(dim=-1)
        total_length = item_x.shape[1]
        batch_size = item_x.shape[0]
        mask = item_x.ne(0)

        # Look up embeddings
        item_x_embedded = self.item_embedding(item_x)
        item_x_embedded = dropout(item_x_embedded, p=self.dropout_rate, training=self.training)

        item_x = pack_padded_sequence(item_x_embedded, lengths, batch_first=True, enforce_sorted=False)

        # Do forward passes GRU stuff
        item_enc_all, item_enc_hidden = self.item_encoder_GRU(item_x)
        item_enc_all = pad_packed_sequence(item_enc_all, total_length=total_length, batch_first=True)
        item_enc_all = item_enc_all[0]  # 0 contains the padded sequences

        item_enc_all = dropout(item_enc_all, p=self.dropout_rate, training=self.training)
        item_enc_hidden = dropout(item_enc_hidden, p=self.dropout_rate, training=self.training)

        # Attention
        item_enc_hidden, attn, norm_attn = self.attention(item_enc_hidden.reshape(batch_size, -1).unsqueeze(1),
                                                          item_enc_all,
                                                          item_enc_all,
                                                          mask=mask.unsqueeze(1))
        global_preference = self.hidden_to_item_tfm(item_enc_hidden).reshape(-1, self.num_item_units)
        item_logits = torch.einsum('bd,vd->bv', global_preference, self.item_embedding.weight)
        if not self.training:
            item_probs = item_logits.softmax(dim=-1)
            _, recommended_list = torch.topk(item_probs, self.recommendation_len, sorted=True)
        else:
            recommended_list = torch.zeros(1, device=self.device)
        recommended_list = recommended_list.requires_grad_(False)
        return Output(preds=recommended_list, logits=item_logits,
                      attention=norm_attn.view(batch_size, -1), context_logits=None, context_preds=None)
