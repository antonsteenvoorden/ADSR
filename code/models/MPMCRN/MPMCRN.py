import torch
import torch.nn as nn
from ..MPMCRN.PSRU import PSRUCell

from torch.nn.functional import dropout
from utils import Output


class MPMCRN(nn.Module):
    def __init__(self, item_num, user_num, num_item_units=128, num_user_units=128, num_gru_units=128,
                 dropout_rate=0.5, recommendation_len=10, device='cpu', sampling=False, BCE=False):
        super(MPMCRN, self).__init__()
        self.sampling = sampling
        self.BCE = BCE

        self.item_num = item_num
        self.user_num = user_num
        self.num_gru_units = num_gru_units
        self.num_user_units = num_user_units
        self.num_item_units = num_item_units

        self.dropout_rate = dropout_rate
        self.recommendation_len = recommendation_len

        # Note that 0 is reserved for the initial decoder input embedding
        self.item_embedding = nn.Embedding(self.item_num, num_item_units, padding_idx=0)

        # GRU and prediction networks
        self.temperature = 0.5
        self.threshold = 0.01
        self.num_purposes = 3
        for i in range(self.num_purposes):
            item_encoder_cell = PSRUCell(num_item_units, num_gru_units, threshold=self.threshold, bias=True)
            setattr(self, f"item_encoder_cell_{i}", item_encoder_cell)

        self.purpose_router = nn.Linear(num_item_units, self.num_purposes, bias=True)
        self.device = device
        self.to(device)
        return

    def forward(self, inputs):
        item_x, _, _, all_samples = inputs
        total_length = item_x.shape[1]
        batch_size = item_x.shape[0]
        lengths = (item_x != 0).sum(dim=-1)-1
        slicing_vector = torch.arange(batch_size)

        # Look up embeddings
        item_x_embedded = self.item_embedding(item_x)
        all_samples_embedded = self.item_embedding(all_samples)
        item_x_embedded = dropout(item_x_embedded, p=self.dropout_rate, training=self.training)

        hidden_states = [torch.zeros(batch_size, total_length, self.num_gru_units, device=self.device)
                         for _ in range(self.num_purposes)]
        concentrations = torch.zeros(batch_size, total_length, self.num_purposes, device=self.device)

        # call GRU sequentially
        hidden = torch.zeros(batch_size, self.num_gru_units, device=self.device)
        # hidden = [torch.zeros(batch_size, self.num_gru_units, device=self.device)
        #           for _ in range(self.num_purposes)]
        for i in range(total_length):
            input = item_x_embedded[:, i, :]
            concentration = self.purpose_router(input)
            concentration = concentration / self.temperature
            concentration = concentration.softmax(dim=-1)
            concentrations[:, i] = concentration

            for j in range(self.num_purposes):
                item_encoder_cell = getattr(self, f"item_encoder_cell_{j}")
                new_hidden = item_encoder_cell(input, hidden, concentration[:, j])
                hidden_states[j][:, i, :] = new_hidden
                hidden = new_hidden
                # new_hidden = item_encoder_cell(input, hidden[j], concentration[:, j])
                # hidden_states[j][:, i, :] = new_hidden
                # hidden[j] = new_hidden

        multi_purpose_context = torch.zeros(batch_size, self.num_gru_units, device=self.device)  # b x d

        # combine hidden states, using the final hidden state (depending on variable sequence length)
        for j in range(self.num_purposes):
            last_hidden_states = hidden_states[j][slicing_vector, lengths] # b x d
            current_concentrations = concentrations[slicing_vector, lengths, j] # b
            multi_purpose_context += torch.einsum('b, bd -> bd', current_concentrations, last_hidden_states)

        # Inner product capturing interaction between them
        if not self.training:
            if self.sampling:
                item_logits = torch.einsum('bd, bnd -> bn', multi_purpose_context, all_samples_embedded)
                if self.BCE:
                    item_probs = item_logits.sigmoid()
                else:
                    item_probs = item_logits.softmax(dim=-1)
                # Select only the samples
                _, indices = torch.topk(item_probs, self.recommendation_len, sorted=True)
                recommended_list = all_samples.gather(1, indices)
            else:
                item_logits = torch.einsum('bd, vd -> bv', multi_purpose_context, self.item_embedding.weight)
                if self.BCE:
                    item_probs = item_logits.sigmoid()
                else:
                    item_probs = item_logits.softmax(dim=-1)
                _, recommended_list = torch.topk(item_probs, self.recommendation_len, sorted=True)

        # training
        else:
            item_logits = torch.einsum('bd, vd -> bv', multi_purpose_context, self.item_embedding.weight)
            recommended_list = torch.zeros(1, device=self.device)

        recommended_list = recommended_list.requires_grad_(False)
        return Output(preds=recommended_list, logits=item_logits,
                      attention=None, context_logits=None, context_preds=None)
