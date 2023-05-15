import torch
from torch.nn.functional import cross_entropy
from utils import split_input, convert_to_one_hot

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def DSR_loss(y_pred, y):
    item_logits = y_pred.logits
    item_y, _ = split_input(y, requires_grad=[False, False], dtype=torch.long, device=device)

    # Calculate relevance loss
    relevance_loss = cross_entropy(item_logits, item_y, reduction='mean')
    return (relevance_loss, relevance_loss)
