import torch
from torch.nn.functional import cross_entropy, binary_cross_entropy_with_logits
from utils import split_input, convert_to_one_hot

device = 'cuda' if torch.cuda.is_available() else 'cpu'


# sigmoid and CE loss
def loss_properties(logits, labels):
    loss = binary_cross_entropy_with_logits(logits, labels, reduction='none')
    return loss.mean(dim=-1)


def get_loss_fn(num_items, BCE=True):
    def DSR_loss(y_pred, y):
        item_logits = y_pred.logits

        if item_logits.shape[-1] != num_items:  # evaluation
            batch_size = item_logits.shape[0]
            item_y = torch.ones(batch_size, device=device, dtype=torch.long)
            if BCE:
                item_y = convert_to_one_hot(item_y, item_logits.shape[-1])
        else:
            item_y, _ = split_input(y, requires_grad=[False, False], dtype=[torch.long, torch.float], device=device)
            if BCE:
                item_y = convert_to_one_hot(item_y, item_logits.shape[-1])

        # Calculate relevance loss
        if BCE:
            relevance_loss = loss_properties(item_logits, item_y.float())
            relevance_loss = relevance_loss.mean()
        else:
            relevance_loss = cross_entropy(item_logits, item_y, reduction='mean')
        return (relevance_loss, relevance_loss)

    return DSR_loss
