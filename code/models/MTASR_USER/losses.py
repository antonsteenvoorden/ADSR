import torch
from torch.nn.functional import cross_entropy
from torch.nn.functional import binary_cross_entropy_with_logits
from utils import split_input, convert_to_one_hot

device = 'cuda' if torch.cuda.is_available() else 'cpu'


# softmax and CE loss
def loss_context(logits, labels):
    loss = cross_entropy(logits, labels.long(), reduction='none')
    return loss


# sigmoid and CE loss
def loss_properties(logits, labels):
    loss = binary_cross_entropy_with_logits(logits, labels, reduction='none')
    return loss.mean(dim=-1)


def get_loss_function_with_parameters(lambda_multitask_loss):
    """
    Because the trainer calls it with just (y_pred, y) this wrapper allows to set the loss variables
    :param lambda_multitask_loss:
    :param lambda_diversity_loss:
    :return: combined_loss(y_pred, y)
    """

    def combined_loss(y_pred, y):
        """
        This receives a combination of the predictions and combination of ground truths
        The ground truth is not sequential, but a single number. This is therefore repeated, to provide a label
        for every intermediate step.
        Returns the means of the losses.

        :param y_pred:
        :param y:
        :return: (combined_loss, relevance_loss, diversity_loss, context_loss)
        """
        item_logits = y_pred.logits
        context_logits=y_pred.context_logits

        item_y, context_y = split_input(y, requires_grad=[False, False], dtype=[torch.long, torch.float], device=device)

        relevance_loss = cross_entropy(item_logits, item_y, reduction='none')
        context_loss = loss_properties(context_logits, context_y.squeeze())

        # Linearly combine
        combined_loss = lambda_multitask_loss * relevance_loss + (1 - lambda_multitask_loss) * context_loss

        # Calculate means
        combined_loss = combined_loss.mean(-1)
        relevance_loss = relevance_loss.mean(-1)
        context_loss = context_loss.mean(-1)
        return (combined_loss, relevance_loss, context_loss)

    return combined_loss
