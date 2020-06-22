import logging
import torch
from ignite.utils import convert_tensor
from torch.nn.init import xavier_uniform_


def _prepare_batch(batch, device=None, non_blocking=False):
    """Prepare batch for training: pass to a device with options.

    """
    x, y = batch
    return (convert_tensor(x, device=device, non_blocking=non_blocking),
            convert_tensor(y, device=device, non_blocking=non_blocking))


class Model:
    name = ""
    train_loader = None
    val_loader = None
    model = None
    trainer = None
    evaluator = None

    def __init__(self, required_keys, **kwargs):
        if required_keys is None:
            raise Exception("Specify in the model which keys are needed")
        for key in required_keys:
            if key not in kwargs:
                raise Exception("Key {} is required to instantiate a model".format(key))

        self.__dict__.update(kwargs)

    def _set_trainer(self, model, optimizer, loss_fn,
                    scheduler=None,
                    device='cpu', non_blocking=False,
                    prepare_batch=_prepare_batch):
        raise NotImplementedError

    def _set_evaluator(self, model, loss_fn, embeddings_path):
        raise NotImplementedError

    def _set_dataloaders(self, train_path, valid_path, test_path, batch_size, num_workers=4):
        raise NotImplementedError

    def get_trainer(self):
        return self.trainer

    def get_evaluator(self):
        return self.evaluator

    def get_dataloaders(self):
        return self.train_loader, self.val_loader, self.test_loader

    def evaluate(self):
        raise NotImplementedError


def gru_forward(gru, input, lengths, state=None, batch_first=True):
    gru.flatten_parameters()
    input_lengths, perm = torch.sort(lengths, descending=True)

    input = input[perm]
    if state is not None:
        state = state[perm].transpose(0, 1).contiguous()

    total_length = input.size(1)
    if not batch_first:
        input = input.transpose(0, 1)  # B x L x N -> L x B x N
    packed = torch.nn.utils.rnn.pack_padded_sequence(input, input_lengths, batch_first)

    outputs, state = gru(packed, state)
    outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(outputs, batch_first=batch_first,
                                                                     total_length=total_length)  # unpack (back to padded)

    _, perm = torch.sort(perm, descending=False)
    if not batch_first:
        outputs = outputs.transpose(0, 1)
    outputs = outputs[perm]
    state = state.transpose(0, 1)[perm]

    return outputs, state


def init_params(model):
    for name, param in model.named_parameters():
        logging.debug(f"Initialising {name} with {param.size()}")
        padding = False
        if "embedding" in name:
            if param.data[0].sum() < 0.1:
                padding = True
        if param.data.dim() > 1:
            xavier_uniform_(param.data)
        if padding:
            param.data[0] = 0
