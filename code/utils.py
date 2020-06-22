import logging
import torch
import numpy as np
import os
from ignite.utils import convert_tensor
from ignite.engine import Events, State
from collections import namedtuple

SAVE_EVENT = "save_model"
TRAINING = "Training"
VALIDATION = "Validation"
TEST = "Test"

Output = namedtuple('Output', ['preds', 'logits', 'attention', 'context_logits', 'context_preds'])


def log_training_progress(engine, log_interval, len_train_loader):
    iter = (engine.state.iteration - 1) % len_train_loader + 1
    if iter > 0 and iter % log_interval == 0:
        logging.info("# Epoch[{}] Iteration[{}/{}]".format(engine.state.epoch, iter, len_train_loader))


def should_terminate_epoch(engine, num_iterations):
    if engine.state.iteration > int(num_iterations/5):
        engine.terminate_epoch()
        engine.remove_event_handler(should_terminate_epoch, Events.ITERATION_COMPLETED)


def log_results(engine, name, loader, evaluator, writer, interval=None):
    if name == TRAINING:
        evaluator.add_event_handler(Events.ITERATION_COMPLETED, should_terminate_epoch, len(loader))

    if name == TEST and (interval is not None) and (engine.state.epoch % interval != 0):
        return

    evaluator.run(loader)
    metrics = evaluator.state.metrics
    results = " ".join(["{} {:.4f}".format(key, val) for (key,val) in metrics.items()])

    if 'mrr' not in metrics:
        raise Exception("Please provide mrr and recall to check if this model is best")
    mrr = metrics['mrr']
    recall = metrics['recall']

    logging.info("# {} Results - Epoch {}: {}".format(name, engine.state.epoch, results))

    if writer is not None:
        for (key, val) in metrics.items():
            writer.add_scalar(f"{name}/{key}", val, engine.state.epoch)
        writer.flush()  # For the logs to appear on the board dynamically

    # Keep track of best model, save model if better
    if not getattr(engine.state, 'best_metrics', False):
        engine.state.best_metrics = {}

    if (name == VALIDATION) and (mrr >= engine.state.best_metrics.get('mrr', 0)) \
            and (recall >= engine.state.best_metrics.get("recall", 0)):
        # write values on state object
        for (key, val) in metrics.items():
            engine.state.best_metrics[key] = val
        engine.state.best_epoch = engine.state.epoch
        logging.info("# Found new best model: emitting save event")
        engine.fire_event(SAVE_EVENT)


def _prepare_batch(batch, device=None, non_blocking=False):
    """Prepare batch for training: pass to a device with options.

    """
    x, y = batch
    return (convert_tensor(x, device=device, non_blocking=non_blocking),
            convert_tensor(y, device=device, non_blocking=non_blocking))


def split_input(inputs, device=None, requires_grad=True, dtype=torch.float):
    tensors = []
    tmp_requires_grad = requires_grad
    tmp_dtype = dtype
    for index, input in enumerate(inputs):
        if isinstance(requires_grad, list):
            tmp_requires_grad = requires_grad[index]
        if isinstance(dtype, list):
            tmp_dtype = dtype[index]
        if isinstance(input, list):
            tensors.append(input)
            continue

        if not input.is_leaf:
            tensor = input.clone().to(device).type(tmp_dtype)
            tensors.append(tensor)
            continue
        tensor = input.clone().requires_grad_(tmp_requires_grad).to(device).type(tmp_dtype)
        tensors.append(tensor)
    return tensors


def convert_to_one_hot(to_convert, output_dim):
    one_hot = torch.zeros(*to_convert.shape, output_dim, device=to_convert.device, dtype=torch.short)
    scatter_dim = len(to_convert.shape)
    return one_hot.scatter_(scatter_dim, to_convert.unsqueeze(-1).long(), 1)


def get_indices_to_reduce(recommended_list, batch_size, item_num, device):
    """
    This converts the list of recommendations we have so far to a vector of dim item size
    it contains 10k for the items in the list
    :param recommended_list:
    :param batch_size:
    :param item_num:
    :return:
    """

    if len(recommended_list) == 0:
        return 0
    if isinstance(recommended_list, list):
        recommended_list = torch.stack(recommended_list, 1)
    to_reduce = convert_to_one_hot(recommended_list, item_num).sum(dim=1, dtype=torch.short)
    to_reduce[:, 0] = 1
    to_reduce *= torch.tensor(1e6, dtype=torch.short, device=device)
    return to_reduce


def print_best_model(engine):
    logging.info("# COMPLETED: best model")
    metrics = engine.state.best_metrics
    results = " ".join(["{} {:.4f}".format(key, val) for (key,val) in metrics.items()])

    logging.info("# Best Validation Results - Epoch {}: {}".format(engine.state.best_epoch, results))


def get_files_to_load(save_path, save_name_prefix, timestamp):
    # sort by epoch as highest epoch is best model
    listed = [f for f in os.listdir(save_path) if not f.startswith('.')]
    natural_key = lambda x: int(x.split(".pth")[0].split("_")[-1])
    saved_models = sorted(listed, reverse=True, key=natural_key)

    current_configuration = save_name_prefix.split("_")
    # skip date
    current_configuration = "_".join(current_configuration[2:])

    model_path = None
    optimizer_path = None
    start_epoch = None

    for saved_model in saved_models:
        tmp_timestamp = saved_model.split("_")[0]
        if tmp_timestamp == timestamp:
            start_epoch = int(saved_model.split("_")[-1][:-len(".pth")])
            old_date = saved_model.split("_")[0]
            old_configuration = "_".join(saved_model.split("_")[2:-2])
            if old_configuration != current_configuration:
                logging.warning(f"# Evaluation CONFIGURATION NOT EQUAL to saved configuration:\n"
                             f"# Current: {str(current_configuration)}\n# Saved: {str(old_configuration)}")
            model_path = f"{old_date}__{old_configuration}_model_{start_epoch}.pth"
            optimizer_path = f"{old_date}__{old_configuration}_optimizer_{start_epoch}.pth"

            model_path = os.path.join(save_path, model_path)
            optimizer_path = os.path.join(save_path, optimizer_path)
            break

    return model_path, optimizer_path, start_epoch


def load_best_model(trainer, model_manager, save_path, save_name_prefix, timestamp, seed, device):
    model_path, optimizer_path, start_epoch = get_files_to_load(save_path, save_name_prefix, timestamp)
    if model_path is not None:
        logging.info(f"# Continue from {model_path}")
        model_manager._model.load_state_dict(torch.load(model_path, map_location=device))
        model_manager._optimizer.load_state_dict(torch.load(optimizer_path, map_location=device))

        set_start_epoch = lambda engine: setattr(engine.state, "epoch", start_epoch)
        trainer.add_event_handler(Events.STARTED, set_start_epoch)
        trainer.add_event_handler(Events.STARTED, set_start_epoch)
    else:
        raise Exception("Can not continue from model. No matching configuration found.")


def evaluate_best_model(trainer, loader, evaluator, model_manager, save_path, save_prefix, timestamp, seed, device):
    load_best_model(trainer, model_manager, save_path, save_prefix, timestamp, seed, device)
    # Re-set random seed to have reproducable evaluation
    torch.random.manual_seed(seed)
    np.random.seed(seed)
    state = State(dataloader=loader, max_epochs=1)
    state.epoch = 1
    setattr(trainer, "state", state)
    log_results(trainer, TEST, loader, evaluator, None)
