import os
import logging
from datetime import datetime

import numpy as np
import torch
import random
from torch.utils.tensorboard import SummaryWriter
import torch.backends.cudnn as cudnn

from ignite.engine import Events, State
from ignite.handlers import ModelCheckpoint

from models.BSR.manager import BSR_Manager
from models.ANAM.manager import ANAM_Manager
from models.MTASR.manager import MTASR_Manager
from models.ADSR.manager import ADSR_Manager

# EXPERIMENTS AFTER CASE STUDY
from models.BSR_USER.manager import BSR_USER_Manager
from models.ANAM_USER.manager import ANAM_USER_Manager
from models.ADSR_USER.manager import ADSR_USER_Manager
from models.MTASR_USER.manager import MTASR_USER_Manager

# BASELINES
from models.GRU.manager import GRU_Manager
from models.MPMCRN.manager import MPMCRN_Manager
from models.STAR.manager import STAR_Manager
from models.STAR_C.manager import STAR_C_Manager

from hyperparameters import HyperParameters
from utils import SAVE_EVENT, TRAINING, VALIDATION, TEST
from utils import log_training_progress, log_results
from utils import print_best_model, load_best_model, evaluate_best_model


def main(ARGS):
    # Prepare logging paths
    save_timestamp = datetime.now().strftime("%Y%m%d%H%M")
    if ARGS.evaluate and ARGS.continue_from:
        raise Exception("# Can not BOTH evaluate and continue training")
    if ARGS.evaluate:
        save_timestamp = ARGS.evaluate
    elif ARGS.continue_from:
        save_timestamp = ARGS.continue_from
    save_name_prefix = "{}__{}_{}_{}-{}_{}_{}_{}_{}_{}_{}-lambdas_{}_{}_{}".format(
        save_timestamp,
        ARGS.model_type,
        ARGS.training_dataset,
        ARGS.random_seed,
        ARGS.num_gru_units,
        ARGS.num_item_units,
        ARGS.num_context_units,
        ARGS.num_user_units,
        ARGS.lr,
        ARGS.dropout_rate,
        ARGS.weight_decay,
        ARGS.lambda_score,
        ARGS.lambda_multitask_loss,
        ARGS.lambda_diversity_loss
    )

    # create output folder for the metrics (stat testing)
    ARGS.metrics_save_path = "./{}/{}/{}".format(ARGS.metrics_log_dir, ARGS.model_type, save_name_prefix)
    os.makedirs("./{}/{}".format(ARGS.metrics_log_dir, ARGS.model_type), exist_ok=True)

    # define tensorboard writer and model saver
    writer_path = "./{}/{}/{}".format(ARGS.tensorboard_log_dir,
                                      ARGS.model_type,
                                      save_name_prefix)

    # If we are only evaluating, do not write to tensorboard
    if not ARGS.evaluate:
        writer = SummaryWriter(writer_path)
    else:
        writer = None

    # define model checkpoint saver. This will be called every time we find a better model
    save_path = "./{}/{}/{}".format(ARGS.model_dir, ARGS.model_type, ARGS.training_dataset)
    saver = ModelCheckpoint(save_path,
                            save_name_prefix,
                            save_interval=1, n_saved=3, require_empty=False, create_dir=True)

    logging.info(f"# Summary path: {writer_path}")
    logging.info(f"# Save path: {save_path}/{save_name_prefix}")

    def save_params(engine, save_dict):
        saver._iteration = engine.state.epoch
        saver(engine, save_dict)


    # define model
    logging.info("# Instantiating model")

    if ARGS.model_type == "GRU":
        model_manager = GRU_Manager(ARGS)
    elif ARGS.model_type == "MPMCRN":
        model_manager = MPMCRN_Manager(ARGS)
    elif ARGS.model_type == "STAR":
        model_manager = STAR_Manager(ARGS)
    elif ARGS.model_type == "STAR_C":
        model_manager = STAR_C_Manager(ARGS)


    # VARIANTS OF THE EXPERIMENTS
    elif ARGS.model_type == "BSR":
        model_manager = BSR_Manager(ARGS)
    elif ARGS.model_type == "ANAM":
        model_manager = ANAM_Manager(ARGS)
    elif ARGS.model_type == "MTASR":
        model_manager = MTASR_Manager(ARGS)
    elif ARGS.model_type == "ADSR":
        model_manager = ADSR_Manager(ARGS)

    elif ARGS.model_type == "BSR_USER":
        model_manager = BSR_USER_Manager(ARGS)
    elif ARGS.model_type == "ANAM_USER":
        model_manager = ANAM_USER_Manager(ARGS)
    elif ARGS.model_type == "MTASR_USER":
        model_manager = MTASR_USER_Manager(ARGS)
    elif ARGS.model_type == "ADSR_USER":
        model_manager = ADSR_USER_Manager(ARGS)
    else:
        raise Exception("# Please specify the model type")

    # Prepare the dataloaders
    train_loader, val_loader, test_loader = model_manager.get_dataloaders()
    len_train_loader = len(train_loader)

    # define ignite trainer
    trainer = model_manager.get_trainer()
    evaluator = model_manager.get_evaluator()

    # decide if to continue training
    if ARGS.continue_from:
        load_best_model(trainer, model_manager, save_path, save_name_prefix, save_timestamp, ARGS.random_seed, device)

    # register handlers
    trainer.register_events(SAVE_EVENT)  # allow the save event to be called from the evaluator
    trainer.add_event_handler(Events.ITERATION_COMPLETED, log_training_progress, ARGS.log_interval, len_train_loader)
    trainer.add_event_handler(Events.EPOCH_COMPLETED, log_results, TRAINING, train_loader, evaluator, writer)
    trainer.add_event_handler(Events.EPOCH_COMPLETED, log_results, VALIDATION, val_loader, evaluator, writer)

    # Evaluate test set based on an interval
    trainer.add_event_handler(Events.EPOCH_COMPLETED, log_results, TEST, test_loader, evaluator, writer, interval=5)

    trainer.add_event_handler(Events.COMPLETED, print_best_model)
    trainer.add_event_handler(Events.COMPLETED, evaluate_best_model, test_loader, evaluator, model_manager,
                              save_path, save_name_prefix, save_timestamp, ARGS.random_seed, device)

    # if we only want to do an evaluation with our saved model, evaluate and exit
    if ARGS.evaluate:
        logging.info("# Running in EVALUATION ONLY mode")
        if len(save_timestamp) > len(datetime.now().strftime("%Y%m%d%H%M")):
            save_name_prefix = save_timestamp
        evaluate_best_model(trainer, test_loader, evaluator, model_manager,
                            save_path, save_name_prefix, save_timestamp, ARGS.random_seed, device)
        logging.info("# Finished")
        return True

    # save every best model
    trainer.add_event_handler(SAVE_EVENT, save_params, {"model": model_manager._model, "optimizer": model_manager._optimizer})

    # kick everything off
    logging.info("# Start training")
    trainer.run(train_loader, max_epochs=ARGS.num_epochs)


if __name__ == "__main__":
    hyperparameters = HyperParameters()
    ARGS = hyperparameters.parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if ARGS.device is not None:
        device = ARGS.device
    else:
        ARGS.device = device

    # Reproducability
    torch.random.manual_seed(ARGS.random_seed)
    np.random.seed(ARGS.random_seed)
    random.seed(ARGS.random_seed)

    if torch.cuda.is_available():
        cudnn.enabled = True
        cudnn.benchmark = True
        cudnn.deterministic = True
        torch.cuda.manual_seed(ARGS.random_seed)
        torch.cuda.manual_seed_all(ARGS.random_seed)

    # File name preparing
    dataset_name = ARGS.train_path.split("/")[-3]
    split_name = ARGS.train_path.split("/")[-1].split(".")[0]
    training_dataset = f'{dataset_name}_{split_name}'  # assuming data/<name>/data/file.csv
    ARGS.training_dataset = training_dataset

    os.makedirs(ARGS.log_dir, exist_ok=True)
    log_filename = "{}/{}_{}_{}.log".format(ARGS.log_dir,
                                            datetime.now().strftime("%Y%m%d%H%M"),
                                            ARGS.model_type,
                                            training_dataset)
    if os.path.exists(log_filename):
        log_filename += "_"
    os.makedirs(ARGS.metrics_log_dir, exist_ok=True)

    logging.basicConfig(level=logging.INFO, filename=log_filename, )

    # Also output to standard out
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    logging.info(f"# Attempting to run on: {device}")
    logging.info(f"# Hyperparameters: {str(ARGS)}")
    # Start training
    main(ARGS)
