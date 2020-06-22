import logging
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_
from ignite.engine import create_supervised_evaluator
from ignite.engine.engine import Engine

from ..Model import Model, _prepare_batch, init_params
from .model import ANAM_USER
from ..data_loader import get_data_loaders
from .losses import DSR_loss
from ..metrics import MRR, Recall, CombinedLoss, RelevanceLoss, ContextLoss, ContextAccuracy, ILD, DiscreteDiversity


class ANAM_USER_Manager(Model):
    name = "The ANAM+user model incorporates attribute information when " \
           "obtaining the representations, as well as the user identifier."

    def __init__(self, args):
        required_keys = ["train_path", "valid_path", "test_path", "batch_size",
                         "device", "num_item_units", "num_context_units", "num_user_units",
                         "num_gru_units", "dropout_rate", "recommendation_len",
                         "lambda_score", "lambda_multitask_loss", "lambda_diversity_loss",
                         "lr", "weight_decay", "properties_path", "metrics_save_path"]

        super(ANAM_USER_Manager, self).__init__(required_keys, **vars(args))
        logging.info("# Getting data loaders")
        item_num, context_num, user_num = self._set_dataloaders(self.train_path,
                                                                self.valid_path,
                                                                self.test_path,
                                                                self.batch_size)
        self.item_num = item_num
        self.context_num = context_num
        logging.info(f"# Unique Items: {item_num} Unique Contexts: {self.context_num}")

        model = ANAM_USER(item_num,
                            self.context_num,
                            user_num,
                            self.num_item_units,
                            self.num_context_units,
                            self.num_user_units,
                            self.num_gru_units,
                            self.dropout_rate,
                            self.recommendation_len,
                            self.lambda_score,
                            self.device,
                            self.properties_path)
        self._model = model
        init_params(self._model)

        logging.info("# Instantiating loss and optimizer")
        _loss_function = DSR_loss

        self._optimizer = Adam(model.parameters(), self.lr)

        logging.info("# Instantiating trainer")
        self._set_trainer(model, self._optimizer, _loss_function, device=self.device)
        self._set_evaluator(model, _loss_function, self.properties_path)

    def _set_trainer(self, model, optimizer, loss_fn,
                     scheduler=None,
                     device='cpu', non_blocking=False,
                     prepare_batch=_prepare_batch):
        if device:
            model.to(device)

        def _update(engine, batch):
            model.train()
            optimizer.zero_grad()
            x, y = prepare_batch(batch, device=device, non_blocking=non_blocking)
            y_pred = model(x)
            (combined_loss, relevance_loss) = loss_fn(y_pred, y)
            combined_loss.backward()
            clip_grad_norm_(model.parameters(), 1)
            optimizer.step()

            # Update LR
            if scheduler is not None:
                scheduler.step()

            combined_loss = combined_loss.item()
            return x, y, y_pred, combined_loss

        self.trainer = Engine(_update)

    def _set_evaluator(self, model, loss_fn, properties_path):
        metrics = {
            'mrr': MRR(self.metrics_save_path),
            'recall': Recall(self.metrics_save_path),
            'combined_loss': CombinedLoss(loss_fn),
            'relevance_loss': RelevanceLoss(loss_fn),
            'ild': ILD(properties_path, self.item_num, self.metrics_save_path),
            'discrete': DiscreteDiversity(properties_path, self.item_num, self.metrics_save_path),
        }

        evaluator = create_supervised_evaluator(model, device=self.device, metrics=metrics, non_blocking=False)
        self.evaluator = evaluator

    def _set_dataloaders(self, train_path, valid_path, test_path, batch_size, num_workers=4):
        train_loader, val_loader, test_loader, \
        item_num, context_num, user_num = get_data_loaders(train_path,
                                                           valid_path,
                                                           test_path,
                                                           batch_size,
                                                           num_workers=num_workers,
                                                           properties_path=self.properties_path)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        return item_num, context_num, user_num
