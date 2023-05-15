from collections import defaultdict
import pandas as pd

import torch
from torch.nn.functional import pdist
from torch.distributions import Categorical

from ignite.exceptions import NotComputableError
from ignite.metrics.metric import Metric
from utils import split_input, convert_to_one_hot

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Keys for the statistics dict
ITEM_ENTROPY = 'item_entropy'
ITEM_MIN = 'item_min'
ITEM_MAX = 'item_max'


class RetrievalMetrics(Metric):
    """
    Calculates the RetrievalMetrics recall, mrr and context prediction accuracy
    `update` must receive output of the form (y_pred, y)
    where y_pred contains multiple prediction variables y_pred = (recommended_list, item_logits, context_logits, _)
    and y is a tuple containing labels for both item sequence and context y = (item_y, context_y)
    """

    def __init__(self, save_path):
        self._ids = None
        self._prediction_lists = None
        self._labels = None

        self._statistics = None

        self.save_path = save_path
        super(RetrievalMetrics, self).__init__()

    def reset(self):
        self._prediction_lists = []
        self._labels = []

        self._statistics = defaultdict(list)

    def update(self, output):
        y_pred, y = output
        recommended_list = y_pred.preds
        item_logits = y_pred.logits

        item_y, _ = split_input(y, dtype=torch.int, requires_grad=False, device=device)

        item_probs = item_logits.softmax(dim=-1)
        item_entropy = Categorical(probs=item_probs).entropy()
        self._statistics[ITEM_ENTROPY].extend(item_entropy)
        self._statistics[ITEM_MIN].extend(item_probs.min(dim=-1).values)
        self._statistics[ITEM_MAX].extend(item_probs.max(dim=-1).values)

        self._prediction_lists.extend(recommended_list)  # extend because we then add the recommendation list
        self._labels.extend(item_y)


    def compute(self):
        if len(self._prediction_lists) == 0:
            raise NotComputableError('MRR must have at least one example before it can be computed')

        mrr = 0
        mrr_list = []

        recall = 0
        recall_list = []

        count = 0

        # go over all predicted lists
        for index, pred_list in enumerate(self._prediction_lists):
            count += 1
            label = self._labels[index]
            if label in pred_list:
                tmp_mrr = 1 / (pred_list.tolist().index(label) + 1)
                mrr += tmp_mrr
                mrr_list.append(tmp_mrr)

                recall += 1
                recall_list.append(1)
            else:
                mrr_list.append(0)
                recall_list.append(0)

        # Save results for statistical testing and inspection
        with open(self.save_path + "_output.txt", 'w') as f:
            for pred_list in self._prediction_lists:
                f.write(",".join([str(p.item()) for p in pred_list]))
                f.write("\n")

        mrr_list = torch.tensor(mrr_list)
        recall_list = torch.tensor(recall_list)
        with open(self.save_path + "_mrr.txt", 'w') as f_mrr:
            for tmp_mrr in mrr_list:
                f_mrr.write(f"{tmp_mrr.item()}")
                f_mrr.write("\n")

        with open(self.save_path + "_recall.txt", 'w') as f_recall:
            for tmp_recall in recall_list:
                f_recall.write(f"{tmp_recall.item()}")
                f_recall.write("\n")

        for key, value_list in self._statistics.items():
            with open(self.save_path + "_"+key+".txt", "w") as file:
                for tmp_val in value_list:
                    file.write(f"{tmp_val.item()}")
                    file.write("\n")
                file.close()

        return {'recall': recall / count, 'mrr': mrr / count}


class MRR(RetrievalMetrics):
    def compute(self):
        retrieval_metrics = super(MRR, self).compute()
        return retrieval_metrics['mrr']


class Recall(RetrievalMetrics):
    def compute(self):
        retrieval_metrics = super(Recall, self).compute()
        return retrieval_metrics['recall']


class Diversity(RetrievalMetrics):
    """
      Calculates the Diversity
      reads the embeddings from the `embeddings_path` and keeps them in memory
      calculates the diversity according to Nguyen et al. 2014: "Exploring the filter bubble:
      The effect of using recommender systems on content diversity"
      """
    embeddings = None

    def __init__(self, embeddings_path, num_items, save_path):
        properties = pd.read_csv(embeddings_path)
        if "ijcai15" in embeddings_path:
            properties = properties.drop(columns=["item_id"])
            self.embeddings = torch.tensor(properties.values, dtype=torch.float)
            self.num_categories = torch.max(self.embeddings).int().item() + 1
        else:
            properties = properties.drop(columns=["item_id", "category"])
            self.embeddings = torch.tensor(properties.values, dtype=torch.float)
        self._padding_indices = num_items - len(self.embeddings)
        super(Diversity, self).__init__(save_path)

    def compute(self):
        diversities = []
        discrete_diversities = []

        for pred_list in self._prediction_lists:
            # calculate diversity using the embeddings of the given ids
            # TODO: create embeddings on the fly for the recommended list
            pred_embeddings = self.get_embeddings(pred_list)
            diversity = self.compute_diversity(pred_embeddings)
            diversities.append(diversity)

            diversity2 = self.compute_discrete_diversity(pred_embeddings)
            discrete_diversities.append(diversity2)

        # Save results for statistical testing
        ild_list = torch.tensor(diversities)
        discrete_list = torch.tensor(discrete_diversities)
        with open(self.save_path + "_ild.txt", 'w') as f_ild:
            for tmp in ild_list:
                f_ild.write(f"{tmp.item()}")
                f_ild.write("\n")

        with open(self.save_path + "_discrete.txt", 'w') as f_discrete:
            for tmp in discrete_list:
                f_discrete.write(f"{tmp.item()}")
                f_discrete.write("\n")

        ILD = torch.tensor(diversities).mean()
        discrete = torch.tensor(discrete_diversities).mean()
        return {'ILD': ILD, 'discrete': discrete}

    def get_embeddings(self, pred_list):
        # look up MF embedding for this prediction, indices start from 0 there.
        pred_list = pred_list - self._padding_indices

        if self.embeddings.shape[-1] <= 1:
            pred_categories = self.embeddings[pred_list].view(pred_list.shape[0])
            pred_embeddings = convert_to_one_hot(pred_categories, self.num_categories)
        else:
            pred_embeddings = self.embeddings[pred_list]
        return pred_embeddings.float()

    def compute_discrete_diversity(self, pred_embeddings):
        bool_per_cat = (pred_embeddings.sum(dim=0) > 0)
        return bool_per_cat.sum().float()

    def compute_diversity(self, pred_embeddings):
        len_recommended_list = len(pred_embeddings)
        distances = pdist(pred_embeddings)
        # calculate diversity according to formula
        diversity = 2 * torch.sum(distances) / (len_recommended_list * (len_recommended_list - 1))
        return diversity


class DiscreteDiversity(Diversity):
    def compute(self):
        retrieval_metrics = super(DiscreteDiversity, self).compute()
        return retrieval_metrics['discrete']


class ILD(Diversity):
    def compute(self):
        retrieval_metrics = super(ILD, self).compute()
        return retrieval_metrics['ILD']


## LOSSES
class CustomLoss(Metric):
    def __init__(self, loss_fn, output_transform=lambda x: x,
                 batch_size=lambda x: len(x[0])):
        super(CustomLoss, self).__init__(output_transform)
        self._loss_fn = loss_fn
        self._batch_size = batch_size

        self._sum = None
        self._num_examples = None

    def reset(self):
        self._sum = defaultdict(int)
        self._num_examples = defaultdict(int)

    def update(self, output):
        if len(output) == 2:
            y_pred, y = output
            kwargs = {}
        else:
            y_pred, y, kwargs = output

        losses = self._loss_fn(y_pred, y, **kwargs)

        (combined_loss, relevance_loss) = losses

        if len(combined_loss.shape) != 0:
            raise ValueError('loss_fn did not return the average loss.')

        N = self._batch_size(y)

        # update sum and num examples
        self._sum['combined_loss'] += combined_loss.item() * N
        self._num_examples['combined_loss'] += N

        self._sum['relevance_loss'] += relevance_loss.item() * N
        self._num_examples['relevance_loss'] += N

    def compute(self):
        if (self._num_examples['combined_loss'] == 0):
            raise NotComputableError(
                'Loss must have at least one example before it can be computed.' + str(self._num_examples))

        combined_loss = self._sum['combined_loss'] / self._num_examples['combined_loss']
        relevance_loss = self._sum['relevance_loss'] / self._num_examples['relevance_loss']

        results = {'combined_loss': combined_loss, 'relevance_loss': relevance_loss}
        return results


class CombinedLoss(CustomLoss):
    def compute(self):
        retrieval_metrics = super(CombinedLoss, self).compute()
        return retrieval_metrics['combined_loss']


class RelevanceLoss(CustomLoss):
    def compute(self):
        retrieval_metrics = super(RelevanceLoss, self).compute()
        return retrieval_metrics.get('relevance_loss', 0)
