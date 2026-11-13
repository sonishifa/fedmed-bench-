import copy
import torch
import torch.nn as nn
import numpy as np
import os
from torch.utils.data import DataLoader
from sklearn.preprocessing import label_binarize
from sklearn import metrics
from utils.data_utils import read_client_data


class Client(object):
    """
    Base class for clients in federated learning.
    """

    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        self.args = args
        self._train_data = None     # cached (possibly label-corrupted) training data
        self.noise_record = None    # corruption targets for the noisy client (applied or hypothetical)
        self.noise_applied = False  # True only if labels were actually changed
        self._corrupt_x = None      # cached inputs of the corrupted subset (diagnostic)
        self._test_data = None      # cached test data
        self.model = copy.deepcopy(args.model)
        self.algorithm = args.algorithm
        self.dataset = args.dataset
        self.device = args.device
        self.id = id  # integer
        self.save_folder_name = args.save_folder_name

        self.num_classes = args.num_classes
        self.train_samples = train_samples
        self.test_samples = test_samples
        self.batch_size = args.batch_size
        self.learning_rate = args.local_learning_rate
        self.local_epochs = args.local_epochs
        self.few_shot = args.few_shot

        # check BatchNorm
        self.has_BatchNorm = False
        for layer in self.model.children():
            if isinstance(layer, nn.BatchNorm2d):
                self.has_BatchNorm = True
                break

        self.train_slow = kwargs['train_slow']
        self.send_slow = kwargs['send_slow']
        self.train_time_cost = {'num_rounds': 0, 'total_cost': 0.0}
        self.send_time_cost = {'num_rounds': 0, 'total_cost': 0.0}

        self.loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        self.learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer, 
            gamma=args.learning_rate_decay_gamma
        )
        self.learning_rate_decay = args.learning_rate_decay


    def get_train_data(self):
        """Read this client's training data once. For the noisy client, draw the corruption
        targets once: applied if noise_rate > 0, otherwise kept as hypothetical targets
        (rate = diag_noise_rate) so clean runs can report the same corrupted-subset diagnostic."""
        if self._train_data is None:
            data = read_client_data(self.dataset, self.id, is_train=True, few_shot=self.few_shot)
            if self.id == self.args.noisy_client:
                apply = self.args.noise_rate > 0
                rate = self.args.noise_rate if apply else getattr(self.args, "diag_noise_rate", 0.0)
                if rate > 0:
                    data = self._corrupt_labels(data, rate, apply)
            self._train_data = data
        return self._train_data

    def _corrupt_labels(self, data, rate, apply=True):
        """Symmetric label noise that always changes the label: y -> (y + U{1..C-1}) mod C.
        Uses its own NumPy generator (seeded by --seed), never the global torch/NumPy RNG."""
        rng = np.random.default_rng(self.args.seed)
        n = len(data)
        indices = rng.choice(n, size=int(rate * n), replace=False)
        original, replacement = [], []
        for index in indices:
            x, y = data[index]
            old_label = int(y)
            offset = int(rng.integers(1, self.num_classes))
            new_label = (old_label + offset) % self.num_classes
            if apply:
                data[index] = (x, torch.tensor(new_label, dtype=torch.long))
            original.append(old_label)
            replacement.append(new_label)
        self.noise_applied = apply
        self.noise_record = {
            "indices": np.asarray(indices, dtype=np.int64),
            "original_labels": np.asarray(original, dtype=np.int64),
            "replacement_labels": np.asarray(replacement, dtype=np.int64),
            "applied": np.bool_(apply),
            "num_train_samples": np.int64(n),
            "noise_rate": np.float64(rate),
            "seed": np.int64(self.args.seed),
        }
        verb = "corrupted" if apply else "drew HYPOTHETICAL (not applied) targets for"
        print(f"[noise] client {self.id}: {verb} {len(indices)}/{n} training labels "
              f"(rate={rate}, seed={self.args.seed})")
        return data

    def num_corrupted_labels(self):
        return int(len(self.noise_record["indices"])) if self.noise_applied else 0

    def save_noise_record(self, path):
        if self.noise_record is not None:
            np.savez(path, **self.noise_record)

    @torch.no_grad()
    def corrupted_subset_metrics(self, batch_size=256):
        """Predictions of the current model (= global model after send_models) on the
        corruption subset: fraction matching the injected label, the original label, or neither.
        Uses plain tensor batching (no DataLoader) so it consumes no RNG."""
        r = self.noise_record
        if r is None:
            return {}
        if self._corrupt_x is None:
            data = self.get_train_data()
            self._corrupt_x = torch.stack([data[i][0] for i in r["indices"]])
        self.model.eval()
        preds = []
        for i in range(0, len(self._corrupt_x), batch_size):
            out = self.model(self._corrupt_x[i:i + batch_size].to(self.device))
            preds.append(out.argmax(dim=1).cpu())
        p = torch.cat(preds).numpy()
        pred_noisy = float((p == r["replacement_labels"]).mean())
        pred_original = float((p == r["original_labels"]).mean())
        k = f"client_{self.id}_corrupted"
        return {f"{k}_pred_noisy": pred_noisy,
                f"{k}_pred_original": pred_original,
                f"{k}_pred_other": 1.0 - pred_noisy - pred_original}

    def load_train_data(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        return DataLoader(self.get_train_data(), batch_size, drop_last=True, shuffle=True, num_workers=0)

    def get_test_data(self):
        if self._test_data is None:
            self._test_data = read_client_data(self.dataset, self.id, is_train=False, few_shot=self.few_shot)
        return self._test_data

    def load_test_data(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        return DataLoader(self.get_test_data(), batch_size, drop_last=False, shuffle=False, num_workers=0,
                          generator=torch.Generator().manual_seed(0))  # keep global RNG untouched

    @torch.no_grad()
    def _loss_acc(self, data, batch_size=256):
        """Returns (correct_predictions, sample_count, summed_cross_entropy_loss)."""
        # A DataLoader iterator draws a base seed even with shuffle=False; give it its own
        # generator so evaluation never advances the global torch RNG used for training.
        loader = DataLoader(data, batch_size, drop_last=False, shuffle=False, num_workers=0,
                            generator=torch.Generator().manual_seed(0))
        self.model.eval()
        correct, n, loss_sum = 0, 0, 0.0
        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)
            output = self.model(x)
            loss_sum += torch.nn.functional.cross_entropy(output, y, reduction="sum").item()
            correct += (output.argmax(dim=1) == y).sum().item()
            n += y.shape[0]
        return correct, n, loss_sum

    def clean_test_metrics(self):
        """Current model on this client's test split (labels are never corrupted)."""
        return self._loss_acc(self.get_test_data())

    def train_loss_metrics(self):
        """Current model on this client's training split, using the labels it trains on
        (i.e. corrupted labels for the noisy client). Plain CE, no prox term, no dropped samples."""
        return self._loss_acc(self.get_train_data())

        
    def set_parameters(self, model):
        for new_param, old_param in zip(model.parameters(), self.model.parameters()):
            old_param.data = new_param.data.clone()

    def clone_model(self, model, target):
        for param, target_param in zip(model.parameters(), target.parameters()):
            target_param.data = param.data.clone()
            # target_param.grad = param.grad.clone()

    def update_parameters(self, model, new_params):
        for param, new_param in zip(model.parameters(), new_params):
            param.data = new_param.data.clone()

    def test_metrics(self):
        testloaderfull = self.load_test_data()
        # self.model = self.load_model('model')
        # self.model.to(self.device)
        self.model.eval()

        test_acc = 0
        test_num = 0
        y_prob = []
        y_true = []
        
        with torch.no_grad():
            for x, y in testloaderfull:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)

                test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                test_num += y.shape[0]

                y_prob.append(output.detach().cpu().numpy())
                nc = self.num_classes
                if self.num_classes == 2:
                    nc += 1
                lb = label_binarize(y.detach().cpu().numpy(), classes=np.arange(nc))
                if self.num_classes == 2:
                    lb = lb[:, :2]
                y_true.append(lb)

        # self.model.cpu()
        # self.save_model(self.model, 'model')

        y_prob = np.concatenate(y_prob, axis=0)
        y_true = np.concatenate(y_true, axis=0)

        auc = metrics.roc_auc_score(y_true, y_prob, average='micro')
        
        return test_acc, test_num, auc

    def train_metrics(self):
        trainloader = self.load_train_data()
        # self.model = self.load_model('model')
        # self.model.to(self.device)
        self.model.eval()

        train_num = 0
        losses = 0
        with torch.no_grad():
            for x, y in trainloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)
                loss = self.loss(output, y)
                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]

        # self.model.cpu()
        # self.save_model(self.model, 'model')

        return losses, train_num

    # def get_next_train_batch(self):
    #     try:
    #         # Samples a new batch for persionalizing
    #         (x, y) = next(self.iter_trainloader)
    #     except StopIteration:
    #         # restart the generator if the previous generator is exhausted.
    #         self.iter_trainloader = iter(self.trainloader)
    #         (x, y) = next(self.iter_trainloader)

    #     if type(x) == type([]):
    #         x = x[0]
    #     x = x.to(self.device)
    #     y = y.to(self.device)

    #     return x, y


    def save_item(self, item, item_name, item_path=None):
        if item_path == None:
            item_path = self.save_folder_name
        if not os.path.exists(item_path):
            os.makedirs(item_path)
        torch.save(item, os.path.join(item_path, "client_" + str(self.id) + "_" + item_name + ".pt"))

    def load_item(self, item_name, item_path=None):
        if item_path == None:
            item_path = self.save_folder_name
        return torch.load(os.path.join(item_path, "client_" + str(self.id) + "_" + item_name + ".pt"))

    # @staticmethod
    # def model_exists():
    #     return os.path.exists(os.path.join("models", "server" + ".pt"))
