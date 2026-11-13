import torch
import os
import numpy as np
import h5py
import copy
import time
import random
from utils.data_utils import read_client_data
from utils.dlg import DLG
import json
import zlib


def rng_snapshot(py_rng):
    """State of every RNG that drives training: global torch, global NumPy, and the server's own
    Python RNG. (The *global* Python `random` is deliberately excluded: wandb's background upload
    thread calls random.random() for retry jitter, so it cannot be relied on.)"""
    return torch.get_rng_state().clone(), np.random.get_state(), py_rng.getstate()


def rng_crc(py_rng):
    """Compact fingerprint of the training RNG state; paired runs must match round by round."""
    t, n, p = rng_snapshot(py_rng)
    crc = zlib.crc32(t.numpy().tobytes())
    crc = zlib.crc32(np.asarray(n[1]).tobytes(), crc)
    crc = zlib.crc32(repr(p).encode(), crc)
    return crc


class Server(object):
    def __init__(self, args, times):
        # Set up the main attributes
        self.args = args
        self.device = args.device
        self.dataset = args.dataset
        self.num_classes = args.num_classes
        self.global_rounds = args.global_rounds
        self.local_epochs = args.local_epochs
        self.batch_size = args.batch_size
        self.learning_rate = args.local_learning_rate
        self.global_model = copy.deepcopy(args.model)
        self.num_clients = args.num_clients
        self.join_ratio = args.join_ratio
        self.random_join_ratio = args.random_join_ratio
        self.num_join_clients = int(self.num_clients * self.join_ratio)
        self.current_num_join_clients = self.num_join_clients
        self.few_shot = args.few_shot
        self.algorithm = args.algorithm
        self.time_select = args.time_select
        self.goal = args.goal
        self.time_threthold = args.time_threthold
        self.save_folder_name = args.save_folder_name
        self.top_cnt = args.top_cnt
        self.auto_break = args.auto_break
        # Server-owned Python RNG (client upload order). Not the global `random`, which other
        # threads (e.g. wandb retry jitter) can advance nondeterministically.
        self.py_rng = random.Random(getattr(args, "seed", 0))

        self.clients = []
        self.selected_clients = []
        self.train_slow_clients = []
        self.send_slow_clients = []

        self.uploaded_weights = []
        self.uploaded_ids = []
        self.uploaded_models = []

        self.rs_test_acc = []
        self.rs_test_auc = []
        self.rs_train_loss = []

        self.times = times
        self.eval_gap = args.eval_gap
        self.client_drop_rate = args.client_drop_rate
        self.train_slow_rate = args.train_slow_rate
        self.send_slow_rate = args.send_slow_rate

        self.dlg_eval = args.dlg_eval
        self.dlg_gap = args.dlg_gap
        self.batch_num_per_client = args.batch_num_per_client

        self.num_new_clients = args.num_new_clients
        self.new_clients = []
        self.eval_new_clients = False
        self.fine_tuning_epoch_new = args.fine_tuning_epoch_new

    def set_clients(self, clientObj):
        for i, train_slow, send_slow in zip(range(self.num_clients), self.train_slow_clients, self.send_slow_clients):
            train_data = read_client_data(self.dataset, i, is_train=True, few_shot=self.few_shot)
            test_data = read_client_data(self.dataset, i, is_train=False, few_shot=self.few_shot)
            client = clientObj(self.args, 
                            id=i, 
                            train_samples=len(train_data), 
                            test_samples=len(test_data), 
                            train_slow=train_slow, 
                            send_slow=send_slow)
            self.clients.append(client)

    # random select slow clients
    def select_slow_clients(self, slow_rate):
        slow_clients = [False for i in range(self.num_clients)]
        idx = [i for i in range(self.num_clients)]
        idx_ = np.random.choice(idx, int(slow_rate * self.num_clients))
        for i in idx_:
            slow_clients[i] = True

        return slow_clients

    def set_slow_clients(self):
        self.train_slow_clients = self.select_slow_clients(
            self.train_slow_rate)
        self.send_slow_clients = self.select_slow_clients(
            self.send_slow_rate)

    def select_clients(self):
        if self.random_join_ratio:
            self.current_num_join_clients = np.random.choice(range(self.num_join_clients, self.num_clients+1), 1, replace=False)[0]
        else:
            self.current_num_join_clients = self.num_join_clients
        selected_clients = list(np.random.choice(self.clients, self.current_num_join_clients, replace=False))

        return selected_clients

    def send_models(self):
        assert (len(self.clients) > 0)

        for client in self.clients:
            start_time = time.time()
            
            client.set_parameters(self.global_model)

            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)

    def receive_models(self):
        assert (len(self.selected_clients) > 0)

        active_clients = self.py_rng.sample(
            self.selected_clients, int((1-self.client_drop_rate) * self.current_num_join_clients))

        self.uploaded_ids = []
        self.uploaded_weights = []
        self.uploaded_models = []
        tot_samples = 0
        for client in active_clients:
            try:
                client_time_cost = client.train_time_cost['total_cost'] / client.train_time_cost['num_rounds'] + \
                        client.send_time_cost['total_cost'] / client.send_time_cost['num_rounds']
            except ZeroDivisionError:
                client_time_cost = 0
            if client_time_cost <= self.time_threthold:
                tot_samples += client.train_samples
                self.uploaded_ids.append(client.id)
                self.uploaded_weights.append(client.train_samples)
                self.uploaded_models.append(client.model)
        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples

    def aggregate_parameters(self):
        assert (len(self.uploaded_models) > 0)

        self.global_model = copy.deepcopy(self.uploaded_models[0])
        for param in self.global_model.parameters():
            param.data.zero_()
            
        for w, client_model in zip(self.uploaded_weights, self.uploaded_models):
            self.add_parameters(w, client_model)

    def add_parameters(self, w, client_model):
        for server_param, client_param in zip(self.global_model.parameters(), client_model.parameters()):
            server_param.data += client_param.data.clone() * w

    def save_global_model(self):
        model_path = os.path.join("models", self.dataset)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model_path = os.path.join(model_path, self.algorithm + "_server" + ".pt")
        torch.save(self.global_model, model_path)

    def load_model(self):
        model_path = os.path.join("models", self.dataset)
        model_path = os.path.join(model_path, self.algorithm + "_server" + ".pt")
        assert (os.path.exists(model_path))
        self.global_model = torch.load(model_path)

    def model_exists(self):
        model_path = os.path.join("models", self.dataset)
        model_path = os.path.join(model_path, self.algorithm + "_server" + ".pt")
        return os.path.exists(model_path)
        
    def save_results(self):
        algo = self.dataset + "_" + self.algorithm
        result_path = "../results/"
        if not os.path.exists(result_path):
            os.makedirs(result_path)

        if (len(self.rs_test_acc)):
            algo = algo + "_" + self.goal + "_" + str(self.times)
            file_path = result_path + "{}.h5".format(algo)
            print("File path: " + file_path)

            with h5py.File(file_path, 'w') as hf:
                hf.create_dataset('rs_test_acc', data=self.rs_test_acc)
                hf.create_dataset('rs_test_auc', data=self.rs_test_auc)
                hf.create_dataset('rs_train_loss', data=self.rs_train_loss)

    def save_item(self, item, item_name):
        if not os.path.exists(self.save_folder_name):
            os.makedirs(self.save_folder_name)
        torch.save(item, os.path.join(self.save_folder_name, "server_" + item_name + ".pt"))

    def load_item(self, item_name):
        return torch.load(os.path.join(self.save_folder_name, "server_" + item_name + ".pt"))

    def test_metrics(self):
        if self.eval_new_clients and self.num_new_clients > 0:
            self.fine_tuning_new_clients()
            return self.test_metrics_new_clients()
        
        num_samples = []
        tot_correct = []
        tot_auc = []
        for c in self.clients:
            ct, ns, auc = c.test_metrics()
            tot_correct.append(ct*1.0)
            tot_auc.append(auc*ns)
            num_samples.append(ns)

        ids = [c.id for c in self.clients]

        return ids, num_samples, tot_correct, tot_auc

    def train_metrics(self):
        if self.eval_new_clients and self.num_new_clients > 0:
            return [0], [1], [0]
        
        num_samples = []
        losses = []
        for c in self.clients:
            cl, ns = c.train_metrics()
            num_samples.append(ns)
            losses.append(cl*1.0)

        ids = [c.id for c in self.clients]

        return ids, num_samples, losses

    # evaluate selected clients
    def evaluate(self, acc=None, loss=None):
        stats = self.test_metrics()
        stats_train = self.train_metrics()

        test_acc = sum(stats[2])*1.0 / sum(stats[1])
        test_auc = sum(stats[3])*1.0 / sum(stats[1])
        train_loss = sum(stats_train[2])*1.0 / sum(stats_train[1])
        accs = [a / n for a, n in zip(stats[2], stats[1])]
        aucs = [a / n for a, n in zip(stats[3], stats[1])]
        
        if acc == None:
            self.rs_test_acc.append(test_acc)
        else:
            acc.append(test_acc)
        
        if loss == None:
            self.rs_train_loss.append(train_loss)
        else:
            loss.append(train_loss)

        print("Averaged Train Loss: {:.4f}".format(train_loss))
        print("Averaged Test Accuracy: {:.4f}".format(test_acc))
        print("Averaged Test AUC: {:.4f}".format(test_auc))
        # self.print_(test_acc, train_acc, train_loss)
        print("Std Test Accuracy: {:.4f}".format(np.std(accs)))
        print("Std Test AUC: {:.4f}".format(np.std(aucs)))

    @torch.no_grad()
    def evaluate_val(self, batch_size=256):
        """Global model on the official PathMNIST validation split (server-held, clean).
        Used only for model selection (rounds, mu, best round) - never for training."""
        if not hasattr(self, "_val_data"):
            from medmnist import PathMNIST
            ds = PathMNIST(split="val", download=False, size=28,
                           root=os.path.join("../dataset", self.dataset, "rawdata"))
            x = torch.tensor(ds.imgs.transpose(0, 3, 1, 2).astype(np.float32) / 255.0)
            y = torch.tensor(ds.labels.reshape(-1).astype(np.int64))
            self._val_data = (x, y)
        x_all, y_all = self._val_data
        self.global_model.eval()
        correct, loss_sum = 0, 0.0
        for i in range(0, len(y_all), batch_size):
            x = x_all[i:i + batch_size].to(self.device)
            y = y_all[i:i + batch_size].to(self.device)
            out = self.global_model(x)
            loss_sum += torch.nn.functional.cross_entropy(out, y, reduction="sum").item()
            correct += (out.argmax(1) == y).sum().item()
        n = len(y_all)
        return {"global_val_acc": correct / n, "global_val_loss": loss_sum / n}

    def evaluate_metrics(self):
        """Evaluate the current model held by every client (call after send_models()).
        Global metrics are sample-weighted over all clients' test sets.
        Guarded: evaluation must not advance any global RNG used by training."""
        rng_before = rng_crc(self.py_rng)
        metrics = {}
        test_correct = test_n = test_loss = 0
        train_n = train_loss = 0
        for c in self.clients:
            correct, n, loss_sum = c.clean_test_metrics()
            metrics[f"client_{c.id}_test_acc"] = correct / n
            metrics[f"client_{c.id}_test_loss"] = loss_sum / n
            test_correct += correct; test_n += n; test_loss += loss_sum

            tr_correct, tr_n, tr_loss_sum = c.train_loss_metrics()
            metrics[f"client_{c.id}_train_loss"] = tr_loss_sum / tr_n
            train_n += tr_n; train_loss += tr_loss_sum

        metrics["global_test_acc"] = test_correct / test_n
        metrics["global_test_loss"] = test_loss / test_n
        metrics["global_train_loss"] = train_loss / train_n
        if self.dataset == "PathMNIST":
            metrics.update(self.evaluate_val())
        for c in self.clients:
            metrics.update(c.corrupted_subset_metrics())   # only the noisy client returns values

        assert rng_crc(self.py_rng) == rng_before, "evaluation advanced a training RNG (would change training order)"

        # keep PFLlib's history lists so save_results()/best-acc printing still work
        self.rs_test_acc.append(metrics["global_test_acc"])
        self.rs_train_loss.append(metrics["global_train_loss"])

        print(f"global test acc {metrics['global_test_acc']:.4f} | "
              f"test loss {metrics['global_test_loss']:.4f} | "
              f"train loss {metrics['global_train_loss']:.4f} | "
              f"val acc {metrics.get('global_val_acc', float('nan')):.4f}")

        print("  per-client test acc: " + "  ".join(
            f"c{c.id}={metrics[f'client_{c.id}_test_acc']:.4f}" for c in self.clients))
        return metrics

    @torch.no_grad()
    def compute_update_stats(self):
        """Call after receive_models() and BEFORE aggregate_parameters().
        delta_i = theta_i - theta_global (model sent this round).
        mean_delta = sum_i w_i * delta_i with w_i = n_i / sum(n) -> exactly the aggregated update.
        client_i_cosine     = cos(delta_i, mean_delta)
        client_i_cosine_loo = cos(delta_i, weighted mean of the OTHER clients' deltas)"""
        def flatten(model):
            return torch.cat([p.detach().reshape(-1).float() for p in model.parameters()])

        global_vector = flatten(self.global_model)
        deltas = [flatten(m) - global_vector for m in self.uploaded_models]

        mean_delta = torch.zeros_like(global_vector)
        for w, d in zip(self.uploaded_weights, deltas):
            mean_delta += w * d
        mean_norm = mean_delta.norm().item()

        def cos(a, b):
            if a.norm().item() > 0 and b.norm().item() > 0:
                return torch.nn.functional.cosine_similarity(a, b, dim=0).item()
            return float("nan")

        metrics = {"global_update_norm": mean_norm}
        for i, (cid, d) in enumerate(zip(self.uploaded_ids, deltas)):
            metrics[f"client_{cid}_update_norm"] = d.norm().item()
            metrics[f"client_{cid}_cosine"] = cos(d, mean_delta) if mean_norm > 0 else float("nan")

            w_rest = 1.0 - self.uploaded_weights[i]
            if w_rest > 0:
                loo = (mean_delta - self.uploaded_weights[i] * d) / w_rest
                metrics[f"client_{cid}_cosine_loo"] = cos(d, loo)
            else:
                metrics[f"client_{cid}_cosine_loo"] = float("nan")
        return metrics

    def log_round(self, metrics):
        """Single sink for per-round metrics: history (-> metrics.csv), metrics.jsonl, W&B."""
        if not hasattr(self, "history"):
            self.history = []
        self.history.append(metrics)
        out = getattr(self, "output_dir", None)
        if out:
            with open(os.path.join(out, "metrics.jsonl"), "a") as f:
                f.write(json.dumps(metrics) + "\n")
        try:
            import wandb
            if wandb.run is not None:
                wandb.log(metrics)
        except ImportError:
            pass

    def run_rounds(self):
        """Round 0 evaluates the initial model; then exactly global_rounds rounds of
        train -> receive -> update stats -> aggregate -> evaluate new global model."""
        self.current_round = 0
        self.best_val_acc, self.best_val_round = -1.0, -1
        self.send_models()
        print("\n------------- Round 0 (initial model) -------------")
        m0 = {"round": 0, "rng_state_crc": rng_crc(self.py_rng), **self.evaluate_metrics()}
        self.log_round({**m0, **self._update_best_val(0, m0)})

        self.Budget = []
        for r in range(1, self.global_rounds + 1):
            s_t = time.time()
            self.current_round = r
            print(f"\n------------- Round {r} -------------")
            crc = rng_crc(self.py_rng)                     # RNG state at the start of this round

            self.selected_clients = self.select_clients()
            self.send_models()
            for client in self.selected_clients:
                client.train()

            self.receive_models()
            update_metrics = self.compute_update_stats()   # must be before aggregation
            self.aggregate_parameters()
            train_time = time.time() - s_t

            self.send_models()                             # evaluation reads client.model
            evaluation_metrics = self.evaluate_metrics()

            best = self._update_best_val(r, evaluation_metrics)

            self.Budget.append(time.time() - s_t)
            self.log_round({"round": r, "rng_state_crc": crc,
                            **evaluation_metrics, **update_metrics, **best,
                            "round_train_time_sec": train_time,
                            "round_total_time_sec": self.Budget[-1]})
            print(f"round time {self.Budget[-1]:.1f}s (train+aggregate {train_time:.1f}s)")

        print(f"\nFinal global test acc (round {self.global_rounds}): {self.rs_test_acc[-1]:.4f}")
        print(f"Best val acc {self.best_val_acc:.4f} at round {self.best_val_round}")
        print(f"Average time per round: {sum(self.Budget) / len(self.Budget):.1f}s")

    def _update_best_val(self, r, metrics):
        """Rolling best-validation checkpoint (strict improvement -> earliest round wins ties)."""
        val = metrics.get("global_val_acc")
        if val is not None and val > self.best_val_acc:
            self.best_val_acc, self.best_val_round = val, r
            out = getattr(self, "output_dir", None)
            if out:
                torch.save(self.global_model.state_dict(), os.path.join(out, "best_val_model.pt"))
        return {"best_val_acc_so_far": self.best_val_acc, "best_val_round_so_far": self.best_val_round}

    def print_(self, test_acc, test_auc, train_loss):
        print("Average Test Accuracy: {:.4f}".format(test_acc))
        print("Average Test AUC: {:.4f}".format(test_auc))
        print("Average Train Loss: {:.4f}".format(train_loss))

    def check_done(self, acc_lss, top_cnt=None, div_value=None):
        for acc_ls in acc_lss:
            if top_cnt is not None and div_value is not None:
                find_top = len(acc_ls) - torch.topk(torch.tensor(acc_ls), 1).indices[0] > top_cnt
                find_div = len(acc_ls) > 1 and np.std(acc_ls[-top_cnt:]) < div_value
                if find_top and find_div:
                    pass
                else:
                    return False
            elif top_cnt is not None:
                find_top = len(acc_ls) - torch.topk(torch.tensor(acc_ls), 1).indices[0] > top_cnt
                if find_top:
                    pass
                else:
                    return False
            elif div_value is not None:
                find_div = len(acc_ls) > 1 and np.std(acc_ls[-top_cnt:]) < div_value
                if find_div:
                    pass
                else:
                    return False
            else:
                raise NotImplementedError
        return True

    def call_dlg(self, R):
        # items = []
        cnt = 0
        psnr_val = 0
        for cid, client_model in zip(self.uploaded_ids, self.uploaded_models):
            client_model.eval()
            origin_grad = []
            for gp, pp in zip(self.global_model.parameters(), client_model.parameters()):
                origin_grad.append(gp.data - pp.data)

            target_inputs = []
            trainloader = self.clients[cid].load_train_data()
            with torch.no_grad():
                for i, (x, y) in enumerate(trainloader):
                    if i >= self.batch_num_per_client:
                        break

                    if type(x) == type([]):
                        x[0] = x[0].to(self.device)
                    else:
                        x = x.to(self.device)
                    y = y.to(self.device)
                    output = client_model(x)
                    target_inputs.append((x, output))

            d = DLG(client_model, origin_grad, target_inputs)
            if d is not None:
                psnr_val += d
                cnt += 1
            
            # items.append((client_model, origin_grad, target_inputs))
                
        if cnt > 0:
            print('PSNR value is {:.2f} dB'.format(psnr_val / cnt))
        else:
            print('PSNR error')

        # self.save_item(items, f'DLG_{R}')

    def set_new_clients(self, clientObj):
        for i in range(self.num_clients, self.num_clients + self.num_new_clients):
            train_data = read_client_data(self.dataset, i, is_train=True, few_shot=self.few_shot)
            test_data = read_client_data(self.dataset, i, is_train=False, few_shot=self.few_shot)
            client = clientObj(self.args, 
                            id=i, 
                            train_samples=len(train_data), 
                            test_samples=len(test_data), 
                            train_slow=False, 
                            send_slow=False)
            self.new_clients.append(client)

    # fine-tuning on new clients
    def fine_tuning_new_clients(self):
        for client in self.new_clients:
            client.set_parameters(self.global_model)
            opt = torch.optim.SGD(client.model.parameters(), lr=self.learning_rate)
            CEloss = torch.nn.CrossEntropyLoss()
            trainloader = client.load_train_data()
            client.model.train()
            for e in range(self.fine_tuning_epoch_new):
                for i, (x, y) in enumerate(trainloader):
                    if type(x) == type([]):
                        x[0] = x[0].to(client.device)
                    else:
                        x = x.to(client.device)
                    y = y.to(client.device)
                    output = client.model(x)
                    loss = CEloss(output, y)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()

    # evaluating on new clients
    def test_metrics_new_clients(self):
        num_samples = []
        tot_correct = []
        tot_auc = []
        for c in self.new_clients:
            ct, ns, auc = c.test_metrics()
            tot_correct.append(ct*1.0)
            tot_auc.append(auc*ns)
            num_samples.append(ns)

        ids = [c.id for c in self.new_clients]

        return ids, num_samples, tot_correct, tot_auc
