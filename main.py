#!/usr/bin/env python
"""Training and evaluation pipeline for skeleton-based sign language recognition.

    python main.py --config config/train/train.yaml
    python main.py --config config/test/test.yaml
"""
from __future__ import print_function

import argparse
import inspect
import os
import time
import numpy as np
import yaml
import pickle
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import shutil
import random
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import wandb
import torchmetrics

from timm.scheduler.cosine_lr import CosineLRScheduler


def init_seed(_):
    torch.cuda.manual_seed_all(1)
    torch.manual_seed(1)
    np.random.seed(1)
    random.seed(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def str2bool(v):
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def import_class(name):
    components = name.split(".")
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing

    def forward(self, x, target):
        confidence = 1. - self.smoothing
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


def get_parser():
    parser = argparse.ArgumentParser(
        description="Skeleton-based sign language recognition"
    )
    parser.add_argument("--work-dir", default="./work_dir/temp")
    parser.add_argument("-model_saved_name", default="")
    parser.add_argument("-Experiment_name", default="")
    parser.add_argument("--config", default="./config/train/train.yaml")

    # processor
    parser.add_argument("--phase", default="train")
    parser.add_argument(
        "--dataset", default="WLASL2000",
        choices=["WLASL100", "WLASL300", "WLASL1000", "WLASL2000",
                 "MSASL100", "MSASL200", "MSASL500", "MSASL1000"],
    )
    parser.add_argument("--save-score", type=str2bool, default=False)

    # debug
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--log-interval", type=int, default=10000)
    parser.add_argument("--save-interval", type=int, default=5,
                        help="evaluate, and save the best model, every N epochs")
    parser.add_argument("--print-log", type=str2bool, default=True)
    parser.add_argument("--show-topk", type=int, default=[1, 5], nargs="+")

    # feeder
    parser.add_argument("--feeder", default="feeder.feeder")
    parser.add_argument("--num-worker", type=int, default=4)
    parser.add_argument("--train-feeder-args", default=dict())
    parser.add_argument("--test-feeder-args", default=dict())

    # model
    parser.add_argument("--model", default=None)
    parser.add_argument("--model-args", type=dict, default=dict())
    parser.add_argument("--weights", default=None)
    parser.add_argument("--ignore-weights", type=str, default=[], nargs="+")

    # optim
    parser.add_argument("--optimizer", default="AdamW")
    parser.add_argument("--base-lr", type=float, default=1e-3)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--warmup-lr", type=float, default=1e-7)
    parser.add_argument("--warmup_prefix", type=str2bool, default=False)
    parser.add_argument("--warm_up_epoch", type=int, default=25)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--lr-scheduler", default="cosine")
    parser.add_argument("--grad-clip", type=str2bool, default=True)
    parser.add_argument("--grad-max", type=float, default=1.0)
    parser.add_argument("--loss-type", type=str, default="LSCE")

    parser.add_argument("--device", type=int, default=0, nargs="+")
    parser.add_argument("--nesterov", type=str2bool, default=False)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--test-batch-size", type=int, default=24)
    parser.add_argument("--start-epoch", type=int, default=0)
    parser.add_argument("--num-epoch", type=int, default=250)

    # wandb
    parser.add_argument("--wandb", type=str2bool, default=False)
    parser.add_argument("--wandb_name", default=None)
    parser.add_argument("--wandb_entity", default=None)
    parser.add_argument("--wandb_project", default=None)

    # auxiliary losses
    parser.add_argument('--csc_weight', type=float, default=0.0,
                        help='weight of the prototype contrastive loss (0 = off)')
    parser.add_argument('--inter_weight', type=float, default=0.0,
                        help='weight of the inter-affinity contrastive loss (0 = off)')

    # evaluation outputs
    parser.add_argument('--score_name', type=str, default='',
                        help='name of the .pkl score file written when save_score '
                             'is true; defaults to score_<split>.pkl')
    parser.add_argument('--feat_name', type=str, default='',
                        help='name of an .npz file to store the features in front '
                             'of the classifier, e.g. for t-SNE (empty = off)')

    return parser


class Processor:
    def __init__(self, arg):
        arg.model_saved_name = "./work_dir/" + arg.Experiment_name + "/save_models/"
        arg.work_dir = "./work_dir/" + arg.Experiment_name
        arg.eval_results_dir = "./work_dir/" + arg.Experiment_name + "/eval_results/"
        arg.train_feeder_args.setdefault("data_path", f"./data/{arg.dataset}/train_data_joint.npy")
        arg.train_feeder_args.setdefault("label_path", f"./data/{arg.dataset}/train_label.pkl")
        arg.test_feeder_args.setdefault("data_path", f"./data/{arg.dataset}/val_data_joint.npy")
        arg.test_feeder_args.setdefault("label_path", f"./data/{arg.dataset}/val_label.pkl")
        self.arg = arg

        if arg.phase == "train":
            if not arg.train_feeder_args["debug"]:
                if os.path.exists(arg.work_dir):
                    print("log_dir: ", arg.work_dir, "already exist")
                    answer = input("delete it? y/n:")
                    if answer == "y":
                        shutil.rmtree(arg.work_dir)
                        print("Dir removed: ", arg.work_dir)
                        os.makedirs(arg.work_dir)
                        os.makedirs(arg.model_saved_name)
                        os.makedirs(arg.eval_results_dir)
                    else:
                        print("Dir not removed: ", arg.work_dir)
                else:
                    os.makedirs(arg.work_dir)
                    os.makedirs(arg.model_saved_name)
                    os.makedirs(arg.eval_results_dir)
        self.save_arg()
        shutil.copy2(__file__, self.arg.work_dir)
        shutil.copy2(arg.config, self.arg.work_dir)

        self.global_step = 0
        self.best_acc = 0
        self.best_acc_5 = 0
        self.best_accuracy_per_class = 0
        self.best_accuracy_5_per_class = 0
        self.best_epoch = 0
        self.load_model()
        self.load_data()
        self.load_optimizer()

        if self.arg.phase == 'train':
            self.load_scheduler(len(self.data_loader['train']))

        self.lr = self.arg.base_lr

        if self.arg.wandb:
            wandb.init(
                name=self.arg.wandb_name,
                entity=self.arg.wandb_entity,
                project=self.arg.wandb_project,
                config=self.arg,
            )

    def load_data(self):
        Feeder = import_class(self.arg.feeder)
        num_class = self.arg.model_args["num_classes"]
        self.data_loader = dict()
        if self.arg.phase == "train":
            self.data_loader["train"] = torch.utils.data.DataLoader(
                dataset=Feeder(
                    **self.arg.train_feeder_args,
                    num_class=num_class,
                ),
                batch_size=self.arg.batch_size,
                shuffle=True,
                num_workers=self.arg.num_worker * len(self.arg.device),
                drop_last=True,
                worker_init_fn=init_seed,
            )
        self.data_loader["test"] = torch.utils.data.DataLoader(
            dataset=Feeder(
                **self.arg.test_feeder_args,
                num_class=num_class,
            ),
            batch_size=self.arg.test_batch_size,
            shuffle=False,
            num_workers=self.arg.num_worker * len(self.arg.device),
            drop_last=False,
            worker_init_fn=init_seed,
        )

    def load_model(self):
        output_device = (
            self.arg.device[0] if type(self.arg.device) is list else self.arg.device
        )
        self.output_device = output_device
        Model = import_class(self.arg.model)
        shutil.copy2(inspect.getfile(Model), self.arg.work_dir)
        self.model = Model(**self.arg.model_args).to(output_device)

        if self.arg.loss_type == 'CE':
            self.loss = nn.CrossEntropyLoss().to(output_device)
        else:
            self.loss = LabelSmoothingCrossEntropy(smoothing=0.1).to(output_device)

        if self.arg.weights:
            self.print_log("Load weights from {}.".format(self.arg.weights))
            if ".pkl" in self.arg.weights:
                with open(self.arg.weights, "r") as f:
                    weights = pickle.load(f)
            else:
                ckpt = torch.load(self.arg.weights,  map_location='cpu')
                if "weights" in ckpt.keys():
                    weights = ckpt["weights"]
                else:
                    weights = ckpt

            weights = OrderedDict(
                [[k.split("module.")[-1], v.to(output_device)] for k, v in weights.items()]
            )

            for w in self.arg.ignore_weights:
                if weights.pop(w, None) is not None:
                    self.print_log("Sucessfully Remove Weights: {}.".format(w))
                else:
                    self.print_log("Can Not Remove Weights: {}.".format(w))

            try:
                self.model.load_state_dict(weights)
            except:
                state = self.model.state_dict()
                diff = list(set(state.keys()).difference(set(weights.keys())))
                print("Can not find these weights:")
                for d in diff:
                    print("  " + d)
                state.update(weights)
                self.model.load_state_dict(state)

            if isinstance(ckpt, dict):
                if 'best_acc' in ckpt:
                    self.best_acc = ckpt['best_acc']
                if 'best_acc_5' in ckpt:
                    self.best_acc_5 = ckpt['best_acc_5']
                if 'best_accuracy_per_class' in ckpt:
                    self.best_accuracy_per_class = ckpt['best_accuracy_per_class']
                if 'best_accuracy_5_per_class' in ckpt:
                    self.best_accuracy_5_per_class = ckpt['best_accuracy_5_per_class']
                if 'epoch' in ckpt:
                    self.test_epoch = ckpt['epoch']
                else:
                    self.test_epoch = 200
            else:
                self.test_epoch = 200

        if type(self.arg.device) is list:
            if len(self.arg.device) > 1:
                self.model = nn.DataParallel(
                    self.model, device_ids=self.arg.device, output_device=output_device
                )

    def load_optimizer(self):
        # parameters the model excludes from weight decay
        model_ref = self.model.module if hasattr(self.model, 'module') else self.model
        nwd = set(model_ref.no_weight_decay()) if hasattr(model_ref, 'no_weight_decay') else set()

        decay, no_decay = [], []
        for name, p in model_ref.named_parameters():
            if not p.requires_grad:
                continue
            (no_decay if name in nwd else decay).append(p)
        param_groups = [
            {'params': decay,    'weight_decay': self.arg.weight_decay},
            {'params': no_decay, 'weight_decay': 0.0},
        ]
        self.print_log(f"param groups: decay={len(decay)} no_decay={len(no_decay)}")

        if self.arg.optimizer == "SGD":
            self.optimizer = optim.SGD(
                param_groups, lr=self.arg.base_lr, momentum=0.9,
                nesterov=self.arg.nesterov, weight_decay=self.arg.weight_decay)
        elif self.arg.optimizer == "AdamW":
            self.optimizer = optim.AdamW(
                param_groups, lr=self.arg.base_lr, weight_decay=self.arg.weight_decay)
        elif self.arg.optimizer == "Adam":
            self.optimizer = optim.Adam(
                param_groups, lr=self.arg.base_lr, weight_decay=self.arg.weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer: {self.arg.optimizer}")

        self.print_log("using warm up, epoch: {}".format(self.arg.warm_up_epoch))

        # resume the optimizer state if the checkpoint has one
        if self.arg.weights:
            ckpt = torch.load(self.arg.weights, map_location='cpu')
            if isinstance(ckpt, dict) and "optimizer" in ckpt:
                try:
                    self.optimizer.load_state_dict(ckpt["optimizer"])
                    self.print_log("Optimizer state loaded from checkpoint.")
                except:
                    self.print_log("Warning: Could not load optimizer state, using fresh optimizer.")

    def load_scheduler(self, n_iter_per_epoch):
        """Cosine schedule with linear warm-up, stepped once per iteration."""
        num_steps = int(self.arg.num_epoch * n_iter_per_epoch)
        warmup_steps = int(self.arg.warm_up_epoch * n_iter_per_epoch)

        self.lr_scheduler = CosineLRScheduler(
            self.optimizer,
            t_initial=(num_steps - warmup_steps) if self.arg.warmup_prefix else num_steps,
            lr_min=self.arg.min_lr,
            warmup_lr_init=self.arg.warmup_lr,
            warmup_t=warmup_steps,
            cycle_limit=1,
            t_in_epochs=False,
            warmup_prefix=self.arg.warmup_prefix,
        )

    def save_arg(self):
        arg_dict = vars(self.arg)
        if not os.path.exists(self.arg.work_dir):
            os.makedirs(self.arg.work_dir)
            os.makedirs(self.arg.work_dir + "/eval_results")
        with open("{}/config.yaml".format(self.arg.work_dir), "w") as f:
            yaml.dump(arg_dict, f)

    def print_time(self):
        localtime = time.asctime(time.localtime(time.time()))
        self.print_log("Local current time :  " + localtime)

    def print_log(self, str, print_time=True):
        if print_time:
            localtime = time.asctime(time.localtime(time.time()))
            str = "[ " + localtime + " ] " + str
        print(str)
        if self.arg.print_log:
            with open("{}/log.txt".format(self.arg.work_dir), "a") as f:
                print(str, file=f)

    def record_time(self):
        self.cur_time = time.time()
        return self.cur_time

    def split_time(self):
        split_time = time.time() - self.cur_time
        self.record_time()
        return split_time

    def train(self, epoch):
        self.model.train()
        self.print_log("Training epoch: {}".format(epoch + 1))
        loader = self.data_loader["train"]
        loss_value = []
        self.record_time()
        timer = dict(dataloader=0.001, model=0.001, statistics=0.001)
        process = tqdm(loader)

        for batch_idx, (data, index_t, label, index) in enumerate(process):
            self.lr_scheduler.step_update(self.global_step)
            self.global_step += 1

            data = data.float().to(self.output_device)
            index_t = index_t.float().to(self.output_device)
            label = label.long().to(self.output_device)
            timer["dataloader"] += self.split_time()

            # forward
            output = self.model(data, index_t)
            loss = self.loss(output, label)

            # auxiliary contrastive losses (training only)
            csc_w = self.arg.csc_weight
            in_w = self.arg.inter_weight
            if csc_w or in_w:
                model_ref = self.model.module if hasattr(self.model, 'module') else self.model
                l_ce = loss.detach()
                L = model_ref.aux_losses(label, output)
                loss = loss + csc_w * L['csc'] + in_w * L['inter']
                if batch_idx % 100 == 0:
                    self.print_log(f"  L_CE: {l_ce.item():.4f}  "
                                   f"csc: {L['csc'].item():.4f}  "
                                   f"inter: {L['inter'].item():.4f}")
            # backward
            self.optimizer.zero_grad()
            loss.backward()
            if self.arg.grad_clip:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.arg.grad_max)
            self.optimizer.step()

            loss_value.append(loss.data.item())
            timer["model"] += self.split_time()

            self.lr = self.optimizer.param_groups[0]["lr"]

            if self.arg.wandb:
                wandb.log({"train_loss": loss.item(), "lr": self.lr})

            if self.global_step % self.arg.log_interval == 0:
                self.print_log(
                    "\tBatch({}/{}) done. Loss: {:.4f}  lr:{:.6f}".format(
                        batch_idx, len(loader), loss.data.item(), self.lr
                    )
                )
            timer["statistics"] += self.split_time()

        # statistics
        proportion = {
            k: "{:02d}%".format(int(round(v * 100 / sum(timer.values()))))
            for k, v in timer.items()
        }
        self.print_log(
            "\tMean training loss: {:.4f}.".format(np.mean(loss_value))
        )
        self.print_log("\tLearning Rate: {:.6f}".format(self.lr))
        self.print_log(
            "\tTime consumption: [Data]{dataloader}, [Network]{model}".format(**proportion)
        )

    def eval(self, epoch, save_score=False, loader_name=["test"], wrong_file=None, result_file=None):
        if wrong_file is not None:
            f_w = open(wrong_file, "w")
        if result_file is not None:
            f_r = open(result_file, "w")
        self.model.eval()

        # Optionally collect the features in front of the classifier (e.g. for
        # t-SNE). A forward hook on the head captures exactly the vector the
        # real forward pass feeds into it.
        feat_name = self.arg.feat_name
        feat_buf, feat_lab, feat_hook = [], [], None
        if feat_name:
            net = self.model.module if hasattr(self.model, 'module') else self.model

            def _grab(_m, inp, _out):
                feat_buf.append(inp[0].detach().float().cpu().numpy())

            feat_hook = net.head.register_forward_hook(_grab)
            self.print_log(f"\tCollecting features at the head -> {feat_name}")

        num_class = self.arg.model_args["num_classes"]

        with torch.no_grad():
            self.print_log("Eval epoch: {}".format(epoch + 1))
            for ln in loader_name:
                loss_value = []
                score_frag = []
                step = 0
                process = tqdm(self.data_loader[ln])
                test_acc = torchmetrics.Accuracy(
                    task="multiclass", num_classes=num_class
                ).to(self.output_device)
                test_recall = torchmetrics.Recall(
                    task="multiclass", average="none", num_classes=num_class
                ).to(self.output_device)
                test_precision = torchmetrics.Precision(
                    task="multiclass", average="none", num_classes=num_class
                ).to(self.output_device)
                test_auc = torchmetrics.AUROC(
                    task="multiclass", average="macro", num_classes=num_class
                ).to(self.output_device)

                for batch_idx, (data, index_t, label, index) in enumerate(process):
                    data = data.float().to(self.output_device)
                    index_t = index_t.float().to(self.output_device)
                    label = label.long().to(self.output_device)

                    with torch.no_grad():
                        output = self.model(data, index_t)

                    loss = self.loss(output, label)
                    score_frag.append(output.data.cpu().numpy())
                    if feat_name:
                        feat_lab.append(label.cpu().numpy())
                    loss_value.append(loss.data.cpu().numpy())
                    test_acc(output.argmax(1), label)
                    test_auc.update(output, label)
                    test_recall(output.argmax(1), label)
                    test_precision(output.argmax(1), label)

                    _, predict_label = torch.max(output.data, 1)
                    step += 1

                    if wrong_file is not None or result_file is not None:
                        predict = list(predict_label.cpu().numpy())
                        true = list(label.data.cpu().numpy())
                        for i, x in enumerate(predict):
                            if result_file is not None:
                                f_r.write(str(x) + "," + str(true[i]) + "\n")
                            if x != true[i] and wrong_file is not None:
                                f_w.write(
                                    str(index[i]) + "," + str(x) + "," + str(true[i]) + "\n"
                                )

                score = np.concatenate(score_frag)

                if feat_name:
                    feat_hook.remove()
                    feat_hook = None
                    Fe = np.concatenate(feat_buf)
                    La = np.concatenate(feat_lab)
                    assert len(Fe) == len(La), \
                        f"feature/label count mismatch: {len(Fe)} vs {len(La)}"
                    fn = feat_name if feat_name.endswith('.npz') else feat_name + '.npz'
                    fdir = os.path.join('./work_dir', self.arg.Experiment_name,
                                        'eval_results')
                    os.makedirs(fdir, exist_ok=True)
                    np.savez_compressed(os.path.join(fdir, fn), feat=Fe, label=La)
                    self.print_log(f"\tSaved features: {os.path.join(fdir, fn)}  "
                                   f"{Fe.shape[0]} samples x {Fe.shape[1]} dims")

                total_acc = test_acc.compute()
                total_recall = test_recall.compute()
                total_precision = test_precision.compute()
                total_auc = test_auc.compute()

                accuracy = self.data_loader[ln].dataset.top_k(score, 1)
                accuracy_5 = self.data_loader[ln].dataset.top_k(score, 5)
                accuracy_per_class = self.data_loader[ln].dataset.per_class_acc_top_k(score, 1)
                accuracy_5_per_class = self.data_loader[ln].dataset.per_class_acc_top_k(score, 5)

                if accuracy > self.best_acc and self.arg.phase == "train":
                    self.best_acc = accuracy
                    self.best_acc_5 = accuracy_5
                    self.best_accuracy_per_class = accuracy_per_class
                    self.best_accuracy_5_per_class = accuracy_5_per_class
                    self.best_epoch = epoch
                    score_dict = dict(
                        zip(self.data_loader[ln].dataset.sample_name, score)
                    )
                    with open(
                        "./work_dir/" + self.arg.Experiment_name
                        + "/eval_results/best_acc.pkl", "wb"
                    ) as f:
                        pickle.dump(score_dict, f)

                    state_dict = self.model.state_dict()
                    weights = OrderedDict(
                        [[k.split("module.")[-1], v.cpu()] for k, v in state_dict.items()]
                    )
                    save_dict = {
                        "weights": weights,
                        "lr": self.lr,
                        "best_acc": self.best_acc,
                        "best_acc_5": self.best_acc_5,
                        "best_accuracy_per_class": self.best_accuracy_per_class,
                        "best_accuracy_5_per_class": self.best_accuracy_5_per_class,
                        "epoch": epoch,
                    }
                    torch.save(save_dict, self.arg.model_saved_name + "best_model.pt")

                self.print_log(
                    "Eval Accuracy: {}, model: {}".format(accuracy, self.arg.model_saved_name)
                )
                self.print_log(f"torch metrics acc: {(100 * total_acc):>0.1f}%\n")
                self.print_log(f"recall of every test dataset class:\n{total_recall}")
                self.print_log(f"precision of every test dataset class:\n{total_precision}")
                self.print_log(
                    f"f1 score: {2*total_recall*total_precision/(total_recall+total_precision)}"
                )
                self.print_log(f"auc: {total_auc.item()}")

                if self.arg.wandb:
                    wandb.log({
                        "eval_accuracy": accuracy,
                        "eval_loss": np.mean(loss_value),
                    })

                score_dict = dict(zip(self.data_loader[ln].dataset.sample_name, score))

                # Scores for ensembling. Unlike best_acc.pkl above, which is only
                # written when the accuracy improves, this is written on every
                # evaluation, so it always belongs to the loaded checkpoint.
                if save_score:
                    name = self.arg.score_name or f'score_{ln}.pkl'
                    if not name.endswith('.pkl'):
                        name += '.pkl'
                    out_dir = os.path.join('./work_dir', self.arg.Experiment_name,
                                           'eval_results')
                    os.makedirs(out_dir, exist_ok=True)
                    with open(os.path.join(out_dir, name), 'wb') as f:
                        pickle.dump(score_dict, f)
                    self.print_log(f"\tSaved scores: {os.path.join(out_dir, name)} "
                                   f"({len(score_dict)} samples)")

                self.print_log(
                    "\tMean {} loss of {} batches: {}.".format(
                        ln, len(self.data_loader[ln]), np.mean(loss_value)
                    )
                )
                for k in self.arg.show_topk:
                    self.print_log(
                        "\tTop{}: {:.2f}%".format(
                            k, 100 * self.data_loader[ln].dataset.top_k(score, k)
                        )
                    )
                    self.print_log(
                        "\tTop{} per-class: {:.2f}%".format(
                            k, 100 * self.data_loader[ln].dataset.per_class_acc_top_k(score, k)
                        )
                    )

        return np.mean(loss_value)

    def start(self):
        if self.arg.phase == "train":
            self.print_log("Parameters:\n{}\n".format(str(vars(self.arg))))
            self.global_step = int(
                self.arg.start_epoch * len(self.data_loader["train"]) / self.arg.batch_size
            )

            def count_parameters(model):
                return sum(p.numel() for p in model.parameters() if p.requires_grad)

            self.print_log(f"# Parameters: {count_parameters(self.model)}")

            for epoch in range(self.arg.start_epoch, self.arg.num_epoch):
                self.train(epoch)
                model_ref = self.model.module if hasattr(self.model, 'module') else self.model
                model_ref.on_epoch_end(epoch)

                if ((epoch + 1) % self.arg.save_interval == 0) or (
                    epoch + 1 == self.arg.num_epoch
                ):
                    self.eval(
                        epoch, save_score=self.arg.save_score, loader_name=["test"]
                    )

            self.print_log(
                "best accuracy: {}, best top-5 accuracy: {}, best accuracy per-class: {}, "
                "best top-5 accuracy per-class: {}, model_name: {}".format(
                    self.best_acc,
                    self.best_acc_5,
                    self.best_accuracy_per_class,
                    self.best_accuracy_5_per_class,
                    self.arg.model_saved_name,
                )
            )

        elif self.arg.phase == "test":
            if not self.arg.test_feeder_args["debug"]:
                wf = self.arg.model_saved_name + "_wrong.txt" if os.path.exists(
                    self.arg.model_saved_name + "_wrong.txt") else None
                rf = self.arg.model_saved_name + "_right.txt" if os.path.exists(
                    self.arg.model_saved_name + "_right.txt") else None
            else:
                wf = rf = None
            if self.arg.weights is None:
                print("Please appoint --weights.")
            self.arg.print_log = False
            self.print_log("Model:   {}.".format(self.arg.model))
            self.print_log("Weights: {}.".format(self.arg.weights))
            self.eval(
                epoch=self.test_epoch,
                save_score=self.arg.save_score,
                loader_name=["test"],
                wrong_file=wf,
                result_file=rf,
            )
            self.print_log("Done.\n")
            self.print_log(
                "best accuracy: {}, best top-5 accuracy: {}, best accuracy per-class: {}, "
                "best top-5 accuracy per-class: {}, model_name: {}".format(
                    self.best_acc,
                    self.best_acc_5,
                    self.best_accuracy_per_class,
                    self.best_accuracy_5_per_class,
                    self.arg.model_saved_name,
                )
            )


if __name__ == "__main__":
    parser = get_parser()

    p = parser.parse_args()
    if p.config is not None:
        with open(p.config, "r") as f:
            default_arg = yaml.safe_load(f)
        key = vars(p).keys()
        for k in default_arg.keys():
            if k not in key:
                print("WRONG ARG: {}".format(k))
                assert k in key
        parser.set_defaults(**default_arg)

    arg = parser.parse_args()
    init_seed(0)
    processor = Processor(arg)
    processor.start()
