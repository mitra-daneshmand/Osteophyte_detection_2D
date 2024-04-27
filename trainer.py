import os
import logging
import numpy as np
import cv2
from collections import defaultdict
from tqdm import tqdm
import torch
import torchmetrics
import torch.nn.functional as F
from torch import nn
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from seed import seed
from train_utils import init_model
from components import checkpoint
from kvs import GlobalKVS
from args import parse_args
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
import matplotlib.pyplot as plt


cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)

logging.basicConfig()
logger = logging.getLogger('train')
logger.setLevel(logging.DEBUG)
seed.set_ultimate_seed()

if torch.cuda.is_available():
    maybe_gpu = 'cuda'
else:
    maybe_gpu = 'cpu'


class ModelTrainer:
    def __init__(self, fold_idx=None):
        args = parse_args()
        self.fold_idx = fold_idx

        self.paths_weights_fold = os.path.join(args.output_dir, args.target_comp, f'fold_{self.fold_idx}')
        os.makedirs(self.paths_weights_fold, exist_ok=True)

        self.path_logs_fold = os.path.join(args.output_dir, args.target_comp, f'fold_{self.fold_idx}')
        os.makedirs(self.path_logs_fold, exist_ok=True)

        self.handlers_ckpt = checkpoint.CheckpointHandler(self.paths_weights_fold)

        paths_ckpt_sel = self.handlers_ckpt.get_last_ckpt()

        # Initialize and configure the model
        self.models = dict()
        self.models['OST'] = init_model().to(maybe_gpu)

        ############################### freezing layers ###############################
        if args.ft_portion == 'complete':
            print('All the model layers require grad!')
        else:
            # child_counter = 1
            # for child in self.models['OST'].children():  # self.models['OST'].encoder.children()
            #     if child_counter < 6:
            #         print("child ", child_counter, " frozen")
            #         for param in child.parameters():
            #             param.requires_grad = False
            #     else:
            #         print("child ", child_counter, " not frozen")
            #     child_counter += 1

            child_counter = 1
            ft_module_names = []
            ft_module_names.append('fc')

            parameters = []
            for k, v in self.models['OST'].named_parameters():
                if child_counter >= 15:
                    parameters.append({'params': v})
                else:
                    parameters.append({'params': v, 'lr': 0.0})
                child_counter += 1

        self.models['OST'] = nn.DataParallel(self.models['OST'])


        # Configure the training
        self.num_epoch = args.epochs
        self.optimizers = (optim.Adam(
            self.models['OST'].parameters(),  # parameters
            lr=args.learning_rate,
            weight_decay=args.weight_decay))

        # self.lr_update_rule = {10: 0.01}

        self.losse = nn.BCEWithLogitsLoss().cuda()

        self.AP = torchmetrics.AveragePrecision()

        self.tensorboard = SummaryWriter(self.path_logs_fold)

    def run_one_epoch(self, epoch_idx, loaders):
        fnames_acc = defaultdict(list)
        metrics_acc = dict()
        metrics_acc['samplew'] = defaultdict(list)
        metrics_acc['batchw'] = defaultdict(list)
        metrics_acc['datasetw'] = defaultdict(list)

        prog_bar_params = {'postfix': {'epoch': epoch_idx}, }

        if self.models['OST'].training:
            # ------------------------ Training regime ------------------------
            loader_ds = loaders['train']

            steps_ds = len(loader_ds)
            prog_bar_params.update({'total': steps_ds,
                                    'desc': f'Train, epoch {epoch_idx}'})

            loader_ds_iter = iter(loader_ds)

            with tqdm(**prog_bar_params) as prog_bar:
                for step_idx in range(steps_ds):
                    self.optimizers.zero_grad()

                    data_batch_ds = next(loader_ds_iter)

                    xs_ds, ys_true_ds = data_batch_ds['mask'], data_batch_ds['Target']

                    # b = xs_ds
                    # a2 = b[0, 1, :, :].cpu().numpy()
                    # a3 = b[0, 2, :, :].cpu().numpy()
                    # plt.imshow(a2)
                    # plt.show()
                    # plt.imshow(a3)
                    # plt.show()

                    fnames_acc['oai'].extend(data_batch_ds['ID_SIDE'])

                    xs_ds = xs_ds.to(maybe_gpu)

                    ys_pred_ds = self.models['OST'](xs_ds)

                    batch_ap = self.AP(F.sigmoid(ys_pred_ds.squeeze().float()),
                                       ys_true_ds.float().to(maybe_gpu))
                    batch_ba = balanced_accuracy_score(
                        (np.array(ys_true_ds.int().squeeze().detach().cpu().numpy())).astype(int),
                        np.array(F.sigmoid(ys_pred_ds).squeeze().detach().cpu().numpy()).round())
                    batch_roc_auc = roc_auc_score(y_true=ys_true_ds.detach().cpu().numpy(),
                                                  y_score=ys_pred_ds.detach().cpu().numpy())
                    batch_loss = self.losse(ys_pred_ds.squeeze().float(), ys_true_ds.float().to(maybe_gpu))


                    metrics_acc['batchw']['ap'].append(batch_ap.item())
                    metrics_acc['batchw']['ba'].append(batch_ba.item())
                    metrics_acc['batchw']['roc_auc'].append(batch_roc_auc.item())
                    metrics_acc['batchw']['loss'].append(batch_loss.item())


                    batch_loss.backward()
                    self.optimizers.step()

                    prog_bar.update(1)
        else:
            # ----------------------- Validation regime -----------------------
            loader_ds = loaders['val']

            steps_ds = len(loader_ds)
            prog_bar_params.update({'total': steps_ds,
                                    'desc': f'Validate, epoch {epoch_idx}'})

            loader_ds_iter = iter(loader_ds)

            with torch.no_grad(), tqdm(**prog_bar_params) as prog_bar:
                for step_idx in range(steps_ds):
                    data_batch_ds = next(loader_ds_iter)

                    xs_ds, ys_true_ds = data_batch_ds['mask'], data_batch_ds['Target']
                    fnames_acc['oai'].extend(data_batch_ds['ID_SIDE'])

                    xs_ds = xs_ds.to(maybe_gpu)

                    ys_pred_ds = self.models['OST'](xs_ds)

                    batch_ap = self.AP(F.sigmoid(ys_pred_ds.squeeze().float()),
                                       ys_true_ds.float().to(maybe_gpu))
                    batch_ba = balanced_accuracy_score(
                        (np.array(ys_true_ds.int().squeeze().detach().cpu().numpy())).astype(int),
                        np.array(F.sigmoid(ys_pred_ds).squeeze().detach().cpu().numpy()).round())
                    batch_roc_auc = roc_auc_score(y_true=ys_true_ds.detach().cpu().numpy(),
                                                  y_score=ys_pred_ds.detach().cpu().numpy())
                    batch_loss = self.losse(ys_pred_ds.squeeze().float(), ys_true_ds.float().to(maybe_gpu))

                    metrics_acc['batchw']['ap'].append(batch_ap.item())
                    metrics_acc['batchw']['ba'].append(batch_ba.item())
                    metrics_acc['batchw']['roc_auc'].append(batch_roc_auc.item())
                    metrics_acc['batchw']['loss'].append(batch_loss.item())

                    prog_bar.update(1)

        return metrics_acc, fnames_acc

    def fit(self, loaders):
        args = parse_args()
        epoch_idx_best = -1
        loss_best = float('inf')
        roc_auc_best = -100.0
        ap_best = -100.0
        ba_best = -100.0
        metrics_train_best = dict()
        fnames_train_best = []
        metrics_val_best = dict()
        fnames_val_best = []

        kvs = GlobalKVS()

        for epoch_idx in range(self.num_epoch):
            kvs.update('cur_epoch', epoch_idx)

            self.models = {n: m.train() for n, m in self.models.items()}
            metrics_train, fnames_train = self.run_one_epoch(epoch_idx, loaders)

            # Process the accumulated metrics
            for k, v in metrics_train['batchw'].items():
                metrics_train['datasetw'][k] = np.mean(np.asarray(v))

            self.models = {n: m.eval() for n, m in self.models.items()}
            metrics_val, fnames_val = \
                self.run_one_epoch(epoch_idx, loaders)

            # Process the accumulated metrics
            for k, v in metrics_val['batchw'].items():
                metrics_val['datasetw'][k] = np.mean(np.asarray(v))

            kvs.update('metrics_train', metrics_train)
            kvs.update('metrics_val', metrics_val)

            # Learning rate update
            # for s, m in self.lr_update_rule.items():
            #     if epoch_idx == s:
            #         for param_group in self.optimizers.param_groups:
            #             param_group['lr'] *= m
            #
            #             # Add console logging
            #             logger.info(f'Epoch: {epoch_idx}')
            #             for subset, metrics in (('train', metrics_train),
            #                                     ('val', metrics_val)):
            #                 logger.info(f'{subset} metrics:')
            #                 for k, v in metrics['datasetw'].items():
            #                     logger.info(f'{k}: \n{v}')

            # Add TensorBoard logging
            for subset, metrics in (('train', metrics_train),
                                    ('val', metrics_val)):
                # Log only dataset-reduced metrics
                for k, v in metrics['datasetw'].items():
                    if isinstance(v, np.ndarray):
                        self.tensorboard.add_scalars(
                            f'fold_{self.fold_idx}/{k}_{subset}',
                            {f'class{i}': e for i, e in enumerate(v.ravel().tolist())},
                            global_step=epoch_idx)
                    elif isinstance(v, (str, int, float)):
                        self.tensorboard.add_scalar(
                            f'fold_{self.fold_idx}/{k}_{subset}',
                            float(v),
                            global_step=epoch_idx)
                    else:
                        logger.warning(f'{k} is of unsupported dtype {v}')

            # Save the model
            loss_curr = metrics_val['datasetw']['loss']
            roc_auc_curr = metrics_val['datasetw']['roc_auc']
            ap_curr = metrics_val['datasetw']['ap']
            ba_curr = metrics_val['datasetw']['ba']
            print('Loss=', loss_curr)
            print('roc_auc=', roc_auc_curr)

            # roc_auc_best
            if roc_auc_curr > roc_auc_best:
                roc_auc_best = roc_auc_curr
                epoch_idx_best = epoch_idx
                metrics_train_best = metrics_train
                metrics_val_best = metrics_val
                fnames_train_best = fnames_train
                fnames_val_best = fnames_val

            # if loss_curr < loss_best:
            #     loss_best = loss_curr
            #     epoch_idx_best = epoch_idx
            #     metrics_train_best = metrics_train
            #     metrics_val_best = metrics_val
            #     fnames_train_best = fnames_train
            #     fnames_val_best = fnames_val

            # if ap_curr > ap_best:
            #     ap_best = ap_curr
            #     epoch_idx_best = epoch_idx
            #     metrics_train_best = metrics_train
            #     metrics_val_best = metrics_val
            #     fnames_train_best = fnames_train
            #     fnames_val_best = fnames_val

            # if ba_curr > ba_best:
            #     ba_best = ba_curr
            #     epoch_idx_best = epoch_idx
            #     metrics_train_best = metrics_train
            #     metrics_val_best = metrics_val
            #     fnames_train_best = fnames_train
            #     fnames_val_best = fnames_val

                self.handlers_ckpt.save_new_ckpt(
                    model=self.models['OST'],
                    model_name=args.backbone,
                    fold_idx=self.fold_idx,
                    epoch_idx=epoch_idx)

        msg = (f'Finished fold {self.fold_idx} '
               f'with the best roc_auc {roc_auc_best:.5f} '
               f'on epoch {epoch_idx_best}, '
               f'weights: ({self.paths_weights_fold})')

        logger.info(msg)
        return (metrics_train_best, fnames_train_best,
                metrics_val_best, fnames_val_best)

