import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import numpy as np
from sklearn import metrics
import time
import random
import pandas as pd
from einops import rearrange, repeat

from sklearn.preprocessing import label_binarize

def cal_time_gap(observed_tp, observed_mask):
    # tmp_time = observed_mask * np.expand_dims(observed_tp,axis=-1) # [B,L,K]
    b,l,k = observed_mask.shape
    if observed_tp.dim() == 2:
        tmp_time = rearrange(observed_tp, 'b l -> b l 1')
    if observed_tp.dim() == 3:
        tmp_time = observed_tp[:]

    tmp_time = observed_mask * tmp_time
    
    tmp_time[observed_mask == 0] = np.nan
    tmp_time = tmp_time.transpose(1,0).numpy() # [L,B,K]
    tmp_time = np.reshape(tmp_time, (l,b*k)) # [L, B*K]
    # tmp_time = rearrange(tmp_time, 'l b k -> l (b k)')

    # padding the missing value with the next value
    df1 = pd.DataFrame(tmp_time)
    df1 = df1.fillna(method='bfill')
    tmp_time = np.array(df1)

    tmp_time = np.reshape(tmp_time, (l,b,k))
    tmp_time = tmp_time.transpose((1,0,2)) # [B,L,K]

    have_e_time = tmp_time.copy()
    have_e_time[:, :-1] = have_e_time[:, 1:] # the time event occured
    have_e_time[observed_mask == 0] = 0
    tmp_time[observed_mask == 1] = 0
    time_gap = tmp_time + have_e_time
    if observed_tp.dim() == 2:
        time_gap = time_gap - np.expand_dims(observed_tp,axis=-1)
    else:
        time_gap = time_gap - observed_tp.numpy()
    time_gap = np.nan_to_num(time_gap)
    return torch.from_numpy(time_gap)



class LabelSmoothingLoss(nn.Module):
    """
    With label smoothing,
    KL-divergence between q_{smoothed ground truth prob.}(w)
    and p_{prob. computed by model}(w) is minimized.
    """

    def __init__(self, label_smoothing, tgt_vocab_size, ignore_index=-100):
        assert 0.0 < label_smoothing <= 1.0
        super(LabelSmoothingLoss, self).__init__()

        self.eps = label_smoothing
        self.num_classes = tgt_vocab_size
        self.ignore_index = ignore_index

    def forward(self, output, target):
        """
        output (FloatTensor): (batch_size) x n_classes
        target (LongTensor): batch_size
        """

        non_pad_mask = target.ne(self.ignore_index).float()

        target[target.eq(self.ignore_index)] = 0
        one_hot = F.one_hot(target, num_classes=self.num_classes).float()
        one_hot = one_hot * (1 - self.eps) + (1 - one_hot) * self.eps / self.num_classes

        log_prb = F.log_softmax(output, dim=-1)
        loss = -(one_hot * log_prb).sum(dim=-1)
        loss = loss * non_pad_mask
        return loss


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, save_path=None, dp_flag=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.save_path = save_path
        self.dp_flag = dp_flag

    def __call__(self, val_loss, model, classifier=None,):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, classifier, dp_flag=self.dp_flag)
        elif score <= self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience} ({self.val_loss_min:.6f} --> {val_loss:.6f})')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, classifier, dp_flag=self.dp_flag)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, classifier=None, dp_flag=False):
        '''
        Saves model when validation loss decrease.
        '''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')

        if dp_flag:
            model_state_dict = model.module.state_dict()
        else:
            model_state_dict = model.state_dict()

        if self.save_path is not None:

            if classifier is None:
                classifier_state_dict = None
            else:
                classifier_state_dict = classifier.state_dict()

            torch.save({
                'model_state_dict':model_state_dict,
                'classifier_state_dict': classifier_state_dict,
            }, self.save_path)
        else:
            print("no path assigned")  

        self.val_loss_min = val_loss


def log_info(opt, phase, epoch, acc, rmse, start, value_rmse=0, auroc=0, auprc=0, loss=0):
    print('  -(', phase, ') epoch: {epoch}, RMSE: {rmse: 8.5f}, acc: {type: 8.5f}, '
                'AUROC: {auroc: 8.5f}, AUPRC: {auprc: 8.5f}, Value_RMSE: {value_rmse: 8.5f}, loss: {loss: 8.5f}, elapse: {elapse:3.3f} min'
                .format(epoch=epoch, type=acc, rmse=rmse, auroc=auroc, auprc=auprc, value_rmse=value_rmse, loss=loss, elapse=(time.time() - start) / 60))

    if opt.log is not None:
        with open(opt.log, 'a') as f:
            f.write(phase + ':\t{epoch}, TimeRMSE: {rmse: 8.5f},  ACC: {acc: 8.5f}, AUROC: {auroc: 8.5f}, AUPRC: {auprc: 8.5f}, ValueRMSE: {value_rmse: 8.5f}, Loss: {loss: 8.5f}\n'
                    .format(epoch=epoch, acc=acc, rmse=rmse, auroc=auroc, auprc=auprc, value_rmse=value_rmse, loss=loss))
                

def load_checkpoints(save_path, model, classifier=None, dp_flag=False, use_cpu=False):
    
    if use_cpu:
        checkpoint = torch.load(save_path,map_location=torch.device('cpu'))
    else:
        checkpoint = torch.load(save_path)
    
    if dp_flag:
        model.module.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint['model_state_dict'])

    if classifier is not None and checkpoint['classifier_state_dict'] is not None:
        classifier.load_state_dict(checkpoint['classifier_state_dict'])


    return model, classifier


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed(seed)  # gpu
    

def evaluate_mc(label, pred, n_class):
    if n_class > 2:
        labels_classes = label_binarize(label, classes=range(n_class))
        pred_scores = pred
        idx = np.argmax(pred_scores, axis=-1)
        preds_label = np.zeros(pred_scores.shape)
        preds_label[np.arange(preds_label.shape[0]), idx] = 1
        acc = metrics.accuracy_score(labels_classes, preds_label)
    else:
        labels_classes = label
        pred_scores = pred[:, 1]
        acc = np.mean(pred.argmax(1) == label)

    auroc = metrics.roc_auc_score(labels_classes, pred_scores, average='macro')
    auprc = metrics.average_precision_score(labels_classes, pred_scores, average='macro')

    return acc, auroc, auprc

def evaluate_ml(true, pred):
    auroc = metrics.roc_auc_score(true, pred, average='macro')
    auprc = metrics.average_precision_score(true, pred, average='macro')
    
    preds_label = np.array(pred > 0.5, dtype=float)
    acc = metrics.accuracy_score(true, preds_label)

    return acc, auroc, auprc