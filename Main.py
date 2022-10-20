import argparse
import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
import Utils
from Utils import *

# from preprocess.Dataset import get_dataloader
from Dataset_MM import get_clints_hii_data
from transformer.Models import Encoder, Classifier, Hie_Encoder
from tqdm import tqdm
import os
import sys
from sklearn import metrics
import gc

eps = 1e-10

def train_epoch(model, training_data, optimizer, pred_loss_func, opt, classifier):
    """ Epoch operation in training phase. """

    model.train()
    losses = []
    sup_preds, sup_labels = [], []
    acc, auroc, auprc = 0,0,0

    for train_batch in tqdm(training_data, mininterval=2,
                      desc='  - (Training)   ', leave=False):
        """ prepare data """
        train_batch, labels = map(lambda x: x.to(opt.device), train_batch)

        observed_tp = train_batch[:, :, 2 * opt.num_types]
        limit_length = torch.argmax(observed_tp, -1).detach().cpu().numpy()
        max_len = max(limit_length)
        observed_data, observed_mask, observed_tp, tau = \
            train_batch[:, :max_len, :opt.num_types], train_batch[:, :max_len, opt.num_types:2 * opt.num_types], train_batch[:, :max_len, 2 * opt.num_types], \
                train_batch[:, :max_len, (2 * opt.num_types)+1: (3 * opt.num_types)+1]
        del train_batch

        """ forward """
        optimizer.zero_grad()

        if classifier is not None:
            out = model(observed_tp, observed_data, observed_mask, tau=tau) # [B,L,K,D]
            sup_pred = classifier(out)
            
            if sup_pred.dim() == 1:
                sup_pred = sup_pred.unsqueeze(0)

            if opt.task == "wbm":
                loss = torch.sum(pred_loss_func((sup_pred), labels.float()))
                sup_pred = torch.sigmoid(sup_pred)

            else:
                loss = torch.sum(pred_loss_func((sup_pred), labels))
                sup_pred = torch.softmax(sup_pred, dim=-1)

            if torch.any(torch.isnan(loss)):
                print("exit nan in pred loss!!!")
                print("sup_pred\n", sup_pred)
                print("sup_pred\n", torch.log(sup_pred))
                print("labels\n", labels)
                sys.exit(0)
            
            losses.append(loss.item())
            loss.backward()
            
            
            sup_preds.append(sup_pred.detach().cpu().numpy())
            sup_labels.append(labels.detach().cpu().numpy())
            
            del out, loss, sup_pred, labels

        optimizer.step()

        del observed_data, observed_mask, observed_tp, tau
        gc.collect()
        torch.cuda.empty_cache()

    train_loss = np.average(losses)

    if len(sup_preds) > 0:
        sup_labels = np.concatenate(sup_labels)
        sup_preds = np.concatenate(sup_preds)
        sup_preds = np.nan_to_num(sup_preds)
        
        if opt.task == 'wbm':
            acc, auroc, auprc = evaluate_ml(sup_labels, sup_preds)
        else:
            acc, auroc, auprc = evaluate_mc(sup_labels, sup_preds, opt.n_classes)

    return acc, auroc, auprc, train_loss



def eval_epoch(model, validation_data, pred_loss_func, opt, classifier):
    """ Epoch operation in evaluation phase. """

    model.eval()

    valid_losses = []
    sup_preds = []
    sup_labels = []
    acc, auroc, auprc = 0,0,0

    with torch.no_grad():
        for batch in tqdm(validation_data, mininterval=2,
                          desc='  - (Validation) ', leave=False):

            """ prepare data """
             #mTAN
            train_batch, labels = map(lambda x: x.to(opt.device), batch)
            observed_tp = train_batch[:, :, 2 * opt.num_types]
            limit_length = torch.argmax(observed_tp, -1).detach().cpu().numpy()
            max_len = max(limit_length)
            # [B,L,K], [B,L,K], [B,L]
            observed_data, observed_mask, observed_tp, tau = \
                train_batch[:, :max_len, :opt.num_types], train_batch[:, :max_len, opt.num_types:2 * opt.num_types], train_batch[:, :max_len, 2 * opt.num_types], \
                    train_batch[:, :max_len, (2 * opt.num_types)+1: (3 * opt.num_types)+1]
            del train_batch

            if classifier is not None:
                out = model(observed_tp, observed_data, observed_mask, tau=tau) # [B,L,K,D]

                sup_pred = classifier(out)
                if sup_pred.dim() == 1:
                    sup_pred = sup_pred.unsqueeze(0)

                if opt.task == "wbm":
                    valid_loss = torch.sum(pred_loss_func((sup_pred + eps), labels.float()))
                    sup_pred = torch.sigmoid(sup_pred)
                else:
                    valid_loss = torch.sum(pred_loss_func((sup_pred + eps), labels))
                    sup_pred = torch.softmax(sup_pred, dim=-1)

                sup_preds.append(sup_pred.detach().cpu().numpy())
                sup_labels.append(labels.detach().cpu().numpy())

            if valid_loss != 0:
                valid_losses.append(valid_loss.item())

            del out, observed_data, observed_mask, observed_tp, tau, valid_loss

            gc.collect()
            torch.cuda.empty_cache()

    valid_loss = np.average(valid_losses)

    if len(sup_preds) > 0:
        sup_labels = np.concatenate(sup_labels, axis=0)
        sup_preds = np.concatenate(sup_preds, axis=0)
        sup_preds = np.nan_to_num(sup_preds)
        
        if opt.task == 'wbm':
            acc, auroc, auprc = evaluate_ml(sup_labels, sup_preds)
        else:
            acc, auroc, auprc = evaluate_mc(sup_labels, sup_preds, opt.n_classes)
        
    return acc, auroc, auprc, valid_loss

def train(model, training_data, validation_data, testing_data, optimizer, scheduler, pred_loss_func, opt, \
                        early_stopping=None, classifier=None, save_path=None):

    """ Start training. """
    for epoch_i in range(opt.epoch):
        
        epoch = epoch_i + 1
        print('[ Epoch', epoch, ']')

        start = time.time()
        train_acc, auroc, auprc, train_loss = train_epoch(model, training_data, optimizer, pred_loss_func, opt, classifier)
        log_info(opt, 'Train', epoch, train_acc, train_time, start, value_rmse=0, auroc=auroc, auprc=auprc, loss=train_loss)

        # if not opt.pretrain:
        start = time.time()
        valid_acc, valid_auroc, valid_auprc, valid_loss = eval_epoch(model, validation_data, pred_loss_func, opt, classifier)
        log_info(opt, 'Valid', epoch, valid_acc, valid_time, start, value_rmse=0, auroc=valid_auroc, auprc=valid_auprc, loss=valid_loss)

        early_stopping(valid_loss, model, classifier)

        if early_stopping.early_stop: #and not opt.pretrain:
            print("Early stopping. Training Done.")
            break

        scheduler.step()

    if save_path is not None:
        print("Testing...")
        model, classifier = load_checkpoints(save_path, model, classifier=classifier, dp_flag=opt.dp_flag)
        start = time.time()
        test_acc, auroc, auprc, _ = eval_epoch(model, testing_data, pred_loss_func, opt, classifier)

        log_info(opt, 'Testing', epoch, test_acc, test_time, start, value_rmse=0, auroc=auroc, auprc=auprc)


def main():
    """ Main function. """

    parser = argparse.ArgumentParser()

    parser.add_argument('--root_path', type=str, default='')
    parser.add_argument('--data_path', type=str, default='path/to/save_data_folder/')

    parser.add_argument('--epoch', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--num_types', type=int, default=23)

    parser.add_argument('--d_model', type=int, default=16)
    parser.add_argument('--d_inner_hid', type=int, default=32)
    parser.add_argument('--d_k', type=int, default=8)
    parser.add_argument('--d_v', type=int, default=8)

    parser.add_argument('--n_head', type=int, default=3)
    parser.add_argument('--n_layers', type=int, default=3)

    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--log', type=str, default='log')

    parser.add_argument('--save_path', type=str, default=None)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--label', type=str, default=None)
    parser.add_argument('--patience', type=int, default=7)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--load_path', type=str, default=None)

    parser.add_argument('--task', type=str, default='nan')
    parser.add_argument('--sample_times', type=int, default=3)
    # enc_tau: "mlp", "tem"
    parser.add_argument('--enc_tau', type=str, default='mlp')
    parser.add_argument('--debug_flag', action='store_true')
    parser.add_argument('--dp_flag', action='store_true')
    # if OOM kill occur in loading data, try to split training set into 5 part and load data in batch
    parser.add_argument('--load_in_batch', action='store_true')
    parser.add_argument('--hie', action='store_true')
    parser.add_argument('--adpt', action='store_true')
    parser.add_argument('--width', type=float, default=0.0)
    parser.add_argument('--new_l', type=int, default=0)
    opt = parser.parse_args()
    seed = opt.seed
    
    # default device is CUDA
    opt.device = torch.device('cuda')
    # opt.device = torch.device('cpu')
    
    if opt.task == "mor" or opt.task == "decom":
        opt.n_classes = 2
    elif opt.task == 'vent' or opt.task == "vaso":
        opt.n_classes = 4
    elif opt.task == 'wbm':
        opt.n_classes = 54
    elif opt.task == 'los':
        opt.n_classes = 9
    else:
        print("invalid task name!")
        sys.exit(0)
    
    opt.log = opt.root_path + opt.log
    if opt.save_path is not None:
        opt.save_path = opt.root_path + opt.save_path
    
    if opt.load_path is not None:
        opt.load_path = opt.root_path + opt.load_path
        
    opt.log += opt.task
    save_name = opt.task
    Utils.setup_seed(seed)


    """ prepare dataloader """
    trainloader, validloader, testloader = get_clints_hii_data(opt)

    """ prepare model """
    if opt.hie:
        model = Hie_Encoder(
            opt=opt,
            num_types=opt.num_types,
            d_model=opt.d_model,
            d_inner=opt.d_inner_hid,
            n_layers=opt.n_layers,
            n_head=opt.n_head,
            d_k=opt.d_k,
            d_v=opt.d_v,
            dropout=opt.dropout,
        )
    else:
        model = Encoder(
            opt=opt,
            num_types=opt.num_types,
            d_model=opt.d_model,
            d_inner=opt.d_inner_hid,
            n_layers=opt.n_layers,
            n_head=opt.n_head,
            d_k=opt.d_k,
            d_v=opt.d_v,
            dropout=opt.dropout,
        )
        
    para_list = list(model.parameters())

    mort_classifier = Classifier(opt.d_model, opt.num_types, opt.n_classes)
    para_list += list(mort_classifier.parameters())

    # load model
    if opt.load_path is not None:
        print("Loading checkpoints...")
            
        model, mort_classifier = load_checkpoints(opt.load_path, model, classifier=mort_classifier)
    
    model = model.to(opt.device)
    
    for mod in [model, mort_classifier]:
        if mod is not None:
            mod = mod.to(opt.device)
    
    if opt.dp_flag:
        model = torch.nn.DataParallel(model)


    if opt.hie:
        opt.log = opt.log + '_hie'
        save_name = save_name + '_hie'
    
        if opt.adpt:
            opt.log = opt.log + '_adpt'
            save_name = save_name + '_adpt'
            
            if opt.width > 0:
                opt.log = opt.log + '_w' + str(opt.width)
                save_name = save_name + '_w' + str(opt.width)
            
            if opt.new_l > 0:
                opt.log = opt.log + '_c' + str(opt.new_l)
                save_name = save_name + '_c' + str(opt.new_l)

    """ optimizer and scheduler """

    params = (para_list)
    optimizer = optim.Adam(params, lr=opt.lr, betas=(0.9, 0.999), eps=1e-05, weight_decay=opt.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 10, gamma=0.5)

    """ prediction loss function """
    if opt.task == 'wbm':
        pred_loss_func = nn.BCEWithLogitsLoss(reduction='none').to(opt.device)
    else:
        pred_loss_func = nn.CrossEntropyLoss(ignore_index=-1, reduction='none').to(opt.device)

    opt.log = opt.log + "_seed" + str(seed) + '.log'
    
    if opt.save_path is not None:
        save_path = opt.save_path + save_name + '_seed' + str(opt.seed) + '.h5'
    
    # setup the log file
    with open(opt.log, 'w') as f:
        f.write('[Info] parameters: {}'.format(opt))

    """ number of parameters """
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('[Info] Number of parameters: {}'.format(num_params))
    print('[Info] parameters: {}'.format(opt))

    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=opt.patience, verbose=True, save_path=save_path, dp_flag=opt.dp_flag)

    """ train the model """
    train(model, trainloader, validloader, testloader, optimizer, scheduler, pred_loss_func, opt, early_stopping, mort_classifier, save_path=save_path)


if __name__ == '__main__':
    main()
