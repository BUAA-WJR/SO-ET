#!/usr/bin/env python
# encoding: utf-8
import sys
import time
import torch
import random
import argparse
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from tree_data import load_tree
from lib.treecnn import TreeCNN
from tree_data import separate_data


criterion = nn.CrossEntropyLoss()


def train(args, model, device, train_trees, optimizer):
    model.train()
    total_iters = args.iters_per_epoch
    loss_accum = 0
    for pos in range(total_iters):
        selected_idx = np.random.permutation(len(train_trees))[:args.batch_size]
        batch_tree = [train_trees[idx] for idx in selected_idx]
        output = model(batch_tree)
        labels = torch.LongTensor([tree['label'] for tree in batch_tree]).to(device)
        loss = criterion(output, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss = loss.detach().cpu().numpy()
        loss_accum += loss
    average_loss = loss_accum/total_iters
    return average_loss


def pass_data_iteratively(model, trees, minibatch_size=64):
    model.eval()
    output = []
    idx = np.arange(len(trees))
    for i in range(0, len(trees), minibatch_size):
        sampled_idx = idx[i:i+minibatch_size]
        if len(sampled_idx) == 0:
            continue
        output.append(model([trees[j] for j in sampled_idx]).detach())
    return torch.cat(output, 0)

def test(args, model, device, train_trees, test_trees):
    model.eval()

    output = pass_data_iteratively(model, train_trees)
    pred = output.max(1, keepdim=True)[1]
    labels = torch.LongTensor([tree['label'] for tree in train_trees]).to(device)
    correct = pred.eq(labels.view_as(pred)).sum().cpu().item()
    acc_train = correct / float(len(train_trees))

    output = pass_data_iteratively(model, test_trees)
    pred = output.max(1, keepdim=True)[1]
    labels = torch.LongTensor([tree['label'] for tree in test_trees]).to(device)
    correct = pred.eq(labels.view_as(pred)).sum().cpu().item()
    acc_test = correct / float(len(test_trees))
    return acc_train, acc_test

def main(args):
    #set up seeds and gpu device
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    global global_data
    if global_data is None:
        global_data = load_tree(args.dataset, args.tree_depth, args.global_degree)
    trees = global_data
    num_classes = len(set([t['label'] for t in trees]))

    train_trees, test_trees = separate_data(trees, args.seed, args.fold_idx)

    tcn = TreeCNN
    model = tcn(args.tree_depth,
                args.num_mlp_layers,
                train_trees[0]['node_features'].shape[1],
                args.hidden_dim,
                num_classes,
                args.final_dropout,
                args.tree_pooling_type,
                device
                ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    accs = []
    for epoch in range(args.epochs):
        avg_loss = train(args, model, device, train_trees, optimizer)
        acc_train, acc_test = test(args, model, device, train_trees, test_trees)
        scheduler.step()
        accs.append(acc_test)
    return accs


if __name__ == '__main__':
    # Training settings
    # Note: Hyper-parameters need to be tuned in order to obtain results reported in the paper.
    parser = argparse.ArgumentParser(description='PyTorch tree convolutional neural net for whole-tree classification')
    parser.add_argument('-d', '--dataset', type=str, default="MUTAG",
                        help='name of dataset (default: MUTAG)')
    parser.add_argument('-k', '--tree_depth', type=int, default=2,
                        help='the depth of coding tree (default: 2)')
    parser.add_argument('-b', '--batch_size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('-e', '--epochs', type=int, default=350,
                        help='number of epochs to train (default: 350)')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.01,
                        help='learning rate (default: 0.01)')
    parser.add_argument('--iters_per_epoch', type=int, default=50,
                        help='number of iterations per each epoch (default: 50)')
    parser.add_argument('-s', '--seed', type=int, default=0,
                        help='random seed for splitting the dataset into 10 (default: 0)')
    parser.add_argument('-fi', '--fold_idx', type=int, default=0,
                        help='the index of fold in 10-fold validation. Should be less then 10.')
    parser.add_argument('-lm', '--num_mlp_layers', type=int, default=2,
                        help='number of layers for MLP EXCLUDING the input one (default: 2). 1 means linear model.')
    parser.add_argument('-hd', '--hidden_dim', type=int, default=64,
                        help='number of hidden units (default: 64)')
    parser.add_argument('-fd', '--final_dropout', type=float, default=0.5,
                        help='final layer dropout (default: 0.5)')
    parser.add_argument('-tp', '--tree_pooling_type', type=str, default="root", choices=["root", "sum", "average"],
                        help='Pooling for over nodes in a tree: root, sum or average')
    parser.add_argument('-gd', '--global_degree', action="store_true",
                        help='add the degree of nodes to features')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    args = parser.parse_args()

    global_data = None
    acc_ss = []
    for fold_idx in range(10):
        args.fold_idx = fold_idx
        accs = main(args)
        acc_ss.append(accs)
    df = pd.DataFrame(acc_ss)
    ss = df.mean()
    acc_max_index = list(ss).index(ss.max())
    print(args)
    print('%.4f\t%s' % (ss.max(), list(ss).index(ss.max())), list(df.iloc[:, acc_max_index]))
    sys.stdout.flush()
