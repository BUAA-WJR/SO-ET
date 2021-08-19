#!/usr/bin/env python
# encoding: utf-8
# author:  ryan_wu
# email:   imitator_wu@outlook.com
# date:    2021-01-01 16:30:35

import os
import copy
import time
import torch
import pickle
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.model_selection import StratifiedKFold


PWD = os.path.dirname(os.path.realpath(__file__))


def extract_tree(graph, tree_depth, global_degree=False):
    leaf_size = len(graph['G'])
    tree = {'label': graph['label'],
            'node_size': [0] * (tree_depth+1),
            'edges': [[] for i in range(tree_depth+1)],
            'node_features': [0] * leaf_size,
            'node_degrees': [0] * leaf_size,
            }
    old_tree = copy.deepcopy(graph['tree'])
    # tree layer mask
    layer_idx = [0]
    for layer in range(tree_depth+1):
        layer_nodes = [i for i, n in old_tree.items() if n['depth']==layer]
        layer_idx.append(layer_idx[-1] + len(layer_nodes))
        tree['node_size'][layer] = len(layer_nodes)

    for i, n in old_tree.items():
        # edge
        if n['depth'] > 0:
            n_idx = n['ID'] - layer_idx[n['depth']]
            c_base = layer_idx[n['depth']-1]
            tree['edges'][n['depth']].extend([(n_idx, c-c_base) for c in n['children']])
            continue
        # leaf: node feature
        graphID = n.get('graphID', n['ID'])
        nid = n['ID']
        tree['node_features'][nid] = int(graph['G'].nodes[graphID].get('tag', 0))
        if global_degree:
            tree['node_degrees'][nid] = graph['G'].degree[graphID]
    return tree


def integrate_label(trees):
    labels = [t['label'] for t in trees]
    labels = list(set(labels))
    labels.sort()
    for t in trees:
        t['label'] = labels.index(t['label'])
    return trees


def one_hot_features(trees):
    label_set = list(set(sum([t['node_features'] for t in trees], [])))
    label_set.sort()
    for t in trees:
        leaf_size = t['node_size'][0]
        node_features = torch.zeros(leaf_size, len(label_set))
        node_features[range(leaf_size), [label_set.index(d) for d in t['node_features']]] = 1
        t['node_features'] = node_features


def add_additional_features(trees, fField):
    fset = list(set(sum([t[fField] for t in trees], [])))
    fset.sort()
    for t in trees:
        leaf_size = t['node_size'][0]
        features = torch.zeros(leaf_size, len(fset))
        features[range(leaf_size), [fset.index(f) for f in t[fField]]] = 1
        t['node_features'] = torch.cat([t['node_features'], features], dim=1)


def load_tree(dataset, tree_depth, global_degree=False):
    t_path = os.path.join(PWD, 'trees', '%s_%s.pickle' % (dataset, tree_depth))
    with open(t_path, 'rb') as fp:
        graphs = pickle.load(fp)
    trees = [extract_tree(g, tree_depth, global_degree) for g in graphs]
    integrate_label(trees)
    one_hot_features(trees)
    if global_degree:
        add_additional_features(trees, 'node_degrees')
    return trees


def separate_data(tree_list, seed, fold_idx):
    assert 0 <= fold_idx and fold_idx < 10, "fold_idx must be from 0 to 9."
    skf = StratifiedKFold(n_splits=10, shuffle = True, random_state = seed)

    labels = [tree['label'] for tree in tree_list]
    idx_list = []
    for idx in skf.split(np.zeros(len(labels)), labels):
        idx_list.append(idx)
    train_idx, test_idx = idx_list[fold_idx]

    train_tree_list = [tree_list[i] for i in train_idx]
    test_tree_list = [tree_list[i] for i in test_idx]

    return train_tree_list, test_tree_list


if __name__ == '__main__':
    load_tree('PTC', 2)
