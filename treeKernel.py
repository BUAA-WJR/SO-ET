#!/usr/bin/env python
# encoding: utf-8
# author:  ryan_wu
# email:   imitator_wu@outlook.com
# date:    2020-11-29 17:00:46

import os
import sys
import pickle
import argparse
import functools
import numpy as np
from sklearn import svm
from sklearn.model_selection import cross_val_score


PWD = os.path.dirname(os.path.realpath(__file__))
bio_dataset = ['MUTAG', 'PROTEINS', 'NCI1', 'PTC']
social_dataset = ['REDDITBINARY', 'IMDBBINARY', 'IMDBMULTI', 'COLLAB', 'REDDITMULTI5K']


def tuple_int_sort(a, b):
    if type(a) == type(b):
        return a > b
    elif isinstance(a, tuple) and isinstance(b, int):
        return 0
    elif isinstance(a, int) and isinstance(b, tuple):
        return 1


def tree_tags(graph, depth, tagV):
    global global_tags
    for i, n in graph['tree'].items():
        graphID = n.get('graphID', n['ID'])
        if n['depth'] != depth:
            continue
        if n['depth'] == 0:
            if tagV == 0:       # tag0: 叶子结点直接为0
                n['tag0'] = 0
            elif tagV == 1:     # tag1: 使用文件里本身tag
                n['tag1'] = int(graph['G'].nodes[graphID].get('tag', 0))
            elif tagV == 2:     # tag2: 使用节点的度
                n['tag2'] = graph['G'].degree[graphID]
            elif tagV == 3:
                n['tag3'] = (int(graph['G'].nodes[graphID].get('tag', 0)), graph['G'].degree[graphID])
            continue
        child_tags = [graph['tree'][c]['tag%s' % tagV] for c in n['children']]
        child_tags.sort()
        child_tags = ','.join(map(str, child_tags))
        if child_tags not in global_tags:
            global_tags[child_tags] = len(global_tags)
        n['tag%s' % tagV] = global_tags[child_tags]


global_tags = {0:0}
def load_data(dataset, tree_depth):
    with open('trees/%s_%s.pickle' % (dataset, tree_depth), 'rb') as fp:
        g_list = pickle.load(fp)
    global global_tags
    global_tags = {0:0}
    tagV = 2 if dataset in social_dataset else 3
    [tree_tags(g, k, tagV) for k in range(tree_depth+1) for g in g_list]
    all_tags = set([n['tag%s' % tagV] for g in g_list for i, n in g['tree'].items()])
    all_tags = list(all_tags)
    if tagV == 3:
        all_tags.sort(key=functools.cmp_to_key(tuple_int_sort))
    else:
        all_tags.sort()
    xs = []
    ys = []
    for g in g_list:
        ys.append(g['label'])
        tags = [n['tag%s' % tagV] for i, n in g['tree'].items()]
        x = [tags.count(t) for t in all_tags]
        xs.append(x)
    return xs, ys


def pool_crossV(input_):
    xs, ys, c, gamma = input_
    clf = svm.SVC(C=c, gamma=gamma)
    scores = cross_val_score(clf, np.array(xs), np.array(ys), cv=10, scoring='accuracy')
    return (c, scores)


def gridSearch(dataset, tree_depth):
    print(dataset, tree_depth)
    xs, ys = load_data(dataset, tree_depth)
    cs = [2**i for i in range(-5, 15)]
    gamma = 'auto' if dataset in ['IMDBBINARY', 'IMDBMULTI', 'REDDITBINARY', 'REDDITMULTI5K'] else 'scale'
    max_acc = 0
    for r in range(10):
        c_accs = []
        for c in cs:
            c_acc = pool_crossV((xs, ys, c, gamma))
            c_accs.append(c_acc)
        c_accs = list(c_accs)
        c_accs.sort(key=lambda ca: np.array(ca[1]).mean(), reverse=True)
        max_c, accs = c_accs[0]
        acc_mean = np.array(accs).mean()
        if acc_mean <= max_acc:
            break
        max_acc = max(max_acc, acc_mean)
        print(r, '%.6f' % max_c, '%.6f' % acc_mean, '[%s]' % ', '.join(['%.4f' % a for a in accs]))
        sys.stdout.flush()
        max_c_i = cs.index(max_c)
        max_c_left = cs[max(max_c_i-1, 0)]
        max_c_right = cs[min(max_c_i+1, len(cs)-1)]
        cs = [max_c_left+(max_c_right-max_c_left)/20*i for i in range(20)]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tree kernel with SVM for whole-tree classification')
    parser.add_argument('-d', '--dataset', type=str, default=None,
                        help='name of dataset (default: None, search all datasets)')
    parser.add_argument('-k', '--tree_depth', type=int, default=None, choices=[2, 3, 4, 5],
                        help='the depth of coding tree (default: None, search all depthes)')
    args = parser.parse_args()
    print(args)
    if args.dataset is not None and args.tree_depth is not None:
        gridSearch(args.dataset, args.tree_depth)
    elif args.dataset is not None and args.tree_depth is None:
        for k in [2, 3, 4, 5]:
            gridSearch(args.dataset, k)
    elif args.dataset is None and args.tree_depth is not None:
        for d in social_dataset + bio_dataset:
            gridSearch(d, args.tree_depth)
    else:
        for d in social_dataset + bio_dataset:
            for k in [2, 3, 4, 5]:
                gridSearch(d, k)
