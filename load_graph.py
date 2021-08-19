#!/usr/bin/env python
# encoding: utf-8
import os
import pickle
import random
import itertools
import numpy as np
import networkx as nx


def load_data(dataset):
    '''
        dataset: name of dataset
        test_proportion: ratio of test train split
        seed: random seed for random splitting of dataset
    '''

    print('loading data')
    g_list = []
    label_dict = {}
    feat_dict = {}

    with open('datasets/%s.txt' % dataset, 'r') as f:
        n_g = int(f.readline().strip())
        for i in range(n_g):
            row = f.readline().strip().split()
            n, l = [int(w) for w in row]
            if not l in label_dict:
                mapped = len(label_dict)
                label_dict[l] = mapped
            g = nx.Graph()
            node_tags = []
            node_features = []
            n_edges = 0
            for j in range(n):
                row = f.readline().strip().split()
                tmp = int(row[1]) + 2
                g.add_node(j, tag=row[0])
                if tmp == len(row):
                    # no node attributes
                    row = [int(w) for w in row]
                    attr = None
                else:
                    row, attr = [int(w) for w in row[:tmp]], np.array([float(w) for w in row[tmp:]])
                if not row[0] in feat_dict:
                    mapped = len(feat_dict)
                    feat_dict[row[0]] = mapped
                node_tags.append(feat_dict[row[0]])

                if tmp > len(row):
                    node_features.append(attr)

                n_edges += row[1]
                for k in range(2, len(row)):
                    g.add_edge(j, row[k])

            if node_features != []:
                node_features = np.stack(node_features)
                node_feature_flag = True
            else:
                node_features = None
                node_feature_flag = False

            assert len(g) == n
            g_list.append({'G': g, 'label': l})
    print("# data: %d\tlabel:%s" % (len(g_list), len(label_dict)))
    with open('graphs/%s.pickle' % dataset, 'wb') as fp:
        pickle.dump(g_list, fp)


if __name__ == '__main__':
    for d in os.listdir('datasets'):
        print(d)
        load_data(d[:-4])
