#!/usr/bin/env python
# encoding: utf-8
# author:  ryan_wu
# email:   imitator_wu@outlook.com
# date:    2020-11-26 16:09:29
# 把networkx的图转成邻接矩阵,然后用结构信息树，给转成各种深度的树

import os
import sys
import copy
import json
import time
import pickle
import itertools
import traceback
import numpy as np
import networkx as nx
from multiprocessing import Pool
from lib.encoding_tree import PartitionTree


PWD = os.path.dirname(os.path.realpath(__file__))


def trans_to_adj(graph):
    graph.remove_edges_from(nx.selfloop_edges(graph))
    nodes = range(len(graph.nodes))
    return nx.to_numpy_array(graph, nodelist=nodes)


def trans_to_tree(adj, k=2):
    undirected_adj = np.array(adj)
    y = PartitionTree(adj_matrix=undirected_adj)
    x = y.build_encoding_tree(k)
    return y.tree_node


def update_depth(tree):
    # set leaf depth
    wait_update = [k for k, v in tree.items() if v.children is None]
    while wait_update:
        for nid in wait_update:
            node = tree[nid]
            if node.children is None:
                node.child_h = 0
            else:
                node.child_h = tree[list(node.children)[0]].child_h + 1
        wait_update = set([tree[nid].parent for nid in wait_update if tree[nid].parent])


def update_node(tree):
    update_depth(tree)
    d_id= [(v.child_h, v.ID) for k, v in tree.items()]
    d_id.sort()
    new_tree = {}
    for k, v in tree.items():
        n = copy.deepcopy(v)
        n.ID = d_id.index((n.child_h, n.ID))
        if n.parent is not None:
            n.parent = d_id.index((n.child_h+1, n.parent))
        if n.children is not None:
            n.children = [d_id.index((n.child_h-1, c)) for c in n.children]
        n = n.__dict__
        n['depth'] = n['child_h']
        new_tree[n['ID']] = n
    return new_tree


def pool_trans(input_):
    g, tree_depth = input_
    adj_mat = trans_to_adj(g['G'])
    tree = trans_to_tree(adj_mat, tree_depth)
    g['tree'] = update_node(tree)
    return g


def pool_trans_disconnected(input_):
    g, tree_depth = input_
    # 一个数据集里也有连通的图
    if nx.is_connected(g['G']):
        return pool_trans((g, tree_depth))
    trees = []
    # gi: 用来标记是第几个子图, graph index
    for gi, sub_nodes in enumerate(nx.connected_components(g['G'])):
        if len(sub_nodes) == 1:
        # 编码树没办法处理单个点, 手动组成单节点树
            node = list(sub_nodes)[0]
            # leaf node, parent的组成：graphIndex_layerIndex_nodeIndex
            js = [{'ID': node, 'parent': '%s_%s_0' % (gi, 1), 'depth': 0, 'children': None}]
            for d in range(1, tree_depth+1):
                js.append({'ID': '%s_%s_0' % (gi, d),
                           'parent': '%s_%s_0' % (gi, d+1) if d<tree_depth else None,
                           'depth': d,
                           'children': [js[-1]['ID']]
                          })
        else:
            sg = g['G'].subgraph(sub_nodes) # sub graph
            nodes = list(sg.nodes)
            nodes.sort()
            nmap = {n: nodes.index(n) for n in nodes}
            sg = nx.relabel_nodes(sg, nmap)
            # 图转树
            adj_mat = trans_to_adj(sg)
            tree = trans_to_tree(adj_mat, tree_depth)
            tree = update_node(tree)
            # relable tree id
            js = list(tree.values())
            rmap = {nodes.index(n): n for n in nodes}  # 叶子节点用原ID
            for j in js:
                if j['depth'] > 0:
                    rmap[j['ID']] = '%s_%s_%s' % (gi, j['depth'], j['ID'])
            for j in js:
                j['ID'] = rmap[j['ID']]
                j['parent'] = rmap[j['parent']] if j['depth']<tree_depth else None
                j['children'] = [rmap[c] for c in j['children']] if j['children'] else None
        trees.append(js)
    # 整树节点id relabel
    id_map = {}
    for d in range(0, tree_depth+1):
        for js in trees:
            for j in js:
                if j['depth'] == d:
                    # 叶子节点维持原图ID
                    id_map[j['ID']] = len(id_map) if d>0 else j['ID']
    tree = {}
    root_ids = []
    for js in trees:
        for j in js:
            n = copy.deepcopy(j)
            n['parent'] = id_map[n['parent']] if n['parent'] else None
            n['children'] = [id_map[c] for c in n['children']] if n['children'] else None
            n['ID'] = id_map[n['ID']]
            tree[n['ID']] = n
            if n['parent'] is None:
                root_ids.append(n['ID'])
    # 根节点合并
    root_id = min(root_ids)
    root_children = list(itertools.chain.from_iterable([tree[i]['children'] for i in root_ids]))
    root_node = {'ID': root_id, 'parent': None, 'children': root_children, 'depth': tree_depth}
    [tree.pop(i) for i in root_ids] # 删掉所有根节点
    for c in root_children: # 修改中间节点到根节点的映射到最新根节点
        tree[c]['parent'] = root_id
    tree[root_id] = root_node # 加入根节
    g['tree'] = tree
    return g


def struct_tree(dataset, tree_depth):
    if os.path.exists('trees/%s_%s.pickle' % (dataset, tree_depth)):
        return
    with open('graphs/%s.pickle' % dataset, 'rb') as fp:
        g_list = pickle.load(fp)
    pool_func = pool_trans_disconnected if dataset in disconnected_dataset else pool_trans
    pool = Pool()
    g_list = pool.map(pool_func, [(g, tree_depth) for g in g_list])
    pool.close()
    pool.join()
    g_list = filter(lambda g: g is not None, g_list)
    with open('trees/%s_%s.pickle' % (dataset, tree_depth), 'wb') as fp:
        pickle.dump(list(g_list), fp)


if __name__ == '__main__':
    disconnected_dataset = ['REDDITBINARY', 'REDDITMULTI5K', 'PROTEINS', 'NCI1']
    for d in os.listdir('datasets'):
        for k in [2, 3, 4, 5]:
            print(d, k)
            struct_tree(d[:-4], k)
