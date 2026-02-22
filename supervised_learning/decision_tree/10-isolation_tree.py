#!/usr/bin/env python3
Node = __import__('8-build_decision_tree').Node
Leaf = __import__('8-build_decision_tree').Leaf
""" Isolation Forest Part I """
import numpy as np


class Isolation_Random_Tree() :
    def __init__(self, max_depth=10, seed=0, root=None) :
        self.rng               = np.random.default_rng(seed)
        if root :
            self.root          = root
        else :
            self.root          = Node(is_root=True)
        self.explanatory       = None
        self.max_depth         = max_depth
        self.predict           = None
        self.min_pop=1

    def __str__(self):
        return str(self.root)

    def depth(self):
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        return self.root.count_nodes_below(only_leaves=only_leaves)

    def update_bounds(self):
        self.root.update_bounds_below()

    def get_leaves(self):
        return self.root.get_leaves_below()

    def update_predict(self):
        self.update_bounds()
        leaves = self.get_leaves()
        for leaf in leaves:
            leaf.update_indicator()
        self.predict = lambda A: np.sum(
            np.array([leaf.indicator(A) * leaf.value for leaf in leaves]),
            axis=0
        )

    def random_split_criterion(self, node):
        diff = 0
        while diff == 0:
            feature = self.rng.integers(0, self.explanatory.shape[1])
            feature_min, feature_max = self.np_extrema(
                self.explanatory[:, feature][node.sub_population]
            )
            diff = feature_max - feature_min
        x = self.rng.uniform()
        threshold = (1 - x) * feature_min + x * feature_max
        return feature, threshold

    def get_leaf_child(self, node, sub_population):
        # value = size of population reaching this leaf (no target!)
        leaf_child = Leaf(np.sum(sub_population))
        leaf_child.depth = node.depth + 1
        leaf_child.sub_population = sub_population
        return leaf_child

    def get_node_child(self, node, sub_population):
        n = Node()
        n.depth = node.depth + 1
        n.sub_population = sub_population
        return n

    def fit_node(self, node):
        node.feature, node.threshold = self.random_split_criterion(node)

        left_population = (
            node.sub_population
            & (self.explanatory[:, node.feature] > node.threshold)
        )
        right_population = (
            node.sub_population
            & (self.explanatory[:, node.feature] <= node.threshold)
        )

        # different from Decision_Tree: no class check, only size and depth
        is_left_leaf = (
            np.sum(left_population) < self.min_pop
            or node.depth + 1 >= self.max_depth
        )

        if is_left_leaf:
            node.left_child = self.get_leaf_child(node, left_population)
        else:
            node.left_child = self.get_node_child(node, left_population)
            self.fit_node(node.left_child)

        is_right_leaf = (
            np.sum(right_population) < self.min_pop
            or node.depth + 1 >= self.max_depth
        )

        if is_right_leaf:
            node.right_child = self.get_leaf_child(node, right_population)
        else:
            node.right_child = self.get_node_child(node, right_population)
            self.fit_node(node.right_child)


    def fit(self,explanatory,verbose=0) :

        self.split_criterion = self.random_split_criterion
        self.explanatory = explanatory
        self.root.sub_population=np.ones_like(explanatory.shape[0],dtype='bool')

        self.fit_node(self.root)
        self.update_predict()

        if verbose==1 :
            print(f"""  Training finished.
    - Depth                     : { self.depth()       }
    - Number of nodes           : { self.count_nodes() }
    - Number of leaves          : { self.count_nodes(only_leaves=True) }""")