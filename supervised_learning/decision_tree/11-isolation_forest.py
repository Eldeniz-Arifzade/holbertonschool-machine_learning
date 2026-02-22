#!/usr/bin/env python3
""" Isolation Forest """
import numpy as np
Isolation_Random_Tree = __import__('10-isolation_tree').Isolation_Random_Tree


class Isolation_Random_Forest:
    def __init__(self, n_trees=100, max_depth=10, seed=0):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.seed = seed
        self.trees = []

    def fit(self, explanatory, verbose=0):
        self.trees = []
        for i in range(self.n_trees):
            tree = Isolation_Random_Tree(max_depth=self.max_depth, seed=self.seed + i)
            tree.fit(explanatory, verbose=0)  # <-- use the tree's fit, it has random_split_criterion
            self.trees.append(tree)
        if verbose == 1:
            depths = [tree.depth() for tree in self.trees]
            nodes = [tree.count_nodes() for tree in self.trees]
            leaves = [tree.count_nodes(only_leaves=True) for tree in self.trees]
            print(f"""  Training finished.
Mean depth                     : {np.mean(depths)}
Mean number of nodes           : {np.mean(nodes)}
Mean number of leaves          : {np.mean(leaves)}""")

    def suspects(self, explanatory, n_suspects):
        """Return indices of the top n_suspects likely outliers"""
        n_individuals = explanatory.shape[0]
        # Compute depths for each individual in each tree
        depths = np.zeros((self.n_trees, n_individuals))
        for t_idx, tree in enumerate(self.trees):
            depths[t_idx, :] = tree.predict(explanatory)
        # Average depth across all trees
        avg_depth = depths.mean(axis=0)
        # Smallest average depth -> most likely outlier
        suspect_indices = np.argsort(avg_depth)[:n_suspects]
        return suspect_indices