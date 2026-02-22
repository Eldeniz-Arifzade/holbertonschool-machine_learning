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
            tree.fit(explanatory)
            self.trees.append(tree)
        if verbose == 1:
            depths = [tree.depth() for tree in self.trees]
            nodes = [tree.count_nodes() for tree in self.trees]
            leaves = [tree.count_nodes(only_leaves=True) for tree in self.trees]
            print(f"""  Training finished.
Mean depth                     : {np.mean(depths)}
Mean number of nodes           : {np.mean(nodes)}
Mean number of leaves          : {np.mean(leaves)}""")

    def suspects(self, explanatory, n_suspects=5):
        # Get depth of each individual in each tree
        all_depths = np.zeros((len(explanatory), len(self.trees)))
        for i, tree in enumerate(self.trees):
            all_depths[:, i] = tree.predict(explanatory)

        mean_depths = all_depths.mean(axis=1)
        idx = np.argsort(mean_depths)[:n_suspects]  # smallest depth = most suspicious
        return explanatory[idx], mean_depths[idx]