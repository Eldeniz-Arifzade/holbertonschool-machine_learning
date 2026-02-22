#!/usr/bin/env python3
""" Isolation Forest """
import numpy as np
Isolation_Random_Tree = __import__('10-isolation_tree').Isolation_Random_Tree


class Isolation_Random_Forest:
    """Isolation Forest using multiple Isolation Random Trees"""
    def __init__(self, n_trees=100, max_depth=10, seed=0):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.seed = seed
        self.trees = []
        self.rng = np.random.default_rng(seed)

    def fit(self, explanatory, verbose=0):
        self.split_criterion = self.random_split_criterion
        self.explanatory = explanatory
        self.root.sub_population = np.ones(explanatory.shape[0], dtype='bool')

        self.fit_node(self.root)
        self.update_predict()

        if verbose == 1:
            print(f"""  Training finished.
    - Depth                     : {self.depth()}
    - Number of nodes           : {self.count_nodes()}
    - Number of leaves          : {self.count_nodes(only_leaves=True)}""")

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