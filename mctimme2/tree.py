"""Functions for working with the phylogenetic tree.
"""

import re
import math
import random
import copy
import warnings
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from ete3 import Tree
from Bio import Phylo
import ete3


def parse_taxonomy(taxonomy):
    """Extract each taxonomic rank from a string.

    Parameters
    ----------
    taxonomy : str
        Sequence of taxonomic ranks delimited with semicolons, with each rank
        preceded by the lower case first letter of the rank name and two
        underscores. 't' (rather than 's') must be used for the preceding letter
        for the strain. Each rank is optional.

        Example - "k__Bacteria; p__Actinobacteria; c__Actinobacteria;
                   o__Bifidobacteriales; f__Bifidobacteriaceae;
                   g__Bifidobacterium; s__Bifidobacterium_aesculapii"

    Returns
    -------
    dict
        Taxonomic ranks, accessible with the keys 'kingdom', 'phylum', 'class',
        'order', 'family', 'genus', 'species', and 'strain'.
    """
    ranks = ('kingdom', 'phylum', 'class', 'order', 'family', 'genus',
             'species', 'strain',)
    regex = ''.join('({}__(?P<{}>[\w_\-\.\[\]\/=\(\),:#\*\+]*)(;|\Z)\s*)?'\
                     .format(x[0] if x != 'strain' else x[1], x) for x in ranks)
    return re.search(regex, taxonomy).groupdict()


def generate_proposal(tree):
    """Randomly reorder some leaves of the tree, keeping the same topology.

    This is used to quickly generate a proposal in the optimization of leaf
    order. It randomly selects a single internal node and then reverses it.

    Parameters
    ----------
    tree : ete3.TreeNode
        The tree
    """
    all_descendants = tree.get_descendants()
    all_internal_nodes = []
    for n in all_descendants:
        if not n.is_leaf():
            all_internal_nodes.append(n)

    nflip = random.choice(all_internal_nodes)
    nflip.children = [nflip.children[1], nflip.children[0]]

    return


def objective(tree, values):
    """Evaluate the leaf ordering objective function.

    Parameters
    ----------
    tree : ete3.TreeNode
        The current tree
    values : dict
        Keys are leaf node names. Values are importance score of that node.
    """

    importances = [values[n.name] for n in tree.iter_leaves()]
    slot_values = np.arange(1, len(importances) + 1)[::-1]

    return slot_values @ importances


def optimize_leaf_order(tree, values):
    """Optimize the order of the leaves in the phylogenetic tree.

    Nodes with higher values will tend to be placed towards the left.
    Optimization is performed with a Monte Carlo method. This method has only
    been tested on small trees (~100 leaves) and may not scale to larger trees.

    Parameters
    ----------
    tree : ete3.TreeNode
        The tree
    values : dict
        Keys are leaf node names. Values are importance score of that node.
    """
    n_chains = 2  # number of chains to run
    n_steps = 400  # number of iterations per chain

    final_Ls = []
    original_tree = tree.copy(method='newick')

    for chain in range(n_chains):
        Ls = []
        Ls.append(objective(tree, values))

        for step in range(n_steps):
            # Make a copy of the tree in case the proposal is not accepted
            old_tree = tree.copy(method='newick')

            generate_proposal(tree)
            L = objective(tree, values)

            if L > Ls[-1]:
                # Accept proposal
                Ls.append(L)
            else: # Reject proposal
                tree = old_tree
                Ls.append(Ls[-1])

        final_Ls.append(Ls[-1])

        # Go back to the original tree for the next chain
        tree = original_tree

    if final_Ls.count(final_Ls[0]) != len(final_Ls):  # final values are different
        warnings.warn('May not have converged.')


    best_chain = final_Ls.index(max(final_Ls))

    return old_tree
