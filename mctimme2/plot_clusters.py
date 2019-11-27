"""Make a figure for perturbation clusters.
"""

import ast
import numpy as np
import matplotlib
matplotlib.rcParams['font.sans-serif'] = "Arial"
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.patches as patches
import pandas

from Bio import Phylo
import io
import tree


class PertCluster:
    def __init__(self):
        self.members = []
        self.species = []
        self.subjects = []

    def transform_traj(self, design='K002'):

        times = np.linspace(-10, 55, len(self.traj))

        # Truncate to -5 -- 35 days
        new_times = times[np.logical_and(times > -5, times < 35)]
        self.traj = np.array(self.traj)
        new_traj = self.traj[np.logical_and(times > -5, times < 35)]
        self.traj = new_traj

def load_clustering(subset=None, keep_clusters=None):
    pdx2fos = ['301-010', '301-016', '301-020', '301-029', '301-012',
               '301-007', '301-019', '301-027', '301-026', '301-030']
    fos2pdx = ['301-001', '301-024', '301-015', '301-023', '301-005',
               '301-003', '301-034', '301-035', '301-031', '301-032']

    posterior_data = 'posterior_data.txt'
    cluster_data = 'cluster_data.txt'

    data = pandas.read_csv(posterior_data, sep='\t')

    all_subjects = []
    clusters = {}
    for i, row in data.iterrows():
        pc = row['pc']

        go = True
        if keep_clusters is not None:
            if pc not in keep_clusters:
                go = False

        if pc != -1:
            otu = row['subject'] + row['otu']
            subj = row['subject']
            all_subjects.append(subj)
            bf = row['bf']

            if subset is not None:
                if subset == 'pdx2fos':
                    go = subj in pdx2fos
                if subset == 'fos2pdx':
                    go = subj in fos2pdx
            if go:
                if pc in clusters.keys():
                    pass
                else:
                    clusters[pc] = PertCluster()
                clusters[pc].members.append(otu)
                clusters[pc].species.append(tree.parse_taxonomy(row['otu'])['species'].replace('[','_').replace(']','_'))
                clusters[pc].subjects.append(row['subject'])


    data = pandas.read_csv(cluster_data, sep='\t')
    for i, row in data.iterrows():
        pc = row['pc']
        traj = ast.literal_eval(row['traj'])

        if pc in clusters.keys():
            clusters[pc].traj = traj

    all_subjects = list(set(all_subjects))
    num_subjects = len(all_subjects)

    return clusters, num_subjects


def filter_clusters(clusters, prop_subjects, num_subjects, keep_otus=None):
    """Remove clusters without num subjects.
    """
    total_subjects = num_subjects

    failed = []

    if keep_otus is None:
        for i, pc in clusters.items():
            if len(set(pc.subjects)) / total_subjects < prop_subjects:
                failed.append(i)
    else:
        for i, pc in clusters.items():
            subjects = []
            for otu, subj in zip(pc.species, pc.subjects):
                if otu in keep_otus:
                    subjects.append(subj)
            if len(set(subjects)) / total_subjects < prop_subjects:
                failed.append(i)
    for i in failed:
        del clusters[i]
    return clusters


def get_otus(clusters, min_number):
    """Get the otus to include and their importance.
    """
    all_otus = []
    for pc in clusters.values():
        all_otus += pc.species

    all_otus = list(set(all_otus))
    otus_to_include = []
    importances = {}
    for otu in all_otus:
        num = 0
        for pc in clusters.values():
            num += pc.species.count(otu)

        if num >= min_number:
            otus_to_include.append(otu)
            importances[otu] = num

    return otus_to_include, importances


def load_tree(include_otus, importances):
    from ete3 import Tree
    import tree
    t = Tree('model_tree.txt')

    sp_in_tree = [n.name for n in t.iter_leaves()]

    for n in t.iter_leaves():
        if n.name not in include_otus:
            n.delete()

    t = tree.optimize_leaf_order(t, importances)
    ordered_otus = [n.name for n in t.iter_leaves()]

    return t, ordered_otus


def plot_clusters(clusters, otus, t):
    """
    enrichment : dict
        Keys are cluster labels. Values are the list of enriched cazymes.
    cazymes : list of str
        cazymes to include in the figure (in order)
    """
    # order the clusters
    pert_clusters = [pc for pc in clusters.values()]
    pert_clusters = [pc for pc in pert_clusters if abs(np.mean(pc.traj)) >= 0.1]
    pert_clusters = [pc for pc in pert_clusters if abs(np.mean(pc.traj)) >= 0.15]
    print('new len:', len(pert_clusters))

    for pc in pert_clusters:
        pc.direction = 1 if np.mean(pc.traj) >= 0 else -1
        try:
            pc.ontime = min(np.nonzero(pc.traj)[0])
        except ValueError:
            pc.ontime = 0

    pos = [pc for pc in pert_clusters if pc.direction > 0]
    neg = [pc for pc in pert_clusters if pc.direction < 0]

    pos = sorted(pos, key=lambda x:x.ontime)
    neg = sorted(neg, key=lambda x:x.ontime)

    pert_clusters = pos + neg

    # write
    with open('plotted_clusters.txt', 'w') as f:
        f.write('{}\t{}\t{}\t{}\n'.format('subject', 'otu', 'cluster', 'effect'))
        for i, pc in enumerate(pert_clusters):
            e = pc.direction
            for j, sp in enumerate(pc.species):
                f.write('{}\t{}\t{}\n'.format(pc.subjects[j], sp, i+1, e))


    x = np.zeros((len(pert_clusters), len(otus)))
    for i, cluster in enumerate(pert_clusters):
        for j, otu in enumerate(otus):
            if otu in cluster.species:
                x[i, j] = cluster.species.count(otu)

    for i, cluster in enumerate(pert_clusters):
        num_this_cluster = sum(x[i,:])
        for j in range(len(otus)):
            pass
            x[i,j] = x[i,j] / num_this_cluster


    x = x.transpose()

    fig = plt.figure(figsize=(11, 11))
    main_grid = fig.add_gridspec(2, 2, height_ratios=[1,3], width_ratios=[1,1])

    ax = fig.add_subplot(main_grid[1, 1])
    im = ax.imshow(x, cmap='bone_r', norm=colors.LogNorm(vmin=0.01, vmax=.5))


    cbaxes = fig.add_axes([.9, .33, .02, .1])  #left, bottom width, height
    cbar = plt.colorbar(im, cax=cbaxes, orientation='vertical')
    cbar.ax.set_xlabel('proportion of\n cluster members')

    ax.set_xticks([])
    ax.set_yticks(np.arange(x.shape[0]))
    ax.set_yticklabels(otus)

    ax.set_yticks([])
    ax.set_yticklabels([])

    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

    ax.set_xticks(np.arange(x.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(x.shape[0]+1)-.5, minor=True)
    ax.grid(which='minor', color='k', linestyle='-', linewidth=1.2)
    ax.tick_params(which='minor', bottom=False, left=False)

    # Add a white box off the edge of the axes to cover up any dotted lines
    # from the phylogenetic tree which are protruding from the grid
    rect = patches.Rectangle((len(pert_clusters)-0.5,-0.5), 1000, 1000, linewidth=1,edgecolor=None,facecolor='white', clip_on=False)
    ax.add_patch(rect)


    zoom = True
    if zoom:
        ax = fig.add_subplot(main_grid[0, 0])

        x = np.zeros((len(pert_clusters), len(pert_clusters[0].traj)))
        for i in range(len(pert_clusters)):
            for j in range(len(pert_clusters[0].traj)):
                x[i, j] = pert_clusters[i].traj[j]

        x = x.transpose()
        aspect = (1/len(pert_clusters[0].traj)) / (3 / len(otus))
        aspect = (4/len(pert_clusters[0].traj)) / (3/len(otus))
        im = ax.imshow(x, aspect=aspect, cmap='bwr_r', vmin=-3.5, vmax=3.5)
        ax.set_xlabel('pert. group')
        ax.xaxis.set_label_position('top')
        pert_start = 5/40 * len(pc.traj)
        day3 = 8/40 * len(pc.traj)
        day10 = 15/40 * len(pc.traj)
        day20 = 25/40 * len(pc.traj)
        day30 = 35/40 * len(pc.traj)
        pert_end = 33/40 * len(pc.traj)
        ax.set_yticks([pert_start, day3])
        ax.set_yticklabels(['0', '3'])
        rect = patches.Rectangle((-.5,pert_start),len(pert_clusters),pert_end-pert_start,linewidth=2,edgecolor='orange',facecolor='none')
        ax.add_patch(rect)

        ax.set_ylabel('time (days)')
        # ax.set_ylim(200, 1000)
        ax.set_ylim(8/40 * len(pc.traj), pert_start-10)

        ax.set_xticks(np.arange(len(pert_clusters)))
        ax.set_xticklabels(np.arange(len(pert_clusters))+1)#, rotation='vertical')
        ax.tick_params(axis='x', which='major', labelsize=6)
        ax.xaxis.tick_top()

        ax.set_xticks(np.arange(x.shape[1]+1)-.5, minor=True)
        ax.set_yticks([], minor=True)
        ax.grid(which='minor', color='k', linestyle='-', linewidth=1)
        ax.tick_params(which='minor', bottom=False, left=False)


    ax = fig.add_subplot(main_grid[0, 1])

    x = np.zeros((len(pert_clusters), len(pert_clusters[0].traj)))
    for i in range(len(pert_clusters)):
        for j in range(len(pert_clusters[0].traj)):
            x[i, j] = pert_clusters[i].traj[j]


    x = x.transpose()
    aspect = (1/len(pert_clusters[0].traj)) / (3 / len(otus))
    im = ax.imshow(x, aspect=aspect, cmap='bwr_r', vmin=-3.5, vmax=3.5)

    ax.set_xlabel('pert. group')
    ax.xaxis.set_label_position('top')

    cbaxes = fig.add_axes([.9, .825, .02, .1])  #left, bottom width, height
    cbar = plt.colorbar(im, cax=cbaxes, orientation='vertical')
    cbar.ax.set_xlabel('perturbation\neffect')

    K003 = False
    # K003 = True

    if not K003:

        pert_start = 5/40 * len(pc.traj)
        day10 = 15/40 * len(pc.traj)
        day20 = 25/40 * len(pc.traj)
        day30 = 35/40 * len(pc.traj)
        pert_end = 33/40 * len(pc.traj)
        ax.set_yticks([pert_start, pert_end])
        ax.set_yticks([pert_start, day10, day20, day30])
        ax.set_yticklabels(['0', '10', '20', '30'])
        rect = patches.Rectangle((-.5,pert_start),len(pert_clusters),pert_end-pert_start,linewidth=2,edgecolor='orange',facecolor='none')
        ax.add_patch(rect)

    elif K003:
        # K003 : perturbation 0--14 days
        ax.set_yticks([77, 615])
        ax.set_yticklabels(['0', '14'])
        rect = patches.Rectangle((-.5,77),len(pert_clusters),538,linewidth=2,edgecolor='orange',facecolor='none')
        ax.add_patch(rect)

    ax.set_ylabel('time (days)')

    ax.set_xticks(np.arange(len(pert_clusters)))
    ax.set_xticklabels(np.arange(len(pert_clusters))+1)#, rotation='vertical')
    ax.tick_params(axis='x', which='major', labelsize=6)
    ax.xaxis.tick_top()

    ax.set_xticks(np.arange(x.shape[1]+1)-.5, minor=True)
    ax.set_yticks([], minor=True)
    ax.grid(which='minor', color='k', linestyle='-', linewidth=1)
    ax.tick_params(which='minor', bottom=False, left=False)

    plt.subplots_adjust(hspace=0.01)

    if not zoom:
        ax = fig.add_subplot(main_grid[0, 0])
        ax.text(0.5, 1, 'B.   K002 (PDX)', size=20)
        plt.axis('off')

    ax = fig.add_subplot(main_grid[1, 0], zorder=-100)

    newick_tree = t.write()
    tree = Phylo.read(io.StringIO(newick_tree), 'newick')
    Phylo.draw(tree, axes=ax, do_show=False, show_confidence=False)
    for text in ax.texts:
        # Add dots from the species name to the grid
        text._text = text._text + '- ' * 100
        text.set_fontsize(8)
    plt.axis('off')
    plt.show()


def main():
    clusters, num_subjects = load_clustering()
    num_otus = None
    prop_subjects = None
    include_otus, importances = get_otus(clusters, num_otus)

    clusters = filter_clusters(clusters, prop_subjects, num_subjects, keep_otus=include_otus)

    t, ordered_otus = load_tree(include_otus, importances)

    for pc in clusters.values():
        pc.transform_traj()

    plotted_clusters = plot_clusters(clusters, ordered_otus, t)

if __name__ == '__main__':
    main()
