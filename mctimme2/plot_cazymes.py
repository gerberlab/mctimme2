



"""For plotting enriched cazymes.
"""

from collections import defaultdict
import ujson
import pandas
import numpy as np
import matplotlib
matplotlib.rcParams['font.sans-serif'] = "Arial"
import matplotlib.pyplot as plt

# Clans of related Glycoside Hydrolase families
# from http://www.cazy.org/Glycoside-Hydrolases.html
GH_CLANS = {'GH-A' : ['GH1', 'GH2', 'GH5', 'GH10', 'GH17', 'GH26', 'GH30', \
                      'GH35', 'GH39', 'GH42', 'GH50', 'GH51', 'GH53', 'GH59', \
                      'GH72', 'GH79', 'GH86', 'GH113', 'GH128', 'GH147', \
                      'GH148', 'GH157', 'GH158'],
            'GH-B' : ['GH6', 'GH16'],
            'GH-C' : ['GH11', 'GH12'],
            'GH-D' : ['GH27', 'GH31', 'GH36'],
            'GH-E' : ['GH33', 'GH34', 'GH83', 'GH93'],
            'GH-F' : ['GH43', 'GH62'],
            'GH-G' : ['GH37', 'GH63', 'GH100', 'GH125'],
            'GH-H' : ['GH13', 'GH70', 'GH77'],
            'GH-I' : ['GH24', 'GH80'],
            'GH-J' : ['GH24', 'GH80'],
            'GH-K' : ['GH18', 'GH20', 'GH85'],
            'GH-L' : ['GH15', 'GH65'],
            'GH-M' : ['GH8', 'GH48'],
            'GH-N' : ['GH28', 'GH49'],
            'GH-O' : ['GH52', 'GH116'],
            'GH-P' : ['GH127', 'GH146'],
            'GH-Q' : ['GH94', 'GH149', 'GF161'],
            'GH-R' : ['GH29', 'GH107']}

GH_CLAN_NAMES = {'GH-A' : '(β/α)_8',
            'GH-B' : 'β-jelly roll',
            'GH-C' : 'β-jelly roll',
            'GH-D' : '(β/α)_8',
            'GH-E' : '6-fold β-propeller',
            'GH-F' : '5-fold β-propeller',
            'GH-G' : '(α/α)_6',
            'GH-H' : '(β/α)_8',
            'GH-I' : 'α+β',
            'GH-J' : '5-fold β-propeller',
            'GH-K' : '(β/α)_8',
            'GH-L' : '(α/α)_6',
            'GH-M' : '(α/α)_6',
            'GH-N' : 'β-helix',
            'GH-O' : '(α/α)_6',
            'GH-P' : '(α/α)_6',
            'GH-Q' : '(α/α)_6',
            'GH-R' : '(β/α)_8',
            'GH-Z' : 'no clan'}

# Add a "null" clan (Z) for all GH which are not in a clan
all_clan_GHs = []
for k in GH_CLANS.keys():
    all_clan_GHs += GH_CLANS[k]
all_GHs = ['GH{}'.format(i) for i in range(1, 166)]
GH_CLANS['GH-Z'] = []
for GH in all_GHs:
    if GH not in all_clan_GHs:
        GH_CLANS['GH-Z'].append(GH)

def load_enrichment(filename):
    """Load enriched cazymes from text file.

    Parameters
    ----------
    filename : str
        Path to file with enrichment results

    Returns
    -------
    dict
        Keys are cluster labels. Values are the list of enriched cazymes.
    """

    data = pandas.read_csv(filename, sep='\t')

    enriched = defaultdict(list)
    for i, row in data.iterrows():
        if row['perturbation_cluster'] != 18:
            enriched[row['perturbation_cluster']].append(row['cazyme'])

    enriched = dict(enriched)

    return enriched


def order_cazymes(cazymes):
    """Order cazymes by Carbohydrate-Active enZYmes Database structure.
    """
    cazyme_ids = []
    for cazyme in cazymes:
        cazyme_id = cazyme.split('.')[0]
        cazyme_ids.append(cazyme_id)

    sorted_cazyme_ids = []

    enzyme_class = 'GH'
    for clan in sorted(GH_CLANS.keys()):
        x = []
        for id in cazyme_ids:
            if id in GH_CLANS[clan]:
                x.append(int(id[2:]))
        x.sort()
        sorted_cazyme_ids += ['{}{}'.format(enzyme_class, id) for id in x]


    classes = ['GT', 'PL', 'CE', 'AA', 'CB']
    for enzyme_class in classes:
        x = []
        for id in cazyme_ids:
            if id[:2] == enzyme_class:
                if enzyme_class != 'CB':
                    x.append(int(id[2:]))
                else:
                    x.append(int(id[3:]))

        x.sort()
        sorted_cazyme_ids += ['{}{}'.format(enzyme_class +'M' if enzyme_class=='CB' else enzyme_class, id) for id in x]

    return sorted_cazyme_ids

def color_cazyme_labels(ax, clans=True):
    """Colorize the cazyme names according to their family.

    Assumes that the tick labels are on the x-axis. Run this function after
    making the other changes to the ticks and labels.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes with labels
    clans : bool
        If True, separate GH clans

    Returns
    -------
    ax : matplotlib.axes.Axes
        Axes with colorized labels
    """

    colors = {'GH': 'grey',
              'GT': 'darkcyan',
              'PL': 'blue',
              'CE': 'green',
              'AA': 'cadetblue',
              'CB': 'red'}

    names = {'GH': 'Glycoside Hydrolases',
             'GT': 'Glycosyl Transferases',
             'PL': 'Polysaccharide Lyases',
             'CE': 'Carbohydrate Esterases',
             'AA': 'Auxiliary Activities',
             'CB': 'Carbohydrate-Binding Modules'}

    extremes = {}
    clan_extremes = {}

    for i, label in enumerate(ax.get_yticklabels()):
        name = label._text
        if name[:2] == 'GH':
            cmap = matplotlib.cm.get_cmap('Greys')
            total_clans = len(GH_CLANS.keys())
            my_clan = -1
            for j, clan in enumerate(GH_CLANS.keys()):
                if name in GH_CLANS[clan]:
                    my_clan = clan
                    pos = (1 - (j/total_clans) ) * .9 + .1
                    pos = .75
                    label.set_color(cmap(pos))
            label.set_color(colors['GH'])

            if my_clan != -1:
                if my_clan in clan_extremes.keys():
                    clan_extremes[my_clan][1] = i
                else:
                    clan_extremes[my_clan] = [i, i+.1]

        elif name[:2] == 'GT':
            label.set_color(colors['GT'])
        elif name[:2] == 'PL':
            label.set_color(colors['PL'])
        elif name[:2] == 'CE':
            label.set_color(colors['CE'])
        elif name[:2] == 'AA':
            label.set_color(colors['AA'])
        elif name[:2] == 'CB':
            label.set_color(colors['CB'])

        if name[:2] in extremes.keys():
            extremes[name[:2]][1] = i
        else:
            extremes[name[:2]] = [i, i+.001]

    for key, spread in extremes.items():
        fraction = -0.2
        fraction = -1 * (1/(spread[1]-spread[0]))
        if key != 'GH':
            ax.annotate('', xy=(-5, spread[0]), xytext=(-5, spread[1]),
                    xycoords='data',
                    arrowprops=dict(connectionstyle='bar, fraction={},armA=-1, armB=-1'.format(fraction),
                    arrowstyle='-', color=colors[key], linewidth=2),
                    annotation_clip=False)
            ax.text(-6, (spread[0] + spread[1]) / 2, names[key], color=colors[key], horizontalalignment='right', verticalalignment='center')
        if key == 'GH':
            ax.annotate('', xy=(-12, spread[0]), xytext=(-12, spread[1]),
                    xycoords='data',
                    arrowprops=dict(connectionstyle='bar, fraction={},armA=-1, armB=-1'.format(fraction),
                    arrowstyle='-', color=colors[key], linewidth=2),
                    annotation_clip=False)
            ax.text(-13, (spread[0] + spread[1]) / 2, names[key], color=colors[key], horizontalalignment='right', verticalalignment='center')

    for key, spread in clan_extremes.items():
        if key != 'GH-Z':
            fraction = -0.2
            fraction = -1 * (1/(spread[1]-spread[0]))
            ax.annotate('', xy=(-4, spread[0]), xytext=(-4, spread[1]),
                    xycoords='data',
                    arrowprops=dict(connectionstyle='bar, fraction={},armA=-1, armB=-1'.format(fraction),
                    arrowstyle='-', color=colors['GH'], linewidth=2),
                    annotation_clip=False)
            ax.text(-5, (spread[0] + spread[1]) / 2, '{} ({})'.format(GH_CLAN_NAMES[key], key), color=colors['GH'], horizontalalignment='right', verticalalignment='center')


    return ax


def plot_cazyme_clusters(enrichment, cazymes):
    """
    enrichment : dict
        Keys are cluster labels. Values are the list of enriched cazymes.
    cazymes : list of str
        cazymes to include in the figure (in order)
    """
    clusters = list(enrichment.keys())
    clusters.sort()

    n_clusters = clusters[:-2] # remove the two aggregated pos/neg response at the end
    n_clusters.sort()
    n_clusters = [str(int(x)) for x in n_clusters]

    # Remove names from the cazymes in enrichment
    for k, v in enrichment.items():
        enrichment[k] = [c.split('.')[0] for c in v]

    x = np.zeros((len(enrichment.keys()), len(cazymes)))
    for i, cluster in enumerate(clusters):
        for j, cazyme in enumerate(cazymes):
            if cazyme in enrichment[cluster]:
                x[i, j] = 1

    x = x.transpose()

    # fig = plt.figure(figsize=(9, 9))
    fig = plt.figure(figsize=(11, 11))

    main_grid = fig.add_gridspec(2, 1, height_ratios=[.1, 3])

    ax = fig.add_subplot(main_grid[1, 0])
    im = ax.imshow(x, cmap='Purples', aspect=0.5, vmin=0, vmax=1)

    ax.set_xticks(np.arange(x.shape[1]))
    ax.set_yticks(np.arange(x.shape[0]))

    ax.set_xticklabels(clusters)
    ax.set_xlabel('pert. group')
    ax.xaxis.set_label_position('top')

    ax.set_yticklabels(cazymes)

    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

    ax.set_xticks(np.arange(x.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(x.shape[0]+1)-.5, minor=True)
    ax.grid(which='minor', color='k', linestyle='-', linewidth=1.2)
    ax.tick_params(which='minor', bottom=False, left=False)

    ax.tick_params(axis='x', which='major', labelsize=9)
    ax.tick_params(axis='y', which='major', labelsize=7)

    ax = color_cazyme_labels(ax)

    plt.show()


def plot_cazyme_species(database, cazymes):
    """Plot the database species -> cazyme.

    Parameters
    ----------
    database : dict
        keys are species, values are list of cazymes for that species
    cazymes : list of str
        cazymes to include in the figure (in order)
    """

    with open('species_kz.json') as f:
        species_cazymes = ujson.load(f)
    keys = list(species_cazymes.keys())

    all_species = sorted(list(species_cazymes.keys()))
    x = np.zeros((len(all_species), len(cazymes)))

    for i, sp in enumerate(all_species):
        for j, cazyme in enumerate(cazymes):
            if cazyme + '.hmm' in species_cazymes[sp]:
                x[i, j] = 1

    import scipy.cluster.hierarchy as sch
    x = x.transpose()

    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(1, 1, 1)
    im = ax.imshow(x, cmap='Purples', aspect=1)

    ax.set_xticks(np.arange(x.shape[1]))
    ax.set_yticks(np.arange(x.shape[0]))

    ax.set_xticklabels([sp.replace('_',' ') for sp in all_species], rotation=90)
    ax.set_yticklabels(cazymes)

    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

    ax.set_xticks(np.arange(x.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(x.shape[0]+1)-.5, minor=True)
    ax.grid(which='minor', color='k', linestyle='-', linewidth=2)
    ax.tick_params(which='minor', bottom=False, left=False)

    ax = color_cazyme_labels(ax)

    plt.show()


def main():
    all_cazymes = []

    enriched = load_enrichment('k1_cazyme_enrichment.txt')
    for cluster, cazymes in enriched.items():
        all_cazymes += cazymes
    all_cazymes = list(set(all_cazymes))


    ordered_cazymes = order_cazymes(all_cazymes)

    with open('species_kz.json') as f:
        database = ujson.load(f)

    plot_cazyme_species(database, ordered_cazymes)
    plot_cazyme_clusters(enriched, ordered_cazymes)


if __name__ == '__main__':
    main()
