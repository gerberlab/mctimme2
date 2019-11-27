"""Perform inference for the mixture model of negative binomial components
for shallow shotgun replicates data.

The input file is a replicates OTU table. Each column should contain the counts
data for one sample. Each row should contain one OTU.

This program is run separately from the MCTIMME2 model. The results from this
run are supplied as input to the MCTIMME2.

Usage
-----
Run from the command line, providing the path to the replicates data.

$ python noise_dp.py data/replicates.txt
"""

import math
import random
import itertools
import argparse
from collections import defaultdict
import warnings
import pandas
import numpy as np
import matplotlib.pyplot as plt
import scipy.special
import sklearn.cluster
import distribution
import sample


random.seed(1234)
sample.seed(5678)

class NegBinComponent:
    """A negative binomial component with its own overdispersion parameter.

    Any OTUs assigned to this component will bring their own mean.

    Note that the logarithm of the overdispersion is used for all computations.
    """
    def __init__(self, log_overdispersion, model):
        """
        Parameters
        ----------
        log_overdispersion : float
            Logarithm of negative binomial overdispersion parameter
        model : Model
            Main model object
        """
        self.model = model
        self.log_overdispersion = log_overdispersion

        self.members = set([])

        self.log_overdispersion_prop_var = 0.1
        self.accepts = 0
        self.rejects = 0


    def loglike_vector(self, ys, ms):
        """Calculate the log likelihood of multiple observations.

        The likelihood is calculated under the negative binomial distribution
        with this component's overdispersion, but with each observation having
        its own mean.

        Parameters
        ----------
        ys : 1darray
            Observed counts
        ms : 1darray
            The mean for each draw of counts in ys.

        Returns
        -------
        float
            The log likelihood.
        """
        r = 1 / math.exp(self.log_overdispersion)
        ll_vector = scipy.special.gammaln(ys + r) - scipy.special.gammaln(ys + 1) \
               - math.lgamma(r) + r * math.log(r) - r * np.log(r + ms) \
               + ys * np.log(ms) - ys * np.log(r + ms)
        return ll_vector.sum()


    def adapt_log_overdispersion_prop_var(self):
        """Adapt the proposal variance for the overdispersion updates.
        """
        try:
            accept_rate = self.accepts / (self.accepts + self.rejects)
        except:
            accept_rate = 0.23

        if accept_rate > 0.23:
            self.log_overdispersion_prop_var *= 1.2

        elif accept_rate < 0.23:
            self.log_overdispersion_prop_var *= 0.8

        self.accepts = 0
        self.rejects = 0


    def update_overdispersion(self):
        """Update overdispersion with an MH step.
        """
        log_overdispersion_old = self.log_overdispersion
        p_0 = self.model.log_overdispersion_prior(log_overdispersion_old)

        # Form a vector of all observations and all means for the OTUs in this
        # cluster
        # Start with empty vectors of the correct size
        counts = np.zeros(sum([otu.n for otu in self.members]))
        totals = np.zeros(sum([otu.n for otu in self.members]))
        # Populate the vectors with the observations
        j = 0
        for i, otu in enumerate(self.members):
            counts[j:j+otu.n] = otu.counts
            totals[j:j+otu.n] = otu.totals * otu.mean
            j += otu.n

        l_0 = self.loglike_vector(counts, totals)

        log_overdispersion_prop = random.gauss(log_overdispersion_old, \
                                   math.sqrt(self.log_overdispersion_prop_var))
        self.log_overdispersion = log_overdispersion_prop

        p_prop = self.model.log_overdispersion_prior(log_overdispersion_prop)
        l_prop = self.loglike_vector(counts, totals)

        log_r = p_prop + l_prop - p_0 - l_0

        if math.log(random.random()) >= log_r:
            # The jump was rejected, switch back
            self.log_overdispersion = log_overdispersion_old
            self.rejects += 1

        else:
            self.accepts += 1


class OTU:
    """A species or OTU which is represented in replicate dataset.
    """
    def __init__(self, species, counts, totals, model):
        """
        Parameters
        ----------
        species : str
            Name of the OTU (must be unique)
        counts : 1darray
            Observed counts for this species in each sample
        totals : 1darray
            Total counts in each sample (lined up with counts)
        model : Model
            Main model object
        """
        self.species = species
        self.counts = counts
        self.totals = totals
        self.mean = 0.01
        self.neg_bin_component = None
        self.model = model
        self.n = len(counts)

        self.mean_prop_var = 0.000001
        self.mean_accepts = 0
        self.mean_rejects = 0


    def update_mean(self):
        """Update the mean abdundance with an MH step.
        """
        mean_old = self.mean
        p_old = self.model.mean_prior(mean_old)
        l_old = self.neg_bin_component.loglike_vector(self.counts, \
                                                      self.totals * mean_old)

        mean_prop = random.gauss(mean_old, math.sqrt(self.mean_prop_var))
        p_prop = self.model.mean_prior(mean_prop)
        if p_prop == -math.inf:
            l_prop = -math.inf
        else:
            l_prop = self.neg_bin_component.loglike_vector(self.counts, \
                                                       self.totals * mean_prop)

        log_r = p_prop + l_prop - p_old - l_old

        if math.log(random.random()) < log_r:
            self.mean = mean_prop
            self.mean_accepts += 1

        else:
            self.mean_rejects += 1


    def adapt_mean_prop_var(self):
        """Adapt the proposal variance for the mean updates.
        """
        accept_rate = self.mean_accepts / (self.mean_accepts+self.mean_rejects)

        if accept_rate > 0.23:
            self.mean_prop_var *= 1.4

        elif accept_rate < 0.23:
            self.mean_prop_var *= 0.6

        self.mean_accepts = 0
        self.mean_rejects = 0


    def update_component(self):
        """Update the negative binomial component to which this OTU is assigned.
        """
        cluster_ps = []
        concentration = self.model.concentration

        for nbc in self.model.neg_bin_components:
            ni = len(nbc.members)
            if self.neg_bin_component == nbc:
                ni -= 1

            ll = nbc.loglike_vector(self.counts, self.totals * self.mean)
            if ni > 0:
                p = math.log(ni) + ll
            else:
                p = math.log(concentration) + ll

            cluster_ps.append(p)

        # Sample from the prior for the case of a new component
        new_cluster_log_overdispersion = self.model.sample_log_overdispersion_prior()
        new_cluster = NegBinComponent(new_cluster_log_overdispersion, self.model)
        ll = new_cluster.loglike_vector(self.counts, self.totals * self.mean)
        p = math.log(concentration) + ll
        cluster_ps.append(p)

        cluster_assign = sample.categorical_log(cluster_ps)

        self.neg_bin_component.members -= {self}

        if cluster_assign < len(self.model.neg_bin_components):
            self.neg_bin_component = self.model.neg_bin_components[cluster_assign]
            self.neg_bin_component.members |= {self}

        else:
            self.model.neg_bin_components.append(new_cluster)
            self.neg_bin_component = new_cluster
            self.neg_bin_component.members |= {self}

        active_clusters = []
        for nbc in self.model.neg_bin_components:
            n_members = len(nbc.members)
            if n_members > 0:
                active_clusters.append(nbc)

        self.model.neg_bin_components = active_clusters


class Model:
    """Main model holding negative binomial components and OTUs.
    """
    def __init__(self):
        # Hyperparameters
        self.log_overdispersion_prior_mean = -2
        self.log_overdispersion_prior_var = 25
        self.a1 = 1e-5
        self.a2 = 1e-5

        # Evaluating / sampling prior distributions
        self.mean_prior = lambda m : 1 if 0 < m < 1 else -math.inf
        self.log_overdispersion_prior = lambda x: distribution.normal_logpdf(x, self.log_overdispersion_prior_mean, math.sqrt(self.log_overdispersion_prior_var))
        self.sample_log_overdispersion_prior = lambda : random.gauss(self.log_overdispersion_prior_mean, math.sqrt(self.log_overdispersion_prior_var))

        # Initialize
        self.concentration = 0.1
        self.otus = []
        self.neg_bin_components = []


    def update_concentration(self):
        """Update Dirichlet process concentration.

        Uses an auxiliary variable technique [1]_

        References
        ----------
        .. [1] Escobar, Michael D., and Mike West. "Bayesian density estimation
            and inference using mixtures." Journal of the American Statistical
            Association 90.430 (1995): 577-588.
        """
        alpha = self.concentration
        a1 = self.a1
        a2 = self.a2
        eta = 1.0

        n_clusters = len(self.neg_bin_components)
        n_data = len(self.otus)

        if n_clusters <= 1 or n_data < 1:
            return

        for i in range(20):
            eta = random.betavariate(alpha+1, n_data)

            pi_eta = [0, 1]
            pi_eta[0] = (a1 + n_clusters - 1) / (n_data * (a2 - math.log(eta)))

            if sample.categorical_log(np.log(pi_eta)) == 0:
                alpha = random.gammavariate(a1 + n_clusters, 1 / (a2 - math.log(eta)))
            else:
                alpha = random.gammavariate(a1 + n_clusters - 1, 1 / (a2 - math.log(eta)))

        self.concentration = alpha


    def init_from_replicates_data(self, replicate_data_filename):
        """Load data from the replicates dataset.

        This creates an OTU object and a NegBinComponent object for each
        species. Each OTU starts in its own cluster.

        Parameters
        ----------
        replicate_data_filename : str
            Path to replicates data
        """
        replicates_data = pandas.read_csv(replicate_data_filename, sep='\t')

        plates = list(replicates_data)[1:]
        totals = np.array([replicates_data[plate].sum() for plate in plates])

        for i, row in replicates_data.iterrows():
            # Filter out very low abundance species
            if np.mean(row[plates] / totals) > 0.001:
                species = row['taxonomy']
                counts = np.array(row[plates])
                counts = counts.astype(np.int64)

                new_otu = OTU(species, counts, totals, self)

                # Initialize mean near the empirical mean from the data
                new_otu.mean = random.gauss(np.mean(counts/totals), 0.0001)

                # Initialize overdispersion using the DeSeq2-style model based
                # on the mean from the data
                init_overdispersion = random.gauss(0.000002 / new_otu.mean + 0.0017, 0.0001)
                new_nbc = NegBinComponent(math.log(init_overdispersion), self)

                new_otu.neg_bin_component = new_nbc
                new_nbc.members |= {new_otu}
                self.otus.append(new_otu)
                self.neg_bin_components.append(new_nbc)


def make_consensus_clustering(cluster_ids):
    """Generate a consensus clustering from the MCMC samples.

    Parameters
    ----------
    cluster_ids : dict
        Keys are species, values are trace of cluster ids for that species

    Returns
    -------
    list
        All species
    list
        The reconstructed cluster assignment for each species
    """
    all_mcids = []
    num_mcmc_iterations = len(next(iter(cluster_ids.values())))
    for _ in range(num_mcmc_iterations):
        all_mcids.append([])
    for i in range(num_mcmc_iterations):
        for ts in cluster_ids.keys():
            all_mcids[i].append(cluster_ids[ts][i])

    num_clusters = [len(set(mcids)) for mcids in all_mcids]
    num_clusters = int(round(np.median(num_clusters)))

    pairs = itertools.combinations(cluster_ids.keys(), 2)
    num_pairs = len(list(itertools.combinations(cluster_ids.keys(), 2)))

    all_otus = list(cluster_ids.keys())
    n = len(all_otus)
    affinity_matrix = np.zeros((n, n))

    for i, pair in enumerate(pairs):
        P_otu0_otu1 = len(np.where(np.array(cluster_ids[pair[0]]) == np.array(cluster_ids[pair[1]]))[0]) / len(cluster_ids[pair[0]])
        D_otu0_otu1 = 1 - P_otu0_otu1

        affinity_matrix[all_otus.index(pair[0]), all_otus.index(pair[1])] = D_otu0_otu1
        affinity_matrix[all_otus.index(pair[1]), all_otus.index(pair[0])] = D_otu0_otu1

        progress = int(round(i / num_pairs) * 20)
        print('\rmaking cocluster counts [' + progress * '=' + (20-progress) * ' ' + ']', end='')
    print('')

    consensus_clustering = sklearn.cluster.AgglomerativeClustering(n_clusters=num_clusters, affinity='precomputed', linkage='average')
    labels = consensus_clustering.fit_predict(affinity_matrix)

    return all_otus, labels


def get_consensus_values(all_otus, labels, overdispersion_samples):
    """Get the consensus overdispersion for each reconstructed cluster.

    Parameters
    ----------
    all_otus : list
        All species
    labels : list
        The reconstructed cluster assignment for each species
    overdispersion_samples : dict
        Keys are species, values are the list of MCMC samples for the
        overdispersion for that species.

    Returns
    -------
    dict
        The log overdispersions of the reconstructed clusters
    dict
        The number of members of the reconstructed clusters
    """
    overdispersions = {}
    sizes = {}
    for cluster in set(labels):
        n = 0
        this_cluster_overdispersions = []
        for otu, label in zip(all_otus, labels):
            if label == cluster:
                n += 1
                this_cluster_overdispersions += overdispersion_samples[otu]

        overdispersions[cluster] = np.mean(this_cluster_overdispersions)
        sizes[cluster] = n

    return overdispersions, sizes


def calc_R(traces, burnin):
    """Calc R convergence statistic.

    Parameters
    ----------
    traces : list of list
        Traces
    burnin : int
        The number of iterations to discard from the beginning of the chain

    Returns
    -------
    float
        R hat
    """
    # Split the chains in half
    chains = []
    for trace in traces:
        full_chain = trace[burnin:]
        chain_len = len(full_chain)
        chains.append(full_chain[:chain_len // 2])
        chains.append(full_chain[chain_len // 2:])

    # Form a matrix with all split chains
    m = len(chains)
    n = len(chains[0])
    psi = np.zeros((n,m))
    for j in range(m):
        psi[:,j] = chains[j]

    # Compute the necessary statistics from the matrix (see Gelman et al.)
    psi_bar_dot_j = np.mean(psi, axis=0)
    psi_bar_dot_dot = np.mean(psi_bar_dot_j)
    B = n / (m-1) * np.sum((psi_bar_dot_j - psi_bar_dot_dot)**2)

    sj2 = 1 / (n-1) * np.sum((psi - psi_bar_dot_j)**2, axis=0)
    W = np.mean(sj2)

    var = (n-1) / n * W + 1 / n * B
    R_hat = math.sqrt(var / W)

    return R_hat


def run(replicates_file, num_iters=10000, num_burnin=5000):
    """Load data and run MCMC inference

    Parameters
    ----------
    replicates_file : str
        Path to replicates OTU table
    num_iters : int
        Total number of MCMC iterations
    num_burnin : int
        Number of MCMC iterations burnin

    Returns
    -------
    dict
        species : list of mean samples
    dict
        species : list of log overdispersion samples
    dict
        species : list of cluster ID samples
    list
        concentration samples
    """
    model = Model()
    model.init_from_replicates_data(replicates_file)

    mean_samples = defaultdict(list)
    overdispersion_samples = defaultdict(list)
    cluster_samples = defaultdict(list)
    concentration_samples = []
    num_cluster_samples = []

    for i in range(num_iters):
        print(i)
        for otu in model.otus:
            otu.update_mean()
            otu.update_component()

        for nbc in model.neg_bin_components:
            for _ in range(100):
                nbc.update_overdispersion()

        if 1 < i < 0.75 * num_burnin and i % 50 == 0:
            for otu in model.otus:
                otu.adapt_mean_prop_var()

            for nbc in model.neg_bin_components:
                nbc.adapt_log_overdispersion_prop_var()

        model.update_concentration()

        for otu in model.otus:
            mean_samples[otu.species].append(otu.mean)
            overdispersion_samples[otu.species].append(otu.neg_bin_component.log_overdispersion)
            cluster_samples[otu.species].append(id(otu.neg_bin_component))

        concentration_samples.append(model.concentration)
        num_cluster_samples.append(len(model.neg_bin_components))

    return mean_samples, \
           overdispersion_samples, \
           cluster_samples, \
           concentration_samples, \
           num_cluster_samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('replicates_file', help='replicates otu table')
    args = parser.parse_args()

    replicates_file = args.replicates_file
    num_iters = 10000
    num_burnin = 5000

    # Run twice and check that it converged
    print('Run 1')
    mean_samples, \
        overdispersion_samples, \
        cluster_samples, \
        concentration_samples, \
        num_cluster_samples = run(replicates_file, num_iters=num_iters, num_burnin=num_burnin)

    print('Run 2')
    mean_samples2, \
        overdispersion_samples2,\
        cluster_samples2, \
        concentration_samples2, \
        num_cluster_samples2 = run(replicates_file, num_iters=num_iters, num_burnin=num_burnin)

    r = calc_R([concentration_samples, concentration_samples2], num_burnin)
    if r > 1.1:
        warnings.warn('Concentration convergence R={}'.format(r))

    r = calc_R([num_cluster_samples, num_cluster_samples2], num_burnin)
    if r > 1.1:
        warnings.warn('Number clusters convergence R={}'.format(r))

    for sp in mean_samples:
        r = calc_R([mean_samples[sp], mean_samples2[sp]], num_burnin)
        if r > 1.1:
            warnings.warn('{} mean convergence R={}'.format(sp, r))

        r = calc_R([overdispersion_samples[sp], overdispersion_samples2[sp]], num_burnin)
        if r > 1.1:
            warnings.warn('{} overdispersion convergence R={}'.format(sp, r))

    all_otus, labels = make_consensus_clustering(cluster_samples)
    log_overdispersions, sizes = get_consensus_values(all_otus, labels, overdispersion_samples)
    overdispersions = {k:math.exp(e) for k, e in log_overdispersions.items()}

    print(overdispersions, sizes)


if __name__ == '__main__':
    main()
