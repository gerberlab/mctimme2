"""Tools for analyzing MCMC samples.

This module contains classes and functions for the visualization and analysis
of the posterior samples contained in one or more hdf5 files.

The hdf5 files to analyze should be provided as command line arguments.
"""

import os
import math
import argparse
import itertools
import copy
import pickle

from tabulate import tabulate
import h5py
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import scipy.interpolate
from sklearn.cluster import AgglomerativeClustering
from scipy.stats import mannwhitneyu
from Bio import Phylo

import load_data


def parse_arguments():
    """Parse the command line arguments.

    Returns
    -------
    argparse.Namespace
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('mcmc_samples_files',
                        help='mcmc_samples_files',
                        nargs='+')
    parser.add_argument('--trajectories', action='store_true',
                        help='plot each time-series')
    parser.add_argument('--tables', action='store_true',
                        help='write results text files')
    parser.add_argument('--clusters', action='store_true',
                        help='plot perturbation clusters')
    parser.add_argument('--convergence', action='store_true',
                        help='assess convergence')
    args = parser.parse_args()
    return args


class ReconstructedPertCluster:
    """
    Attributes
    ----------
    members : list of ReconstructedMicrobeCluster
        Reconstructed microbe clusters assigned to this pc.
    """
    def __init__(self, mcmc_samples):
        """...

        Parameters
        ----------
        mcmc_samples : MCMCSamples
            The MCMC samples this cluster is formed from.
        """
        self.mcmc_samples = mcmc_samples
        self.members = []


    def calc_consensus_perturbation(self, compound, study_design='K002'):
        """Compute a summary perturbation from members.

        Each time-series assigned to this reconstructed perturbation cluster
        (via a reconstructed microbe cluster) has a potentially different
        posterior distribution of perturbations, but a single summary
        perturbation is desired for visual. This function first calculates the
        median perturbation trajectory for each time-series in this cluster.
        Then, it calculates the median of those median perturbation trajectories
        as the summary perturbation for the cluster.

        Parameters
        ----------
        compound : str
        """

        if study_design in ['K001', 'K002']:
            # From study design
            first_time = -10
            last_time = 55
            pert_start = 0
            pert_end = https://github.com/rcw5890/test.git28
            self.mcmc_samples.open_file()

            # Collect the median perturbation from each time-series assigned here
            my_pert_trajs = []
            for mc in self.members:
                for ts in mc.members:
                    dp_samples = self.mcmc_samples.file['time_series'][ts[0]+ts[1]]

                    all_pert_trajs = dp_samples['pert_traj'][compound]

                    traj_times = dp_samples.attrs['traj_times']
                    med_pert_traj = np.median(all_pert_trajs, axis=0)

                    my_pert_trajs.append((traj_times, med_pert_traj))

            # Draw all the perturbations on the same time grid
            my_pert_trajs_aligned = []
            pert_period = []
            global_grid = np.linspace(first_time, last_time, 1000)
            for time, pert in my_pert_trajs:
                fi = scipy.interpolate.interp1d(time, pert, fill_value=0, bounds_error=False)
                pert_on_grid = fi(global_grid)
                my_pert_trajs_aligned.append(pert_on_grid)
            for time in global_grid:
                pert_period.append(1 if pert_start < time < pert_end else 0)

            median_pert_traj = np.median(my_pert_trajs_aligned, axis=0)

            self.pert_times = global_grid
            self.median_pert = median_pert_traj
            self.pert_period = np.array(pert_period)

            self.mcmc_samples.close_file()

        if study_design in ['K003']:
            # from study design
            first_time = -10
            last_time = 55
            pert_start = 0
            pert_end = 28

            pdx2fos = ['301-010', '301-016', '301-020', '301-029', '301-012',
                       '301-007', '301-019', '301-027', '301-026', '301-030']
            fos2pdx = ['301-001', '301-024', '301-015', '301-023', '301-005',
                       '301-003', '301-034', '301-035', '301-031', '301-032']

            self.mcmc_samples.open_file()

            my_pert_trajs = []
            for mc in self.members:
                for ts in mc.members:
                    dp_samples = self.mcmc_samples.file['time_series'][ts[0]+ts[1]]
                    all_pert_trajs = dp_samples['pert_traj'][compound]
                    traj_times = dp_samples.attrs['traj_times']
                    med_pert_traj = np.median(all_pert_trajs, axis=0)

                    # get this compound
                    subj = ts[0]
                    if (compound == 'PDX' and subj in pdx2fos) or (compound=='FOS' and subj in fos2pdx):
                        pert_start = 0
                        pert_end = 14
                    elif (compound == 'FOS' and subj in pdx2fos) or (compound=='PDX' and subj in fos2pdx):
                        pert_start = 42
                        pert_end = 56

                    selected_pert_traj = []
                    selected_traj_times = []
                    for t, p in zip(traj_times, med_pert_traj):
                        if pert_start - 2 < t < pert_end + 10:
                            selected_pert_traj.append(p)
                            selected_traj_times.append(t - pert_start)

                    my_pert_trajs.append((selected_traj_times, selected_pert_traj))

            # Draw all the perturbations on the same time grid
            my_pert_trajs_aligned = []
            pert_period = []
            global_grid = np.linspace(-2, 24, 1000)
            for time, pert in my_pert_trajs:
                fi = scipy.interpolate.interp1d(time, pert, fill_value=0, bounds_error=False)
                pert_on_grid = fi(global_grid)
                my_pert_trajs_aligned.append(pert_on_grid)
            for time in global_grid:
                pert_period.append(1 if pert_start < time < pert_end else 0)

            median_pert_traj = np.median(my_pert_trajs_aligned, axis=0)

            self.pert_times = global_grid
            self.median_pert = median_pert_traj
            self.pert_period = np.array(pert_period)

            self.mcmc_samples.close_file()


    def get_number_subjects(self):
        """Get the number of subjects who have some representation in this cluster.
        """
        subjects = []
        for mc in self.members:
            for ts in mc.members:
                subjects.append(ts[0])

        self.num_subjects = len(set(subjects))


    def __getstate__(self):
        """For pickling, must remove the hdf5.
        """
        state = self.__dict__.copy()
        state['mcmc_samples'] = 'Must reset MCMC samples'
        return state


    def __setstate__(self, newstate):
        """For pickling, must remove the hdf5.
        """
        newstate['mcmc_samples'] = 'Must reset MCMC samples'
        self.__dict__.update(newstate)


class ReconstructedMicrobeCluster:
    """
    Attributes
    ----------
    members : list of tuple
        Each element in the list is a tuple representing a time-series. The
        first element in the tuple is the subject name and the second element
        is the otu name (both str).
    """
    def __init__(self, mcmc_samples):
        """...

        Parameters
        ----------
        mcmc_samples : MCMCSamples
            The MCMC samples this cluster is formed from.
        """
        self.mcmc_samples = mcmc_samples
        self.pert_cluster = None
        self.members = []


    def find_central_member(self, cluster, distance_matrix, distance_matrix_key):
        pass
        ds = []
        for dp in self.members:
            ds.append(0)
            for dp2 in self.members:
                idx0 = distance_matrix_key.index(dp[0] + dp[1])
                idx1 = distance_matrix_key.index(dp2[0] + dp2[1])
                ds[-1] += distance_matrix[idx0, idx1]
        cluster.central_member = cluster.members[ds.index(min(ds))]


    def __getstate__(self):
        """For pickling, must remove the hdf5.
        """
        state = self.__dict__.copy()
        state['mcmc_samples'] = 'Must reset MCMC samples'
        return state


    def __setstate__(self, newstate):
        """For pickling, must remove the hdf5.
        """
        newstate['mcmc_samples'] = 'Must reset MCMC samples'
        self.__dict__.update(newstate)


class MCMCSamples:
    """Holds functions used to analyze results saved in the MCMC output file.

    This class holds the functions used for post-hoc analysis of the MCMC
    samples. It is not used during the MCMC sampling.

    Attributes
    ----------
    hdf5_filename : str
        Path to the hdf5 file containing the MCMC samples
    write_directory : str
        Default directory into which to save output figues
    burn_in : int
    """

    def __init__(self, hdf5_filename, write_directory):
        self.hdf5_filename = hdf5_filename
        self.write_directory = write_directory

        self.open_file()
        self.scale_factor = self.file['model'].attrs['scale_factor']
        self.burnin = self.file['model'].attrs['num_burnin']
        self.close_file()


    def open_file(self):
        """Open the hdf5 file for reading.
        """
        self.file = h5py.File(self.hdf5_filename, 'r')


    def close_file(self):
        """Close the hdf5 file.
        """
        self.file.close()


    def move_burnin(self, new_burnin):
        """Change the burnin period.

        By default, the burnin is saved in the hdf5 file attributes.

        Parameters
        ----------
        new_burnin : int
            The new burnin iteration number
        """
        self.burnin = new_burnin


    def reconstruct_clusters(self,
                             compound,
                             bf_threshold=0,
                             load_clusters_from_pickle=False,
                             save=False,
                             study_design='K002'):
        """Use the coclustering counts to form a consensus double clustering.

        When forming microbe clusters, the distance between each time-series is
        given by one minus their microbe cluster coclustering probability.

        The reconstructed microbe clusters are aggregated into consensus
        perturbation clusters using the perturbation cluster coclustering
        proportions of each time-series. The distance between each reconstructed
        microbe cluster is defined as one minus the mean of the coclustering
        probabilities between all pairs of time-series in each reconstructed
        microbe cluster.

        This function does not return anything, but it populates two attributes
        self.microbe_clusters and self.pert_clusters, each of which is a list
        of Reconstructed[]Cluster objects.

        Parameters
        ----------
        compound : str
            Compound for which to reconstruct. Currently only one compound at a
            time is implemented.
        bf_threshold : float, optional (0)
            Time-series with bayes factor below this threshold will not be
            included in the clustering
        """
        if load_clusters_from_pickle:
            fname = os.path.join(self.write_directory, 'pcs.pkl')
            with open(fname, 'rb') as f:
                self.pert_clusters = pickle.load(f)

            fname = os.path.join(self.write_directory, 'mcs.pkl')
            with open(fname, 'rb') as f:
                self.microbe_clusters = pickle.load(f)

            return

        self.open_file()

        # Get the time-series which pass the Bayes factor threshold and will be
        # placed into reconstruct_clusters
        time_series = []
        for ts in self.file['time_series'].keys():
            print(ts)
            try:
                bf = self.calc_bayes_factor(ts)
            except:
                bf = {compound: -1}
            if bf[compound] >= bf_threshold:
                time_series.append(ts)

        # mc_ids contains for each key (subjectotu) the list of
        # microbe cluster ids for the MCMC chain
        mc_ids = {}
        for ts in self.file['time_series'].keys():
            if ts in time_series:
                mc_ids[ts] = np.array(self.file['time_series'][ts]['dyn_cluster'][compound][self.burnin:])

        # Get the number of clusters to agglomerate to
        num_mcs = self.get_number_clusters(mc_ids)

        # Form a distance matrix
        distance_matrix, all_dps = self.calc_distance_matrix(mc_ids)

        # Use agglomerative cluster to get cluster assignments for each time series
        scikit_cluster = AgglomerativeClustering(n_clusters=num_mcs, affinity='precomputed', linkage='average')
        scikit_assignments = scikit_cluster.fit_predict(distance_matrix)

        # Instantiate the reconstructed clusters
        reconstructed_microbe_clusters = []
        for _ in set(scikit_assignments):
            reconstructed_microbe_clusters.append(ReconstructedMicrobeCluster(self))

        # Add the time-series as members to the reconstructed clusters
        for i, assignment in enumerate(scikit_assignments):
            dp = all_dps[i]
            subj_name = self.file['time_series'][dp].attrs['subject_name'].decode()
            otu_name = self.file['time_series'][dp].attrs['otu'].decode()
            reconstructed_microbe_clusters[assignment].members.append((subj_name, otu_name))

        # pc_ids contains for each key (subjectotu) the list of perturbation
        # cluster ids for the MCMC chain
        pc_ids = {}
        for ts in self.file['time_series'].keys():
            if ts in time_series:
                pc_ids[ts] = np.array(self.file['time_series'][ts]['pert_cluster'][compound][self.burnin:])

        # Get the number of clusters to agglomerate to
        num_pcs = self.get_number_clusters(pc_ids)

        # Form a distance matrix between reconstructed microbe clusters
        distance_matrix, all_mcs = self.calc_distance_matrix(pc_ids, reconstructed_microbe_clusters)

        # Use agglomerative clustering to get cluster assignments for each mc
        scikit_cluster = AgglomerativeClustering(n_clusters=num_pcs, affinity='precomputed', linkage='average')
        scikit_assignments = scikit_cluster.fit_predict(distance_matrix)

        # Instantiate the reconstructed perturbtion clusters
        reconstructed_pert_clusters = []
        for _ in set(scikit_assignments):
            reconstructed_pert_clusters.append(ReconstructedPertCluster(self))

        # Add the microbe clusters as member to the reconstructed pert clusters
        for i, assignment in enumerate(scikit_assignments):
            reconstructed_pert_clusters[assignment].members.append(all_mcs[i])
            all_mcs[i].pert_cluster = reconstructed_pert_clusters[assignment]

        # Save the reconstructed cluster dicts
        self.microbe_clusters = reconstructed_microbe_clusters
        self.pert_clusters = reconstructed_pert_clusters

        # Get the properties of each reconstructed cluster
        # Get on-time and number of subjects present for each pert cluster
        for pc in self.pert_clusters:
            pc.calc_consensus_perturbation(compound, study_design=study_design)
            pc.get_number_subjects()

        # Save the reconstructed clustering for fast loading
        if save:
            fname = os.path.join(self.write_directory, 'pcs.pkl')
            with open(fname, 'wb') as f:
                pickle.dump(self.pert_clusters, f)

            fname = os.path.join(self.write_directory, 'mcs.pkl')
            with open(fname, 'wb') as f:
                pickle.dump(self.microbe_clusters, f)

        self.close_file()


    def get_number_clusters(self, cluster_ids, measure='mode', verbose=False):
        """Get the number of clusters to use in the consensus clustering.

        Parameters
        ----------
        cluster_ids : dict
            Keys are the items being clustered (otus or microbe clusters). For
            each key, the value is a list or array of the id of the cluster to
            which that item was assigned at each MCMC iteration.
        measure : {'mode', 'median', 'mean'}, optional
            Which central tendency of the number of clusters at each MCMC step
            to return.

        Returns
        -------
        int
            Number of clusters
        """
        num_mcmc_iterations = len(next(iter(cluster_ids.values())))

        all_cluster_ids = [[] for _ in range(num_mcmc_iterations)]
        for i in range(num_mcmc_iterations):
            for ts in cluster_ids.keys():
                all_cluster_ids[i].append(cluster_ids[ts][i])

        num_clusters = [len(set(ids)) for ids in all_cluster_ids]

        if measure not in ['mode', 'median', 'mean']:
            warnings.warn('Unrecognized number of clusters %s. Using mode.' % measure)
            measure = 'mode'

        if measure == 'mode':
            num_clusters = scipy.stats.mode(num_clusters)[0][0]
        elif measure == 'median':
            num_clusters = np.median(num_clusters)
        elif measure == 'mean':
            num_clusters = np.mean(num_clusters)

        return num_clusters


    def calc_distance_matrix(self, cluster_ids, microbe_level_clustering=None):
        """Calculate a distance matrix based on coclustering probabilities.

        Parameters
        ----------
        cluster_ids : dict
            Keys are the items for which coclustering probabilities can be
            calculated. For each key, the value is a list or array of the id of
            the cluster to which that item was assigned at each MCMC iteration.
        microbe_level_clustering : list, optional
            If None, the keys of cluster_ids are themselves clustered.
            Alternatively, a list of ReconstructedMicrobeCluster objects can be
            provided as this parameter. If supplied, the
            ReconstructedMicrobeCluster objects will be clustered using the
            coclustering probabilities of their members.

        Returns
        -------
        ndarray
            distance matrix
        list
            List of all otus, indices lining up with distance matrix
        """
        if microbe_level_clustering is None:
            all_dps = list(cluster_ids.keys())
            n_dps = len(cluster_ids.keys())
            distance_matrix = np.zeros((n_dps, n_dps))
            pairs = itertools.combinations(cluster_ids.keys(), 2)
            total_pairs = len(list(copy.copy(pairs)))
            index_map = {}
            for dp in all_dps:
                index_map[dp] = all_dps.index(dp)
        else:
            all_mcs = microbe_level_clustering
            all_dps = all_mcs
            n_mcs = len(microbe_level_clustering)
            distance_matrix = np.zeros((n_mcs, n_mcs))
            pairs = itertools.combinations(all_mcs, 2)
            total_pairs = len(list(copy.copy(pairs)))

            index_map = {}
            for dp in all_dps:
                index_map[dp] = all_dps.index(dp)

        for i, pair in enumerate(pairs):
            if microbe_level_clustering is None:
                cluster_ids_dp0 = cluster_ids[pair[0]]
                cluster_ids_dp1 = cluster_ids[pair[1]]

                # Calculate the probability of dp0 and dp1 coclustering
                P_dp0_dp1 = len(np.where(cluster_ids_dp0 == cluster_ids_dp1)[0]) / len(cluster_ids_dp0)
            else:
                pairwise_ps = []
                mc0_members = pair[0].members
                mc1_members = pair[1].members
                for dp0 in mc0_members:
                    for dp1 in mc1_members:
                        cluster_ids_dp0 = cluster_ids[dp0[0] + dp0[1]]
                        cluster_ids_dp1 = cluster_ids[dp1[0] + dp1[1]]
                        P_dp0_dp1 = len(np.where(cluster_ids_dp0 == cluster_ids_dp1)[0]) / len(cluster_ids_dp0)
                        pairwise_ps.append(P_dp0_dp1)

                P_dp0_dp1 = np.sum(pairwise_ps) / len(pairwise_ps)

            # A distance between dp0 and dp1 can be formed by
            D_dp0_dp1 = 1 - P_dp0_dp1

            idx0 = index_map[pair[0]]
            idx1 = index_map[pair[1]]
            distance_matrix[idx0, idx1] = D_dp0_dp1
            distance_matrix[idx1, idx0] = D_dp0_dp1

        return distance_matrix, all_dps


    def plot_trajectory(self,
                        ax,
                        otu,
                        plot_data=True,
                        shade_compound=True,
                        trajectory=True,
                        aux_trajectory=True):
        """Plot the data and inferred trajectory.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axis on which to plot trajectory
        otu : str
            Name of time-series
        plot_data : bool, optional
            Whether or not to plot the data points
        shade_compound : bool, optional
            Whether or not to shade the period of compound consumption
        trajectory : bool, optional
            Whether or not to plot the latent trajectory
        aux_trajectory : bool, optional
            Whether or not to plot the auxiliary trajectory

        Returns
        -------
        matplotlib.axes.Axes
            ax with the trajectory plotted
        """
        otu_samples = self.file['time_series'][otu]
        traj_times = otu_samples.attrs['traj_times']
        data_times = otu_samples.attrs['data_times']
        rel_abundances = otu_samples.attrs['rel_abundances']

        if aux_trajectory:
            all_tss = otu_samples['aux_trajectory'][self.burnin:]
            ax = self.plot_time_series_posterior(ax,
                                                 traj_times,
                                                 all_tss,
                                                 'green',
                                                 lower_pctle=2.5,
                                                 upper_pctle=97.5,
                                                 label='aux_traj')

        if trajectory:
            all_tss = otu_samples['trajectory'][self.burnin:]
            ax = self.plot_time_series_posterior(ax,
                                                 traj_times,
                                                 all_tss,
                                                 'red',
                                                 lower_pctle=2.5,
                                                 upper_pctle=97.5,
                                                 label='traj')

        if plot_data:
            ax.plot(data_times, rel_abundances * self.scale_factor,
                    'x-', color='k', label='data')

        ax.set_ylabel('norm. abundance')
        ax.set_xlabel('time (days)')
        if shade_compound:
            ax = self.shade_compound_consumption(ax, otu_samples.attrs['subject_name'])
        ax.legend(fontsize=7)
        return ax


    def plot_pharmacokinetics(self, ax, subject_name, colors=['grey', 'blue']):
        """Plot the inferred pharmacokinetic profile.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axes on which to plot pharmacokinetics
        subject_name : str
            Subject for which to plot
        colors : list of str, optional
            Colors for each compound. Should be aligned with the attribute
            'compounds' of the model group.

        Returns
        -------
        matplotlib.axes.Axes
            ax with the pharmacokinetics plotted
        """
        all_compounds = self.file['model'].attrs['compounds']
        all_compounds = [c.decode() for c in all_compounds]

        subject_attrs = self.file['subjects'][subject_name].attrs
        traj_times = subject_attrs['traj_times']

        for compound, color in zip(all_compounds, itertools.cycle(colors)):
            if compound in subject_attrs.keys():
                all_tss = self.file['subjects'][subject_name]['pharma_traj'][compound][self.burnin:]
                ax = self.plot_time_series_posterior(ax,
                                                     traj_times,
                                                     all_tss,
                                                     color,
                                                     label='{} level'.format(compound))

        ax.set_ylabel('compound concentration')
        ax.set_xlabel('time (days)')
        ax = self.shade_compound_consumption(ax, subject_name)
        ax.legend(fontsize=7)

        return ax


    def plot_time_series_posterior(self, ax, times, all_tss, color,
                                   lower_pctle=2.5, upper_pctle=97.5, label=None):
        """Plot the posterior interval of a time series.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axis on which to plot.
        times : array_like
            Time points
        all_tss : array_like
            Each row contains one sample of the trajectory
        color : str
            Color in which to draw lines and shade.
        lower_pctle : float, optional
            Lower bound
        upper_pctle : float, optional
            Upper bound
        label : str, optional
            Label to put in the legend

        Returns
        -------
        matplotlib.axes.Axes
            ax with the time series plotted
        """
        tss_median = np.median(all_tss, axis=0)
        tss_lower = np.percentile(all_tss, lower_pctle, axis=0)
        tss_upper = np.percentile(all_tss, upper_pctle, axis=0)

        ax.plot(times, tss_lower, color)
        ax.plot(times, tss_upper, color)
        ax.plot(times, tss_median, color, label=label)
        ax.fill_between(times, tss_lower, tss_upper, alpha=0.3, color=color)
        return ax


    def shade_compound_consumption(self, ax, subject, colors=['orange', 'teal']):
        """Color the periods of compound consumption.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axis on which to color consumption
        subject : str
            Name of subject
        colors : list of str, optional
            Colors for each compound. Should be aligned with the attribute
            'compounds' of the model group.

        Returns
        -------
        matplotlib.axes.Axes
            ax with consumption shaded
        """
        all_compounds = self.file['model'].attrs['compounds']
        all_compounds = [c.decode() for c in all_compounds]

        subject_attrs = self.file['subjects'][subject].attrs

        for compound, color in zip(all_compounds, itertools.cycle(colors)):
            if compound in subject_attrs.keys():
                compound_times = subject_attrs[compound]

                ax.axvspan(min(compound_times), max(compound_times),
                           alpha=0.3,
                           color=color,
                           zorder=-10,
                           label=compound)

        return ax


    def plot_carrying_capacity(self, ax, otu, plot_data=True, shade_compound=True):
        """Plot the data and the inferred carrying capacity.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axes on which to plot carrying capacity
        otu : str
            Time-series to plot
        plot_data : bool, optional
            Whether or not to plot the data points
        shade_compound : bool, optional
            Whether or not to shade the period of compound consumption

        Returns
        -------
        matplotlib.axes.Axes
            ax with the carrying capacity plotted
        """
        otu_samples = self.file['time_series'][otu]
        traj_times = otu_samples.attrs['traj_times']
        data_times = otu_samples.attrs['data_times']
        rel_abundances = otu_samples.attrs['rel_abundances']

        growths = np.array(otu_samples['growth_rate'][self.burnin:])
        self_interacts = np.array(otu_samples['self_interact'][self.burnin:])

        all_compounds = list(otu_samples['pert_traj'].keys())
        total_pert_trajs = np.array(otu_samples['pert_traj'][all_compounds[0]][self.burnin:])
        for compound in all_compounds[1:]:
            total_pert_trajs += np.array(otu_samples['pert_traj'][compound][self.burnin:])

        persist_traj = np.array(otu_samples['persist_pert_traj'][self.burnin:])

        # Compute the carrying capacity based on dynamical equations
        carrying_capacity = (1 + total_pert_trajs + persist_traj) * growths[:, np.newaxis] / -self_interacts[:, np.newaxis]

        ax = self.plot_time_series_posterior(ax, traj_times, carrying_capacity, 'dodgerblue', label='carrying capacity')

        if plot_data:
            ax.plot(data_times, rel_abundances * self.scale_factor,
                    'x-', color='k', label='data')

        ax.set_ylabel('norm. abundance')
        ax.set_xlabel('time (days)')
        if shade_compound:
            ax = self.shade_compound_consumption(ax, otu_samples.attrs['subject_name'])
        ax.legend(fontsize=7)

        return ax


    def calc_bayes_factor(self, otu):
        """Calculate the Bayes factor for the time-series being perturbed.

        Parameters
        ----------
        otu : str
            Time-series key (subjotu)

        Returns
        -------
        dict
            Keys are compounds. For each compound, the Bayes factor for whether
            this time-series had a perturbation for that compound.
        """

        alpha = self.file['model'].attrs['perturb_select_prior_alpha']
        beta = self.file['model'].attrs['perturb_select_prior_beta']

        otu_samples = self.file['time_series'][otu]
        traj_times = otu_samples.attrs['traj_times']

        bayes_factors = {}

        for compound in otu_samples['pert_traj'].keys():
            pert_trajs = otu_samples['pert_traj'][compound].value
            pert_trajs = pert_trajs[self.burnin:]
            num_samples = len(pert_trajs)

            pert_trajs_bool = np.array(pert_trajs, dtype=bool)
            proportion_traj = np.array(pert_trajs_bool, dtype=float).sum(0) / num_samples

            bayes_factor_traj = (proportion_traj * beta) / ((1-proportion_traj) * alpha)

            bayes_factor = max(bayes_factor_traj)
            bayes_factor = min(1000, bayes_factor)

            bayes_factors[compound] = bayes_factor

        return bayes_factors


    def plot_bayes_factor(self, ax, otu):
        """Plot the Bayes factors for a perturbation being present.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axes on which to plot carrying capacity
        otu : str

        Returns
        -------
        matplotlib.axes.Axes
            ax with the carrying capacity plotted
        """
        bayes_factors = self.calc_bayes_factor(otu)

        compounds = list(bayes_factors.keys())

        bfs = [bayes_factors[compound] for compound in compounds]

        ax.bar(range(len(bfs)), bfs, color='orange', edgecolor='k', linewidth=1.5, width=.2)
        ax.set_ylabel('Bayes factor')
        ax.set_xlabel('compound')
        ax.set_xlim(-0.5, len(bfs)-0.5)
        ax.set_xticks(range(len(bfs)))
        ax.set_xticklabels(compounds)
        return ax


    def plot_otu_results(self):
        """Plot data and relevant posterior distributions for each time-series.

        The figures are saved as pdfs in the write_directory, with each
        filename containing the subject name and OTU name.
        """
        self.open_file()

        for ts in self.file['time_series'].keys():
            try:
                subject_name = self.file['time_series'][ts].attrs['subject_name'].decode()
                otu_name = self.file['time_series'][ts].attrs['otu'].decode()

                fig = plt.figure(figsize=(10, 4.5))

                ax = fig.add_subplot(2, 2, 1)
                ax = self.plot_trajectory(ax, ts)

                taxon = load_data.Taxon(otu_name)
                ax.set_title('{} {}'.format(subject_name, getattr(taxon, 'taxon', otu_name)))

                ax = fig.add_subplot(2, 2, 2)
                ax = self.plot_pharmacokinetics(ax, subject_name)

                ax = fig.add_subplot(2, 2, 3)
                ax = self.plot_carrying_capacity(ax, ts)

                ax = fig.add_subplot(2, 2, 4)
                ax = self.plot_bayes_factor(ax, ts)

                fig.set_tight_layout(True)

                # plt.show()
                plt.savefig(os.path.join(self.write_directory, '{}_{}.pdf'.format(subject_name, otu_name)))

                plt.close()

            except KeyError:
                pass

        self.close_file()


    def tabulate2(self):
        self.open_file()
        compounds = list(self.file['model']['num_microbe_clusters'].keys())
        compound = compounds[0]
        self.close_file()

        for compound in compounds:
            self.reconstruct_clusters(compound, bf_threshold=100, study_design='K002')

            self.open_file()

            fname = os.path.join(self.write_directory, '{}_posterior_data.txt'.format(compound))

            with open(fname, 'w') as f:
                f.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format('subject', 'otu', 'bf', 'ontime', 'duration', 'effect_median', 'effect_mean', 'mean', 'pc', 'cc0', 'cc1'))
                for ts in self.file['time_series'].keys():
                    print(ts)
                    try:
                        subject_name = self.file['time_series'][ts].attrs['subject_name'].decode()
                        otu_name = self.file['time_series'][ts].attrs['otu'].decode()
                        bf = self.calc_bayes_factor(ts)[compound]
                        ontimes = self.file['time_series'][ts]['on_time'][compound][self.burnin:]
                        effects = self.file['time_series'][ts]['effect'][compound][self.burnin:]
                        durations = self.file['time_series'][ts]['duration'][compound][self.burnin:]
                        true_durations = []
                        true_ontimes = []
                        for d, t, e in zip(durations, ontimes, effects):
                            if e != 0:
                                true_ontimes.append(t)
                                true_durations.append(d)
                        ontime = np.median(true_ontimes)
                        duration = np.median(true_durations)

                        effect = np.median(self.file['time_series'][ts]['effect'][compound][self.burnin:])
                        effect_mean = np.mean(self.file['time_series'][ts]['effect'][compound][self.burnin:])

                        abu = np.mean(self.file['time_series'][ts].attrs['rel_abundances'])

                        pc = -1
                        for i, rpc in enumerate(self.pert_clusters):
                            for mc in rpc.members:
                                if (subject_name, otu_name) in mc.members:
                                    pc = i
                        otu_samples = self.file['time_series'][ts]
                        traj_times = otu_samples.attrs['traj_times']
                        data_times = otu_samples.attrs['data_times']
                        rel_abundances = otu_samples.attrs['rel_abundances']
                        growths = np.array(otu_samples['growth_rate'][self.burnin:])
                        self_interacts = np.array(otu_samples['self_interact'][self.burnin:])
                        all_compounds = list(otu_samples['pert_traj'].keys())
                        total_pert_trajs = np.array(otu_samples['pert_traj'][all_compounds[0]][self.burnin:])
                        for compound2 in all_compounds[1:]:
                            total_pert_trajs += np.array(otu_samples['pert_traj'][compound2][self.burnin:])
                        persist_traj = np.array(otu_samples['persist_pert_traj'][self.burnin:])
                        carrying_capacity = (1 + total_pert_trajs + persist_traj) * growths[:, np.newaxis] / -self_interacts[:, np.newaxis]
                        carrying_capacity = np.median(carrying_capacity, axis=0)
                        # only for K001/K002 (parameters come from study design)
                        ccs_0 = []
                        ccs_1 = []
                        for t, cc in zip(traj_times, carrying_capacity):
                            if 0 < t < 14:
                                ccs_0.append(cc)
                            elif 14 < t< 28:
                                ccs_1.append(cc)
                        cc_0 = np.median(ccs_0)
                        cc_1 = np.median(ccs_1)

                        f.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(subject_name, otu_name, bf, ontime, duration, effect, effect_mean, abu, pc, cc_0, cc_1))

                    except KeyError:
                        pass

            fname = os.path.join(self.write_directory, '{}_cluster_data.txt'.format(compound))

            with open(fname, 'w') as f:
                for i, rpc in enumerate(self.pert_clusters):
                    traj = rpc.median_pert
                    f.write('{}\t{}\n'.format(i, [float(x) for x in traj]))

            self.close_file()



def calc_R(traces, variable_path, index=None, plot=False):
    """Compute the Rhat MCMC convergence statistic for a scalar variable.

    A. Gelman, H. S. Stern, J. B. Carlin, D. B. Dunson, A. Vehtari, and
    D. B. Rubin, Bayesian Data Analysis. Chapman and Hall/CRC, 2013.

    Parameters
    ----------
    traces : list of MCMCSamples
        Multiple separate runs of the MCMC inference, with potentially different
        initial conditions.
    variable_path : str
        hdf5 path to the variable for which to assess convergence
    index : int, optional
        When variable_path points to a vector quantity, this parameter is used
        to specify which element in that vector to calculate Rhat for.
    plot : bool, optional (False)
        If True, plot the histogram and trace for each chain

    Returns
    -------
    float
        The Rhat measure as defined in the textbook
    """

    for mcmc_samples in traces:
        mcmc_samples.open_file()

    if plot:
        fig = plt.figure(figsize=(8, 2 * len(traces)))

    # Split the chains in half, and extract the specified scalar element if the
    # parameter is a vector
    chains = []
    for i, mcmc_samples in enumerate(traces):
        full_chain = np.array(mcmc_samples.file[variable_path])

        # If the parameter is a vector, get the specified index
        if index is not None:
            full_chain = np.array([traj[index] for traj in full_chain])

        # Plot the chain if requested
        if plot:
            ax = fig.add_subplot(len(traces), 2, 2*i+1)
            ax.plot(full_chain)
            ax.axvline(mcmc_samples.burnin, color='red')
            ax.set_title('Samples {}'.format(i))
            ax.set_ylabel(variable_path.split('/')[-1])

        # Remove the burnin period
        full_chain = full_chain[mcmc_samples.burnin:]

        # Plot the distribution if requested (only samples after burnin)
        if plot:
            ax = fig.add_subplot(len(traces), 2, 2*i+2)
            ax.hist(full_chain, alpha=0.5, density=True)

        chain_len = len(full_chain)
        chains.append(full_chain[:chain_len // 2])
        chains.append(full_chain[chain_len // 2:])

    if plot:
        # Adjust the plot limits the chains can be compared on the same scale
        x_lowest = min([ax.get_xlim()[0] for ax in fig.axes if ax.colNum == 1])
        x_highest = max([ax.get_xlim()[1] for ax in fig.axes if ax.colNum == 1])
        for ax in fig.axes:
            if ax.colNum == 1:
                ax.set_xlim(x_lowest, x_highest)

        y_lowest = min([ax.get_ylim()[0] for ax in fig.axes if ax.colNum == 0])
        y_highest = max([ax.get_ylim()[1] for ax in fig.axes if ax.colNum == 0])
        for ax in fig.axes:
            if ax.colNum == 0:
                ax.set_ylim(y_lowest, y_highest)

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

    for mcmc_samples in traces:
        mcmc_samples.close_file()

    if plot:
        fig.suptitle(r'$\hat{R}$ = ' + str(round(R_hat, 3)))
        fig.set_tight_layout(True)
        plt.show()

    return R_hat


def get_datasets(group, datasets):
    """Get all datasets in this hdf5 group or file.

    Parameters
    ----------
    group : h5py._hl.group.Group
        The group to scan for datasets
    datasets : list
        The current list of parameters which have been found

    Returns
    -------
    list
        The datasets list with the datasets from this group appended
    """
    for k in group:
        if type(group[k]) == h5py._hl.dataset.Dataset:
            datasets.append(group[k].name)
        else:
            datasets = get_datasets(group[k], datasets)
    return datasets


def run_analysis():
    """Parse command line arguments and start the analysis.
    """
    args = parse_arguments()

    print(args)

    output_dir = ''

    if args.trajectories:
        for filename in args.mcmc_samples_files:
            trace = MCMCSamples(filename, output_dir)
            trace.plot_otu_results()

    if args.clusters:
        for filename in args.mcmc_samples_files:
            trace = MCMCSamples(filename, output_dir)
            trace.plot_clusters()

    if args.tables:
        for filename in args.mcmc_samples_files:
            trace = MCMCSamples(filename, output_dir)
            trace.tabulate2()
if __name__ == '__main__':
    run_analysis()
