"""Run MCMC inference for the model.
"""

import os
import math
import time
import random
import multiprocessing as mp
from collections import defaultdict
import numpy as np
import scipy.interpolate
import scipy.integrate
import scipy.misc
import h5py
from numba import jit
import mctimme2.load_data
import mctimme2.sample
import distribution
import time

class Subject:
    """A subject in the study.

    This class is used to handle the subject-specific model parameters
    (pharmacokinetic elimination rates and compound concentration) and the
    subject-specific data (time points and compound dosing).

    Attributes
    ----------
    model : Model
        The main model object.
    name : str
        A unique name for the subject. Must be unique, as it is used in some
        dictionaries as a key.
    data_times : list of float
        The time associated with each data point.
    consumption_times : dict
        Times at which each dose was taken.
    doses : dict
        Quantity of each adminstered dose in any user-desired units.
    traj_times : 1darray
        A grid of time values -- presumably finer than data_times -- on which
        the trajectories for OTUs are calculated.
    traj_times_on_pert : 1darray
        Boolean array,
    ks : dict
        Keys are compounds, values are pharmacokinetic elimination rate.
    pharma_storage : dict
        Keys are compounds, values are lists of pharmacokinetic profiles.
    k_storage : dict
        Keys are compounds, values are elimination rates.

    """
    def __init__(self, name, data_times, consumption_times, doses, model):
        self.model = model
        self.name = name
        self.data_times = data_times
        self.consumption_times = consumption_times
        self.doses = doses

        self.ks = {}

        self.k_accepts = {}
        self.k_rejects = {}
        self.k_prop_vars = {}

        self.pharma_storage = {}
        self.k_storage = {}

        for compound in self.consumption_times.keys():
            self.ks[compound] = 1
            self.pharma_storage[compound] = []
            self.k_storage[compound] = []

            self.k_accepts[compound] = 0
            self.k_rejects[compound] = 0
            self.k_prop_vars[compound] = 0.1

        self.subj_compounds = list(self.consumption_times.keys())


    def create_on_off_periods(self):
        """Calculate when the compound administration period is active.

        This function populates self.traj_times_on_pert, a list of booleans
        lining up with self.traj_times indiciating True when the perturbation
        period is active and False at other times.
        """
        traj_times_on_pert = [False] * len(self.traj_times)

        for ti in range(len(self.traj_times)):
            on_pert = False
            t = self.traj_times[ti]
            for compound in self.consumption_times.keys():
                if t >= min(self.consumption_times[compound]) and \
                   t <= max(self.consumption_times[compound]):
                    on_pert = True

            traj_times_on_pert[ti] = on_pert

        self.traj_times_on_pert = traj_times_on_pert

        self.traj_times_on_pert_compounds = {}
        for compound in self.consumption_times.keys():
            self.traj_times_on_pert_compounds[compound] = [False] * len(self.traj_times)
            for ti in range(len(self.traj_times)):
                on_pert = False
                t = self.traj_times[ti]
                if t >= min(self.consumption_times[compound]) and \
                   t <= max(self.consumption_times[compound]):
                    on_pert = True

                self.traj_times_on_pert_compounds[compound][ti] = on_pert

            self.traj_times_on_pert_compounds[compound] = np.array(self.traj_times_on_pert_compounds[compound], dtype=np.int)



    def create_traj_times(self, spacing, fixed=[]):
        """Find time points to use for latent trajectory inference.

        Inference for the trajectory will be performed on a denser grid of
        points than the data time points, and this grid needs to include all
        of the original data time points. Using self.data_times, this function
        populates self.traj_times with time points which have approximate
        spacing as specified by the argument, and also includes the data time
        points as well as any other desired time points.

        Parameters
        ----------
        spacing : float
            Approximate spacing between each time point.
        fixed : list of float, optional
            Time points to include in traj_times. This can be used to require
            that trajectory inference will be performed at specified time
            points.
        """

        # Start forming the arrays with the first time point
        self.traj_times = [self.data_times[0]]
        self.traj_times_is_data = [True]
        self.traj_times_data_idx = [0]

        rti = 1
        ti = 1

        data_and_fixed = self.data_times.copy()
        for tx in fixed:
            bisect.insort(data_and_fixed, tx)
        offset = 0

        while ti < len(data_and_fixed):
            ct = self.traj_times[rti-1] + spacing
            if np.greater(ct,data_and_fixed[ti]) | np.less(np.abs(data_and_fixed[ti]-ct), spacing): ## measured time-point is better, so use it
                self.traj_times.append(data_and_fixed[ti])
                if data_and_fixed[ti] in self.data_times:
                    self.traj_times_is_data.append(True)
                    self.traj_times_data_idx.append(ti-offset)
                else:
                    self.traj_times_is_data.append(False)
                    self.traj_times_data_idx.append(-1)
                    offset += 1
                rti = rti + 1
                ti = ti + 1
            else:
                self.traj_times.append(ct)
                self.traj_times_is_data.append(False)
                self.traj_times_data_idx.append(-1)
                rti = rti + 1

        self.traj_times = np.array(self.traj_times)
        self.traj_times_is_data = np.array(self.traj_times_is_data)
        self.traj_times_data_idx = np.array(self.traj_times_data_idx)
        self.delta_t = np.diff(self.traj_times)


    def calc_pharma_trajectory(self):
        """Compute the current pharmacokinetic profile.

        Computes the level of the compounds in this subject over time, using the
        current values for the elimination rate. The updated levels are stored
        in self.pharma_trajectories.
        """
        self.pharma_trajectories = {}
        for compound in self.consumption_times.keys():
            t = self.traj_times
            tm = np.tile(t, (len(self.doses[compound]), 1))
            cm = np.exp(-self.ks[compound]*(tm.transpose()-np.array(self.consumption_times[compound])).transpose())
            cm[cm>1] = 0
            cm = (np.array(self.doses[compound])*cm.transpose()).transpose()
            c = np.sum(cm, axis=0)
            self.pharma_trajectories[compound] = c


    def adapt_k_proposals(self):
        """Tune the variance of the elimination rate (k) proposal distributions.

        The MH acceptance rate is calculated from the iterations taken since this
        function was last called. The proposal variance is tuned upwards if the
        acceptance rate is above 0.234 and downwards if the acceptance rate is
        below 0.234. This function should be used during the burn-in period
        to accelerate mixing.
        """
        for compound in self.consumption_times.keys():
            accepts = self.k_accepts[compound]
            rejects = self.k_rejects[compound]
            current_prop_var =self.k_prop_vars[compound]

            accept_rate = accepts / (accepts + rejects)
            if accept_rate > 0.234:
                prop_var = min(current_prop_var * 1.5, 5)
            elif accept_rate < 0.234:
                prop_var = max(current_prop_var * 0.75, 0.05)

            self.k_prop_vars[compound] = prop_var
            self.k_accepts[compound] = 0
            self.k_rejects[compound] = 0


    def update_k(self):
        """Update the pharmacokinetic elimination rates with MH steps.

        An adaptive Gaussian jumping kernal is used.
        """
        for compound in self.consumption_times.keys():
            k_old = self.ks[compound]
            p_0 = self.model.pharma_k_prior(k_old)
            l_0 = 0
            for dp in self.model.time_series:
                if dp.subject == self:
                    l_0 += dp.calc_traj_loglike()

            # Save old pharma profile for fast reverting if jump not accepted
            old_pharma = self.pharma_trajectories[compound].copy()

            k_prop = random.gauss(k_old, math.sqrt(self.k_prop_vars[compound]))
            self.ks[compound] = k_prop
            self.calc_pharma_trajectory()

            p_prop = self.model.pharma_k_prior(k_prop)
            l_prop = 0
            for dp in self.model.time_series:
                if dp.subject == self:
                    for pert_compound in dp.dyn_clusters.keys():
                        dp.dyn_clusters[pert_compound].pert_cluster.calc_pert_trajs(self)
                    l_prop += dp.calc_traj_loglike()

            log_r = p_prop + l_prop - p_0 - l_0
            if math.log(random.random()) >= log_r:
                # MH proposal was not accepted, so revert back
                self.ks[compound] = k_old
                self.pharma_trajectories[compound] = old_pharma
                for dp in self.model.time_series:
                    if dp.subject == self:
                        for pert_compound in dp.dyn_clusters.keys():
                            dp.dyn_clusters[pert_compound].pert_cluster.calc_pert_trajs(dp.subject)
                self.k_rejects[compound] += 1

            else:
                self.k_accepts[compound] += 1


    def save_current_iteration(self):
        """Save the current parameter values to object storage.

        Typically, this function is called at every iteration after the end of
        the burn-in period.
        """
        for compound in self.consumption_times.keys():
            self.k_storage[compound].append(self.ks[compound])
            self.pharma_storage[compound].append(self.pharma_trajectories[compound].copy())


class DynCluster:
    """Microbe cluster (lower level of double Dirichlet Process).

    At the bottom level of the double Dirichlet Process, time-series are
    clustered, incorporating phylogenetic distances.

    Attributes
    ----------
    model : Model
        The main model object.
    pert_cluster : PertCluster
        Perturbation cluster (or null cluster) to which this microbe cluster
        is assigned in the top level of the double Dirichlet process.
    members : set of TimeSeries
        The time-series currently assigned to this cluster.
    prototype : TimeSeries
        A prototype member for the cluster. Phylogenetic distances to this
        member are used to calculate the likelihood of time-series under this
        cluster.
    compound : str
        The compound to which this cluster represents a response.
    """

    def __init__(self, model, compound):
        self.model = model
        self.pert_cluster = None
        self.compound = compound
        self.members = set([])
        self.prototype = None


    def update_prototype(self):
        """Update the prototype member with a Gibbs step.
        """
        otus = []  # Each timeseries which could be the prototype
        otu_ps = []  # The unnormalized log probability of that timeseries being
                     # the prototype
        for otu in self.members:
            p = 0
            for otu2 in self.members:
                # p += self.model.potential_function(otu, otu2)
                p += self.model.log_potentials_cache[otu][otu2]
            otu_ps.append(p)
            otus.append(otu)

        self.prototype = otus[sample.categorical_log(otu_ps)]


    def update_pert_cluster2(self):
        """Update the effect selection and perturbation cluster assignment with
        a Gibbs step.

        Assigns to a perturbation cluster or to the null response cluster.
        """
        p = [0, 0]  # [P(no effect), P(effect)]
        p[0] = math.log(self.model.perturb_select_prior_zero_freq)
        y = []
        X = []
        S_diag = []
        for dp in self.members:
            obs = (dp.trajectory[1:] - dp.trajectory[:-1]) / dp.delta_t - dp.growth_rate * (1 + dp.total_persist_pert_traj[:-1]) * dp.trajectory[:-1] - dp.self_interact * dp.trajectory[:-1]**2
            for other_cmpd in self.model.all_compounds:
                if other_cmpd != self.compound:
                    obs -= dp.growth_rate * dp.trajectory[:-1] * dp.dyn_clusters[other_cmpd].pert_cluster.total_pert_trajs[dp.subject_name][:-1]
            y = np.hstack((y, obs))
            design = dp.growth_rate * dp.trajectory[:-1] * self.pert_cluster.pert_trajs[dp.subject_name][:-1]
            X = np.hstack((X, design))
            S_diag = np.hstack((S_diag, dp.traj_var / dp.delta_t))
        S_0 = self.model.perturb_effect_prior_var
        S_0_m1 = 1 / self.model.perturb_effect_prior_var
        S_m1 = 1 / S_diag
        S_3_m1 = np.dot(X * S_m1, X) + S_0_m1
        S_3 = 1 / S_3_m1
        mu_3 = S_3 * np.dot(X * S_m1, y)
        ll = 0.5 * math.log(S_3) - 0.5 * math.log(S_0) + 0.5 * mu_3 * S_3_m1 * mu_3
        p[1] = math.log(1-self.model.perturb_select_prior_zero_freq) + ll
        u = sample.categorical_log(p)
        if u == 0:
            self.pert_cluster.members.remove(self)
            null = self.model.null_clusters[self.compound]
            self.pert_cluster = null
            null.members |= {self}
            self.model.purge_empty_clusters()
            return
        else:
            pass

        conc_pert = self.model.pert_concentrations[self.compound]
        all_pert_clusters = self.model.pert_cluster_sets[self.compound]
        k = len(all_pert_clusters)
        p = []

        # Use the fast method to calculate the likelihood for each pert cluster
        if k > 0:
            lls = np.zeros(len(all_pert_clusters))
            for dp in self.members:
                lls += dp.calc_traj_loglike_given_all_pcs(self.compound)
        else:
            lls = []

        # Get the probabilities for being assigned to each existing pert cluster
        for i in range(k):
            pc = all_pert_clusters[i]
            ni = len(pc.members)

            if self.pert_cluster == pc:
                ni -= 1

            ll = lls[i]

            if ni != 0:
                p.append(math.log(ni) + ll)

            elif ni == 0:
                p.append(math.log(conc_pert) + ll)

        # Sample from the prior for the case of a new cluster
        effect = self.model.sample_pert_effect_prior()
        r = self.model.sample_transfer_shape_prior()
        on_time = self.model.sample_pert_on_time_prior()
        duration = self.model.sample_pert_duration_prior()

        # Calculate the perturbation trajectories for this sample of parameters
        total_pert_trajs = {}
        for subject in self.model.subjects.values():
            total_pert_trajs[subject.name] = np.zeros(len(subject.traj_times))
            duration_days = duration * (max(subject.consumption_times[self.compound]) - min(subject.consumption_times[self.compound]))
            subj_on_time = on_time + min(subject.consumption_times[self.compound])
            step_function = (subject.traj_times > subj_on_time) & (subject.traj_times < subj_on_time+duration_days)
            step_function = step_function.astype(np.int)
            pert_traj = step_function * (2.0 / (1.0 + np.exp(-r*subject.pharma_trajectories[self.compound])) - 1.0)
            total_pert_trajs[subject.name] += effect * pert_traj.copy()

        # Calculate the likelihood of the new cluster
        ll = 0
        for dp in self.members:
            npc = PertCluster(self.model, self.compound)
            npc.total_pert_trajs = total_pert_trajs
            ll += dp.calc_traj_loglike_given_pc_m(npc, self.compound)
        p.append(math.log(conc_pert) + ll)

        p = [math.log(1-self.model.perturb_select_prior_zero_freq) + x for x in p]

        cluster_update = sample.categorical_log(p)

        if cluster_update < k:
            # Assign to an existing pert cluster
            self.pert_cluster.members -= {self}
            self.pert_cluster = all_pert_clusters[cluster_update]
            self.pert_cluster.members |= {self}

        elif cluster_update == k:
            # Assign to the new pert cluster
            new_pc = PertCluster(self.model, self.compound)
            new_pc.effect = effect
            new_pc.transfer_shape = r
            new_pc.on_time = on_time
            new_pc.duration = duration
            new_pc.calc_all_pert_trajs()

            self.model.pert_cluster_sets[self.compound].append(new_pc)

            self.pert_cluster.members -= {self}
            self.pert_cluster = new_pc
            new_pc.members |= {self}

        elif cluster_update == k+1:
            self.pert_cluster.members.remove(self)
            null = self.model.null_clusters[self.compound]
            self.pert_cluster = null
            null.members |= {self}
            self.model.purge_empty_clusters()

        self.model.purge_empty_clusters()


class PertCluster:
    """Perturbation cluster (upper level of double Dirichlet process).

    Attributes
    ----------
    model : Model
        The main model object.
    compound : str
        The compound to which this cluster represents a response.
    members : set of DynCluster
        The microbe clusters currently assigned to this perturbation cluster.
    effect : float
        The magnitude of the perturbation effect
    transfer_shape : float
        The transfer function shape parameter
    on_time : float
        The on time relative to the start of the dosing period (days)
    duration : float
        The duration as a proportion of the dosing period
    total_pert_trajs : dict
        For each subject name (key), the total perturbation trajectory over time (value)
    pert_trajs : dict
        For each subject name (key), the perturbation trajectory before being
        multiplied by effect parameter
    """

    def __init__(self, model, compound):
        self.model = model
        self.compound = compound
        self.members = set([])
        self.effect = 0
        self.transfer_shape = 1
        self.on_time = 1
        self.duration = 1.2

        self.total_pert_trajs = {} # with effect magnitude
        self.pert_trajs = {} # before multiplying by effect

        self.calc_all_pert_trajs()


    def calc_all_pert_trajs(self):
        """Calculate the perturbation trajectories for all subjects.
        """
        for subject in self.model.subjects.values():
            self.calc_pert_trajs(subject)


    def calc_pert_trajs(self, subject):
        """Calculate the perturbation trajectory for a single subject.

        Parameters
        ----------
        subject : Subject
            Subject for which to calculate the perturbation trajectory
        """
        r = self.transfer_shape
        on_time = self.on_time
        on_time += min(subject.consumption_times[self.compound])
        duration = self.duration
        duration_days = duration * (max(subject.consumption_times[self.compound]) - min(subject.consumption_times[self.compound]))

        step_function = (subject.traj_times > on_time) & (subject.traj_times < on_time+duration_days)
        step_function = step_function.astype(np.int)

        pert_traj = step_function * (2.0 / (1.0 + np.exp(-r*subject.pharma_trajectories[self.compound])) - 1.0)
        total_pert_traj = self.effect * pert_traj

        self.total_pert_trajs[subject.name] = total_pert_traj
        self.pert_trajs[subject.name] = pert_traj


    def update_on_times(self):
        """Update the on time with a MH step.

        A Gaussian jumping kernel is used as the proposal.
        """
        on_time_0 = self.on_time
        p_0 = self.model.pert_on_time_prior(on_time_0)

        l_0 = 0
        for mc in self.members:
            for dp in mc.members:
                l_0 += dp.calc_traj_loglike()

        on_time_prop = random.gauss(on_time_0, math.sqrt(self.model.on_time_prop_var))

        self.on_time = on_time_prop
        self.calc_all_pert_trajs()

        p_prop = self.model.pert_on_time_prior(on_time_prop)
        l_prop = 0
        for mc in self.members:
            for dp in mc.members:
                l_prop += dp.calc_traj_loglike()

        log_r = p_prop + l_prop - p_0 - l_0
        if math.log(random.random()) >= log_r:
            # MH proposal was not accepted, so revert back
            self.on_time = on_time_0
            self.calc_all_pert_trajs()


    def update_transfer_shapes(self,):
        """Update the transfer function shape parameter with a MH step.

        A Gaussian jumping kernel is used as the proposal.
        """
        r_0 = self.transfer_shape
        p_0 = self.model.transfer_shape_prior(r_0)

        l_0 = 0
        for mc in self.members:
            for dp in mc.members:
                l_0 += dp.calc_traj_loglike()

        r_prop = random.gauss(r_0, math.sqrt(self.model.transfer_shape_prop_var))

        self.transfer_shape = r_prop
        self.calc_all_pert_trajs()

        p_prop = self.model.transfer_shape_prior(r_prop)

        l_prop = 0
        for mc in self.members:
            for dp in mc.members:
                l_prop += dp.calc_traj_loglike()

        log_r = p_prop + l_prop - p_0 - l_0
        if math.log(random.random()) >= log_r:
            # MH proposal was not accepted, so revert back
            self.transfer_shape = r_0
            self.calc_all_pert_trajs()


    def update_durations(self):
        """Update the perturbation duration parameter with a MH step.

        A Gaussian jumping kernel is used as the proposal.
        """
        duration_0 = self.duration
        p_0 = self.model.pert_duration_prior(duration_0)

        l_0 = 0
        for mc in self.members:
            for dp in mc.members:
                l_0 += dp.calc_traj_loglike()

        duration_prop = random.gauss(duration_0, math.sqrt(self.model.duration_prop_var))

        self.duration = duration_prop
        self.calc_all_pert_trajs()

        p_prop = self.model.pert_duration_prior(duration_prop)

        l_prop = 0
        for mc in self.members:
            for dp in mc.members:
                l_prop += dp.calc_traj_loglike()

        log_r = p_prop + l_prop - p_0 - l_0
        if math.log(random.random()) >= log_r:
            # MH proposal was not accepted, so revert back
            self.duration = duration_0
            self.calc_all_pert_trajs()


    def update_effects(self):
        """Update the perturbation effect parameter with a Gibbs step.
        """
        self.calc_all_pert_trajs()
        y = []
        x = []
        s1 = []
        for mc in self.members:
            for dp in mc.members:
                obs = (dp.trajectory[1:] - dp.trajectory[:-1]) / dp.delta_t - dp.growth_rate * (1 + dp.total_persist_pert_traj[:-1]) * dp.trajectory[:-1] - dp.self_interact * dp.trajectory[:-1]**2

                for other_cmpd in self.model.all_compounds:
                    if other_cmpd != self.compound:
                        obs -= dp.growth_rate * dp.trajectory[:-1] * dp.dyn_clusters[other_cmpd].pert_cluster.pert_trajs[dp.subject_name][:-1]

                y = np.hstack((y, obs))
                x = np.hstack((x, dp.growth_rate * dp.trajectory[:-1] * self.pert_trajs[dp.subject_name][:-1]))
                s1 = np.hstack((s1, dp.traj_var / dp.delta_t))

        precis_matrix_diag = 1/s1
        precis = np.dot(x * precis_matrix_diag, x) + 1 / self.model.perturb_effect_prior_var
        mean = np.dot(x * precis_matrix_diag, y)
        mean *= 1/precis

        self.effect = random.gauss(mean, math.sqrt(1/precis))

        self.calc_all_pert_trajs()


class NullPertCluster(PertCluster):
    """The null effect group to which microbe clusters are assigned when
    they choose no effect.
    """
    def update_on_time(self):
        """Update the perturbation on time.
        """
        self.on_time = self.model.sample_pert_on_time_prior()


    def update_duration(self):
        """Update the perturbation duration.
        """
        self.duration = self.model.sample_pert_duration_prior()


    def update_transfer_shape(self,):
        """Update the perturbation transfer function shape parameter.
        """
        self.transfer_shape = self.model.sample_transfer_shape_prior()


@jit(nopython=True, fastmath=True)
def fast_update_var_param(trajectory,
                          delta_t,
                          self_interact,
                          growth_rate,
                          total_persist_pert_traj,
                          total_pert_traj,
                          period,
                          a,
                          b):
    """Perform the calculations for the update of process variance.

    Parameters
    ----------
    trajectory : ndarray
        Latent trajectory
    delta_t : ndarray
    """
    observations = (trajectory[1:] - trajectory[:-1]) / np.sqrt(delta_t) - self_interact * trajectory[:-1] ** 2 * np.sqrt(delta_t) - growth_rate * trajectory[:-1] * (1 + total_persist_pert_traj[:-1]) * np.sqrt(delta_t)
    observations -= growth_rate * trajectory[:-1] * total_pert_traj[:-1] * np.sqrt(delta_t)
    observations = observations[period]

    ap = a + len(observations) / 2
    bp = b + np.sum(observations**2) / 2

    return ap, 1/bp


# @jit(nopython=True, fastmath=True)
def fast_update_growth_rate(model_growth_prior_var, trajectory,
                            total_pert_traj,
                            total_persist_pert_traj,
                            delta_t,
                            traj_var,
                            model_growth_prior_mean,
                            self_interact):
    v = 1 / model_growth_prior_var
    mtt = np.ones(len(trajectory[:-1]))
    mtt += total_pert_traj[:-1]
    mtt += total_persist_pert_traj[:-1]
    mtt *= trajectory[:-1]
    v += np.sum(mtt**2 * delta_t / traj_var)
    v = 1/v

    m = model_growth_prior_mean / model_growth_prior_var
    mt = np.diff(trajectory) / delta_t
    mt -= self_interact * trajectory[:-1]**2
    mtt = np.ones(len(trajectory[:-1]))
    mtt += total_pert_traj[:-1]
    mtt += total_persist_pert_traj[:-1]
    mt = trajectory[:-1] * mtt * mt
    m += np.sum(mt * delta_t / traj_var)
    m *= v
    return m, v


# @jit(nopython=True, cache=True, fastmath=True)
def calc_rhs_p(traj_pair, growth_rate, persist_pair, delta_t_pair, self_interact, pert_traj_pair):
    rhs = traj_pair + growth_rate * (1 + persist_pair) * traj_pair * delta_t_pair + self_interact * traj_pair**2 * delta_t_pair
    rhs += growth_rate * pert_traj_pair * traj_pair * delta_t_pair
    return rhs


def logpdf_normal_sum(x, loc, scale):
    return -x.size/2 * np.log(2*np.pi) + (-np.log(scale)
           - np.square((x-loc)/(math.sqrt(2)*scale))).sum()


class TimeSeries:
    """The time series object for one taxon in a subject.

    Attributes
    ----------
    model : Model
        Main model
    subject : Subject
        subject associated with
    otu : str
        Taxonomic string for this otu
    taxon : str
        Extracted rank
    etc.
    data_times : ndarray
    """
    def __init__(self,
                 otu,
                 subject,
                 data_times,
                 rel_abundances,
                 counts,
                 totals,
                 model):

        # Information about which time series this is
        self.otu = otu
        self.taxon = load_data.Taxon(otu)
        self.species = self.taxon.species
        self.subject = subject
        self.subject_name = self.subject.name
        self.model = model
        self.compounds = self.subject.subj_compounds

        # Convert raw data to numpy arrays
        self.data_times = np.array(data_times)
        self.rel_abundances = np.array(rel_abundances)
        self.counts = np.array(counts)
        self.totals = np.array(totals)

        # Get trajectory time point grid from subject and calculate delta t
        self.traj_times = self.subject.traj_times
        self.traj_times_is_data = self.subject.traj_times_is_data
        self.traj_times_data_idx = self.subject.traj_times_data_idx
        self.traj_times_on_pert = np.array(self.subject.traj_times_on_pert, dtype=np.int)
        self.delta_t = np.diff(self.traj_times)

        # Initialize trajectory and aux trajectory
        self.trajectory = np.zeros(len(self.traj_times))
        self.aux_trajectory = np.zeros(len(self.traj_times))
        self.is_zero = np.zeros(len(self.aux_trajectory))
        sm = self.smooth_data()
        for ti in range(len(self.trajectory)):
            self.trajectory[ti] = sample.truncated_normal(sm[ti], sm[ti]+10, self.model.aux_traj_lower_bound, self.model.aux_traj_upper_bound)
            self.aux_trajectory[ti] = sample.truncated_normal(self.trajectory[ti], math.sqrt(self.model.var_q_b*self.trajectory[ti]**2 + self.model.var_q_c), self.model.aux_traj_lower_bound, self.model.aux_traj_upper_bound)

        self.zero_inflatable = np.zeros(len(self.traj_times), dtype=bool)
        for ti in range(len(self.zero_inflatable)):
            self.check_zero_inflatable(ti)

        # Initialize microbe clusters dict
        self.dyn_clusters = {}

        # Initialize object storage
        self.trajectory_storage = []
        self.aux_trajectory_storage = []
        self.growth_rate_storage = []
        self.self_interact_storage = []
        self.var_c_storage = []
        self.var_c_pert_storage = {}
        self.persist_effect_storage = []
        self.persist_select_storage = []
        self.persist_duration_storage = []
        self.persist_pert_traj_storage = []
        self.pert_traj_storage = {}
        self.effect_storage = {}
        self.transfer_shape_storage = {}
        self.on_time_storage = {}
        self.duration_storage = {}
        self.dyn_cluster_storage = {}
        self.pert_cluster_storage = {}
        for compound in self.compounds:
            self.pert_traj_storage[compound] = []
            self.effect_storage[compound] = []
            self.transfer_shape_storage[compound] = []
            self.on_time_storage[compound] = []
            self.duration_storage[compound] = []
            self.dyn_cluster_storage[compound] = []
            self.pert_cluster_storage[compound] = []

            self.var_c_pert_storage[compound] = []

        self.get_on_pert = [bool(x) for x in self.traj_times_on_pert[:-1]]
        self.get_off_pert = [not x for x in self.traj_times_on_pert[:-1]]

        self.trajectory_history = []
        self.aux_trajectory_history = []

    def set_initial_values(self, effect_setting='calculate'):
        """Set initial values for parameters and proposals.

        Should be called after the model has set up initial clusters.

        Parameters
        ----------
        effect_setting : {'calculate', 0}, optional ('calculute')
            calculate the initial effect. or set it to zero
        """
        # Growth rate, self interaction, and perturbation effect
        abundance_on = self.trajectory[:-1][self.get_on_pert]
        abundance_off = self.trajectory[:-1][self.get_off_pert]
        self.self_interact = -0.005
        self.growth_rate = -self.self_interact * np.mean(abundance_off)
        init_effect = -self.self_interact * np.mean(abundance_on) / self.growth_rate - 1
        for compound in self.compounds:
            if effect_setting == 'calculate':
                self.dyn_clusters[compound].pert_cluster.effect = init_effect
            else:
                self.dyn_clusters[compound].pert_cluster.effect = 0

        # Trajectory proposals
        self.traj_prop_covs = []  # a list of covariance matrices for each time point
        self.cs = []  # Scale for each time point
        for i in range(len(self.aux_trajectory)):
            #self.traj_prop_covs.append(np.array([[100, 90], [90, 100]]))
            self.traj_prop_covs.append(.01*np.array([[1, .9], [.9, 1]]))
            self.cs.append(1)

        self.both_accepts = np.zeros(len(self.traj_times))
        self.both_rejects = np.zeros(len(self.traj_times))

        # All other parameters
        self.var_c = np.mean(abundance_on) ** 2
        self.var_c_pert = {}
        for compound in self.compounds:
            self.var_c_pert[compound] = np.mean(abundance_on) ** 2
        self.persist_select = 0
        self.persist_effect = 0
        self.persist_on_time = 0
        self.persist_duration = 50

        self.calc_persist_traj()
        self.calc_traj_var()


    def smooth_data(self):
        """Interpolate the data points.

        Returns
        -------
        ndarray
            Interpolation of data at trajectory times.
        """
        f = scipy.interpolate.interp1d(self.data_times,
                self.counts * self.model.settings['scale_factor'] / self.totals)
        sm = np.zeros(len(self.traj_times))
        for ti in range(len(self.traj_times)):
            sm[ti] = f(self.traj_times[ti])
        return sm


    def check_zero_inflatable(self, ti):
        """Check if this time point could have q = structural zero.

        Populates self.zero_inflatable[ti] with True if q could = 0 and False
        otherwise. The update for zero inflation will be performed at those
        time points where self.zero_inflatable[ti] == True.

        Parameters
        ----------
        ti : int
            Index of time point
        """
        data_index = self.traj_times_data_idx[ti]
        if data_index == -1:
            i = ti
            left_counts = -1
            while self.traj_times_data_idx[i] == -1:
                i -= 1
                if i < 0:
                    left_counts = 0
                    break
            if left_counts == -1:
                left_counts = self.counts[self.traj_times_data_idx[i]]

            if left_counts == 0:
                self.zero_inflatable[ti] = True
            else:
                self.zero_inflatable[ti] = False


        elif data_index > -1:
            if self.counts[data_index] == 0:
                self.zero_inflatable[ti] = True


    def adapt_traj_proposals(self):
        """Tune the covariance of the trajectory/aux trajectory proposals.

        This function calculates the empirical covariance from the currently
        accepted samples, and sets the proposal covariance equal to this
        multiplied by a scalar to adjust the acceptance rate.
        """
        filename = os.path.join(self.model.settings['output_directory'],
                                self.model.settings['mcmc_name'])
        # mcmc_samples = h5py.File(filename, 'r')

        key = self.subject_name + self.otu

        self.trajectory_history = self.trajectory_history.copy()[-400:]
        self.aux_trajectory_history = self.aux_trajectory_history.copy()[-400:]
        traj_samples2 = list(self.trajectory_history)
        aux_traj_samples2 = list(self.aux_trajectory_history)

        for i in range(len(self.aux_trajectory)):

            xs = [traj_samples2[j][i] for j in range(len(traj_samples2))]
            qs = [aux_traj_samples2[j][i] for j in range(len(aux_traj_samples2))]

            accepts = self.both_accepts[i]
            rejects = self.both_rejects[i]

            accept_rate = accepts / (accepts + rejects)

            empirical_cov = np.cov([xs, qs])

            if accept_rate > 0.234:
                self.cs[i] = min(1.2 * self.cs[i], 2)

            elif accept_rate < 0.234:
                self.cs[i] = max(.8 * self.cs[i], 0.5)

            self.traj_prop_covs[i] = self.cs[i]**2 * empirical_cov + .001 * np.array([[1, .9],[.9, 1]])

            self.both_accepts[i] = 0
            self.both_rejects[i] = 0


    def calc_traj_var(self):
        """Calculate the process variance based on current parameters.
        """
        if len(self.compounds) == 1:
            c = self.compounds[0]
            self.traj_var = self.var_c * (1 - self.traj_times_on_pert[:-1]) + self.var_c_pert[c] * self.traj_times_on_pert[:-1]

        else:
            self.traj_var = self.var_c * (1 - self.traj_times_on_pert[:-1])
            for compound in self.compounds:
                self.traj_var += self.var_c_pert[compound] * self.subject.traj_times_on_pert_compounds[compound][:-1]

    def update_var_parameters(self):
        """Update all process variance parameters.
        """
        if len(self.compounds) == 0:
            var_params = ['var_c', 'var_c_pert']

            for var_param in var_params:
                self.update_var_parameter_gibbs(var_param)

        else:
            period = self.get_off_pert
            compound = self.model.all_compounds[0]
            pert_traj = self.dyn_clusters[compound].pert_cluster.total_pert_trajs[self.subject_name].copy()
            for cmpd in self.model.all_compounds[1:]:
                pert_traj += self.dyn_clusters[cmpd].pert_cluster.total_pert_trajs[self.subject_name]
            aact, binv = fast_update_var_param(self.trajectory, self.delta_t, self.self_interact, self.growth_rate, self.total_persist_pert_traj, pert_traj, np.array(period), self.model.process_var_prior_a, self.model.process_var_prior_b)

            X = random.gammavariate(aact, binv)
            setattr(self, 'var_c', 1/X)
            self.calc_traj_var()

            for var_compound in self.compounds:
                period = [bool(x) for x in self.subject.traj_times_on_pert_compounds[var_compound][:-1]]
                compound = self.model.all_compounds[0]
                pert_traj = self.dyn_clusters[compound].pert_cluster.total_pert_trajs[self.subject_name].copy()
                for cmpd in self.model.all_compounds[1:]:
                    pert_traj += self.dyn_clusters[cmpd].pert_cluster.total_pert_trajs[self.subject_name]
                aact, binv = fast_update_var_param(self.trajectory, self.delta_t, self.self_interact, self.growth_rate, self.total_persist_pert_traj, pert_traj, np.array(period), self.model.process_var_prior_a, self.model.process_var_prior_b)

                X = random.gammavariate(aact, binv)
                self.var_c_pert[var_compound] = 1/X
                self.calc_traj_var()


    def update_var_parameter_gibbs(self, var_param):
        """Update a process variance term with a Gibbs step.

        Parameters
        ----------
        var_param : {'var_c', 'var_c_pert'}
            The parameter to update
        """

        if '_pert' in var_param:
            period = self.get_on_pert
        else:
            period = self.get_off_pert

        compound = self.model.all_compounds[0]
        pert_traj = self.dyn_clusters[compound].pert_cluster.total_pert_trajs[self.subject_name].copy()
        for cmpd in self.model.all_compounds[1:]:
            pert_traj += self.dyn_clusters[cmpd].pert_cluster.total_pert_trajs[self.subject_name]
        aact, binv = fast_update_var_param(self.trajectory, self.delta_t, self.self_interact, self.growth_rate, self.total_persist_pert_traj, pert_traj, np.array(period), self.model.process_var_prior_a, self.model.process_var_prior_b)

        X = random.gammavariate(aact, binv)
        setattr(self, var_param, 1/X)
        self.calc_traj_var()


    def update_both_trajectories(self, seed=None):
        """Update the trajectory and auxiliary trajectory with MH steps.
        """
        # For reproduciblity in multiprocessing
        if seed is not None:
            random.seed(seed)
            sample.seed(seed*2)

        self.calc_traj_var()
        for i in range(len(self.aux_trajectory)):
            self.update_both_trajectories_one_step(i)
        self.calc_traj_var()


    def update_zero_inflation(self, ti):
        """Update whether or not q is structural 0 at this time point.

        This method proposes a jump from the current setting to the other
        setting at a time point.

        Parameters
        ----------
        ti : int
            Time index
        """
        x = self.trajectory[ti]
        q = self.aux_trajectory[ti]
        aux_var = self.model.var_q_c + self.model.var_q_b * q**2

        log_prior_q = lambda x: math.log(1 / (self.model.aux_traj_upper_bound - self.model.aux_traj_lower_bound))

        if q == 0:
            new_q = sample.truncated_normal(0, 1, self.model.aux_traj_lower_bound, self.model.aux_traj_upper_bound)

            # Calculate acceptance ratio
            log_r = distribution.normal_logpdf(x, new_q, math.sqrt(aux_var)) \
                    + log_prior_q(q) \
                    + math.log(1 - self.model.zero_inflation_prior_zero_freq)

            log_r -= distribution.normal_logpdf(x, 0, math.sqrt(self.model.var_q_c))
            log_r -= distribution.truncated_normal_logpdf(new_q, 0, 1, self.model.aux_traj_lower_bound, self.model.aux_traj_upper_bound)
            log_r -= math.log(self.model.zero_inflation_prior_zero_freq)

            if math.log(random.random()) < log_r:
                # move was accepted
                self.aux_trajectory[ti] = new_q
                return

        elif q != 0:
            # Consider a jump to the model where q == 0
            log_r = distribution.normal_logpdf(x, 0, math.sqrt(self.model.var_q_c)) \
                  + distribution.truncated_normal_logpdf(q, 0, 1, self.model.aux_traj_lower_bound, self.model.aux_traj_upper_bound) \
                  + math.log(self.model.zero_inflation_prior_zero_freq)
            log_r -= distribution.normal_logpdf(x, q, math.sqrt(aux_var))
            log_r -= log_prior_q(q)
            log_r -= math.log(1 - self.model.zero_inflation_prior_zero_freq)

            if math.log(random.random()) < log_r:
                # Move was accepted
                self.aux_trajectory[ti] = 0
                return


    def update_both_trajectories_one_step(self, ti, allow_zero_inflation=True):
        """Update the traj and aux traj at a time point with an MH step.

        This function uses a multivariate MH proposal to simultaneously update
        the trajectory and auxiliary trajectory.

        Parameters
        ----------
        ti : int
            Index of time point at which to update
        allow_zero_inflation : bool, optional
            Whether to consider zero inflation at this time point
        """
        x_old = self.trajectory[ti]
        q_old = self.aux_trajectory[ti]
        aux_var = self.model.var_q_b * self.aux_trajectory[ti] ** 2 + self.model.var_q_c
        data_index = self.traj_times_data_idx[ti]

        zi_possible = self.zero_inflatable[ti] and allow_zero_inflation

        # If zero inflation is possible here, choose which update to perform
        if zi_possible:
            if q_old == 0:
                # With probability 0.5, update parameters within this case
                if random.random() < 0.5:
                    self.update_trajectory_one_step(ti)
                    return

                # With probability 0.5, propose a jump to the other case
                else:
                    self.update_zero_inflation(ti)
                    return

            elif q_old != 0:
                # With probability 0.5, update parameters within this case
                if random.random() < 0.5:
                    pass

                # With probability 0.5, propose a jump tp q == 0
                else:
                    self.update_zero_inflation(ti)
                    return

        self.proposal_cov = self.traj_prop_covs[ti]

        q_old = self.aux_trajectory[ti]
        x_old = self.trajectory[ti]

        traj_sample = sample.multivariate_normal_fast(np.array([x_old, q_old]), self.traj_prop_covs[ti])

        x_new = traj_sample[0]
        q_new = traj_sample[1]

        if q_new < self.model.aux_traj_lower_bound:
            self.both_rejects[ti] += 1
            return

        if 0 < ti < len(self.trajectory) - 1:
            compound0 = self.model.all_compounds[0]
            send_pert_traj = self.dyn_clusters[compound0].pert_cluster.total_pert_trajs[self.subject_name][ti-1:ti+1].copy()

            for compound in self.model.all_compounds[1:]:
                send_pert_traj += self.dyn_clusters[compound].pert_cluster.total_pert_trajs[self.subject_name][ti-1:ti+1].copy()

            x0 = self.trajectory[ti-1]
            x2 = self.trajectory[ti+1]
            growth_rate = self.growth_rate
            self_interact = self.self_interact
            dt0 = self.delta_t[ti-1]
            dt1 = self.delta_t[ti]

            post_old = self.calc_traj_posterior_c(ti, x_old, aux_var, data_index, q_old, send_pert_traj[0], send_pert_traj[1], x0, x2, growth_rate, self_interact, dt0, dt1)

        else:
            post_old = self.calc_traj_posterior(ti)

        self.aux_trajectory[ti] = q_new
        self.trajectory[ti] = x_new

        if 0 < ti < len(self.trajectory) - 1:
            aux_var = self.model.var_q_b * self.aux_trajectory[ti] ** 2 + self.model.var_q_c

            post_new = self.calc_traj_posterior_c(ti, x_new, aux_var, data_index, q_new, send_pert_traj[0], send_pert_traj[1], x0, x2, growth_rate, self_interact, dt0, dt1)

        else:
            post_new = self.calc_traj_posterior(ti)

        log_r = post_new - post_old

        if math.log(random.random()) >= log_r or math.isnan(x_new) or math.isnan(q_new):
            # Revert back
            self.aux_trajectory[ti] = q_old
            self.trajectory[ti] = x_old
            self.both_rejects[ti] += 1

        else:
            self.both_accepts[ti] += 1


    def calc_traj_posterior_c(self, ti, x, aux_var, data_index, q, send_pert_traj0, send_pert_traj1, x0, x2, growth_rate, self_interact, dt0, dt1):

        post_old_c = distribution.traj_logpdf(x0,
                                              x,
                                              x2,
                                              growth_rate,
                                              self_interact,
                                              dt0,
                                              dt1,
                                              send_pert_traj0,
                                              send_pert_traj1,
                                              self.total_persist_pert_traj[ti-1],
                                              self.total_persist_pert_traj[ti],
                                              self.traj_var[ti-1],
                                              self.traj_var[ti],
                                              aux_var,
                                              q)
        if data_index > -1:
            total_reads = self.totals[data_index]
            scale_factor = self.model.settings['scale_factor']
            counts = self.counts[data_index]
            rel_abundance_data = counts / total_reads
            post_old_c += distribution.noise_model(counts, total_reads * self.aux_trajectory[ti] / scale_factor)

        return post_old_c

    def calc_traj_posterior(self, ti):
        """Calculate the trajectory posterior after updating ti.

        Parameters
        ----------
        ti : int
            Index

        Returns
        -------
        float
            log prob
        """
        x_old = self.trajectory[ti]
        q_old = self.aux_trajectory[ti]
        aux_var = self.model.var_q_b * self.aux_trajectory[ti] ** 2 + self.model.var_q_c
        data_index = self.traj_times_data_idx[ti]

        if 0 < ti < len(self.trajectory) - 1:
            compound0 = self.model.all_compounds[0]
            send_pert_traj = self.dyn_clusters[compound0].pert_cluster.total_pert_trajs[self.subject_name][ti-1:ti+1].copy()
            for compound in self.model.all_compounds[1:]:
                send_pert_traj += self.dyn_clusters[compound].pert_cluster.total_pert_trajs[self.subject_name][ti-1:ti+1]
            rhs = calc_rhs_p(self.trajectory[ti-1:ti+1], self.growth_rate, self.total_persist_pert_traj[ti-1:ti+1], self.delta_t[ti-1:ti+1], self.self_interact, send_pert_traj)
            p = distribution.normal_logpdf(self.trajectory[ti], rhs[0], math.sqrt(self.traj_var[ti-1] * self.delta_t[ti-1])) \
              + distribution.normal_logpdf(self.trajectory[ti+1], rhs[1], math.sqrt(self.traj_var[ti] * self.delta_t[ti])) \
              + distribution.normal_logpdf(x_old, q_old, math.sqrt(aux_var))

        elif ti == 0:
            rhs = self.trajectory[ti] + self.growth_rate * (1 + self.total_persist_pert_traj[ti]) * self.trajectory[ti] * self.delta_t[ti] + self.self_interact * self.trajectory[ti]**2 * self.delta_t[ti]
            for compound in self.model.all_compounds:
                rhs += self.growth_rate * self.dyn_clusters[compound].pert_cluster.total_pert_trajs[self.subject_name][ti] * self.trajectory[ti] * self.delta_t[ti]
            p = distribution.normal_logpdf(self.trajectory[ti+1], rhs, math.sqrt(self.traj_var[ti] * self.delta_t[ti])) \
              + distribution.normal_logpdf(x_old, q_old, math.sqrt(aux_var))

            p += distribution.normal_logpdf(self.trajectory[ti], self.model.init_traj_prior_mean, self.model.init_traj_prior_var)

        elif ti == len(self.trajectory) - 1:
            rhs = self.trajectory[ti-1] + self.growth_rate * (1 + self.total_persist_pert_traj[ti-1]) * self.trajectory[ti-1] * self.delta_t[ti-1] + self.self_interact * self.trajectory[ti-1]**2 * self.delta_t[ti-1]
            for compound in self.model.all_compounds:
                rhs += self.growth_rate * self.dyn_clusters[compound].pert_cluster.total_pert_trajs[self.subject_name][ti-1] * self.trajectory[ti-1] * self.delta_t[ti-1]
            p = distribution.normal_logpdf(self.trajectory[ti], rhs, math.sqrt(self.traj_var[ti-1] * self.delta_t[ti-1])) \
              + distribution.normal_logpdf(x_old, q_old, math.sqrt(aux_var))

        if data_index > -1:
            total_reads = self.totals[data_index]
            scale_factor = self.model.settings['scale_factor']
            counts = self.counts[data_index]
            rel_abundance_data = counts / total_reads

            p += distribution.noise_model(counts, total_reads * q_old / scale_factor)

        return p


    def update_trajectory_one_step(self, ti):
        # only used for q == 0
        aux_val = 0
        aux_var = self.model.var_q_b * self.aux_trajectory[ti] ** 2 + self.model.var_q_c
        #calculate proposal
        if ti == 0:
            v = 1 / (1 / self.model.init_traj_prior_var + 1 / aux_var)
            m = v * (self.model.init_traj_prior_mean / self.model.init_traj_prior_var + aux_val / aux_var)
        else:
            rhs = self.compute_rhs(ti, self.trajectory[ti-1])
            ssv = self.traj_var[ti-1] * self.delta_t[ti - 1]

        m = self.trajectory[ti]
        x_new = random.gauss(m, 5)
        x_old = self.trajectory[ti]

        pp_new =0
        pp_old = 0

        l_new = distribution.normal_logpdf(x_new, aux_val, math.sqrt(aux_var))
        l_old = distribution.normal_logpdf(x_old, aux_val, math.sqrt(aux_var))

        if ti == 0:
            p_new = distribution.normal_logpdf(x_new, self.model.init_traj_prior_mean, math.sqrt(self.model.init_traj_prior_var))
            p_old = distribution.normal_logpdf(x_old, self.model.init_traj_prior_mean, math.sqrt(self.model.init_traj_prior_var))
        else:
            p_new = distribution.normal_logpdf(x_new, rhs, math.sqrt(ssv))
            p_old = distribution.normal_logpdf(x_old, rhs, math.sqrt(ssv))
        if ti < len(self.delta_t):
            rhs2_new = self.compute_rhs(ti+1, x_new)
            ssv2 = self.traj_var[ti] * self.delta_t[ti]
            p_new += distribution.normal_logpdf(self.trajectory[ti+1], rhs2_new, math.sqrt(ssv2))
            rhs2_old = self.compute_rhs(ti+1, x_old)
            p_old += distribution.normal_logpdf(self.trajectory[ti+1], rhs2_old, math.sqrt(ssv2))
        log_r = p_new + l_new - pp_new - p_old - l_old + pp_old
        if math.log(random.random()) < log_r and not math.isnan(x_new):
            self.trajectory[ti] = x_new

    def compute_rhs(self,ti,prev_step_value):
        ## compute right-hand side based on previous time-point
        compound = self.model.all_compounds[0]
        rhs = prev_step_value * self.growth_rate * self.delta_t[ti - 1] * (1 + self.dyn_clusters[compound].pert_cluster.total_pert_trajs[self.subject_name][ti-1] + self.total_persist_pert_traj[ti-1]) + prev_step_value**2 * self.self_interact * self.delta_t[ti - 1]
        for compound in self.model.all_compounds[1:]:
            rhs += prev_step_value * self.growth_rate * self.delta_t[ti - 1] * self.dyn_clusters[compound].pert_cluster.total_pert_trajs[self.subject_name][ti-1]
        rhs = prev_step_value + rhs
        return rhs


    def calc_traj_loglike(self):
        """Calculate the trajectory likelihood with current parameter settings.

        Returns
        -------
        float
            The trajectory likelihood
        """
        rhs = self.trajectory[:-1] + self.growth_rate * (1 + self.total_persist_pert_traj[:-1]) * self.trajectory[:-1] * self.delta_t + self.self_interact * self.trajectory[:-1]**2 * self.delta_t
        for compound in self.model.all_compounds:
            rhs += self.growth_rate * self.dyn_clusters[compound].pert_cluster.total_pert_trajs[self.subject_name][:-1] * self.trajectory[:-1] * self.delta_t
        return logpdf_normal_sum(self.trajectory[1:], rhs, np.sqrt(self.traj_var * self.delta_t))


    def calc_traj_loglike_ab(self, a, b):
        """Calculate the trajectory likelihood with current parameter settings.

        Returns
        -------
        float
            The trajectory likelihood
        """
        rhs = self.trajectory[:-1] + a * (1 + self.total_persist_pert_traj[:-1]) * self.trajectory[:-1] * self.delta_t + b * self.trajectory[:-1]**2 * self.delta_t
        for compound in self.model.all_compounds:
            rhs += a * self.dyn_clusters[compound].pert_cluster.total_pert_trajs[self.subject_name][:-1] * self.trajectory[:-1] * self.delta_t
        return logpdf_normal_sum(self.trajectory[1:], rhs, np.sqrt(self.traj_var * self.delta_t))


    def calc_traj_loglike_ab_no_pert(self, a, b):
        """Calculate the trajectory likelihood with current parameter settings.

        Returns
        -------
        float
            The trajectory likelihood
        """
        rhs = self.trajectory[:-1] + a * (1 + self.total_persist_pert_traj[:-1]) * self.trajectory[:-1] * self.delta_t + b * self.trajectory[:-1]**2 * self.delta_t
        return logpdf_normal_sum(self.trajectory[1:], rhs, np.sqrt(self.traj_var * self.delta_t))


    def calc_traj_loglike_no_pert(self, compound):
        """Calculate the trajectory likelihood with no perturbation effects on the given compd.

        Returns
        -------
        float
            The trajectory likelihood
        """
        rhs = self.trajectory[:-1] + self.growth_rate * (1 + self.total_persist_pert_traj[:-1]) * self.trajectory[:-1] * self.delta_t + self.self_interact * self.trajectory[:-1]**2 * self.delta_t
        for other_compound in self.model.all_compounds:
            if other_compound != compound:
                rhs += self.dyn_clusters[other_compound].pert_cluster.total_pert_trajs[self.subject_name][:-1] * self.delta_t * self.growth_rate * self.trajectory[:-1]

        return logpdf_normal_sum(self.trajectory[1:], rhs, np.sqrt(self.traj_var * self.delta_t))


    def calc_traj_loglike_given_mc(self, mc):
        """Calculate the trajectory likelihood under a specified microbe cluster.

        Parameters
        ----------
        mc : DynCluster
            Any microbe cluster under which to evaluate my likelihood

        Returns
        -------
        float
            The trajectory likelihood
        """
        rhs = self.trajectory[:-1] + self.growth_rate * (1 + mc.pert_cluster.total_pert_trajs[self.subject_name][:-1] + self.total_persist_pert_traj[:-1]) * self.trajectory[:-1] * self.delta_t + self.self_interact * self.trajectory[:-1]**2 * self.delta_t
        return logpdf_normal_sum(self.trajectory[1:], rhs, np.sqrt(self.traj_var * self.delta_t))


    def calc_traj_loglike_given_pc_m(self, pc, compound):
        return self.calc_traj_loglike_given_pc(pc, compound)


    def calc_traj_loglike_given_all_pcs(self, compound):
        """Calculate my trajectory likelihood under all perturbation clusters.

        Parameters
        ----------
        compound : str
            The compound for which these perturbation clusters are for.

        Returns
        -------
        ndarray
            The ll for each pc in self.model.pert_cluster_sets[compound]
        """
        all_pcs = self.model.pert_cluster_sets[compound]

        lls = []
        for pc in all_pcs:
            ll = self.calc_traj_loglike_given_pc_m(pc, compound)
            lls.append(ll)

        return lls

    def calc_traj_loglike_given_all_pcs2(self, compound):
        all_pcs = self.model.pert_cluster_sets[compound]
            # will produce a list of lls paralleling this with the ll for self in that cluster

        pert_trajs = np.array([pc.total_pert_trajs[self.subject_name][:-1] for pc in all_pcs])
        rhs = (1 + pert_trajs + self.total_persist_pert_traj[:-1]) * self.delta_t * self.growth_rate * self.trajectory[:-1] + self.trajectory[:-1] + self.self_interact * self.trajectory[:-1]**2 * self.delta_t
        for other_compound in self.model.all_compounds:
            if other_compound != compound:
                rhs += self.dyn_clusters[other_compound].pert_cluster.total_pert_trajs[self.subject_name][:-1] * self.delta_t * self.growth_rate * self.trajectory[:-1]

        arg = self.trajectory[1:] - rhs
        scale = np.sqrt(self.traj_var * self.delta_t)

        lls = -arg.shape[1]/2 * math.log(2*math.pi) + (-np.log(scale) - np.square(arg/(math.sqrt(2)*scale))).sum(axis=1)

        return lls


    def calc_traj_loglike_given_all_mcs2(self, compound):

        all_mcs = self.model.dyn_cluster_sets[compound]

        pert_trajs = np.array([mc.pert_cluster.total_pert_trajs[self.subject_name][:-1] for mc in all_mcs])
        rhs = (1 + pert_trajs + self.total_persist_pert_traj[:-1]) * self.delta_t * self.growth_rate * self.trajectory[:-1] + self.trajectory[:-1] + self.self_interact * self.trajectory[:-1]**2 * self.delta_t

        # conditional on the other clusterings remaining fixed
        for other_compound in self.model.all_compounds:
            if other_compound != compound:
                rhs += self.dyn_clusters[other_compound].pert_cluster.total_pert_trajs[self.subject_name][:-1] * self.delta_t * self.growth_rate * self.trajectory[:-1]


        arg = self.trajectory[1:] - rhs
        scale = np.sqrt(self.traj_var * self.delta_t)

        lls = -arg.shape[1]/2 * math.log(2*math.pi) + (-np.log(scale) - np.square(arg/(math.sqrt(2)*scale))).sum(axis=1)

        return lls


    def calc_traj_loglike_given_pc(self, pc, pc_compound):
        """Calculate my trajectory likelihood under a specified pert cluster.

        Parameters
        ----------
        pc : PertCluster
            Any perturbation cluster under which to evaluate my likelihood

        Returns
        ------
        float
            The trajectory likelihood with that perturbation cluster
        """
        rhs = self.trajectory[:-1] \
            + self.growth_rate * (1 \
                                    + self.total_persist_pert_traj[:-1])\
                               * self.trajectory[:-1] * self.delta_t \
            + self.self_interact * self.trajectory[:-1]**2 * self.delta_t

        for compound in self.model.all_compounds:
            if compound is not pc_compound:
                rhs += self.growth_rate * self.dyn_clusters[compound].pert_cluster.total_pert_trajs[self.subject_name][:-1] * self.trajectory[:-1] * self.delta_t
            else:
                rhs += self.growth_rate * pc.total_pert_trajs[self.subject_name][:-1] * self.trajectory[:-1] * self.delta_t

        return logpdf_normal_sum(self.trajectory[1:],
                                 rhs,
                                 np.sqrt(self.traj_var * self.delta_t))


    def calc_traj_loglike_given_pert_traj(self, pert_traj, compound):
        """Calculate the trajectory likelihood under a specified pert traj.

        Parameters
        ----------
        pert_traj : ndarray
            General pert traj

        Returns
        ------
        float
            The trajectory likelihood
        """
        rhs = self.trajectory[:-1] + self.growth_rate * (1 + pert_traj[:-1] + self.total_persist_pert_traj[:-1]) * self.trajectory[:-1] * self.delta_t + self.self_interact * self.trajectory[:-1]**2 * self.delta_t
        # add other compounds:
        for cmpd in self.model.all_compounds:
            if cmpd != compound:
                other_pert_traj = self.dyn_clusters[cmpd].pert_cluster.total_pert_trajs[self.subject_name][:-1].copy()
                rhs += self.growth_rate * other_pert_traj * self.trajectory[:-1] * self.delta_t
        return logpdf_normal_sum(self.trajectory[1:], rhs, np.sqrt(self.traj_var * self.delta_t))


    def update_growth_rate(self):
        compound = self.model.all_compounds[0]
        pert_traj = self.dyn_clusters[compound].pert_cluster.total_pert_trajs[self.subject_name].copy()
        for cmpd in self.model.all_compounds[1:]:
            pert_traj += self.dyn_clusters[cmpd].pert_cluster.total_pert_trajs[self.subject_name].copy()
        m, v = fast_update_growth_rate(self.model.growth_prior_var,
                                self.trajectory,
                                pert_traj,
                                self.total_persist_pert_traj,
                                self.delta_t,
                                self.traj_var,
                                self.model.growth_prior_mean,
                                self.self_interact,)

        self.growth_rate = sample.truncated_normal(m, math.sqrt(v), 0, math.inf)


    def update_self_interact(self):
        """Update the self interact with a Gibbs step.
        """
        v = 1 / self.model.self_interact_prior_var
        v += np.sum(self.trajectory[:-1]**4 * self.delta_t / self.traj_var)
        v = 1/v

        m = self.model.self_interact_prior_mean / self.model.self_interact_prior_var
        mt = np.diff(self.trajectory) / self.delta_t
        mt -= self.growth_rate * self.trajectory[:-1]
        compound = self.model.all_compounds[0]
        mt -= (self.dyn_clusters[compound].pert_cluster.total_pert_trajs[self.subject_name][:-1] + self.total_persist_pert_traj[:-1]) * self.trajectory[:-1] * self.growth_rate
        for compound in self.model.all_compounds[1:]:
            mt -= self.dyn_clusters[compound].pert_cluster.total_pert_trajs[self.subject_name][:-1] * self.trajectory[:-1] * self.growth_rate
        mt *= self.trajectory[:-1]**2
        m += np.sum(mt * self.delta_t / self.traj_var)
        m = m * v

        self.self_interact = sample.truncated_normal(m, math.sqrt(v), -math.inf, 0)


    def update_dyn_clusters(self):
        """Update the microbe cluster for each compound.
        """
        for cmpd in self.model.all_compounds:
            self.update_dyn_cluster_compound(cmpd)

    def update_dyn_cluster_compound(self, compound):
        """Update the microbe cluster assignment with a Gibbs step.
        """
        conc_pert = self.model.pert_concentrations[compound]
        conc_dyn = self.model.microbe_concentrations[compound]

        all_pert_clusters = self.model.pert_cluster_sets[compound]
        all_dyn_clusters = self.model.dyn_cluster_sets[compound]

        n = len(self.model.time_series)

        k = len(all_dyn_clusters)

        lls = self.calc_traj_loglike_given_all_mcs2(compound)

        p = []

        # Get probabilities for being assigned to each existing microbe cluster
        # for i in range(k):
        for mc, ll in zip(all_dyn_clusters, lls):
            ni = len(mc.members)

            if self.dyn_clusters[compound] == mc:
                ni -= 1

            log_potential = self.model.log_potentials_cache[self][mc.prototype]

            if ni != 0:
                p.append(log_potential + math.log(ni) + ll)
            elif ni == 0:
                p.append(log_potential + math.log(conc_dyn) + ll)

        # Case of a new microbe cluster
        log_terms_to_sum = []

        # First go through all existing pert clusters
        m = len(all_dyn_clusters)
        m = 0
        for mc in all_dyn_clusters:
            if mc.pert_cluster.effect != 0:
                m += 1
        if len(all_pert_clusters) > 0:
            pert_lls = self.calc_traj_loglike_given_all_pcs2(compound)
        else:
            pert_lls = []
        for j, pc in enumerate(all_pert_clusters):
            mk = len(pc.members)
            tw = math.log(1-self.model.perturb_select_prior_zero_freq) + math.log(mk) - math.log(m + conc_pert) + pert_lls[j]
            log_terms_to_sum.append(tw)

        # Null effect pert cluster
        tw = math.log(self.model.perturb_select_prior_zero_freq) + self.calc_traj_loglike_no_pert(compound)
        log_terms_to_sum.append(tw)

        # Sample from prior for new pert cluster
        effect = self.model.sample_pert_effect_prior()
        r = self.model.sample_transfer_shape_prior()
        on_time = self.model.sample_pert_on_time_prior()
        duration = self.model.sample_pert_duration_prior()

        subject = self.subject
        on_time += min(subject.consumption_times[compound])

        # Calculate the perturbation trajectories for this sample of parameters
        total_pert_trajs = {}
        subject = self.subject
        total_pert_trajs[subject.name] = np.zeros(len(subject.traj_times))
        duration_days = duration * (max(subject.consumption_times[compound]) - min(subject.consumption_times[compound]))
        step_function = (subject.traj_times > on_time) & (subject.traj_times < on_time+duration_days)
        step_function = step_function.astype(np.int)
        pert_traj = step_function * (2.0 / (1.0 + np.exp(-r*subject.pharma_trajectories[compound])) - 1.0)
        total_pert_trajs[subject.name] += effect * pert_traj

        tw = math.log(1-self.model.perturb_select_prior_zero_freq) + math.log(conc_pert) - math.log(m + conc_pert) + self.calc_traj_loglike_given_pert_traj(total_pert_trajs[subject.name], compound)
        log_terms_to_sum.append(tw)

        log_potential = self.model.potential_function(self, self)
        p_new = log_potential + math.log(conc_dyn) + scipy.misc.logsumexp(log_terms_to_sum)
        p.append(p_new)

        cluster_update = sample.categorical_log(p)

        if cluster_update < k:
            self.dyn_clusters[compound].members -= {self}
            self.dyn_clusters[compound] = all_dyn_clusters[cluster_update]
            self.dyn_clusters[compound].members |= {self}

        else:
            pert_cluster_update = sample.categorical_log(log_terms_to_sum)

            if pert_cluster_update < len(all_pert_clusters):
                new_mc = DynCluster(self.model, compound)
                new_pc = all_pert_clusters[pert_cluster_update]

                self.model.dyn_cluster_sets[compound].append(new_mc)
                self.dyn_clusters[compound].members -= {self}
                self.dyn_clusters[compound] = new_mc
                new_mc.members |= {self}
                new_mc.prototype = self

                new_mc.pert_cluster = new_pc
                new_pc.members |= {new_mc}

            elif pert_cluster_update == len(all_pert_clusters):
                new_mc = DynCluster(self.model, compound)
                new_pc = self.model.null_clusters[compound]

                self.model.dyn_cluster_sets[compound].append(new_mc)
                self.dyn_clusters[compound].members -= {self}
                self.dyn_clusters[compound] = new_mc
                new_mc.members |= {self}
                new_mc.prototype = self

                new_mc.pert_cluster = new_pc
                new_pc.members |= {new_mc}

            else:
                new_mc = DynCluster(self.model, compound)
                new_pc = PertCluster(self.model, compound)
                new_pc.effect = effect
                new_pc.transfer_shape = r
                new_pc.on_time = on_time
                new_pc.durations= duration
                new_pc.calc_all_pert_trajs()

                self.model.dyn_cluster_sets[compound].append(new_mc)

                self.dyn_clusters[compound].members -= {self}
                self.dyn_clusters[compound] = new_mc
                new_mc.members |= {self}
                new_mc.prototype = self

                self.model.pert_cluster_sets[compound].append(new_pc)
                new_mc.pert_cluster = new_pc
                new_pc.members |= {new_mc}

        self.model.purge_empty_clusters()

    def calc_persist_traj(self):
        """Calculate the persistent effect trajectory.
        """
        # Find last compound
        # The persistent effect starts after the last compound administration
        last_dose = -math.inf
        for compound in self.model.all_compounds:
            last_dose_compound = max(self.subject.consumption_times[compound])
            if last_dose_compound > last_dose:
                last_dose = last_dose_compound

        persist_step_function = (self.subject.traj_times > last_dose + self.persist_on_time) & (self.subject.traj_times < last_dose + self.persist_on_time + self.persist_duration)
        persist_step_function = persist_step_function.astype(np.int)

        persist_pert_traj = persist_step_function * self.persist_effect

        self.persist_pert_traj = persist_step_function
        self.total_persist_pert_traj = persist_pert_traj


    def update_persist_select(self):
        """Update the persistent effect selection with a Gibbs step.
        """
        p = [0, 0]
        p[0] = math.log(self.model.persist_perturb_select_prior_zero_freq)

        y = []  # observation vector
        X = []  # design matrix
        S_diag = []  # diagonal of covariance matrix

        # Form matrices for regression
        dp = self
        obs = (dp.trajectory[1:] - dp.trajectory[:-1]) / dp.delta_t - dp.growth_rate * dp.trajectory[:-1] - dp.self_interact * dp.trajectory[:-1]**2
        for cmpd in self.model.all_compounds:
            obs -= dp.growth_rate * dp.trajectory[:-1] * dp.dyn_clusters[cmpd].pert_cluster.total_pert_trajs[dp.subject_name][:-1]
        y = np.hstack((y, obs))
        design = dp.growth_rate * dp.trajectory[:-1] * self.persist_pert_traj[:-1]
        X = np.hstack((X, design))
        S_diag = np.hstack((S_diag, dp.traj_var / dp.delta_t))

        S_0 = self.model.perturb_effect_prior_var
        S_0_m1 = 1 / self.model.perturb_effect_prior_var
        S_m1 = 1 / S_diag
        S_3_m1 = np.dot(X * S_m1, X) + S_0_m1
        S_3 = 1 / S_3_m1
        mu_3 = S_3 * np.dot(X * S_m1, y)
        ll = 0.5 * math.log(S_3) - 0.5 * math.log(S_0) + 0.5 * mu_3 * S_3_m1 * mu_3

        p[1] = math.log(1-self.model.persist_perturb_select_prior_zero_freq) + ll
        u = sample.categorical_log(p)
        if u == 0:
            self.persist_effect = 0
            self.persist_select = 0
        elif u == 1:
            self.persist_select = 1
        self.calc_persist_traj()


    def update_persist_effect(self):
        """Update the persistent effect with a Gibbs step.
        """
        if self.persist_select == 1:
            y = []
            X = []
            S_diag = []

            dp = self
            obs = (dp.trajectory[1:] - dp.trajectory[:-1]) / dp.delta_t - dp.growth_rate * dp.trajectory[:-1] - dp.self_interact * dp.trajectory[:-1]**2
            for cmpd in self.model.all_compounds:
                obs -= dp.growth_rate * dp.trajectory[:-1] * dp.dyn_clusters[cmpd].pert_cluster.total_pert_trajs[dp.subject_name][:-1]
            y = np.hstack((y, obs))
            design = dp.growth_rate * dp.trajectory[:-1] * self.persist_pert_traj[:-1]
            X = np.hstack((X, design))
            S_diag = np.hstack((S_diag, dp.traj_var / dp.delta_t))

            x = X

            precis_matrix_diag = 1/S_diag
            precis = np.dot(x * precis_matrix_diag, x) + 1 / self.model.perturb_effect_prior_var
            mean = np.dot(x * precis_matrix_diag, y)
            mean *= 1/precis

            self.persist_effect = random.gauss(mean, math.sqrt(1/precis))
            self.calc_persist_traj()


    def update_persist_duration(self):
        """Update the persistent duration parameter with a MH step.
        """
        duration_0 = self.persist_duration
        p_0 = distribution.truncated_normal_logpdf(duration_0, self.model.persist_duration_mean, math.sqrt(self.model.persist_duration_var), self.model.persist_duration_min, math.inf)
        l_0 = 0
        l_0 += self.calc_traj_loglike()

        duration_prop = sample.uniform(5, 35)
        self.persist_duration = duration_prop
        self.calc_persist_traj()

        p_prop = distribution.truncated_normal_logpdf(duration_prop, self.model.persist_duration_mean, math.sqrt(self.model.persist_duration_var), self.model.persist_duration_min, math.inf)
        l_prop = 0
        l_prop += self.calc_traj_loglike()
        log_r = p_prop + l_prop - p_0 - l_0
        if math.log(random.random()) >= log_r:
            self.persist_duration = duration_0
            self.calc_persist_traj()


    def save_current_trajs(self):
        """Add the trajectory values.

        Used for adaptive trajectory proposal to prevent slow lookup from file.
        """
        self.trajectory_history.append(self.trajectory.copy())
        self.aux_trajectory_history.append(self.aux_trajectory.copy())


    def save_current_iteration(self, save_trajs=True):
        """Save the current values to object storage.
        """
        if save_trajs:
            self.trajectory_storage.append(self.trajectory.copy())
            self.aux_trajectory_storage.append(self.aux_trajectory.copy())

        self.growth_rate_storage.append(self.growth_rate)
        self.self_interact_storage.append(self.self_interact)
        self.var_c_storage.append(self.var_c)

        for compound in self.dyn_clusters.keys():
            self.var_c_pert_storage[compound].append(self.var_c_pert[compound])
            self.effect_storage[compound].append(self.dyn_clusters[compound].pert_cluster.effect)
            self.pert_traj_storage[compound].append(self.dyn_clusters[compound].pert_cluster.total_pert_trajs[self.subject_name])
            self.transfer_shape_storage[compound].append(self.dyn_clusters[compound].pert_cluster.transfer_shape)
            self.on_time_storage[compound].append(self.dyn_clusters[compound].pert_cluster.on_time)
            self.duration_storage[compound].append(self.dyn_clusters[compound].pert_cluster.duration)
            self.dyn_cluster_storage[compound].append(id(self.dyn_clusters[compound]))
            self.pert_cluster_storage[compound].append(id(self.dyn_clusters[compound].pert_cluster))

        self.persist_effect_storage.append(self.persist_effect)
        self.persist_select_storage.append(self.persist_select)
        self.persist_duration_storage.append(self.persist_duration)
        self.persist_pert_traj_storage.append(self.total_persist_pert_traj)


class Model:
    """Main model handling clusters, priors, and instantiation.

    Attributes
    ----------
    all_compounds : list of str
        All compounds administered to any subject
    all_taxons : list of load_data.Taxon
        All taxons being analyzed by the model in any subject
    subjects : dict
        Keys are subject names. Values are Subject objects.
    time_series : list of TimeSeries
        Each time-series
    """
    def __init__(self, dataset, settings, init_clustering='individual'):
        """Load data into model.

        Parameters
        ----------
        dataset : load_data.Dataset
            The loaded dataset. As returned by load_data.load_dataset().
        settings : dict
            Config settings
        init_clustering : {'individual', 'together'}, optional ('individual')
            How to initialize the clustering. There are two options implemented
            in this function. 'individual' places each time-series in its own
            microbe cluster, and each microbe cluster in its own perturbation
            cluster.
            'together' places all time-series in a single microbe
            cluster, and that microbe cluster in the null response cluster,
            with no initial perturbation clusters.
        """
        self.settings = settings

        # get correct distances
        self.phylogenetic_distances = dataset.dist_matrix

        # Prior hyperparameters and some initial conditions
            # Growth rate
        self.growth_prior_mean = .1
        self.growth_prior_var = 100

            # Self interaction
        self.self_interact_prior_mean = -.02
        self.self_interact_prior_var = 0.04

        self.growth_prior_var = 100
        self.self_interact_prior_var = 1

            # Perturb effect (mean = 0)
        self.perturb_effect_prior_var = 100

            # Perturb select
        self.perturb_select_prior_alpha = 1
        self.perturb_select_prior_beta = 50
        self.perturb_select_prior_zero_freq = 0.5

            # Persistent effect
        self.persist_select_prior_alpha = 1
        self.persist_select_prior_beta = 3000
        self.persist_perturb_select_prior_zero_freq = 1.0 - 1e-5

        self.persist_duration_mean = 30
        self.persist_duration_var = 50
        self.persist_duration_min = 10

            # Trajectory
        self.init_traj_prior_mean = 100
        self.init_traj_prior_var = 10e6

            # Trajectory zero inflation
        self.zero_inflation_prior_alpha = 1
        self.zero_inflation_prior_beta = 1
        self.zero_inflation_prior_zero_freq = 0.5

            # Auxiliary trajectory coupling variance
        self.var_q_b = 0.01
        self.var_q_c = 0.01

            # Auxiliary trajectory bounds
        self.aux_traj_lower_bound = 0.1
        self.aux_traj_lower_bound = 0.001
        self.aux_traj_upper_bound = 1e5

            # Process variance
        self.process_var_prior_a = 1e-5
        self.process_var_prior_b = 1

            # Perturb shape and duration
        self.perturb_transfer_rate = 5
        self.perturb_on_time_mean = 0
        self.perturb_on_time_var = 20
        self.perturb_on_time_min = 0
        self.perturb_on_time_max = 20
        self.perturb_duration_mean = 1.25
        self.perturb_duration_var = 0.25
        self.perturb_duration_min = 0.4
        self.perturb_duration_max = 1.25

            # Phylogenetic clustering
        self.zeta0 = 3.33
        self.zeta1 = 1.82

            # Dirichlet process concentration
        self.a1 = {'microbe': 1e-3, 'perturbation': 1e-3}
        self.a2 = {'microbe': 1e-3, 'perturbation': 1e-3}

            # Pharmacokinetics
        self.k_mean = 1
        self.k_var = 1
        self.k_min = 0.25
        self.k_max = math.inf

        # Lambda functions to generate samples from the prior
            # Perturbation
        self.sample_pert_effect_prior = lambda : random.gauss(0, math.sqrt(self.perturb_effect_prior_var))
        self.sample_transfer_shape_prior = lambda : random.expovariate(self.perturb_transfer_rate)
        self.sample_pert_on_time_prior = lambda : sample.truncated_normal(self.perturb_on_time_mean, math.sqrt(self.perturb_on_time_var), self.perturb_on_time_min, self.perturb_on_time_max)
        self.sample_pert_duration_prior = lambda : sample.truncated_normal(self.perturb_duration_mean, math.sqrt(self.perturb_duration_var), self.perturb_duration_min, self.perturb_duration_max)

        self.sample_growth_rate_prior = lambda : random.gauss(self.growth_prior_mean, math.sqrt(self.growth_prior_var))
        self.sample_self_interact_prior = lambda : random.gauss(self.self_interact_prior_mean, math.sqrt(self.self_interact_prior_var))

        self.sample_growth_rate_prior = lambda : sample.truncated_normal(self.growth_prior_mean, math.sqrt(self.growth_prior_var), 0, math.inf)
        self.sample_self_interact_prior = lambda : sample.truncated_normal(self.self_interact_prior_mean, math.sqrt(self.self_interact_prior_var), -math.inf, 0)


        # Lambda functions to evaluate the prior density at a value
            # Perturbation
        self.pert_effect_prior = lambda x : distribution.normal_logpdf(x, 0, math.sqrt(self.perturb_effect_prior_var))
        self.transfer_shape_prior = lambda x : math.log(self.perturb_transfer_rate) - self.perturb_transfer_rate * x if x > 0 else -math.inf
        self.pert_on_time_prior = lambda x : distribution.truncated_normal_logpdf(x, self.perturb_on_time_mean, math.sqrt(self.perturb_on_time_var), self.perturb_on_time_min, self.perturb_on_time_max)
        self.pert_duration_prior = lambda x : distribution.truncated_normal_logpdf(x, self.perturb_duration_mean, math.sqrt(self.perturb_duration_var), self.perturb_duration_min, self.perturb_duration_max)

            # Dynamics
        self.growth_rate_prior = lambda x : distribution.truncated_normal_logpdf(x, self.growth_prior_mean, math.sqrt(self.growth_prior_var), 0, math.inf)
        self.self_interact_prior = lambda x : distribution.truncated_normal_logpdf(x, self.self_interact_prior_mean, math.sqrt(self.self_interact_prior_var), -math.inf, 0)

            # Pharmacokinetics
        self.pharma_k_prior = lambda x: distribution.truncated_normal_logpdf(x, self.k_mean, math.sqrt(self.k_var), self.k_min, self.k_max)

        # Proposal variances
        self.on_time_prop_var = 9
        self.transfer_shape_prop_var = 0.01
        self.duration_prop_var = 0.25

        self.all_compounds = []
        self.all_taxons = []

        # Instantiate time-series and subject objects from the dataset
        self.subjects = {}
        self.time_series = []
        for data_subj_key in sorted(dataset.subjects.keys()):
        # for data_subj in dataset.subjects.values():
            data_subj = dataset.subjects[data_subj_key]
            if len(data_subj.counts.keys()) > 0:
                new_subj = Subject(data_subj.name,
                                   data_subj.times,
                                   data_subj.consumption_times,
                                   data_subj.doses,
                                   self)
                new_subj.create_traj_times(self.settings['spacing'])
                new_subj.create_on_off_periods()
                new_subj.calc_pharma_trajectory()
                self.subjects[data_subj.name] = new_subj

                for k in sorted(data_subj.doses.keys()):
                    self.all_compounds.append(k)

                for otu in sorted(data_subj.rel_abundances.keys()):
                    new_ts = TimeSeries(otu, new_subj,
                                        data_subj.times,
                                        data_subj.rel_abundances[otu],
                                        data_subj.counts[otu],
                                        data_subj.totals, self)

                    self.time_series.append(new_ts)

        self.all_compounds = list(set(self.all_compounds))

        # Instantiate one null cluster for each compound
        self.null_clusters = {}
        for compound in self.all_compounds:
            self.null_clusters[compound] = NullPertCluster(self, compound)

        if init_clustering == 'individual':
            # Instantiate one perturbation and microbe cluster for each timeseries
            self.dyn_cluster_sets = defaultdict(list)
            self.pert_cluster_sets = defaultdict(list)
            for ts in self.time_series:
                for compound in self.all_compounds:
                    new_mc = DynCluster(self, compound)
                    new_pc = PertCluster(self, compound)

                    self.dyn_cluster_sets[compound].append(new_mc)
                    ts.dyn_clusters[compound] = new_mc
                    new_mc.members |= {ts}
                    new_mc.prototype = ts

                    self.pert_cluster_sets[compound].append(new_pc)
                    new_mc.pert_cluster = new_pc
                    new_pc.members |= {ts.dyn_clusters[compound]}

            # Convert from defaultdict back to normal dict
            self.dyn_cluster_sets = dict(self.dyn_cluster_sets)
            self.pert_cluster_sets = dict(self.pert_cluster_sets)

            # Initialize time-series values
            for ts in self.time_series:
                ts.set_initial_values()

        elif init_clustering == 'together':
            # Instantiate one cluster that all time-series will go in
            self.dyn_cluster_sets = defaultdict(list)
            self.pert_cluster_sets = defaultdict(list)

            for compound in self.all_compounds:
                first_mc = DynCluster(self, compound)
                self.dyn_cluster_sets[compound].append(first_mc)
                self.pert_cluster_sets[compound] = []
                for ts in self.time_series:
                    ts.dyn_clusters[compound] = first_mc
                    first_mc.members |= {ts}
                    first_mc.prototype = ts
                null = self.null_clusters[compound]
                first_mc.pert_cluster = null
                null.members |= {first_mc}

            # Convert from defaultdict back to normal dict
            self.dyn_cluster_sets = dict(self.dyn_cluster_sets)
            self.pert_cluster_sets = dict(self.pert_cluster_sets)

            # Initialize time-series values
            for ts in self.time_series:
                ts.set_initial_values(effect_setting=0)

        # Initialize concentration parameters at 1
        self.pert_concentrations = {}
        self.microbe_concentrations = {}
        for compound in self.all_compounds:
            self.pert_concentrations[compound] = 1
            self.microbe_concentrations[compound] = 1

        # Initialize empty object storage for model parameters
        self.microbe_concentrations_storage = {}
        self.pert_concentrations_storage = {}
        for compound in self.all_compounds:
            self.microbe_concentrations_storage[compound] = []
            self.pert_concentrations_storage[compound] = []

        self.num_microbe_clusters_storage = {}
        self.num_pert_clusters_storage = {}
        for compound in self.all_compounds:
            self.num_microbe_clusters_storage[compound] = []
            self.num_pert_clusters_storage[compound] = []

        self.perturb_select_prior_zero_freq_storage = []
        self.persist_perturb_select_prior_zero_freq_storage = []

        self.cache_potential_lookups()

        if len(self.all_compounds) > 1:
            self.perturb_on_time_max = 10


    def purge_empty_clusters(self):
        """Remove from the model any empty perturbation or microbe clusters.
        """
        # Remove empty microbe clusters from the pert cluster member sets
        # Otherwise, a pert cluster could appear to have members even though
        # they were only empty microbe clusters
        for compound in self.all_compounds:
            for pc in self.pert_cluster_sets[compound] + [self.null_clusters[compound]]:
                empty_mcs = set([])
                for mc in pc.members:
                    if len(mc.members) == 0:
                        empty_mcs |= {mc}
                pc.members -= empty_mcs

            # Remove empty microbe clusters and pert clusters from the model lists
            self.dyn_cluster_sets[compound] = [mc for mc in self.dyn_cluster_sets[compound] if len(mc.members) > 0]
            self.pert_cluster_sets[compound] = [pc for pc in self.pert_cluster_sets[compound] if len(pc.members) > 0]


    def cache_potential_lookups(self):
        log_potentials = {}
        for otu1 in self.time_series:
            log_potentials[otu1] = {}
            for otu2 in self.time_series:
                log_potentials[otu1][otu2] = self.potential_function(otu1, otu2)
        self.log_potentials_cache = log_potentials


    def potential_function(self, otu1, otu2, log=True):
        """Calculate the value of the phylogenetic potential function.

        Parameters
        ----------
        otu1 : TimeSeries
            Species 1
        otu2 : TimeSeries
            Species 2
        log : bool, optional (True)
            If True, return the logarithm of the potential function. If False,
            return the potential function itself.

        Returns
        -------
        float
            The potential
        """
        try:
            sp1 = otu1.species
            sp2 = otu2.species
            d = self.phylogenetic_distances[sp1][sp2]
        except KeyError:
            print('Warning: {}, {} not found in tree'.format(sp1, sp2))
            d = 1

        if not log:
            return math.exp(-self.zeta0 * d**2 + self.zeta1)
        else:
            return -self.zeta0 * d +self.zeta1


    def update_pert_select_prior(self):
        """Update the perturbation effect selection prior.
        """
        n = 0
        sum_x = 0

        for compound in self.all_compounds:
            for mc in self.dyn_cluster_sets[compound]:
                n += 1
                sum_x += 1 if mc.pert_cluster.effect != 0 else 0

        p = random.betavariate(self.perturb_select_prior_alpha + sum_x,
                               self.perturb_select_prior_beta + n - sum_x)

        self.perturb_select_prior_zero_freq = 1.0 - p


    def update_persist_prior(self):
        """Update the persistent effect selection prior.
        """
        n = 0
        sum_x = 0

        for dp in self.time_series:
            n += 1
            sum_x += 1 if dp.persist_select != 0 else 0

        p = random.betavariate(self.persist_select_prior_alpha + sum_x,
                               self.persist_select_prior_beta + n - sum_x)

        self.persist_perturb_select_prior_zero_freq = 1.0 - p


    def update_zero_inflation_prior(self):
        """Update the auxiliary trajectory zero-inflation prior.
        """
        n = 0
        sum_x = 0
        for dp in self.time_series:
            for i, q in enumerate(dp.aux_trajectory):
                possible_zero = dp.zero_inflatable[i]
                if possible_zero:
                    n += 1
                    if q != 0:
                        sum_x += 1

        p = random.betavariate(self.zero_inflation_prior_alpha + sum_x , self.zero_inflation_prior_beta + n -sum_x)
        self.zero_inflation_prior_zero_freq = 1.0 - p


    def update_concentrations(self):
        """Update the Dirichlet process concentration parameters.
        """
        for compound in self.all_compounds:
            self.update_concentration('microbe', compound, 20)
            self.update_concentration('perturbation', compound, 20)


    def update_concentration(self, level, compound, iterations):
        """Update a single Dirichlet process concentration.

        Uses an auxiliary variable technique [1]_

        .. [1] Escobar, Michael D., and Mike West. "Bayesian density estimation
            and inference using mixtures." Journal of the American Statistical
            Association 90.430 (1995): 577-588.

        Parameters
        ----------
        level : {'microbe', 'perturbation'}
            Which level of clustering to update the concentration for
        compound : str
            Compound clustering for which to update concentration
        iterations : int
            Number of steps
        """
        if level == 'microbe':
            alpha = self.microbe_concentrations[compound]
            n_data = len(self.time_series)
            n_clusters = len(self.dyn_cluster_sets[compound])

        elif level == 'perturbation':
            # self.pert_concentrations[compound] = 0.01
            # return
            alpha = self.pert_concentrations[compound]
            n_data = len(self.dyn_cluster_sets[compound])
            for mc in self.dyn_cluster_sets[compound]:
                if mc.pert_cluster.effect == 0:
                    n_data -= 1
            n_clusters = len(self.pert_cluster_sets[compound])

        eta = 1

        if n_clusters <= 1 or n_data < 1:
                # reject update
            alpha = self.a1[level] / self.a2[level]
            if alpha == 0:
                return

        else:
            for i in range(iterations):
                eta = random.betavariate(alpha+1, n_data)

                pi_eta = [0, 1]
                pi_eta[0] = (self.a1[level] + n_clusters - 1) / (self.a2[level] - math.log(eta))
                pi_eta[1] = n_data

                if sample.categorical_log(np.log(pi_eta)) == 0:
                    alpha = random.gammavariate(self.a1[level] + n_clusters, 1 / (self.a2[level] - math.log(eta)))
                else:
                    alpha = random.gammavariate(self.a1[level] + n_clusters - 1, 1 / (self.a2[level] - math.log(eta)))

        if level == 'microbe':
            self.microbe_concentrations[compound] = alpha

        elif level == 'perturbation':
            self.pert_concentrations[compound] = alpha



    def save_current_iteration(self):
        """Save the values at the current iteration to object storage.
        """
        for compound in self.all_compounds:
            self.microbe_concentrations_storage[compound].append(self.microbe_concentrations[compound])
            self.pert_concentrations_storage[compound].append(self.pert_concentrations[compound])

            self.num_microbe_clusters_storage[compound].append(len(self.dyn_cluster_sets[compound]))
            self.num_pert_clusters_storage[compound].append(len(self.pert_cluster_sets[compound]))

        self.perturb_select_prior_zero_freq_storage.append(self.perturb_select_prior_zero_freq)
        self.persist_perturb_select_prior_zero_freq_storage.append(self.persist_perturb_select_prior_zero_freq)


class Sampler:
    """Handles execution and storage of the MCMC sampling chain.

    Attributes
    ----------
    model : Model
        The main model object for which MCMC samples are being generated.
    mcmc_chunk_size : int
        For all model parameters (including time-series, etc.), MCMC samples are
        saved in memory at each iteration. Every mcmc_chunk_size iterations,
        the samples are saved to the output file. Larger values of this
        setting enable slightly faster performance at the cost of higher memory
        usage.
    mcmc_chunk_progress : int
        The current number of non-burnin iterations performed since the last
        dump of the MCMC samples to the output file.
    """
    def __init__(self, model):
        self.model = model

        self.mcmc_chunk_size = self.model.settings['mcmc_chunk_size']
        self.mcmc_chunk_progress = 0

        self.set_up_trace_file()

        self.times_per_iteration = []


    def run_MCMC_iteration(self, i):
        """Run one iteration of the MCMC chain.

        Parameters
        ----------
        i : int
            Iteration number
        """
        if self.model.settings['verbose']:
            print(i)

        # Each iteration, randomize the order in which the parameters are
        # updated
        blocks = ['trajectory',
                  'dynamics',
                  'microbe_cluster',
                  'pert_cluster',
                  'perturbation',
                  'subject', 'model']
        random.shuffle(blocks)

        for block in blocks:
            if block == 'trajectory':
                if i % 2 == 0:
                    ## Update trajectory and auxiliary trajectory
                    # If parallel workers are not selected, update with loop over all
                    if not self.model.settings['parallel_workers']:
                        for ts in self.model.time_series:
                            ts.update_both_trajectories()

                    # If parallel workers are selected, distribute:
                    else:
                        n_processes = self.model.settings['parallel_workers']
                        print('parallel workers:{}'.format(n_processes))
                        arg_queue = mp.Queue()
                        out_queue = mp.Queue()
                        workers = [mp.Process(target=worker, args=(self.model, arg_queue, out_queue)) for j in range(n_processes)]

                        for j in range(len(self.model.time_series)):
                            # Send a seed to the worker using random.random()
                            # Ensures reproduciblity with the same initial seeds, as long
                            # as the number of workers is the same
                            s = random.random()
                            arg_queue.put((j, s))
                        for j in range(n_processes):
                            arg_queue.put(('end', ''))
                        for work in workers:
                            work.start()
                        res_lst = []
                        for j in range(len(self.model.time_series)):
                            res_lst.append(out_queue.get())
                        for work in workers:
                            work.join()
                        for result in res_lst:
                            tsi = result[0]
                            self.model.time_series[tsi].trajectory = result[1]
                            self.model.time_series[tsi].aux_trajectory = result[2]
                            self.model.time_series[tsi].both_accepts = result[3]
                            self.model.time_series[tsi].both_rejects = result[4]
                            self.model.time_series[tsi].traj_var = result[5]

                        for work in workers:
                            work.terminate()

                        arg_queue.close()
                        out_queue.close()

            elif block == 'dynamics':
                # Update other time-series specific parameters
                for ts in self.model.time_series:
                    ts.update_growth_rate()
                    ts.update_self_interact()
                    ts.update_var_parameters()

                    if i > 1:
                        ts.update_persist_select()
                        ts.update_persist_duration()
                        ts.update_persist_effect()

            elif block == 'microbe_cluster':
                if i > 1:
                    if i % 10 == 0:
                        for ts in self.model.time_series:
                            ts.update_dyn_clusters()

            elif block == 'pert_cluster':
                # Update microbe cluster and perturbation cluster parameters
                for compound in self.model.all_compounds:
                    for mc in self.model.dyn_cluster_sets[compound]:
                        mc.update_prototype()
                        if i > 20:
                            mc.update_pert_cluster2()


            elif block == 'perturbation':
                for compound in self.model.all_compounds:
                    for pc in self.model.pert_cluster_sets[compound]:
                        pc.update_effects()
                        pc.update_transfer_shapes()
                        pc.update_on_times()
                        pc.update_durations()


            elif block == 'subject':
                # Update subject specific parameters
                for subject in self.model.subjects.values():
                    subject.update_k()

            elif block == 'model':
                # Update global model specific parameters
                self.model.update_pert_select_prior()
                self.model.update_persist_prior()
                self.model.update_zero_inflation_prior()

                if i > 1:
                    self.model.update_concentrations()

                for compound in self.model.null_clusters.keys():
                    self.model.null_clusters[compound].update_on_times()
                    self.model.null_clusters[compound].update_durations()
                    self.model.null_clusters[compound].update_transfer_shapes()
                    # The overridden null cluster samplers just set to prior sample,
                    # must manually update perturbation traj
                    self.model.null_clusters[compound].calc_all_pert_trajs()

        self.mcmc_chunk_progress += 1
        for ts in self.model.time_series:
            ts.save_current_iteration()
            if i < 0.75 * self.model.settings['num_burnin']:
                ts.save_current_trajs()

        for subject in self.model.subjects.values():
            subject.save_current_iteration()

        self.model.save_current_iteration()

        # Adaptive MH proposals every 200 iterations during the first 75% burn in only
        if (i + 1) % 200 == 0 and i < 0.75 * self.model.settings['num_burnin']:
            for subj in self.model.subjects.values():
                subj.adapt_k_proposals()

            for ts in self.model.time_series:
                # Only update after starting to get some samples in
                if i > 101:
                    ts.adapt_traj_proposals()

        # Save samples to disk at the specified interval or at the end of the chain
        if (self.mcmc_chunk_progress == self.mcmc_chunk_size) \
           or (i+1 == self.model.settings['num_mcmc_iterations']):
            self.append_samples_to_hdf5()


    def append_samples_to_hdf5(self):
        """Append the current samples to the file.
        """
        filename = os.path.join(self.model.settings['output_directory'],
                                self.model.settings['mcmc_name'])
        mcmc_samples = h5py.File(filename, 'r+')

        g = mcmc_samples['model']['concentration_microbe']
        for compound in self.model.all_compounds:
            d = g[compound]
            d.resize(d.shape[0] + self.mcmc_chunk_progress, axis=0)
            d[-self.mcmc_chunk_progress:] = self.model.microbe_concentrations_storage[compound]
            self.model.microbe_concentrations_storage[compound] = []

        g = mcmc_samples['model']['concentration_pert']
        for compound in self.model.all_compounds:
            d = g[compound]
            d.resize(d.shape[0] + self.mcmc_chunk_progress, axis=0)
            d[-self.mcmc_chunk_progress:] = self.model.pert_concentrations_storage[compound]
            self.model.pert_concentrations_storage[compound] = []

        g = mcmc_samples['model']['num_microbe_clusters']
        for compound in self.model.all_compounds:
            d = g[compound]
            d.resize(d.shape[0] + self.mcmc_chunk_progress, axis=0)
            d[-self.mcmc_chunk_progress:] = self.model.num_microbe_clusters_storage[compound]
            self.model.num_microbe_clusters_storage[compound] = []

        g = mcmc_samples['model']['num_pert_clusters']
        for compound in self.model.all_compounds:
            d = g[compound]
            d.resize(d.shape[0] + self.mcmc_chunk_progress, axis=0)
            d[-self.mcmc_chunk_progress:] = self.model.num_pert_clusters_storage[compound]
            self.model.num_pert_clusters_storage[compound] = []

        model_group = mcmc_samples['model']
        scalar_variables = ('perturb_select_prior_zero_freq', 'persist_perturb_select_prior_zero_freq',)

        for s in scalar_variables:
            d = model_group[s]
            d.resize(d.shape[0] + self.mcmc_chunk_progress, axis=0)
            d[-self.mcmc_chunk_progress:] = getattr(self.model, s + '_storage')
            setattr(self.model, s + '_storage', [])

        for ts in self.model.time_series:
            ts_group = mcmc_samples['time_series'][ts.subject_name + ts.otu]

            vector_variables = ('trajectory', 'aux_trajectory', 'persist_pert_traj')
            scalar_variables = ('growth_rate', 'self_interact', 'var_c', 'persist_select', 'persist_effect')

            for v in vector_variables:
                d = ts_group[v]
                d.resize(d.shape[0] + self.mcmc_chunk_progress, axis=0)
                d[-self.mcmc_chunk_progress:,:] = getattr(ts, v + '_storage')
                setattr(ts, v + '_storage', [])

            for s in scalar_variables:
                d = ts_group[s]
                d.resize(d.shape[0] + self.mcmc_chunk_progress, axis=0)
                d[-self.mcmc_chunk_progress:] = getattr(ts, s + '_storage')
                setattr(ts, s + '_storage', [])

            pert_vector_variables = ('pert_traj',)
            pert_scalar_variables = ('effect', 'transfer_shape', 'on_time', 'duration', 'dyn_cluster', 'pert_cluster', 'var_c_pert')

            for v in pert_vector_variables:
                for compound in self.model.all_compounds:
                    d = ts_group[v][compound]
                    d.resize(d.shape[0] + self.mcmc_chunk_progress, axis=0)
                    d[-self.mcmc_chunk_progress:,:] = getattr(ts, v + '_storage')[compound]
                    getattr(ts, v + '_storage').update({compound: []})

            for s in pert_scalar_variables:
                for compound in self.model.all_compounds:
                    d = ts_group[s][compound]
                    d.resize(d.shape[0] + self.mcmc_chunk_progress, axis=0)
                    d[-self.mcmc_chunk_progress:] = getattr(ts, s + '_storage')[compound]
                    getattr(ts, s + '_storage').update({compound: []})


        for subject in self.model.subjects.values():
            subj_group = mcmc_samples['subjects'][subject.name]

            for compound in subject.pharma_trajectories.keys():
                d = subj_group['pharma_traj'][compound]
                d.resize(d.shape[0] + self.mcmc_chunk_progress, axis=0)
                d[-self.mcmc_chunk_progress:,:] = subject.pharma_storage[compound]
                subject.pharma_storage[compound] = []

                d = subj_group['k'][compound]
                d.resize(d.shape[0] + self.mcmc_chunk_progress, axis=0)
                d[-self.mcmc_chunk_progress:] = subject.k_storage[compound]
                subject.k_storage[compound] = []

        mcmc_samples.close()
        self.mcmc_chunk_progress = 0


    def set_up_trace_file(self):
        """Instantiate the HDF5 file for storing MCMC samples.
        """
        fn = self.model.settings['mcmc_name']
        filename = os.path.join(self.model.settings['output_directory'], fn)

        if os.path.isfile(filename):
            if not self.model.settings['overwrite']:
                print('Warning: MCMC samples file already exists.')
                print('Exiting to avoid overwriting data...')
                exit()

        mcmc_samples = h5py.File(filename, 'w')
        mcmc_samples.create_group('model')
        mcmc_samples.create_group('time_series')
        mcmc_samples.create_group('subjects')

        ## Attributes are used to store fixed information about this run
        # Config settings
        model_group = mcmc_samples['model']
        for k, v in self.model.settings.items():
            print(k, v)
            if type(v) is str:
                model_group.attrs.create(k, v, dtype='<S999')

            elif type(v) in [float, int]:
                model_group.attrs.create(k, np.array(v))

            elif type(v) is list:
                try:
                    if type(v[0]) is str:
                        try:
                            model_group.attrs.create(k, v, dtype='<S999')
                        except RuntimeError:
                            pass
                    else:
                        model_group.attrs.create(k, np.array(v))
                except TypeError:
                    model_group.attrs.create(k, repr(v), dtype='<S999')
                except IndexError:
                    pass

        # Hyperparameters
        hyperparameters = ['growth_prior_mean',
                           'growth_prior_var',
                           'self_interact_prior_mean',
                           'self_interact_prior_var',
                           'perturb_effect_prior_var',
                           'perturb_select_prior_alpha',
                           'perturb_select_prior_beta',
                           'perturb_transfer_rate',
                           'perturb_on_time_mean',
                           'perturb_on_time_var',
                           'perturb_on_time_min',
                           'perturb_on_time_max',
                           'perturb_duration_mean',
                           'perturb_duration_var',
                           'perturb_duration_min',
                           'perturb_duration_max',
                           'persist_select_prior_alpha',
                           'persist_select_prior_beta',
                           'zeta0',
                           'zeta1',
                           'k_mean',
                           'k_var',
                           'k_min',
                           'k_max',]

        for param in hyperparameters:
            model_group.attrs.create(param, np.array(getattr(self.model, param)))

        model_group.attrs.create('a1_microbe', np.array(self.model.a1['microbe']))
        model_group.attrs.create('a2_microbe', np.array(self.model.a2['microbe']))
        model_group.attrs.create('a1_perturbation', np.array(self.model.a1['perturbation']))
        model_group.attrs.create('a2_perturbation', np.array(self.model.a2['perturbation']))

        model_group.attrs.create('compounds', np.array(self.model.all_compounds), dtype='<S999')

        # Set up traces for model parameters
        mcmc_samples['model'].create_dataset('perturb_select_prior_zero_freq',
                                             (0,),
                                             maxshape=(100000,),
                                             dtype='f')

        mcmc_samples['model'].create_dataset('persist_perturb_select_prior_zero_freq',
                                             (0,),
                                             maxshape=(100000,),
                                             dtype='f')


        # For compound-specific parameters including concentrations, there is
        # one dataset per compound for each parameter
        mcmc_samples['model'].create_group('concentration_microbe')
        for compound in self.model.all_compounds:
            mcmc_samples['model']['concentration_microbe'].create_dataset(compound, (0,), maxshape=(100000,), dtype='f')

        mcmc_samples['model'].create_group('concentration_pert')
        for compound in self.model.all_compounds:
            mcmc_samples['model']['concentration_pert'].create_dataset(compound, (0,), maxshape=(100000,), dtype='f')

        mcmc_samples['model'].create_group('num_microbe_clusters')
        for compound in self.model.all_compounds:
            mcmc_samples['model']['num_microbe_clusters'].create_dataset(compound, (0,), maxshape=(100000,), dtype='f')

        mcmc_samples['model'].create_group('num_pert_clusters')
        for compound in self.model.all_compounds:
            mcmc_samples['model']['num_pert_clusters'].create_dataset(compound, (0,), maxshape=(100000,), dtype='f')

        for ts in self.model.time_series:
            traj_len = len(ts.trajectory)
            mcmc_samples['time_series'].create_group(ts.subject_name + ts.otu)
            ts_group = mcmc_samples['time_series'][ts.subject_name + ts.otu]

            ts_group.attrs.create('subject_name', ts.subject_name, dtype='<S999')
            ts_group.attrs.create('otu', ts.otu, dtype='<S999')
            ts_group.attrs.create('traj_times', ts.traj_times)
            ts_group.attrs.create('data_times', ts.data_times)
            ts_group.attrs.create('rel_abundances', ts.rel_abundances)

            ts_group.create_dataset('trajectory',
                                    (0, traj_len),
                                    dtype='f',
                                    maxshape=(100000, traj_len),
                                    chunks=(self.mcmc_chunk_size, traj_len))

            ts_group.create_dataset('aux_trajectory',
                                    (0, traj_len),
                                    dtype='f',
                                    maxshape=(100000, traj_len),
                                    chunks=(self.mcmc_chunk_size, traj_len))

            ts_group.create_dataset('growth_rate', (0,), maxshape=(100000,), dtype='f')
            ts_group.create_dataset('self_interact', (0,), maxshape=(100000,), dtype='f')

            ts_group.create_dataset('var_c', (0,), maxshape=(100000,), dtype='f')

            ts_group.create_group('var_c_pert')
            for compound in ts.dyn_clusters.keys():
                ts_group['var_c_pert'].create_dataset(compound,(0,), maxshape=(100000,), dtype='f')


            # Perturbation datasets are organized by compound
            ts_group.create_group('effect')
            for compound in ts.dyn_clusters.keys():
                ts_group['effect'].create_dataset(compound, (0,), maxshape=(100000,), dtype='f')

            ts_group.create_group('pert_traj')
            for compound in ts.dyn_clusters.keys():
                ts_group['pert_traj'].create_dataset(compound,
                                        (0, traj_len),
                                        dtype='f',
                                        maxshape=(100000, traj_len),
                                        chunks=(self.mcmc_chunk_size, traj_len))

            ts_group.create_group('transfer_shape')
            for compound in ts.dyn_clusters.keys():
                ts_group['transfer_shape'].create_dataset(compound, (0,), maxshape=(100000,), dtype='f')

            ts_group.create_group('on_time')
            for compound in ts.dyn_clusters.keys():
                ts_group['on_time'].create_dataset(compound, (0,), maxshape=(100000,), dtype='f')

            ts_group.create_group('duration')
            for compound in ts.dyn_clusters.keys():
                ts_group['duration'].create_dataset(compound, (0,), maxshape=(100000,), dtype='f')

            ts_group.create_group('dyn_cluster')
            for compound in ts.dyn_clusters.keys():
                ts_group['dyn_cluster'].create_dataset(compound, (0,), maxshape=(100000,), dtype='i8')

            ts_group.create_group('pert_cluster')
            for compound in ts.dyn_clusters.keys():
                ts_group['pert_cluster'].create_dataset(compound, (0,), maxshape=(100000,), dtype='i8')

            ts_group.create_dataset('persist_pert_traj',
                                    (0, traj_len),
                                    dtype='f',
                                    maxshape=(100000, traj_len),
                                    chunks=(self.mcmc_chunk_size, traj_len))

            ts_group.create_dataset('persist_select', (0,), maxshape=(100000,), dtype='i8')
            ts_group.create_dataset('persist_effect', (0,), maxshape=(100000,), dtype='f')

        for subject in self.model.subjects.values():
            traj_len = len(subject.traj_times)

            mcmc_samples['subjects'].create_group(subject.name)
            subj_group = mcmc_samples['subjects'][subject.name]

            for compound, times in subject.consumption_times.items():
                subj_group.attrs.create(compound, times)

            for compound, doses in subject.doses.items():
                subj_group.attrs.create(compound + '_doses', doses)

            subj_group.attrs.create('traj_times', subject.traj_times)

            subj_group.create_group('pharma_traj')
            for compound in subject.consumption_times.keys():
                subj_group['pharma_traj'].create_dataset(compound,
                                        (0, traj_len),
                                        dtype='f',
                                        maxshape=(100000, traj_len),
                                        chunks=(self.mcmc_chunk_size, traj_len))

            subj_group.create_group('k')
            for compound in subject.consumption_times.keys():
                subj_group['k'].create_dataset(compound, (0,), maxshape=(100000,), dtype='f')

        mcmc_samples.close()


def worker(model, arg_queue, result_queue):
    """Update trajectories for some time-series.

    Parameters
    ----------
    model : Model
    arg_queue :
    """
    while True:
        commands = arg_queue.get()
        tsi = commands[0]
        seed = commands[1]
        random.seed(seed)
        sample.seed(2*seed)
        if tsi == 'end':
            return
        ts = model.time_series[tsi]
        ts.update_both_trajectories(seed=seed)
        result_queue.put((tsi, ts.trajectory, ts.aux_trajectory, ts.both_accepts, ts.both_rejects, ts.traj_var))


def get_dataset(settings):
    """Load the dataset.

    The possible datasets are a newly generated synthetic dataset, a JSON
    format dataset, or loading from data files.

    Parameters
    ----------
    settings : dict
        Config settings in dict.

    Returns
    -------
    load_data.Dataset
        Dataset
    """
    if settings['synthetic_data']:
        dataset = synthetic_data.generate_synthetic_dataset(settings)

    else:
        dataset = load_data.load_dataset(settings)

    if settings['verbose']:
        print('Loaded data')

    return dataset


def master(settings):
    """Run the master process.

    Parameters
    ----------
    settings : dict
        Config settings in dict. Returned by run.parse_config_file
    """
    # Make output directory if necessary
    if 'output_directory' in settings.keys():
        if not os.path.exists(settings['output_directory']):
            os.makedirs(settings['output_directory'])

    # Get data
    dataset = get_dataset(settings)

    # Choose particular OTUs on which to perform inference (optional)
    if 'keep_otus' in settings.keys():
        dataset.keep_otus(settings['keep_otus'], taxon=settings['keep_otus_rank'])


    # With fixed seeds, the same MCMC chain can only be guaranteed when the number
    # of processors is the same. With different numbers of processors, the same
    # seeds could lead to different MCMC samples.
    random.seed(4321)
    sample.seed(8765)


    # Load data into Model object
    model = Model(dataset, settings, init_clustering='together')
    sampler = Sampler(model)
    model.sampler = sampler

    for i in range(settings['num_mcmc_iterations']):
        sampler.run_MCMC_iteration(i)

    if settings['verbose']:
        print('MCMC sampling finished')
