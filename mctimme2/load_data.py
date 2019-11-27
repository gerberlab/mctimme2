"""Load microbiome counts and subject data from raw files or from JSON.
"""

import math
import re
import os
import warnings
from collections import defaultdict
from operator import itemgetter
import ujson
import pandas
import numpy as np
import matplotlib.pyplot as plt


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


class Taxon:
    """Container for taxonomic information.

    When certain ranks are not available, they are set equal to None.

    Attributes
    ----------
    kingdom : str or None
        kingdom
    phylum : str or None
        phylum
    class_ : str or None
        class
    order : str or None
        order
    family : str or None
        family
    genus : str or None
        genus
    species : str or None
        species
    strain : str or None
        strain
    name : str
        The most precise name available (typically genus or species)
    lowest_rank : {'kingdom', 'phylum', 'class', 'order', 'family', 'genus',
             'species', 'strain'}
        The taxonomic level of name
    full_taxonomy : str
        All taxonomic information using the k__kingdom;p__phylum;... notation

    See Also
    --------
    parse_taxonomy : function for reading the ranks into a dictionary
    """
    def __init__(self, taxonomy):
        """
        Parameters
        ----------
        taxonomy : str
            Sequence of taxonomic ranks delimited with semicolons, with each
            rank preceded by the lower case first letter of the rank name and
            two underscores. 't' (rather than 's') is used for the preceding
            letter for the strain. Each rank is optional.
        """
        self.full_taxonomy = taxonomy

        ranks = ('kingdom', 'phylum', 'class', 'order', 'family', 'genus',
                 'species', 'strain',)
        ranks_dict = parse_taxonomy(taxonomy)

        for rank in ranks:
            if ranks_dict[rank] != '':
                setattr(self, rank + '_' if rank == 'class' else rank, ranks_dict[rank])
            else:
                setattr(self, rank + '_' if rank == 'class' else rank, None)

        for rank in ranks[::-1]:
            if getattr(self, rank + '_' if rank == 'class' else rank) is not None:
                self.taxon = getattr(self, rank + '_' if rank == 'class' else rank)
                self.lowest_rank = rank + '_' if rank == 'class' else rank
                break


    def update(self):
        """Update the full taxonomy.
        """
        self.full_taxonomy = 'k__{}; p__{}; c__{}; o__{}; f__{}; g__{}; s__{}'\
                              .format(self.kingdom, self.phylum, self.class_,
                                      self.order, self.family, self.genus,
                                      self.species)


    def binomial_name(self, abbreviate_genus=False):
        """Get the binomial name without underscores.

        This function is applicable when the species attribute is holding
        "[Genus]_species" or "Genus_species".

        Parameters
        ----------
        abbreviate_genus : bool, optional (False)
            Whether to abbreviate the genus to its first letter.

        Returns
        -------
        str
            "Genus species" or "G. species"
        """
        name = self.species.split('_')[0]
        if abbreviate_genus:
            name = name[0] + '.'

        name += ' ' + '_'.join(self.species.split(' ')[1:])
        name = name.replace('[', '').replace(']', '')

        return name


    def __str__(self):
        return self.full_taxonomy


    def __eq__(self, other):
        ranks = ('kingdom', 'phylum', 'class_', 'order', 'family', 'genus',
                 'species', 'strain',)

        for rank in ranks:
            if getattr(self, rank) != getattr(other, rank):
                return False
        return True


class DataSubject:
    """Holds microbiome counts and compound dosing data for a single subject.

    These objects are used for data input and initial processing, not model
    inference.

    Attributes (fully populated after running Dataset.load_data)
    ----------
    name : str
        A unique label identifying the subject.
    sample_ids : list of str
        The sample ids associated with the subject.
    times : list of float
        The times associated with each of the sample ids in sample_ids.
    counts : dict
        Keys are a unique name for each otu (for example, the taxonomic string).
        For each key, the value is a list of counts for that otu at each sample.
    rel_abundances : dict
        Same as counts, but values are lists of relative abundances.
    totals : list of int
        The total counts at each sample.
    consumption_times : dict
        Keys are the compound names. For each compound, the value is a list of
        times at which each dose was taken
    doses : dict
        Keys are the compound names. For each compound, the value is a list of
        doses corresponding to the time in consumption_times.
    epochs : dict
        Keys are the compound names. For each compound, the value is a list of
        bool (aligned with times) which is True when that compound is being
        dosed and False otherwise. This can be generated automatically using the
        first and last dose of each compound as the boundaries
        (generate_epochs_from_dosing). Epochs are used for fold-change
        filtering, visualization, and variance parameters.
    epoch_boundaries : dict
        Keys are compounds. For each compound, the value is a tuple whose two
        elements give the start and end time of that compound's dosing period.
    run_in_epoch : list of bool
        Represents time points before any compound has been administered. List
        aligned with times of True when before and False otherwise.
    wash_out_epoch : list of bool
        Represents time points after all doses have benn administered. List
        aligned with times of True when after and False otherwise
    """
    def __init__(self, name):
        """
        Parameters
        ----------
        name : str
            A unique label identifying the subject.
        """
        self.name = name
        self.times = []
        self.sample_ids = []
        self.totals = []
        self.counts = {}
        self.rel_abundances = {}
        self.consumption_times = {}
        self.doses = {}


    def generate_epochs_from_dosing(self):
        """Find epochs based on first and last dose of each compound.

        Populates self.epochs, self.epoch_boundaries, self.run_in_epoch,
        and self.wash_out_epoch.
        """
        self.epoch_boundaries = {}
        self.epochs = {}
        for compound in self.consumption_times.keys():
            m0 = min(self.consumption_times[compound])
            m1 = max(self.consumption_times[compound])
            self.epoch_boundaries[compound] = (m0, m1)
            self.epochs[compound] = (np.array(self.times) > m0) & \
                                    (np.array(self.times) < m1)
            self.epochs[compound] = self.epochs[compound].tolist()

        # Get the first and last dose across all compounds
        # Run in is defined as the period before any compound is consumed
        # Wash out is the period after all compounds have been consumed
        first_dose = min((x[0] for x in self.epoch_boundaries.values()))
        last_dose = max((x[1] for x in self.epoch_boundaries.values()))
        self.run_in_epoch = (np.array(self.times) < first_dose).tolist()
        self.wash_out_epoch = (np.array(self.times) > last_dose).tolist()


class Dataset:
    """Used to load data from files and form DataSubject objects.

    Attributes
    ----------
    name : str
        A name for the study
    data_directory : str
        Directory where otu_table, otu_table_raw, sample_info, and
        treatment_info files are located
    faulty_samples : list
        List of sample ids to automatically exclude from loading
    faulty_subjects : list
        List of subjects to automatically exclude from loading
    subjects : dict
    """
    def __init__(self, name, settings):
        """
        Parameters
        ----------
        name : str
            A name for the study
        settings : dict
            Config settings
        """
        self.name = name
        try:
            self.data_directory = settings['data_directory']
            self.faulty_samples = settings['faulty_samples']
            self.faulty_subjects = settings['faulty_subjects']
        except KeyError:
            self.data_directory = ''
            self.faulty_samples = []
            self.faulty_subjects = []
        self.subjects = {}


    def load_data(self, settings):
        """Populate the dataset from data files.

        Parameters
        ----------
        settings : dict
            config settings
        """
        self.load_samples_data(settings['sample_info'])
        self.load_treatments_data(settings['treatment_info'])
        self.load_otu_table(settings['otu_table'])

        # Remove any subject which had no samples in the otu table
        self.subjects = dict(((k,v) for k, v in self.subjects.items() \
                                             if len(v.sample_ids) > 0))

        for subj in self.subjects.values():
            subj.generate_epochs_from_dosing()


    def load_samples_data(self, samples_file, sep='\t'):
        r"""Load samples and subject data from raw file.

        Each row in the file should contain the information for a single
        sample. The required columns are 'sample_id', 'subject_id', and
        'sample_study_day'. The sample names in 'sample_id' must match sample
        ids in the otu table. The subject names in 'subject_id' must match the
        subject ids in the treatments data file.

        This function populates self.subjects, a dictionary of subject names ->
        DataSubject objects. Each DataSubject also has its list of times and
        sample_ids populated.

        Parameters
        ----------
        samples_file : str
            Filename of the samples data file relative to self.data_directory
        sep : str, optional ('\t')
            The separator between entries in the file. Typically '\t' (tab
            separated values) or ',' (comma separated values).
        """
        sample_metadata = pandas.read_csv(os.path.join(self.data_directory,
                                          samples_file), sep=sep)

        sample_metadata = sample_metadata.sort_values(['subject_id',
                                                       'sample_study_day'])

        for _, row in sample_metadata.iterrows():
            subj_name = row['subject_id']

            if subj_name not in self.faulty_subjects:
                if subj_name in self.subjects.keys():
                    subj_obj = self.subjects[subj_name]

                else:
                    subj_obj = DataSubject(subj_name)
                    self.subjects[subj_name] = subj_obj

                sample_id = row['sample_id']

                if sample_id not in self.faulty_samples:
                    time = row['sample_study_day']
                    if time not in subj_obj.times:
                        subj_obj.times.append(row['sample_study_day'])
                        subj_obj.sample_ids.append(sample_id)
                    else:
                        warnings.warn('Duplicate sample for subject {}'.format(subj_name))


    def load_treatments_data(self, treatments_file, sep='\t'):
        r"""Load compound administration data.

        Each row in the file should contain the information for a single
        administration of a compound to one subject. The required columns are
        'subject_id', 'treatment_study_day', and 'actual_dose'. A column for
        'compound' is optional in the case of a study with one compound and
        required for a study of multiple compounds. If the 'compound' column is
        not present, the compound will be called 'Cmpd1'. Optionally, a cohort
        for each subject can be provided in a column with header
        'treatment_group'.

        Parameters
        ----------
        treatments_file : str
            filename of the treatments data file relative to self.data_directory
        sep : str, optional ('\t')
            The separator between entries in the file. Typically '\t' (tab
            separated values) or ',' (comma separated values).
        """
        treatments_map = pandas.read_csv(os.path.join(self.data_directory,
                                         treatments_file), sep='\t')

        # Clear dosing information and instantiate defaultdicts in each subject
        for subj_obj in self.subjects.values():
            subj_obj.consumption_times = defaultdict(list)
            subj_obj.doses = defaultdict(list)

        # Add the dosing information to each subject.
        for i, row in treatments_map.iterrows():
            subj_name = row['subject_id']
            if subj_name in self.subjects.keys():
                subj_obj = self.subjects[subj_name]

                if 'compound' in treatments_map.columns:
                    compound = row['compound']
                else:
                    compound = 'Cmpd1'

                if 'treatment_group' in treatments_map.columns:
                    subj_obj.cohort = row['treatment_group']

                subj_obj.consumption_times[compound].append(\
                                                    row['treatment_study_day'])
                subj_obj.doses[compound].append(row['actual_dose'])


        # Sort the doses and times
        for subj_obj in self.subjects.values():
            for compound in subj_obj.doses.keys():
                subj_obj.doses[compound], subj_obj.consumption_times[compound]=\
                (list(z) for z in zip(*sorted(zip(subj_obj.doses[compound],\
                    subj_obj.consumption_times[compound]), key=itemgetter(0))))

        # Convert subject doses and times back to standard dicts (not
        # defaultdict)
        # Standard dict can cause fewer problems if saving to JSON
        for subj_obj in self.subjects.values():
            subj_obj.doses = dict(subj_obj.doses)
            subj_obj.consumption_times = dict(subj_obj.consumption_times)


    def load_otu_table(self, otu_table_file, otu_table_raw_file=None, sep='\t'):
        r"""Load counts data from an otu table.

        The otu_table is a text file whose columns are samples and rows are
        OTUs, species, genuses, KEGG data, or other units to analyze with the
        model. The first column should contain the name of each OTU and the
        first row should contain a name for each sample.

        Parameters
        ----------
        otu_table_file : str
            filename of otu table relative to self.data_directory
        otu_table_raw_file : str, optional (None)
            filename of raw otu table relative to self.data_directory. The raw
            table includes records which are not intended for analysis by the
            model (for example, counts which were not mapped at the desired
            taxonomic level), but whose counts are to be included in the
            calculation of total counts per sample.
        sep : str, optional ('\t')
            The separator between entries in the file. Typically '\t' (tab
            separated values) or ',' (comma separated values).
        """
        otu_table = pandas.read_csv(os.path.join(self.data_directory,
                                    otu_table_file), sep='\t')

        if otu_table_raw_file is not None:
            otu_table_raw = pandas.read_csv(os.path.join(self.data_directory,
                                            otu_table_raw_file), sep='\t')
        else:
            otu_table_raw = otu_table

        for column in otu_table.columns:
            if column not in otu_table_raw.columns:
                otu_table = otu_table.drop(column, axis=1)
        otu_table_raw = otu_table

        otu_column_name = otu_table.columns[0]

        # The subjects got their sample ids from samples_id
        # Remove any of these sample ids which are not found in the otu table
        for subj_obj in self.subjects.values():
            not_found = []
            for i, sample_id in enumerate(subj_obj.sample_ids):
                if sample_id not in otu_table.columns:
                    not_found.append(i)
            if len(not_found) > 0:
                message = '{} samples of subject {} were not found in the OTU table:\n\t'.format(len(not_found), subj_obj.name) \
                          + '\n\t'.join([subj_obj.sample_ids[i] for i in not_found])
                warnings.warn(message)
            subj_obj.times = [t for i, t in enumerate(subj_obj.times) if i not in not_found]
            subj_obj.sample_ids = [id for i, id in enumerate(subj_obj.sample_ids) if i not in not_found]

        for subj_obj in self.subjects.values():
            keys = otu_table[otu_column_name]
            # Remove any spaces in between ranks to be consistent with
            # venti.tax
            keys = [x.replace(' ', '') for x in keys]
            data = otu_table[subj_obj.sample_ids].values.astype('float64')
            totals = otu_table_raw[subj_obj.sample_ids].values.sum(axis=0)

            subj_obj.counts = dict(zip(keys, [list(ts) for ts in data]))
            subj_obj.rel_abundances = dict(zip(subj_obj.counts.keys(),
                            [(x / totals).tolist() for x in subj_obj.counts.values()]))
            subj_obj.totals = totals.tolist()


    def apply_abundance_filter(self, threshold):
        """Remove any time-series with mean abundance below the threshold.

        Parameters
        ----------
        threshold : float
            Mean value below which the time-series will be filtered out.

        Returns
        -------
        delete : list of tuple
            Tuples (subject_name, otu) to filter out
        """
        delete = []

        for name, subj_obj in self.subjects.items():
            for k, d in subj_obj.rel_abundances.items():
                if not np.mean(d) >= threshold:
                    delete.append((name, k))

        return delete


    def apply_abundance_filter_across_subjects(self, threshold, min_subjects):
        """Remove low abundance time-series, unless enough other subjects have
        that OTU with a higher abundance.

        If the OTU passes the threshold in at least min_subjects, it will be
        kept for all subjects.

        If the OTU does not pass the threshold in at least min_subjects, it will
        be filtered out for all subjects.

        Parameters
        ----------
        threshold : float
            Mean value below which the time series will be filtered out.
        min_subjects : int
            Minimum number of subjects such that if the OTU passes in that many
            subjects, it is kept in all subjects.

        Returns
        -------
        delete : list of tuple
            Tuples (subject_name, otu) to filter out
        """
        delete = []
        first_subj_obj = next(iter(self.subjects.values()))
        num_otus = len(first_subj_obj.rel_abundances.keys())
        for otu in first_subj_obj.rel_abundances.keys():
            means = []
            subj_names = []
            for name, subj_obj in self.subjects.items():
                ts = subj_obj.rel_abundances[otu]
                means.append(np.mean(ts))
                subj_names.append(name)
            num_subjects_passed = np.sum(np.array(means)>=threshold)

            if num_subjects_passed < min_subjects:
                for subj_name, mean in zip(subj_names, means):
                    delete.append((subj_name, otu))

        return delete


    def apply_foldchange_filter(self, threshold):
        """Remove time-series without a sufficient fold change on perturbation.

        Parameters
        ----------
        threshold : float
            Log2 fold change value below which time series will be filtered out

        Returns
        -------
        delete : list of tuple
            Tuples (subject_name, otu) to filter out
        """
        delete = []
        for name, subj in self.subjects.items():
            for otu, data in subj.rel_abundances.items():
                fc = self._ts_foldchange(data, subj)
                if fc < threshold:
                    delete.append((name, otu))

        return delete


    def apply_foldchange_filter_across_subjects(self, threshold, min_subjects):
        delete = []
        first_subj_obj = next(iter(self.subjects.values()))
        for otu in first_subj_obj.rel_abundances.keys():
            fold_changes = []
            subj_names = []
            for name, subj in self.subjects.items():
                data = subj.rel_abundances[otu]
                fold_changes.append(self._ts_foldchange(data, subj))
                subj_names.append(name)
            num_subjects_passed = np.sum(np.array(fold_changes) >= threshold)

            if num_subjects_passed < min_subjects:
                for subj_name in subj_names:
                    delete.append((subj_name, otu))

        return delete


    def _ts_foldchange(self, data, subj):
        """Calculate the log2 fold change for a single time series.

        Parameters
        ----------
        data : list
            The time series data points
        subj : DataSubject
            subject

        Returns
        -------
        float
            The largest log2 foldchange between periods
        """
        offset = 1/10000
        log2_data = np.log2(np.array(data) + offset)

        fold_changes = []
        for cmpd in subj.doses.keys():
            fc1 = np.abs(np.median(log2_data[subj.run_in_epoch]) - np.median(log2_data[subj.epochs[cmpd]]))
            fold_changes.append(fc1)

            fc2 = np.abs(np.median(log2_data[subj.epochs[cmpd]]) - np.median(log2_data[subj.wash_out_epoch]))
            fold_changes.append(fc2)

        return max(fold_changes)


    def apply_counts_filter(self, min_counts):
        """Remove time-series without sufficient counts"""
        delete = []

        for name, subj_obj in self.subjects.items():
            for otu, counts in subj_obj.counts.items():
                passed = False
                for i in range(len(counts) - 3):
                    window = counts[i:i+3]
                    if min(window) > min_counts:
                        passed = True
                if not passed:
                    delete.append((name, otu))

        return delete


    def apply_subjects_sample_locations_filter(self, n_before, n_during, n_after):
        """Remove subjects without enough samples in each experiment phase.

        Parameters
        ----------
        n_before : int
            Minimum number of samples each subject must have before they consume
            the first dose of any compound.
        n_during : int
            Minimum number of samples each subject must have in between the
            consumption of the first and last dose.
        n_after : int
            Minimum number of samples each subject must have after they consume
            the last dose.
        """
        delete = []
        for subj_name, subj_obj in self.subjects.items():
            times = np.array(subj_obj.times)

            start_pert = math.inf
            end_pert = -math.inf
            for compound in subj_obj.consumption_times.keys():
                start_pert = min(start_pert, min(subj_obj.consumption_times[compound]))
                end_pert = max(end_pert, max(subj_obj.consumption_times[compound]))

            pts_before = sum(times <= start_pert)
            pts_after = sum(times > end_pert)
            pts_during = sum((times > start_pert) & (times <= end_pert))

            if (pts_before < n_before) or (pts_during < n_during) or (pts_after < n_after):
                delete.append(subj_name)

        for dl in delete:
            del self.subjects[dl]


    def apply_filters(self, settings):
        """Apply all filters as specified in the settings.

        Parameters
        ----------
        settings : dict
            Config settings
        """
        sample_locs_settings = ['min_samples_before', 'min_samples_during', 'min_samples_after']
        if any(loc_setting in settings.keys() for loc_setting in sample_locs_settings):

            self.apply_subjects_sample_locations_filter(settings.setdefault('min_samples_before', 0),
                                                   settings.setdefault('min_samples_during', 0),
                                                   settings.setdefault('min_samples_after', 0))

        abundance_failed = []
        foldchange_failed = []
        counts_failed = []

        if 'abundance_filter_min_subjects' in settings.keys():
            # Convert the config setting (proportion of subjects) to an actual
            # number of subjects in this datasets
            min_number_subjects = settings['abundance_filter_min_subjects'] \
                                  * len(self.subjects.keys())
            min_number_subjects = math.floor(min_number_subjects)

            abundance_failed = self.apply_abundance_filter_across_subjects(
                                        settings['abundance_threshold'],
                                        min_number_subjects)
        elif 'abundance_threshold' in settings.keys():
            abundance_failed = self.apply_abundance_filter(settings['abundance_threshold'])


        if 'foldchange_filter_min_subjects' in settings.keys():
            min_number_subjects = settings['foldchange_filter_min_subjects'] \
                                  * len(self.subjects.keys())
            min_number_subjects = math.floor(min_number_subjects)

            foldchange_failed = self.apply_foldchange_filter_across_subjects(
                                        settings['foldchange_threshold'],
                                        min_number_subjects)
        elif 'foldchange_threshold' in settings.keys():
            foldchange_failed = self.apply_foldchange_filter(settings['foldchange_threshold'])

        if 'min_counts' in settings.keys():
            counts_failed = self.apply_counts_filter(settings['min_counts'])

        for dl in set(abundance_failed + foldchange_failed + counts_failed):
            del self.subjects[dl[0]].counts[dl[1]]
            del self.subjects[dl[0]].rel_abundances[dl[1]]


    def keep_otus(self, to_keep, taxon=None):
        """Delete all time series from dataset except those specified.

        Parameters
        ----------
        to_keep : list
            Tuples to keep (subj, otu)
        taxon : str, optional
            If present, apply parse
        """
        new_subjects = {}

        delete = []

        for subject in self.subjects.values():
            for otu in subject.rel_abundances.keys():
                if taxon is not None:
                    if (subject.name, parse_taxonomy(otu)[taxon]) not in to_keep:
                        delete.append((subject.name, otu))
                else:
                    if (subject.name, otu) not in to_keep:
                        delete.append((subject.name, otu))

        for dl in set(delete):
            del self.subjects[dl[0]].counts[dl[1]]
            del self.subjects[dl[0]].rel_abundances[dl[1]]

        delete = []
        for subject in self.subjects.values():
            if len(subject.counts.keys()) == 0:
                delete.append(subject.name)
        for dl in set(delete):
            del self.subjects[dl]

        n = self.num_time_series()
        if n == 0:
            warnings.warn('No time-series remaining after running keep_otus')


    def num_time_series(self):
        """Count the number of time-series currently in the dataset.

        Returns
        -------
        int
            Number of time-series in dataset
        """
        n = 0
        for subj_obj in self.subjects.values():
            n += len(subj_obj.rel_abundances.keys())
        return n


    def get_taxa(self):
        """Get the species currently in the dataset.
        """
        taxa = []
        for subj_obj in self.subjects.values():
            taxa += list(subj_obj.rel_abundances.keys())
        taxa = list(set(taxa))

        return taxa
        n = 0
        for tax in taxa:
            n += 1
            print(Taxon(tax).species)
        return n


    def plot_data(self, settings, colors=None):
        """Plot all data time-series in the dataset.

        Plots are saved to the output_directory in settings.

        Parameters
        ----------
        settings : dict
            config settings
        colors : dict (optional)
            Map from compound names to colors. When not provided, the same
            default color is used for all compounds.
        """
        colors = {'PDX': 'orange', 'FOS': 'teal'}
        for s in self.subjects.keys():
            t = self.subjects[s].times
            for k in self.subjects[s].rel_abundances.keys():
                fig = plt.figure(figsize=(5, 2.75))
                ax = fig.add_subplot(1, 1, 1)
                x = np.array(self.subjects[s].rel_abundances[k]) * \
                                                  settings['scale_factor']

                ax.plot(t, x, 'x-', label='data', color='k')

                for compound in self.subjects[s].doses.keys():
                    m = min(self.subjects[s].consumption_times[compound])
                    M = max(self.subjects[s].consumption_times[compound])
                    plt.axvspan(m, M,
                               alpha=0.3,
                          color=colors[compound] if colors != None else 'teal',
                               zorder=-10,
                               label=compound)

                title = '{} {}'.format(s, Taxon(k).taxon)
                ax.set_title(title)
                ax.set_xlabel('time (days)')
                ax.set_ylabel('norm. abundance')
                ax.legend()
                fig.tight_layout()

                filename = '{}_{}_data.pdf'.format(s, k)
                filename = filename.replace('/', 'SLASH')
                output_path = os.path.join(settings['output_directory'],\
                                                                      filename)
                plt.savefig(output_path)
                plt.close()


    def save_to_json(self):
        """Save the dataset to JSON format.

        Returns
        str
            A JSON of the dataset.
        """
        return ujson.dumps(self)


    def load_data_from_json(self, settings):
        """Populate the dataset from saved JSON dataset.

        Parameters
        ----------
        settings : dict
            config settings
        """
        with open(settings['load_dataset']) as f:
            json_data = ujson.load(f)

        for subj_dict in json_data['subjects'].values():
            self.subjects[subj_dict['name']] = DataSubject(subj_dict['name'])
            subj_obj = self.subjects[subj_dict['name']]
            subj_obj.times = subj_dict['times']
            subj_obj.sample_ids = subj_dict['sample_ids']
            subj_obj.counts = subj_dict['counts']
            subj_obj.rel_abundances = subj_dict['rel_abundances']
            subj_obj.totals = subj_dict['totals']
            subj_obj.consumption_times = subj_dict['consumption_times']
            subj_obj.doses = subj_dict['doses']


def load_phylogenetic_distances(distance_matrix, taxa):
    """Load phylogenetic distances from text matrix to python dict.

    Parameters
    ----------
    distance_matrix : str
        Path to distance matrix text file
    taxa : list of str
        The taxa for which to get the distances

    Returns
    -------
    dict
        Double dictionary such that d[otu1][otu2] = the distance between
    """
    taxa_species_level = [Taxon(x).species for x in taxa]

    d = pandas.read_csv(distance_matrix, sep='\t')
    all_species = list(d.columns[1:])

    distance_dict = defaultdict(dict)

    for sp in all_species:
        if sp in taxa_species_level:
            dists = list(d[sp])
            for i, sp2 in enumerate(all_species):
                if sp2 in taxa_species_level:
                    distance_dict[sp][sp2] = dists[i]
                    distance_dict[sp2][sp] = dists[i]

    return dict(distance_dict)


def load_dataset(settings):
    """Load dataset from data files.

    Parameters
    ----------
    settings : dict
        Config options

    Returns
    -------
    Dataset
        The dataset with filters applied
    """
    d = Dataset('d1', settings)

    if 'faulty_subjects' in settings.keys():
        d.faulty_subjects += settings['faulty_subjects']

    if 'faulty_samples' in settings.keys():
        d.faulty_samples += settings['faulty_samples']

    if settings['load_dataset']:
        d.load_data_from_json(settings)
    else:
        d.load_data(settings)


    d.apply_filters(settings)

    if settings['verbose']:
        num_time_series = d.num_time_series()
        print('{} time series loaded'.format(num_time_series))

        taxa = d.get_taxa()
        print('{} taxa loaded'.format(len(taxa)))
        for tax in taxa:
            print(tax)

    dist_matrix = load_phylogenetic_distances(os.path.join(settings['data_directory'],settings['phylogenetic_distances']), taxa)
    d.dist_matrix = dist_matrix


    if settings['plot_data']:
        d.plot_data(settings)

    if settings['save_dataset']:
        json_study = d.save_to_json()
        with open(os.path.join(settings['output_directory'], settings['save_dataset']), 'w') as f:
            f.write(json_study)

    return d
