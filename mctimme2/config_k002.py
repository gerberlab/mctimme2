"""Configuration settings for running the model.

Each setting should consist of a variable assignment.
"""
faulty_samples = ['G213.SS001971', 'G213.SS004450', 'G213.SS003631']

data_directory = 'K002_data'
otu_table = 'K002_venti_otutable_raw-species_summary.txt'
treatment_info = 'K002_treatments_map.txt'
sample_info = 'K002_sample_metadata.txt'
faulty_subjects = ['201-006', '201-007', '201-009', '201-016', '201-010', '201-031', '201-033']
phylogenetic_distances = '../dist_matrix.txt'

output_directory = '/'
mcmc_name = 'mcmc_samples.hdf5'
overwrite = False

num_mcmc_iterations=15000
num_burnin=7500
mcmc_chunk_size = 100

min_samples_before = 1
verbose = True

abundance_filter_min_subjects = 0.25
abundance_threshold = 5e-4
min_counts = 15

parallel_workers = 12
