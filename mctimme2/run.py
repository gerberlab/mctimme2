"""Entry point for loading settings/data and launching model from command line.

Usage
-----
This program can be run from the command line. A single argument should be
provided giving the path to the configuration settings file.

Examples
--------
$ python run.py config.py
"""

import argparse
import importlib.util
import mctimme2.model


def parse_arguments():
    """Parse the command line arguments.

    Returns
    -------
    argparse.Namespace
        Holds the attribute 'config_file' with the path to the config file
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('config_file', help='config settings file')
    args = parser.parse_args()
    return args


def parse_config_file(filename):
    """Read the config options in a python file.

    Parameters
    ----------
    filename : str
        Filename of a python file containing the settings

    Returns
    -------
    dict
        The settings in a dictionary
    """
    # The importlib is used so that the filename of the module to import can be
    # user-specified
    module_spec = importlib.util.spec_from_file_location('config_module',
                                                         filename)

    config_file = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(config_file)

    settings = vars(config_file)
    unwanted_keys = ['__builtins__', '__cached__', '__doc__', '__file__', \
                     '__loader__', '__name__', '__package__', '__spec__']
    for key in unwanted_keys:
        if key in settings.keys():
            del settings[key]

    # Add defaults settings
    defaults = {'verbose': False,
                'plot_data': False,
                'save_dataset': False,
                'load_dataset': False,
                'parallel_workers': False,
                'synthetic_data': False,
                'overwrite' : False,
                'faulty_samples': [],
                'faulty_subjects': [],
                'keep_otus_rank': None,
                'spacing': 0.33 ,
                'scale_factor': 10000,
               }

    for key, value in defaults.items():
        settings.setdefault(key, value)

    return settings


def run_model():
    """Parse command line arguments and start the model.
    """
    args = parse_arguments()
    settings = parse_config_file(args.config_file)
    model.master(settings)


if __name__ == '__main__':
    run_model()
