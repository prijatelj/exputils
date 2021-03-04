"""The general input and output of experiments, including a commonly used
argparse setup with nested namespaces.
"""
import argparse
from copy import deepcopy
from datetime import datetime
import importlib.util
import json
import logging
import os
import sys

# TODO import h5py
from packaging import version
import numpy as np


# TODO currently NestedNamespace still requires all args to be uniquely
# identified. There is perhaps a better solution that is akin to subparsers,
# but allows for multiple "subparsers" to be called per script call and to
# allow for the subparsers to have overlapping arg names because they are
# inherently nested.

class NestedNamespace(argparse.Namespace):
    """An extension of the Namespace allowing for nesting of namespaces.

    Notes
    -----
    Modified version of hpaulj's answer at
        https://stackoverflow.com/a/18709860/6557057

    Use by specifying the full `dest` parameter when adding the arg. then pass
    the NestedNamespace as the `namespace` to be used by `parser.parse_args()`.
    """
    def __setattr__(self, name, value):
        if '.' in name:
            group, _, name = name.partition('.')
            namespace = getattr(self, group, NestedNamespace())
            setattr(namespace, name, value)
            self.__dict__[group] = namespace
        else:
            self.__dict__[name] = value

    def __getattr__(self, name):
        if '.' in name:
            group, _, name = name.partition('.')

            try:
                namespace = self.__dict__[group]
            except KeyError:
                raise AttributeError

            return getattr(namespace, name)
        else:
            getattr(super(NestedNamespace, self), name)


class NumpyJSONEncoder(json.JSONEncoder):
    """Encoder that handles common Numpy values, and general objects. This also
    works for JSON serializing NestedNamespaces.
    """
    def default(self, o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)

        # TODO either remove or make some form of check if valid.
        return o.__dict__


def filename_append(filename, suffix):
    """Appends text to the end of the filename before the file extention."""
    root, ext = os.path.splitext(filename)

    if ext:
        # TODO add optional to ignore ext if fits some pattern.
        # Perhaps add given extenstion to ease respecting the extention in
        # cases where the existing setup fails

        return f'{root}{suffix}{ext}'

    # If there is no extention, simply append suffix
    return f'{filename}{suffix}'


def create_filepath(
    filepath,
    overwrite=False,
    datetime_fmt='_%Y-%m-%d_%H-%M-%S.%f',
):
    """Ensures the directories along the filepath exists. If the file exists
    and overwrite is False, then the datetime is appeneded to the filename
    while respecting the extention.

    Note
    ----
    If there is no file extension, determined via existence of a period at the
    end of the filepath with no filepath separator in the part of the path that
    follows the period, then the datetime is appeneded to the end of the file
    if it already exists.
    """
    # Check if file already exists
    if not overwrite and os.path.isfile(filepath):
        logging.warning(
            ' '.join([
                '`overwrite` is False to prevent overwriting existing files',
                'and there is an existing file at the given filepath: `%s`',
            ]),
            filepath,
        )

        # NOTE beware possibility of program writing the same file in parallel
        filepath = filename_append(
            filepath,
            datetime.now().strftime(datetime_fmt),
        )

        logging.warning('The filepath has been changed to: %s', filepath)
    else:
        # Ensure the directory exists
        dir_path = os.path.dirname(filepath)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)

    return filepath


def create_dirs(
    filepath,
    overwrite=False,
    datetime_fmt='_%Y-%m-%d_%H-%M-%S.%f',
):
    """Ensures the directory path exists. Creates a sub folder with current
    datetime if it exists and overwrite is False.
    """
    # Check if dir already exists
    if not overwrite and os.path.isdir(filepath):
        logging.warning(
            ' '.join([
                '`overwrite` is False to prevent overwriting existing',
                'directories and there is an existing file at the given',
                'filepath: %s',
            ]),
            filepath,
        )

        # NOTE beware possibility of a parallel program writing same filename
        filepath = os.path.join(
            filepath,
            datetime.now().strftime(datetime_fmt),
        )
        os.makedirs(filepath, exist_ok=True)

        logging.warning('The filepath has been changed to: %s', filepath)
    else:
        os.makedirs(filepath, exist_ok=True)

    return filepath


def write(file, mode='x', *args, **kwargs):
    """Convenience wrapper for python open file and safely write using
    create_filepath, appending datetime if already existing. This simply
    handles the exclusive write fail case.
    """
    if 'x' not in mode or 'r' in mode or 'w' in mode or '+' in mode:
        raise ValueError(
            "exputils.io.write() only supports exclusive creation. `mode='x'`",
        )

    # Using create_filepath ensures any directories are made and no overwriting
    return open(create_filepath(file), mode, *args, **kwargs)


def save_json(
    filepath,
    results,
    deep_copy=True,
    overwrite=False,
    datetime_fmt='_%Y-%m-%d_%H-%M-%S.%f',
):
    """Saves the content in results and additional info to a JSON.

    Parameters
    ----------
    filepath : str
        The filepath to the resulting JSON.
    results : dict
        The dictionary to be saved to file as a JSON.
    deep_copy : bool
        Deep copies the dictionary prior to saving due to making the contents
        JSON serializable.
    overwrite :
        If file already exists and False, appends datetime to filename,
        otherwise that file is overwritten.
    datetime_fmt : str
        Format of the datetime string that is appeneded to a file whose given
        filepath already exists and overwrite is False.
    """
    if deep_copy:
        results = deepcopy(results)

    # Check if file already exists
    if not overwrite and os.path.isfile(filepath):
        logging.warning(
            ' '.join([
                '`overwrite` is False to prevent overwriting existing files',
                'and there is an existing file at the given filepath: `%s`',
            ]),
            filepath,
        )

        # NOTE beware possibility of a program writing the same file in parallel
        parts = filepath.rpartition('.')
        filepath = parts[0] + datetime.now().strftime(datetime_fmt) \
            + parts[1] + parts[2]

        logging.warning(f'The filepath has been changed to: {filepath}')

    # Ensure the directory exists
    dir_path = filepath.rpartition(os.path.sep)
    if dir_path[0]:
        os.makedirs(dir_path[0], exist_ok=True)

    with open(filepath, 'w') as summary_file:
        json.dump(
            results,
            summary_file,
            indent=4,
            sort_keys=True,
            cls=NumpyJSONEncoder,
        )


# TODO save hd5f


def multi_typed_arg(*types):
    """Returns a callable to check if a variable is any of the types given."""
    # TODO needs unit tested
    def multi_type_conversion(value):
        for arg_type in types:
            try:
                return arg_type(value)
            except TypeError as error:
                print(f'\n{error}\n')
            except ValueError as error:
                print(f'\n{error}\n{type(error)}\n')
        raise argparse.ArgumentTypeError(' '.join([
            f'Arg of {type(value)} is not convertable to any of the types:',
            f'{types}',
        ]))
    return multi_type_conversion


def check_argv(value, arg, optional_arg=True):
    """Checks if the arg was given and checks if its value is one in the given
    iterable. If true to both, then the arg in question is required. This is
    often used to check if another arg is required dependent upon the value of
    another argument, however if the arg in question of being required has a
    default value, then setting it to required is unnecessary.

    Parameters
    ----------
    value : list() | type | object, optional
        Value is expected to be a list of values to check as the value of the
        given arg. If given vlaue is not iterable, then it is treated as a
        single value to be checked. If a type is given, then its type is
        checked.
    arg : str, optional
        The name of the argument whose value is being checked.
    optional_arg : bool, optional
        Flag indicating if the arg being checked is optional or required by
        default. If the arg is optional and is lacking the initial '--', then
        that is added before checking if it exists in sys.argv. Defaults to
        True.
    """
    if optional_arg and arg[:2] != '--':
        arg = '--' + arg

    if arg in sys.argv:
        idx = sys.argv.index(arg)
        if isinstance(value, type):
            print('\n')
            print(f'value ({value}) is of type {type(value)}.')
            print(f'{sys.argv[idx + 1]} is of type {type(sys.argv[idx + 1])}')
            print('\n')
            try:
                value(sys.argv[idx + 1])
                return True
            except:
                return False
        if not hasattr(value, '__iter__'):
            return value == sys.argv[idx + 1]
        return any([x == sys.argv[idx + 1] for x in value])
    return False


def add_logging_args(parser, log_level='WARNING'):
    """Adds the logging args to the arg parser."""
    log_args = parser.add_argument_group('logging', 'Python logging arguments')

    log_args.add_argument(
        '--log_level',
        default=log_level,
        help='The log level to be logged.',
        choices=['NOTSET', 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        dest='logging.level',
    )

    log_args.add_argument(
        '--log_filename',
        default=None,
        help='The log file to be written to.',
        dest='logging.filename',
    )

    log_args.add_argument(
        '--log_filemode',
        default='a',
        choices=['a', 'w', 'w+'],
        help='The filemode for writing to the log file.',
        dest='logging.filemode',
    )

    log_args.add_argument(
        '--log_format',
        default='%(asctime)s; %(levelname)s: %(message)s',
        help='The logging format.',
        dest='logging.format',
    )

    log_args.add_argument(
        '--log_datefmt',
        default=None,
        #default='%Y-%m-%d_%H-%M-%S',
        help='The logging date/time format.',
        dest='logging.datefmt',
    )

    log_args.add_argument(
        '--log_overwrite',
        action='store_true',
        help=' '.join([
            'If file already exists, giving this flag overwrites that log',
            'file if filemode is "w", otherwise the datetime is appended to',
            'the log filename to avoid overwriting the existing log file.',
        ]),
        dest='logging.overwrite',
    )


def set_logging(log_args):
    """Given an argparse.Namespace, initialize the python logging."""
    # Set logging configuration
    numeric_level = getattr(logging, log_args.level.upper(), None)

    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level given: {log_args.level}')

    if log_args.filename is not None:
        overwrite = (
            log_args.overwrite
            or log_args.filemode not in {'w', 'w+'}
        )

        log_args.filename = create_filepath(log_args.filename, overwrite)

        logging.basicConfig(
            filename=log_args.filename,
            filemode=log_args.filemode,
            level=numeric_level,
            format=log_args.format,
            datefmt=log_args.datefmt,
        )

        if log_args.filemode == 'a':
            # Adding some form of line break for ease of searching logs.
            logging.info('Start of new logging session.')
    else:
        logging.basicConfig(
            level=numeric_level,
            format=log_args.format,
            datefmt=log_args.datefmt,
        )


def add_hardware_args(parser, ml_libs=None):
    """Adds the arguments detailing the hardware to be used."""
    # TODO consider packaging as a dict/NestedNamespace
    # TODO consider a boolean or something to indicate when to pass a
    # tensorflow session or to use it as default
    hardware = parser.add_argument_group(
        'hardware',
        'Hardware uage constraints.',
    )

    hardware.add_argument(
        '--cpus',
        default=1,
        type=int,
        help='The number of available CPUs.',
        dest='hardware.cpus',
    )
    hardware.add_argument(
        '--cpu_cores',
        default=1,
        type=int,
        help='The number of available cores per CPUs.',
        dest='hardware.cpu_cores',
    )
    hardware.add_argument(
        '--gpus',
        default=0,
        type=int,
        help='The number of available GPUs. Pass negative value if no CUDA.',
        dest='hardware.gpus',
    )

    # TODO implement general way of setting which GPU is used for TF & torch
    #hardware.add_argument(
    #    '--which_gpu',
    #    default=None,
    #    type=int,
    #    help='The specifer(s) of which GPU(s) to be used.',
    #)

    hardware.add_argument(
        '--ml_libs',
        default=ml_libs,
        nargs='+',
        help='The machine learning libraries to expect and to init hardware.',
        choices=[
            None,
            'infer',
            'jax',
            'keras',
            'tensorflow',
            'tf',
            'pytorch',
            'torch',
        ],
        dest='hardware.ml_libs',
    )



def set_hardware(args):
    """Sets machine learning librarys' default hardware configuration."""
    # Setup list of ml_libs whose hardware is to be configured
    if args.ml_libs is None:
        # Skip default hardware configuration
        return

    if 'infer' in args.ml_libs:
        # Find ml libraries are installed and initialize for all.
        ml_libs = []

        for lib in ['jax', 'tensorflow', 'keras', 'torch']:
            if importlib.util.find_spec(lib) is not None:
                ml_libs.append(lib)
    else:
        ml_libs = args.ml_libs

    logging.info('Setting the default hardware configuration for %s', ml_libs)

    if 'jax' in ml_libs:
        # TODO add option to provide default config, only add when necessary
        logging.error('JAX default hardware config not implemented yet.')

    if 'keras' in ml_libs:
        import keras
        import tensorflow as tf

        logging.info('Keras version being used: %s', keras.__version__)
        logging.info('Tensorflow version being used: %s', tf.__version__)

        if version.parse(tf.__version__) < version.parse('2.0.0'):
            # Tensorflow 1.15 backend
            keras.backend.set_session(tf.Session(config=get_tf_config(
                args.cpu_cores,
                args.cpus,
                args.gpus,
            )))
        else:
            # TODO handle tf 2.+ config of hardware. Add option to provide
            # default config, only add when necessary
            logging.error('Tensorflow 2.+ default config not implemented yet.')

        # NOTE in the future, may have to add support for other Keras backends
    elif 'tensorflow' in ml_libs or 'tf' in ml_libs:
        import tensorflow as tf

        logging.info('Tensorflow version being used: %s', tf.__version__)

        if version.parse(tf.__version__) < version.parse('2.0.0'):
            # TODO handle tf 1.15 config of hardware. Add option to provide
            # default config, only add when necessary
            logging.error(
                'Tensorflow 1.15 default config not implemented w/o Keras.',
            )
        else:
            # TODO handle tf 2.+ config of hardware. Add option to provide
            # default config, only add when necessary
            logging.error('Tensorflow 2.+ default config not implemented yet.')

    if 'torch' in ml_libs or 'pytorch' in ml_libs:
        # TODO add option to provide default config, only add when necessary
        logging.error('PyTorch default config not implemented yet.')


def get_tf_config(cpu_cores=1, cpus=1, gpus=0, allow_soft_placement=True):
    """Convenient creation of Tensorflow < 2.0 ConfigProto."""
    # TODO Currently only tf 1.15, needs to be updated for 2.+
    from tensorflow import ConfigProto

    return ConfigProto(
        intra_op_parallelism_threads=cpu_cores,
        inter_op_parallelism_threads=cpu_cores,
        allow_soft_placement=allow_soft_placement,
        device_count={
            'CPU': cpus,
            'GPU': gpus,
        } if gpus >= 0 else {'CPU': cpus},
    )


def add_kfold_cv_args(parser):
    """Adds args to the arg parser specific to K-fold cross validation."""

    # TODO simplify this to the kfold params provided in exputils

    kfold_cv = parser.add_argument_group(
        'kfold_cv',
        'K fold cross validation args for evaluating models.',
    )

    kfold_cv.add_argument(
        '--kfolds',
        default=2,
        type=int,
        help='The number of K folds to use in K fold cross validation.',
        dest='kfold_cv.kfolds',
    )

    kfold_cv.add_argument(
        '--no_shuffle_data',
        action='store_false',
        help='Add flag to disable shuffling of data.',
        dest='kfold_cv.shuffle',
    )

    kfold_cv.add_argument(
        '--stratified',
        action='store_true',
        help='Stratified K fold cross validaiton will be used.',
        dest='kfold_cv.stratified',
    )

    kfold_cv.add_argument(
        '--train_focus_fold',
        action='store_true',
        help=' '.join([
            'The focus fold in K fold cross validaiton will be used for ',
            'training and the rest will be used for testing.',
        ]),
        dest='kfold_cv.train_focus_fold',
    )

    # NOTE may remove this from arg group.
    kfold_cv.add_argument(
        '--focus_fold',
        default=None,
        type=int,
        help=' '.join([
            'The focus fold to split the data on to form train and test sets',
            'for a singlemodel train and evaluate session (No K-fold Cross ',
            'Validation; Just evaluates one partition).',
        ]),
        dest='kfold_cv.focus_fold',
    )


def parse_args(
    arg_set=None,
    custom_args=None,
    description=None,
    log_level='WARNING',
    ml_libs=None,
):
    """Creates the args to be parsed and the handling for each.

    Parameters
    ----------
    arg_set : iterable, optional
        contains the additional argument types to be parsed.
    custom_args : function | callable, optional
        Given a function that expects a single argument to be
        `argparse.ArgumentParser`, this function adds arguments to the parser.
    description : str, optional
        The argparse.ArgumentPartser description.

    Returns
    -------
    argparse.namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(description=description)

    # Setup and configure parser
    add_logging_args(parser, log_level=log_level)
    add_hardware_args(parser, ml_libs=ml_libs)

    if arg_set and 'kfold' in arg_set:
        add_kfold_cv_args(parser)

    # TODO add optional profiling args function
    #if arg_set and 'profiling' in arg_set:
    #    add_profiling_args(parser)

    # Add custom args
    if custom_args and callable(custom_args):
        custom_args(parser)

    # Parse args
    args = parser.parse_args(namespace=NestedNamespace())

    # Perform necessary default args setup
    set_logging(args.logging)
    set_hardware(args.hardware)

    # TODO handle random seeds whenever it becomes a necessary thing in scripts
    # e.g. deterministic results instead of statistical reproduction.
    # LOAD the randomseeds from file if it is of type str!!!
    #if isinstance(args.random_seeds, list) and len(args.random_seeds) <= 1:
    #    if os.path.isfile(args.random_seeds[0]):
    #        raise NotImplementedError('Load the random seeds from file.')
    #    else:
    #        args.random_seeds[0] = int(args.random_seeds[0])
    #elif isinstance(args.random_seeds, list):
    #    args.random_seeds = [int(r) for r in args.random_seeds]

    return args
