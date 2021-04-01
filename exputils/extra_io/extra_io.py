"""The general Input / Output of experiments."""
import argparse
from copy import deepcopy
from datetime import datetime
import json
import logging
import os
import sys

import h5py
import numpy as np
import keras
import tensorflow as tf


def add_output_args(parser):
    """Adds the arguments for specifying how to save output."""
    parser.add_argument(
        '-o',
        '--output_dir',
        default='./',
        help='Filepath to the output directory.',
    )

    parser.add_argument(
        '-s',
        '--summary_path',
        default='summary/',
        help='Filepath appeneded to `output_dir` for saving the summaries.',
    )


def add_data_args(parser):
    """Adds the data arguments defining what data is loaded and used."""
    # NOTE may depend on model.parts
    data = parser.add_argument_group('data', 'Arguments pertaining to the '
        + 'data loading and handling.')

    # TODO add dataset id here, and have code expect that.

    data.add_argument(
        '-d',
        '--dataset_id',
        default='LabelMe',
        help='The dataset to use',
        choices=['LabelMe', 'FacialBeauty', 'All_Ratings'],
        #dest='' # TODO make the rest of the code expect this to be together ..?
        # mostly requires changing `predictors.py` as dataset_handler.load_dataset()
    )
    data.add_argument(
        'data.dataset_filepath',
        help='The filepath to the data directory',
        #dest='data.dataset_filepath',
    )

    # TODO add to data args and expect it in Data Classes.
    data.add_argument(
        '-l',
        '--label_src',
        default='majority_vote',
        help='The source of labels to use for training.',
        choices=['majority_vote', 'frequency', 'ground_truth', 'annotations'],
        #dest='data.label_src',
    )


def add_mle_args(parser):
    mle = parser.add_argument_group('mle', 'Arguments pertaining to the '
        + 'Maximum Likelihood Estimation.')

    mle.add_argument(
        '--max_iter',
        default=10000,
        type=int,
        help='The maximum number of iterations for finding MLE.',
        dest='mle.max_iter',
    )

    mle.add_argument(
        '--num_top_likelihoods',
        default=1,
        type=int,
        help='The number of top MLEs to be saved for each distribution.',
        dest='mle.num_top_likelihoods',
    )

    mle.add_argument(
        '--const_params',
        default=None,
        nargs='+',
        type=str,
        help='The model\'s parameters to be kept constant throughout the '
            + 'estimate of the MLE.',
        dest='mle.const_params'
    )

    mle.add_argument(
        '--alt_distrib',
        action='store_true',
        help=' '.join([
            'Whether to use the alternate parameterization of the given',
            'distribution, if it exists (ie. mean and precision for the',
            'Dirichlet)',
        ]),
        dest='mle.alt_distrib',
    )

    # Tolerances
    mle.add_argument(
        '--tol_param',
        default=1e-8,
        type=float,
        help='The threshold of parameter difference to the prior parameters '
            + 'set before declaring convergence and terminating the MLE '
            + 'search.',
        dest='mle.tol_param',
    )
    mle.add_argument(
        '--tol_loss',
        default=1e-8,
        type=float,
        help='The threshold of difference to the prior negative log likelihood '
            + 'set before declaring convergence and terminating the MLE '
            + 'search.',
        dest='mle.tol_loss',
    )
    mle.add_argument(
        '--tol_grad',
        default=1e-8,
        type=float,
        help='The threshold of difference to the prior gradient set before '
            + 'declaring convergence and terminating the MLE search.',
        dest='mle.tol_grad',
    )

    mle.add_argument(
        '--tol_chain',
        default=3,
        type=int,
        help=' '.join([
            'The number of iterations that a tolerance must be surpassed in',
            'order to be considered as convergence. Default is 1, meaning as',
            'soon as the tolerance threshold is surpassed, it is considered',
            'to have converged. This is a soft chain of tolerances, meaning',
            'that the tally of number of surpassed tolerances only increments',
            'and decrements by one every iteration, staying within the range ',
            'of [0. tol_chain]. The tally does not reset to 0 after a single',
            'iteration of not surpassing the tolerance threshold.',
        ]),
        dest='mle.tol_chain',
    )

    # optimizer_args
    mle.add_argument(
        '--learning_rate',
        #default=1e-3,
        default=.8,
        type=float,
        help='A Tensor or a floating point vlaue. The learning rate.',
        dest='mle.optimizer_args.learning_rate',
    )
    mle.add_argument(
        '--beta1',
        default=0.9,
        type=float,
        help='A float value or a constant float tensor. The exponential decay '
            + 'rate for the 1st moment estimates.',
        dest='mle.optimizer_args.beta1',
    )
    mle.add_argument(
        '--beta2',
        default=0.999,
        type=float,
        help='A float value or a constant float tensor. The exponential decay '
            + 'rate for the 2nd moment estimates',
        dest='mle.optimizer_args.beta2',
    )
    mle.add_argument(
        '--epsilon',
        default=1e-08,
        type=float,
        help='A small constant for numerical stability. This epsilon is '
            + '"epsilon hat" in the Kingma and Ba paper (in the formula just '
            + 'before Section 2.1), not the epsilon in Algorithm 1 of the '
            + 'paper.',
        dest='mle.optimizer_args.epsilon',
    )
    mle.add_argument(
        '--use_locking',
        action='store_true',
        help='Use locks for update operations.',
        dest='mle.optimizer_args.use_locking',
    )

    # tb_summary_dir ?? handled by output dir? or summary dir

def add_model_args(parser):
    """Adds the model arguments for `predictors.py`."""
    model = parser.add_argument_group(
        'model',
        'Arguments of the model to be loaded, trained, or evaluated.',
    )

    model.add_argument(
        '-m',
        '--model_id',
        default='vgg16',
        help='The model to use',
        choices=['vgg16', 'resnext50'],
        dest='model.model_id',
    )
    model.add_argument(
        '-p',
        '--parts',
        default='labelme',
        help='The part of the model to use, if parts are allowed (vgg16)',
        choices=['full', 'vgg16', 'labelme'],
        dest='model.parts',
    )

    # Init / Load
    model.add_argument(
        '--crowd_layer',
        action='store_true',
        help='Use crowd layer in ANNs.',
        dest='model.init.crowd_layer',
    )
    model.add_argument(
        '--kl_div',
        action='store_true',
        help='Uses Kullback Leibler Divergence as loss instead of Categorical '
            + 'Cross Entropy',
        dest='model.init.kl_div',
    )

    # TODO consider adding into model or putting into general (non-grouped) args
    # allow to be given a str
    parser.add_argument(
        '-r',
        '--random_seeds',
        default=None,
        nargs='+',
        #type=int,
        #type=multi_typed_arg(int, str),
        help='The random seed to use for initialization of the model.',
    )

    # Train
    model.add_argument(
        '-b',
        '--batch_size',
        default=32,
        type=int,
        help='The size of the batches in training.',
        dest='model.train.batch_size',
    )
    model.add_argument(
        '-e',
        '--epochs',
        default=1,
        type=int,
        help='The number of epochs.',
        dest='model.train.epochs',
    )

    model.add_argument(
        '-w',
        '--weights_file',
        default=None,
        help='The file containing the model weights to set at initialization.',
        dest='model.init.weights_file',
    )


def add_sjd_args(parser):
    """Adds the test SJD arguments to the argparser."""
    sjd = parser.add_argument_group(
        'sjd',
        'Arguments pertaining to tests evaluating the'
        + 'SupervisedJointDistribution in fitting simulated data.',
    )

    sjd.add_argument(
        '--target_distrib',
        type=multi_typed_arg(
            str,
            json.loads,
        ),
        help=' '.join([
            'Either a str identifer of a distribution or a dict with',
            '"distirb_id" as a key and the parameters of that distribution',
            'that serves as the target distribution.',
        ]),
        dest='sjd.target_distrib',
    )

    sjd.add_argument(
        '--transform_distrib',
        type=multi_typed_arg(
            str,
            json.loads,
        ),
        help=' '.join([
            'Either a str identifer of a distribution or a dict with',
            '"distirb_id" as a key and the parameters of that distribution',
            'that serves as the transform distribution.',
        ]),
        dest='sjd.transform_distrib',
    )

    sjd.add_argument(
        '--data_type',
        help='Str identifier of the type of data.',
        dest='sjd.data_type',
        default='nominal',
        choices=['nominal', 'ordinal', 'continuous'],
    )

    sjd.add_argument(
        '--independent',
        action='store_true',
        help=' '.join([
            'Indicates if the Supervised Joint Distribution\'s second random',
            'variable is independent of the first. Defaults to False.',
        ]),
        dest='sjd.independent',
    )

    # KNN desnity estimate parameters
    sjd.add_argument(
        '--knn_num_neighbors',
        type=float,
        help=' '.join([
            'A positive int for the number of neighbors to use in the K',
            'Nearest Neighbors density estimate of the transform pdf.',
        ]),
        default=10,
        dest='sjd.knn_num_neighbors',
    )

    sjd.add_argument(
        '--knn_num_samples',
        type=int,
        help=' '.join([
            'Number of samples to draw to approximate the transform pdf for ',
            'the K Nearest Neighbors density estimation.',
        ]),
        default=int(1e6),
        dest='sjd.knn_num_samples',
    )


def expand_mle_optimizer_args(args):
    """Put all mle-related args in a single dictionary."""
    if args.mle.optimizer_args and isinstance(args.mle.optimizer_args, NestedNamespace):
        args.mle.optimizer_args = vars(args.mle.optimizer_args)
    elif args.mle.optimizer_args:
        raise TypeError(f'`args.mle.optimizer` has expected type NestedNamespace, but instead was of type {type(args.mle.optimizer)} with value: {args.mle.optimizer}')


def expand_model_args(args):
    """Turns the init and train attributes into dicts."""
    if args.model.train and isinstance(args.model.train, NestedNamespace):
        args.model.train = vars(args.model.train)
    elif args.mle.train:
        raise TypeError(f'`args.mle.optimizer` has expected type NestedNamespace, but instead was of type {type(args.mle.optimizer)} with value: {args.mle.optimizer}')

    if args.model.init and isinstance(args.model.init, NestedNamespace):
        args.model.init = vars(args.model.init)
    elif args.mle.init:
        raise TypeError(f'`args.mle.optimizer` has expected type NestedNamespace, but instead was of type {type(args.mle.optimizer)} with value: {args.mle.optimizer}')


def parse_args(arg_set=None, custom_args=None, description=None):
    """Creates the args to be parsed and the handling for each.

    Parameters
    ----------
    arg_set : iterable, optional
        contains the additional argument types to be parsed.
    custom_args : function | callable, optional
        Given a function that expects a single argument to be
        `argparse.ArgumentParser`, this function adds arguments to the parser.

    Returns
    -------
    (argparse.namespace, None|int|list(ints))
        Parsed argumentss and random seeds if any.
    """
    if description:
        parser = argparse.ArgumentParser(description=description)
    else:
        parser = argparse.ArgumentParser(description='Run proof of concept.')

    add_logging_args(parser)
    add_hardware_args(parser)

    add_data_args(parser)
    add_output_args(parser)

    add_model_args(parser)
    add_kfold_cv_args(parser)

    if arg_set and 'mle' in arg_set:
        add_mle_args(parser)

    if arg_set and 'sjd' in arg_set:
        add_sjd_args(parser)

    # Add custom args
    if custom_args and callable(custom_args):
        custom_args(parser)

    args = parser.parse_args(namespace=NestedNamespace())

    # Set logging configuration
    numeric_level = getattr(logging, args.log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % args.log_level)
    if args.log_file is not None:
        dir_part = args.log_file.rpartition(os.path.sep)[0]
        os.makedirs(dir_part, exist_ok=True)
        logging.basicConfig(filename=args.log_file, level=numeric_level)
    else:
        logging.basicConfig(level=numeric_level)

    # Set the Hardware
    keras.backend.set_session(tf.Session(config=get_tf_config(
        args.cpu_cores,
        args.cpu,
        args.gpu,
    )))

    # TODO LOAD the randomseeds from file if it is of type str!!!
    if isinstance(args.random_seeds, list) and len(args.random_seeds) <= 1:
        if os.path.isfile(args.random_seeds[0]):
            raise NotImplementedError('Load the random seeds from file.')
        else:
            args.random_seeds[0] = int(args.random_seeds[0])
    elif isinstance(args.random_seeds, list):
        args.random_seeds = [int(r) for r in args.random_seeds]

    # TODO fix this mess here and its usage in `predictors.py`
    #if args.random_seeds and len(args.random_seeds) == 1:
    #    args.kfold_cv.random_seed = args.random_seeds[0]
    #    random_seeds = None
    #else:
    #    random_seeds = args.random_seeds

    #if args is not an int, draw from file.
    #if type(args.random_seeds, str) and os.path.isfile(args.random_seeds):

    #if type(args.random_seeds, list):
    #    print('random_seeds is a list!')

    expand_model_args(args)

    # expand early stopping args:
    args.kfold_cv.early_stopping = vars(args.kfold_cv.early_stopping)

    if arg_set and 'mle' in arg_set:
        expand_mle_optimizer_args(args)

        if arg_set and 'sjd' in arg_set:
            args.sjd.mle_args = vars(args.mle)
            args.sjd.n_jobs = args.cpu_cores

    #return args, random_seeds
    return args
