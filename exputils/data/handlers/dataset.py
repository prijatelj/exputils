"""Generic Dataset Loader intended to serve as the base for specific loders."""
import os

import pandas as pd

# TODO PostgreSQL config parser w/ psycopg2

# TODO this may not be a real generic, as it does not easily turn into all the
# other PyTorch Dataset classes...
# So perhaps some basic tools commonly used when making Dataset classes and
# data loaders would be best?

class GenericDataset(object):
    """Generic Dataset specification intended to be the backbone to all dataset
    specifications and loaders, across Tensorflow and Pytorch. Simply put, the
    generic code for loading from common data sources, such as csv, tsv, hdf5,
    and databases.
    """

    def __init__(self, filepath, *args, **kwargs):
        if isinstance(filepath, str):
            basename, ext = os.path.splitext(filepath)
            # TODO add generic support for HDF5 and .tar.gz for WebDataset???
            #   Perhaps separate these into their own base dataset clases?
            if ext not in {'csv', 'tsv', 'ini'}:
                raise ValueError(' '.join([
                    "Expected one of the filetypes:{'csv', 'tsv', 'ini'}, but",
                    f"recieved: {ext} at `filepath`: {filepath}",
                ]))
            # TODO load dataset from CSV or TSV

            # TODO perhaps, Save the column headers as a NamedTuple or
            # DataClass for an item description?
        #elif isinstance(filepath, PostgreSQLConfig)
        else:
            raise TypeError(
                '`filepath` expected to be type `str`, not {type(filepath)}.'
            )

    def __len__(self):
        raise NotImplementedError()

    def __getitem__(self):
        raise NotImplementedError()

        # TODO to implement slicing, the databackend must support it. This is
        # why there is IterableDataset in PyTorch. If a backend is akin to a
        # dict/map, then perhaps treat it as an OrderedDict, or make a more
        # specific generic subclass that will.
        # NOTE to have both iterable and map by id, you essentially want the
        # pandas.DataFrame, just efficient for I/O things

    def __add__(self, other):
        """Generic concatenation of datasets, similar to python lists."""
        raise NotImplementedError()

        # NOTE this is a bad idea, beacuase in the end, it is probably better
        # to treat this as a tensor, thus wanting to preserve math ops for
        # tensor math. So simply add general append, stack, etc... funcs or
        # methods.

    # TODO implement numpy style slicing??

    # TODO implement partitioning and kfold partitions as a Dataset method?

#class GenericIterDataset(GenericDataset):
#    """A generic iterable dataset."""

# TODO consider adding a Streaming dataset, or iterable I/O dataset that only
# reads in portions of the dataset from the source at a time, such as from a DB
# or from part of a file.
# If not using a Torch or TF dataset/loader, the consider Dask as a streaming
# alternative to Pandas for large data files (and parallel). Albeit, not an
# alternative to a database like PostgreSQL. However, since research code is a
# mess, its nice for the option to parallelie many csvs or tabular data.
