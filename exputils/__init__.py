from exputils import io
from exputils.profile import TimeData, time_func, proc_time_func, log_time
# TODO decide if kfold is necessary to install here. Perhaps just an example.


# Currently this is the best way to blacklist the file's imports, afaik.
# Basically everything in io is wanted in the namespace except for io's
# imports.
del io.argparse
del io.deepcopy
del io.datetime
del io.json
del io.logging
del io.os
del io.sys

# TODO del io.h5py once imported
del io.np
