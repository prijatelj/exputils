from exputils import io
from exputils import profile
# TODO decide if kfold is necessary to install here. Perhaps just an example.


# Currently this is the best way to blacklist the file's imports, afaik.
# Basically everything in io is wanted in the namespace except for io's
# imports.

# blacklist io imports
del io.argparse
del io.deepcopy
del io.datetime
del io.json
del io.logging
del io.os
del io.sys

# TODO del io.h5py once imported
del io.np

# blacklist profile imports
del profile.datetime
del profile.contextmanager
del profile.logging
del profile.time
del profile.process_time
