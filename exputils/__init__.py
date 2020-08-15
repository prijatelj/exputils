from exputils import io
# TODO from exputils.profile import time_call
# TODO decide if kfold is necessary to install here.


# Currently this is the best way to blacklist the file's imports, afaik
del io.argparse
del io.deepcopy
del io.datetime
del io.json
del io.logging
del io.os
del io.sys

# TODO del io.h5py once imported
del io.np
