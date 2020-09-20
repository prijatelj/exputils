from exputils import io
#from exputils.io import NestedNamespace
#from exputils.io import NumpyJSONEncoder
#from exputils.io import create_filepath
#from exputils.io import create_dirs
#from exputils.io import save_json
#from exputils.io import multi_typed_arg
#from exputils.io import check_argv
#from exputils.io import add_logging_args
#from exputils.io import set_logging
#from exputils.io import add_hardware_args
#from exputils.io import set_hardware
#from exputils.io import get_tf_config
#from exputils.io import add_kfold_cv_args
#from exputils.io import parse_args

from exputils import profile
#from exputils.profile import TimeData
#from exputils.profile import time_func
#from exputils.profile import proc_time_func
#from exputils.profile import log_time

# TODO decide if kfold is necessary to install here. Perhaps just an example.

# TODO Cannot del from the namespace, otherwise it breaks the code. Need to
# figure out another way to get exactly desired namespace and to avoid
# `exputils.io.np` or `exputils.io.json`

__all__ = ['io', 'profile']
