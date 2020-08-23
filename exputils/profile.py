"""Convenience functions for profiling python code. Please remember that both
cProfile and profile packages exist in the Python standard library. They may be
used to profile the compute times of all function calls within a program. These
are for convenience and simplicity, especially when the recorded times will be
used by the program, but the results from cProfile may also be used similarly.
"""
from datetime import datetime
from contextlib import contextmanager
import logging
from time import time, process_time

class TimeData(object):
    """Contains the different kinds of time data when timing content.

    Attributes
    ----------
    start_time : float
        Start time in seconds.
    end_time : float
        End time in seconds.
    elapsed_time : float
        Elapsed time between start and end in seconds.
    start_datetime : float
    end_datetime : float
    elapsed_datetime : float
    """
    def __init__(self, time_method=time):
        self.time = time_method
        # TODO ensure this is as low overhead as possible for timing

    @property
    def start_time(self):
        return self._start_time

    @property
    def end_time(self):
        return self._end_time

    @property
    def elapsed_time(self):
        return self._end_time - self._start_time

    @property
    def start_datetime(self):
        return self._start_datetime

    @property
    def end_datetime(self):
        return self._end_datetime

    @property
    def elapsed_datetime(self):
        return self._end_datetime - self._start_datetime

    def start(self):
        """Record the starting time. The timers are ordered by grainularity."""
        self._start_datetime = datetime.now()
        self._start_time = time()

    def end(self):
        """Record the ending time. The timers are ordered by grainularity."""
        self._end_time = time()
        self._end_datetime = datetime.now()


def time_func(func, *args, **kwargs):
    """Given a function or callable, times its wall runtime in seconds."""
    timedata = TimeData()

    timedata.start()
    results = func(*args, **kwargs)
    timedata.end()

    return timedata


# TODO consider adding wall_time and making these two partial funcs of
# time_func(time_method, func, *args, **kwargs)
#def wall_time_func()


def proc_time_func(func, *args, **kwargs):
    """Given a function or callable, times its process runtime in seconds."""
    timedata = TimeData(process_time)

    timedata.start()
    results = func(*args, **kwargs)
    timedata.end()

    return timedata


@contextmanager
def log_time(start_message, end_message=None, time_method=time):
    """Logs the wall time of the content within the `with` block and logs it.

    Parameters
    ----------
    start_message : str
        Message to be written to the log before timing begins.
    end_message : str, optional
        Message to be written to the log after timing ends.
    time_method : callable
        A callable that operate like python's `time.time()`
    """
    # TODO implement writing to a specific, different log file than current one
    logging.info(start_message)

    start = time_method()
    yield
    stop = time_method()

    if end_message is not None:
        logging.info(end_message)

    logging.info("Elapsed time: %d", stop - start)

# TODO consider, if feasible, profiling cpu, gpu, memory usage.
