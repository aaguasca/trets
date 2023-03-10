#!/usr/bin/env python
# Licensed under a 3-clause BSD style license - see LICENSE

from astropy.time import Time
import astropy.units as u
import ray
import numpy as np
from gammapy.data import Observations
import math

__all__ = [
    "get_intervals",
    "conditional_ray",
    "split_observations",
    "subrun_split",
    "fraction_outside_interval",
    "variance_error_prop_calculation"
]


def get_intervals(data, n):
    """
    Return the values whose arg satisfies that it is a
    multiple of n. The value in the last position is also
    returned.
    
    Parameters
    ----------
    data: np.array
        Array of interest
    n: int
        Integer value we want the arg to be multiple of.
    Returns
    -------
    selected_data: np.array
        Array that satisfies the condition.
    """

    if isinstance(data, np.ndarray):
        if len(np.shape(data)) != 1:
            raise ValueError("data has more than one dimension")
    else:
        data = np.array(data)
        if len(np.shape(data)) != 1:
            raise ValueError("data has more than one dimension")
            
    selected_data = data[::n]
    if (len(data)-1) % n != 0:
        selected_data = np.concatenate((selected_data, [data[-1]]))
    return selected_data


def conditional_ray(attr):
    """
    Conditional ray decorator
    """

    def decorator(func):

        def inner(*args, **kwargs):

            is_ray = getattr(args[0], attr)

            if is_ray:
                return ray.remote(func)
            else:
                return func

        return inner

    return decorator


def split_observations(observations, threshold_time):
    """
    Split the dataset of observations into subsets where the interval between observations is
    lower than threshold_time.

    Parameters
    ----------
    observations:
        Observations object desired to split.
    threshold_time: astropy.Quantity
        Threshold time between observations to consider them as in the same dataset.

    Returns
    -------
    split_obs: list
        list of Observations objects split according to threshold_time
    """
    split_obs = []
    joined_runs = []
    last = False
    prev_run_join = False
    for i in range(len(observations)-1):
        if not prev_run_join:
            joined_runs.append(observations[i])

        sep_time = (observations[i+1].gti.time_start - observations[i].gti.time_stop).to(threshold_time.unit)
        # print(observations[i].obs_id,observations[i+1].obs_id,sep_time)

        if sep_time < threshold_time:
            joined_runs.append(observations[i+1])
            prev_run_join = True
            if i == len(observations)-2:
                last = True
        else:
            split_obs.append(Observations(joined_runs))
            joined_runs = []
            prev_run_join = False
            if i == len(observations)-2:
                last = True

        if last is True and prev_run_join is True:
            split_obs.append(Observations(joined_runs))

        if last is True and prev_run_join is False:
            split_obs.append(Observations([observations[i+1]]))

    return split_obs


def subrun_split(interval_subrun, time_interval_obs, atol=1e-6):
    """
    Obtain the time intervals required to divide a run into subruns with a gti of interval_subrun,
    intervals in the extremes of the run account the extra or infra-time of the run to obtain an
    integer number of subruns.
        
    Parameters
    ----------
    interval_subrun: astropy.Quantity
        The number of time we want the subruns to have.
    time_interval_obs: list
        List of [t_start,tstop] for each observation object.
    atol: float
        Resolution.

    Returns
    -------
    time_intervals: list
        list of [tstart,tstop] of each subrun.   
        
    """

    time_intervals = []
    for time_run in time_interval_obs:

        intervals = []
        t0 = time_run.tt.mjd[0]
        tf = time_run.tt.mjd[-1]

        dt = ((tf-t0)*u.d).to_value("min")
        sections = dt/interval_subrun.to_value("min")

        if sections < 1:
            time_intervals.append(Time([t0, tf], format="mjd", scale="tt"))
        else:            
            if round(sections)-round(sections, int(-math.log10(atol))) == 0:
                ti = t0+interval_subrun.to_value("d")
                t_end = tf-interval_subrun.to_value("d")
            else:
                if np.modf(sections)[0] > 0.6 or np.modf(sections)[0] < 0.4 or sections < 4:
                    if np.modf(sections)[0] < 0.4:
                        ti = t0+interval_subrun.to_value("d")*(1+np.modf(sections)[0])
                    else:
                        ti = t0+interval_subrun.to_value("d")*(np.modf(sections)[0])
                    if int(sections) == 1:
                        t_end = ti
                    else:
                        t_end = tf-interval_subrun.to_value("d")
                else:
                    # select the time that correspond to the end of the first subrun if we add half of the
                    # residual to this run
                    ti = t0+interval_subrun.to_value("d")*(1+abs(int(sections)-sections)/2)
                    # select the time that correspond to the end of the last subrun if we add half of the
                    # residual to this run
                    t_end = tf-interval_subrun.to_value("d")*(1+abs(int(sections)-sections)/2)
            
            # obtain the initial and final time of subruns with the same time interval as "interval_subrun"
            if int(sections) == 1:  # for values 1<sections<2
                interval = np.linspace(ti, t_end, 1)
            else:
                if np.modf(sections)[0] > 0.6 or np.modf(sections)[0] < 0.4 or sections < 4:
                    if np.modf(sections)[0] < 0.4:
                        interval = np.linspace(ti, t_end, int(sections)-1)
                    else:
                        interval = np.linspace(ti, t_end, int(sections))
                else:
                    if round(sections)-round(sections, int(-math.log10(atol))) == 0:
                        interval = np.linspace(ti, t_end, int(sections))
                    else:
                        interval = np.linspace(ti, t_end, int(sections)-1)

            if int(sections) == 1 and np.modf(sections)[0] < 0.4:
                time_intervals.append(Time([t0, tf], format="mjd", scale="tt"))
            else:
                for i in range(len(interval)):
                    # replace the initial time of the first subrun to consider the residual
                    if i == 0:
                        intervals.append(Time(t0, format='mjd', scale="tt"))

                    intervals.append(Time(interval[i], format='mjd', scale="tt"))
                    # replace the final time of the first subrun to consider the residual
                    if i == len(interval)-1:
                        intervals.append(Time(tf, format='mjd', scale="tt"))

                time = [Time([tstart, tstop]) for tstart, tstop in zip(intervals[:-1], intervals[1:])]

                for i in time:
                    time_intervals.append(i)

    return time_intervals


def fraction_outside_interval(x, xmin, xmax):
    """
    The normalized fraction of the interval [x[0],x[1]]
    that is outside the interval [xmin,xmax].

    If [x[0],x[1]]>[xmin,xmax], return > 0
    Else, return = 0.

    Parameters
    ----------
    x: list or np.array
        Array where the value of the first argument is
        the start value of the interval and the value
        in the second argument the stop value of the interval.
    xmin: float or int
        Minimum value.
    xmax: float or int
        Maximum value.

    Returns
    -------
    Sum normalized fraction outside xmin-xmax.
    """
    frac_sup = np.max(x) - xmax
    frac_inf = xmin - np.min(x)
    if frac_sup > 0:
        sup = (np.max(x)-xmax)/(np.max(x)-np.min(x))
    else:
        sup = 0
    if frac_inf > 0:
        inf = (np.max(x)-xmin)/(np.max(x)-np.min(x))
    else:
        inf = 0
    return inf+sup


def variance_error_prop_calculation(errors, weights):
    """
    Compute the squared error of the weighted mean
    through error propagation.
    
    Parameters
    ----------
    errors:
        Intrinsic errors of each measurement.
    weights:
        weighted value for each measurement.
        
    Returns
    -------
    squared_mean_error:
        squared error of the weighted mean.
    """
    weights = np.array(weights)
    if np.sum(weights) != 1:
        norm_weights = weights/np.sum(weights)
    else:
        norm_weights = weights
    squared_mean_error = np.sum((np.array(errors)*norm_weights)**2)
    
    return squared_mean_error
