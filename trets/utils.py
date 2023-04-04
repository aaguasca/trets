#!/usr/bin/env python
# Licensed under a 3-clause BSD style license - see LICENSE

from astropy.time import Time
import astropy.units as u
from astropy.io import fits
from astropy.table import (
    Table,
    hstack,
)
import ray
import numpy as np
from gammapy.data import Observations
from gammapy.estimators import (
    FluxPoints,
)
import math

__all__ = [
    "del_asymmetric_errors",
    "get_intervals",
    "get_TRETS_table",
    "get_TRETS_significance_threshold",
    "get_TRETS_binIterator",
    "get_TRETS_flux_significance",
    "conditional_ray",
    "read_TRETS_fluxpoints",
    "split_observations",
    "subrun_split",
    "fraction_outside_interval",
    "weighted_average_error_calculation",
    "write_TRETS_fluxpoints",
    "get_intervals_sum",
    "split_data_from_intervals"

]


def del_asymmetric_errors(flux_points):
    tab = flux_points.to_table(sed_type="flux", format="lightcurve")
    meta_TRETS = flux_points.meta.copy()

    for colname in tab.colnames:
        if "errn" in colname or "errp" in colname:
            del tab[colname]

    fluxpoints = FluxPoints.from_table(tab, sed_type="flux", format="lightcurve")
    fluxpoints.meta = meta_TRETS
    return fluxpoints

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

def get_TRETS_table(flux_points):
    """
    Add the significance detection in the flux_points table to
    the flux_points object.
    """
    tab = flux_points.to_table(sed_type="flux", format="lightcurve")
    sig_detection_column = get_TRETS_flux_significance(flux_points)
    tab = hstack([tab, sig_detection_column])

    meta = flux_points.meta.copy()
    del meta["sig_detection"]
    tab.meta = meta
    return tab

def get_TRETS_significance_threshold(flux_points):
    """
    Obtain the significance threshold used to compute TRETS
    flux points in flux_points object.
    """
    if "sig-thd" in flux_points.meta.keys():
        return flux_points.meta["sig-thd"]

def get_TRETS_flux_significance(flux_points):
    """
    Obtain the significance detection for each time bin used to
    compute the flux points in flux_points object.
    """
    if "sig_detection" in flux_points.meta.keys():
        return flux_points.meta["sig_detection"]

def get_TRETS_binIterator(flux_points):
    """
    Obtain the bin interval value used to compute TRETS
    flux points in flux_points object.
    If event-bin-method, event-wise iterator is used, if
    time-bin-method, fixed-time interval iterator is used.
    """
    for k in flux_points.meta.keys():
        if "method" in k:
            return flux_points.meta[k]

def get_TRETS_flux_significance_thd(flux_points):
    """
    Obtain the flux significance threshold used to compute TRETS
    flux points in flux_points object.
    """
    if "sig-flux-thd" in flux_points.meta.keys():
        return flux_points.meta["sig-flux-thd"]

def conditional_ray(attr):
    """
    Conditional ray decorator.
    Source: https://github.com/ray-project/ray/issues/8105#issuecomment-863940313
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


def read_TRETS_fluxpoints(filename):
    tab = Table.read(filename, format="fits")
    if "TIME-BIN-METHOD" in tab.meta.keys():
        tab.meta["TIME-BIN-METHOD"] = tab.meta["TIME-BIN-METHOD"] * u.s
    del tab.meta["EXTNAME"]
    # lowercase all metadata keys
    for k in list(tab.meta.keys()):
        tab.meta[k.lower()] = tab.meta.pop(k)

    # move sig_detection column to metadata
    tab.meta["sig_detection"] = tab["sig_detection"]

    fluxpoints = FluxPoints.from_table(tab, sed_type="flux", format="lightcurve")
    return fluxpoints


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
    that is outside the interval xmin,xmax.

    If x[0]<xmin or x[1]>xmax, return > 0, respectively.
    Else, return = 0, respectively.

    Parameters
    ----------
    x: list or np.array
        Array where the value of the first argument is
        the start value of the interval and the value
        in the second argument the stop value of the interval.
    xmin: float or int
        The reference minimum value.
    xmax: float or int
        Ther reference maximum value.

    Returns
    -------
    Sum normalized fraction outside xmin-xmax.
    """
    frac_sup = np.max(x) - xmax
    frac_inf = xmin - np.min(x)
    if frac_sup > 0 and frac_inf <= 0:
        sup = (np.max(x)-xmax)/(np.max(x)-np.min(x))
        inf = 0
    elif frac_inf > 0 and frac_sup <= 0:
        inf = 1-(np.max(x)-xmin)/(np.max(x)-np.min(x))
        sup = 0
    elif frac_inf > 0 and frac_sup > 0:
        sup = (np.max(x)-xmax)/(np.max(x)-np.min(x))
        inf = 1-(np.max(x)-xmin)/(np.max(x)-np.min(x))
    else:
        sup = 0
        inf = 0
    return inf+sup

def weighted_average_error_calculation(errors, weights):
    """
    Compute the squared error of the weighted average
    through error propagation of each term.

    \delta_E = (w_1\delta a_1)**2 + (w_2\delta a_2)**2 + ...
    
    Parameters
    ----------
    errors:
        Intrinsic errors of each measurement.
    weights:
        Weighted value for each measurement.
        
    Returns
    -------
    squared_weighted_average_error:
        squared error of the weighted mean.
    """
    weights = np.array(weights)
    if np.sum(weights) != 1:
        norm_weights = weights/np.sum(weights)
    else:
        norm_weights = weights
    squared_weighted_average_error = np.sum((np.array(errors)*norm_weights)**2)
    
    return squared_weighted_average_error


def write_TRETS_fluxpoints(filename, flux_points, **kwargs):
    tab = get_TRETS_table(flux_points)
    header = fits.Header()

    # delete gpy keys
    del tab.meta["sed_type_init"]
    del tab.meta["SED_TYPE"]

    if "time-bin-method" in tab.meta.keys():
        tab.meta["time-bin-method"] = tab.meta["time-bin-method"].to("s")

    # add metadata to header
    for k, v in zip(tab.meta.keys(), tab.meta.values()):
        if isinstance(v, u.quantity.Quantity):
            header[f"{k}"] = (
                v.to_value(v.unit),
                f"[{v.unit.to_string()}]",
            )
        else:
            header[f"{k}"] = (v)
    tab.meta.clear()

    fluxes = fits.BinTableHDU(tab, header=header, name="FLUXES")

    hdulist = fits.HDUList(
        [fits.PrimaryHDU(), fluxes]
    )
    hdulist.writeto(filename, **kwargs)


def get_intervals_sum(start, stop, thd_sum, digit_res=5):
    """
    always keep the minimum time interval checking the next dataset
    """

    if isinstance(thd_sum, u.Quantity):
        dt = (stop - start).to(thd_sum.unit)
        sum_dt = 0 * thd_sum.unit
    else:
        dt = stop - start
        sum_dt = 0

    arg_ini = 0
    intervals = []

    sum_dt = sum_dt + dt[0]
    for i in range(1, len(dt)):
        # keep the value of start and stop of the interval where the sum is smaller than thd
        if round((sum_dt + dt[i]).to_value(thd_sum.unit), digit_res) * thd_sum.unit > thd_sum:
            arg_end = i - 1
            intervals.append(
                [start[arg_ini], stop[arg_end]]
            )
            # restart
            arg_ini = i
            min_coef = 0
            sum_dt = dt[i]
        else:
            sum_dt = sum_dt + dt[i]

    intervals.append(
        [start[arg_ini], stop[-1]]
    )

    return intervals


def split_data_from_intervals(data, intervals, start, stop):
    split_data = []
    for i, (ini, end) in enumerate(intervals):
        arg_i = np.argwhere(start == ini)[0, 0]
        arg_e = np.argwhere(stop == end)[0, 0]

        if arg_i == 0:
            arg_i = None
        if arg_e == len(start) - 1 and i == len(intervals) - 1:
            arg_e = None

        split_data.append(data[arg_i:arg_e])
    return split_data