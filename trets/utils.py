#!/usr/bin/env python
# Licensed under a 3-clause BSD style license - see LICENSE

from astropy.time import Time
import astropy.units as u
import ray
import numpy as np
from gammapy.data import Observations

__all__=[
    "aaa",
    "conditional_ray",
    "split_observations",
    "subrun_split"
]

def aaa():
    print("hola")

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


def split_observations(observations,threshold_time):
    """
    Split the dataset of observations into subsets where the interval between observations is
    lower than threshold_time.

    parameters
    ----------
    observations:
        Observations object desired to split.
    threshold_time: astropy.Quantity
        Threshold time between observations to considere them as in the same dataset.

    return
    ------
    splitted_obs: list
        list of Observations objects splitted according to threshold_time
    """
    splitted_obs=[]
    joined_runs=[]
    last=False
    prev_run_join=False
    for i in range(len(observations)-1):
        if not prev_run_join:
            joined_runs.append(observations[i])

        sep_time = (observations[i+1].gti.time_start - observations[i].gti.time_stop).to(threshold_time.unit)
        #print(observations[i].obs_id,observations[i+1].obs_id,sep_time)

        if sep_time < threshold_time:
            joined_runs.append(observations[i+1])
            prev_run_join=True
            if i == len(observations)-2:
                last=True
        else:
            splitted_obs.append(Observations(joined_runs))
            joined_runs=[]
            prev_run_join=False
            if i == len(observations)-2:
                last=True

        if last==True and prev_run_join==True:
            splitted_obs.append(Observations(joined_runs))

        if last==True and prev_run_join==False:
            splitted_obs.append(Observations([observations[i+1]]))

    return splitted_obs


def subrun_split(interval_subrun, time_interval_obs):
    """
    Obtain the time intervals required to divide a run into subruns with a gti of interval_subrun,
    intervals in the extremes of the run account the extra or infratime of the run to obtain an
    interger number of subruns.
        
    parameters
    ----------
    interval_subrun: astopy.Quantity
        The number of time we want the subruns to have.
    time_interval_obs: list
        List of [t_start,tstop] for each observation object.
        
    return
    ------
    time_intervals: list
        list of [tstart,tstop] of each subrun.   
        
    """

    time_intervals=[]
    for time_run in time_interval_obs:

        intervals=[]
        t0=time_run.mjd[0]
        tf=time_run.mjd[-1]

        dt=tf-t0
        sections=dt/interval_subrun.to_value("d")

        #select the time that correspond to the end of the first subrun if we add half of the
        #residual to this run
        ti=t0+interval_subrun.to_value("d")*(1+abs(int(sections)-sections)/2)
        #select the time that correspond to the end of the last subrun if we add half of the
        #residual to this run
        t_end=tf-interval_subrun.to_value("d")*(1+abs(int(sections)-sections)/2)

        #obtain the initial and final time of subruns with the same time interval as "interval_subrun"
        interval=np.linspace(ti,t_end,int(sections)-1)
        for i in range(len(interval)):
            #replace the initial time of the first subrun to consider the resiudal
            if i==0:
                intervals.append(Time(t0,format='mjd',scale="utc"))

            intervals.append(Time(interval[i],format='mjd',scale="utc"))
            #replace the final time of the first subrun to consider the resiudal        
            if i==len(interval)-1:
                intervals.append(Time(tf,format='mjd',scale="utc"))


        time = [Time([tstart, tstop]) for tstart, tstop in zip(intervals[:-1], intervals[1:])]

        for i in time:
            time_intervals.append(i)
    return time_intervals
