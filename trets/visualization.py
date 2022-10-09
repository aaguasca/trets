#!/usr/bin/env python
# Licensed under a 3-clause BSD style license - see LICENSE

import matplotlib.pyplot as plt
import numpy as np

__all__=[
    temporal_resolution_hist,
    significance_distribution
]

def temporal_resolution_hist(ax,flux_points,temporal_units):
    """
    Plot the histogram of the time bins of the fluxes in the
    flux container flux_points.

    Parameters
    ----------
    ax:
        Axis
    flux_points:astropy.table.table.Table
        Container Table for flux points as gammapy. Required column names
        time_min, time_max, is_ul.
    temporal_units: astropy.units
        Units to express the x axis
    Returns
    -------
    ax:
        Axis
    """

    #flux_points=flux_points.to_table(sed_type="flux", format="lightcurve")
    mask_ul=flux_points["is_ul"]==True

    #flux points
    time_array=(flux_TRETS["time_max"][~mask_ul.reshape(-1)]-flux_TRETS["time_min"][~mask_ul.reshape(-1)])*u.day
    #UL
    time_array_ul=(flux_TRETS["time_max"][mask_ul.reshape(-1)]-flux_TRETS["time_min"][mask_ul.reshape(-1)])*u.day

    time_array=time_array.to_value(temporal_units)
    n=ax.hist(time_array,bins=50,label="Flux point")
    time_array_ul=time_array_ul.to_value(temporal_units)
    n2=plt.hist(time_array_ul,bins=n[1],label="UL")
    ax.set_title("Temporal resolution histogram")
    ax.set_xlabel("Time interval [{}]".format(temporal_units))
    ax.set_ylabel("# of fluxes")
    plt.legend()
    return ax


def significance_distribution(ax,flux_points_table):
    """
    Plot the significance of the fluxes of the flux container 
    flux_points as a function of time.

    Parameters
    ----------
    ax:
        Axis
    flux_points_table: astropy.table.table.Table
        Container Table for flux points as gammapy. Required column names
        time_min, time_max, sig_detection, is_ul.

    Returns
    -------
    ax:
        Axis
    """

    #flux_points=flux_points.to_table(sed_type="flux", format="lightcurve")
    mask_ul=flux_points["is_ul"]==True

    x=(flux_points["time_max"]+flux_points["time_min"])/2

    plt.plot(x,flux_points["sig_detection"],"o",label="Flux point")
    plt.plot(x[mask_ul.reshape(-1)],flux_points["sig_detection"][mask_ul.reshape(-1)],"o",label="UL")

    xmin,xmax=ax.get_xlim()
    ax.hlines(flux_points.meta["sig-THD"],xmin,xmax,label="Threshold")
    ax.set_ylabel("Significance")
    ax.set_xlabel("Time [MJD]")
    plt.legend(loc="best")

    return ax

