#!/usr/bin/env python
# Licensed under a 3-clause BSD style license - see LICENSE

import matplotlib.pyplot as plt
import astropy.units as u
from .utils import (
    get_TRETS_table,
    get_TRETS_significance_threshold,
)

__all__ = [
    "temporal_resolution_hist",
    "significance_distribution",
    "temporal_resolution_vs_sig",
]


def temporal_resolution_hist(ax, flux_points, temporal_units):
    """
    Plot the histogram of the time bins of the fluxes from the
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

    flux_points_tab = flux_points.to_table(sed_type="flux", format="lightcurve")
    mask_ul = flux_points_tab["is_ul"].value is True
    mask_ul = mask_ul.reshape(-1)

    # flux points
    time_array = (
        flux_points_tab["time_max"][~mask_ul] - flux_points_tab["time_min"][~mask_ul]
    ) * u.day
    # UL
    time_array_ul = (
        flux_points_tab["time_max"][mask_ul] - flux_points_tab["time_min"][mask_ul]
    ) * u.day

    time_array = time_array.to_value(temporal_units)
    n = ax.hist(time_array, bins=50, label="Flux point")
    time_array_ul = time_array_ul.to_value(temporal_units)
    ax.hist(time_array_ul, bins=n[1], label="UL")
    ax.set_title("Temporal resolution histogram")
    ax.set_xlabel("Time interval [{}]".format(temporal_units))
    ax.set_ylabel("# of fluxes")
    plt.legend()
    return ax


def significance_distribution(ax, flux_points):
    """
    Plot the significance of the fluxes of the flux container
    flux_points as a function of time.

    Parameters
    ----------
    ax:
        Axis
    flux_points: astropy.table.table.Table
        Container Table for flux points as gammapy. Required column names
        time_min, time_max, sig_detection, is_ul.

    Returns
    -------
    ax:
        Axis
    """

    flux_points_tab = get_TRETS_table(flux_points)
    sig_thd = get_TRETS_significance_threshold(flux_points)

    mask_ul = flux_points_tab["is_ul"].value is True
    mask_ul = mask_ul.reshape(-1)
    x = (flux_points_tab["time_max"] + flux_points_tab["time_min"]) / 2

    plt.plot(x, flux_points_tab["sig_detection"], "o", label="Flux point")
    plt.plot(x[mask_ul], flux_points_tab["sig_detection"][mask_ul], "o", label="UL")

    xmin, xmax = ax.get_xlim()
    ax.hlines(sig_thd, xmin, xmax, label="Threshold", color="k")
    ax.set_ylabel(r"Significance [$\sigma$]")
    ax.set_xlabel("Time [MJD]")
    plt.legend(loc="best")

    return ax


def temporal_resolution_vs_sig(ax, list_tab, temporal_units):
    """
    Plot the median value of the temporal resolution histogram
    as a function of the detection significance threshold.
    By default, edges of the boxes delimiters the range between
    the 25th and 75th interpercentile, and the error bars delimiters
    the 5th to 95th interpercentile range.

    Parameters
    ----------
    ax:
        Axis
    list_tab: list
        List with the table of flux points for different source detection
        significance thresholds.
    temporal_units: astropy.units
        Units to express the y-axis

    Returns
    -------
    ax:
        Axis
    """

    list_sig_detection = []
    list_t_bins = []
    for tab in list_tab:
        list_t_bins.append(
            ((tab["time_max"] - tab["time_min"]).value * u.d).to_value(temporal_units)
        )
        list_sig_detection.append(tab.meta["SIG-THD"])

    ax.boxplot(list_t_bins, positions=list_sig_detection, whis=[5, 95], sym="")

    ax.set_xlabel(r"Detection statistical significance [$\sigma$]")
    ax.set_ylabel(r"Median time-bin distribution [{}]".format(temporal_units))

    return ax
