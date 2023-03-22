#!/usr/bin/env python
# Licensed under a 3-clause BSD style license - see LICENSE

from ..utils import (
    subrun_split,
    get_intervals_sum,
    split_data_from_intervals
)
import numpy as np
from astropy.time import (
    Time
)
from gammapy.makers import (
    SpectrumDatasetMaker
)
from gammapy.maps import (
    RegionGeom
)
from gammapy.datasets import (
    Datasets,
    SpectrumDataset,
)
from gammapy.estimators import (
    LightCurveEstimator,
)

__all__ = ['intrarun']


def intrarun(
    is_simu,
    E1,
    E2,
    e_reco,
    e_true,
    on_region,
    observations,
    bkg_maker_reflected,
    best_fit_spec_model,
    time_bin=None
):

    """
    Compute the integral fluxes using subrun events.

    Parameters
    ----------
    time_bin: astropy.Quantity
        The number of time we want the subruns to have.
    E1: astropy.Units
        Minimum energy bound used to compute the integral flux. It must be considered with one center
        in e_reco.
    E2: astropy.Units
        Maximum energy bound used to compute the integral flux. It must be considered with one center
        in e_reco.       
    e_reco:
        Reconstructed energy axis used in the SpectrumDatasetOnOff object
    e_true:
        True energy axis used in the SpectrumDatasetOnOff object
    on_region:
        On region located in the source position, it must has the same size used in the IRFs.
    observations: `gammapy.data.Observations`
        Observation object with the runs used to compute the light curve.
    bkg_maker_reflected:
        Background maker to estimate the background.
    best_fit_spec_model:
        Assumed SkyModel of the source. Only spectral model.

    Returns
    -------
    lc_subrun: LightCurve
        Light curve object.    
    """

    if not is_simu:
        geom = RegionGeom.create(region=on_region, axes=[e_reco])

        # create the dataset container object of the spectrum on the ON region
        dataset_empty = SpectrumDataset.create(
            geom, energy_axis_true=e_true
        )

        # maker to produce data reduction to DL4
        dataset_maker = SpectrumDatasetMaker(
            containment_correction=False, selection=["counts", "exposure", "edisp"]  # make this maps
        )

    # prepare again the Dataset
    datasets_short = Datasets()

    if time_bin is not None:

        #data at DL3
        if not is_simu:
            time_interval_obs = []
            for run in range(len(observations)):
                time_interval_obs.append(
                    Time(
                        [observations[run].events.observation_time_start.tt.mjd,
                         observations[run].events.observation_time_stop.tt.mjd],
                        format="mjd",
                        scale="tt"
                    )
                )

            time_intervals = subrun_split(time_bin, time_interval_obs)

            # divide the runs into different subruns
            short_observations = observations.select_time(time_intervals)

        #data at DL4
        else:
            gti_time_start_obs = observations.gti.time_start
            gti_time_stop_obs = observations.gti.time_stop
            time_intervals = get_intervals_sum(gti_time_start_obs, gti_time_stop_obs, time_bin)
            observations_sets = split_data_from_intervals(
                observations,
                time_intervals,
                gti_time_start_obs,
                gti_time_stop_obs
            )
            print(len(observations_sets))

            short_observations = Datasets()
            for obs_set in observations_sets:
                short_observations.append(Datasets(obs_set).stack_reduce())
            short_observations = Datasets(short_observations)

    else:
        # only to match observations
        short_observations = observations
        if not is_simu:
            time_intervals = observations.ids
        else:
            time_intervals = observations.names

    # loop for each run
    for obs_id, observation in zip(np.arange(len(time_intervals)), short_observations):

        if not is_simu:
            dataset = dataset_maker.run(dataset_empty.copy(name=str(obs_id)), observation)
            dataset_on_off = bkg_maker_reflected.run(dataset, observation)
        # No need to do the DL3->DL4 stage
        else:
            dataset_on_off = observation

        # collect the SpectrumDatasetOnOff containers of all the observations
        datasets_short.append(dataset_on_off)

    # loop in order to add the model to the SpectrumDataset container
    for dataset in datasets_short:
        dataset.models = best_fit_spec_model

    # build the estimator of the light curve with the assigned parameters
    if time_bin is not None:
        lc_subrun = LightCurveEstimator(
            energy_edges=[E1, E2],
            reoptimize=False,
            selection_optional='all',
            n_sigma=1,
            n_sigma_ul=2,
            time_intervals=time_intervals
        )
    else:
        lc_subrun = LightCurveEstimator(
            energy_edges=[E1, E2],
            reoptimize=False,
            selection_optional='all',
            n_sigma=1,
            n_sigma_ul=2
        )

    # run the estimator using all the data
    lc_subrun = lc_subrun.run(datasets_short)
    return lc_subrun
