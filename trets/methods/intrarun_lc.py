#!/usr/bin/env python
# Licensed under a 3-clause BSD style license - see LICENSE

import sys
from ..utils import subrun_split

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

__all__ = ['intrarun_lightcurve']

def intrarun_lightcurve(interval_subrun,
                        E1,
                        E2,
                        e_reco,
                        e_true,
                        on_region,
                        observations,
                        bkg_maker_reflected,
                        best_fit_spec_model
):

    """
    Compute the integral fluxes using subrun events.

    parameters
    ----------
    interval_subrun: astopy.Quantity
        The number of time we want the subruns to have.
    E1: astopy.Units
        Minimum energy bound used to compute the integral flux. It must be conside with one center
        in e_reco.
    E2: astopy.Units
        Maximum energy bound used to compute the integral flux. It must be conside with one center
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
        Assumed Skymodel of the source. Only spectral model.

    return
    ------
    lc_subrun: LightCurve
        Light curve object.    
    """

    geom=RegionGeom.create(region=on_region, axes=[e_reco])

    # create the dataset container object of the spectrum on the ON region
    dataset_empty = SpectrumDataset.create( 
        geom, energy_axis_true=e_true
    )
    
    #maker to produce data reduction to DL4
    dataset_maker = SpectrumDatasetMaker( 
        containment_correction=False, selection=["counts", "exposure", "edisp"] # make this maps
    )       

    time_interval_obs=[]
    for run in range(len(observations)):
        time_interval_obs.append(Time([observations[run].events.observation_time_start,
                                    observations[run].events.observation_time_stop]))


    time_intervals=subrun_split(interval_subrun, time_interval_obs)

    #divide the runs into different subruns
    short_observations = observations.select_time(time_intervals)

    # prepair again the Dataset
    datasets_short = Datasets()

    #loop for each run
    for obs_id, observation in zip(np.arange(len(time_intervals)), short_observations):

        dataset = dataset_maker.run(dataset_empty.copy(name=str(obs_id)), observation) 
        dataset_on_off = bkg_maker_reflected.run(dataset, observation)     
        #collect the SpectrumDatasetOnOff containers of all the observations
        datasets_short.append(dataset_on_off)

    #loop in order to add the model to the SpectrumDataset container
    for dataset in datasets_short:
        dataset.models = best_fit_spec_model

    #build the estimator of the light curve with the assigned parameteres
    lc_subrun=LightCurveEstimator(energy_edges=[E1, E2], 
                           reoptimize=False, 
                           selection_optional='all',
                           n_sigma=1,
                           n_sigma_ul=2,
                           time_intervals=time_intervals)

    #run the estimator using all the data
    lc_subrun=lc_subrun.run(datasets_short)
    return lc_subrun
