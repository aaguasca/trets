#!/usr/bin/env python
# Licensed under a 3-clause BSD style license - see LICENSE

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


__all__ = ['runwise_lightcurve']

def runwise_lightcurve(E1,
                        E2,
                        e_reco,
                        e_true,
                        on_region,
                        observations,
                        bkg_maker_reflected,
                        best_fit_spec_model
):

    """
    Compute the integral fluxes using all the events in a run.

    parameters
    ----------
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

    E1 = self.e_inf_flux
    E2 = self.e_sup_flux
    e_reco = self.e_reco
    e_true = self.e_true
    on_region = self.on_region
    observations = self.observations
    best_fit_spec_model=self.sky_model
    bkg_maker_reflected=self.bkg_maker_reflected
    geom=RegionGeom.create(region=on_region, axes=[e_reco])
    
    # create the dataset container
    dataset_empty = SpectrumDataset.create( 
        geom, energy_axis_true=e_true
    )
    
    #maker to produce data reduction DL4
    dataset_maker = SpectrumDatasetMaker(
        containment_correction=False, selection=["counts", "exposure", "edisp"]
    )    

    datasets = Datasets()
    #loop for each run
    for obs_id, observation in zip(np.arange(len(observations)), observations):
        dataset = dataset_maker.run(dataset_empty.copy(name=str(obs_id)), observation) 
        dataset_on_off = bkg_maker_reflected.run(dataset, observation)     
        #collect the SpectrumDatasetOnOff containers of all the observations
        datasets.append(dataset_on_off)

    #loop in order to add the model to the SpectrumDataset container
    for dataset in datasets:
        dataset.models = best_fit_spec_model

    #build the estimator of the light curve with the assigned parameteres
    lc=LightCurveEstimator(energy_edges=[E1, E2], 
                           reoptimize=False, 
                           selection_optional='all',
                           n_sigma=1,
                           n_sigma_ul=2)

    #run the LC estimator 
    lightcurve=lc.run(datasets)

    return lightcurve
