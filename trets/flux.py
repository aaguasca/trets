#!/usr/bin/env python
# Licensed under a 3-clause BSD style license - see LICENSE

import time
import ray
from .utils import split_observations
from .methods.intrarun_lc import intrarun_lightcurve
from .methods.runwise_lc import runwise_lightcurve
from .methods.variable import TRETS
import astropy.units as u
from astropy.table import (
    vstack
)
from gammapy.estimators import (
    FluxPoints,
)

__all__=[
    "lightcurve_methods"
]    

class lightcurve_methods:
    """
    Two ways to obtain a light curve:
    - Using a fixed statistical significance of the source (TRETS)
    - Using a fixed time interval (intrarun and runwise light curve)
    
    Three methods to obtain a light curve:
    - TRETS
    - intrarun_lc
    - runwise_lc
    """
    
    def __init__(self,script_name, **kwargs):
        
        #name of the method used to obtain the light curve (TRETS, intrarun_lc or runwise_lc)
        self.script_name=script_name
        
        allowed_keys = self._allowed_keys_script()
        self.__dict__.update((key, value) for key, value in kwargs.items() if key in allowed_keys)
        
        assert self.e_inf_flux.to(self.e_reco.edges.unit).value and self.e_sup_flux.to(self.e_reco.edges.unit).value in self.e_reco.edges.value
        if self.script_name=="TRETS":
            assert len(self.e_reco.center)==3

    def _allowed_keys_script(self):
        """
        Allowed keys for each method
        """  
        
        if self.script_name=="TRETS":
            allowed_keys = {"e_inf_flux",     # Minimum energy bound used to compute the integral flux.
                "e_sup_flux",                 #Maximum energy bound used to compute the integral flux.
                "e_reco",                     #Reconstructed energy axis used in the SpectrumDatasetOnOff object
                "e_true",                     #True energy axis used in the SpectrumDatasetOnOff object
                "on_region",                  #On region located in the source position
                "sqrt_TS_LiMa_threshold",     #Li & Ma significance threshold
                "sqrt_TS_flux_UL_threshold",  #Flux point Fit significance threshold
                "print_check",                #Value that shows different values to check the script. 
                "thres_time_twoobs",          #Threshold time to consider two consecutive runs
                "bin_event",                  #Number of events added in each iteration.
                "observations",               #Observation object with the runs used to compute the light curve.
                "bkg_maker_reflected",        #Background maker to estimate the background.
                "sky_model",                  #Assumed Skymodel of the source
                "parallelization"             #Boolean to use parallelization or not
                }
            
        if self.script_name=="intrarun_lc":
            allowed_keys = {
                "e_inf_flux",                 #""
                "e_sup_flux",                 #""
                "e_reco",                     #""
                "e_true",                     #""
                "interval_subrun",            #The number of time we want the subruns to have.
                "on_region",                  #""
                "observations",               #""
                "bkg_maker_reflected",        #""
                "sky_model"                   #""
                }
            
        if self.script_name=="runwise_lc":
            allowed_keys = {
                "e_inf_flux",                 #""
                "e_sup_flux",                 #""
                "e_reco",                     #""
                "e_true",                     #""
                "on_region",                  #""
                "observations",               #""
                "bkg_maker_reflected",        #""
                "sky_model"                   #""
                }
            
        return allowed_keys            
            
    def run(self):
        """
        Run the light curve methods
        """
        
        t_start = time.time() 
        if self.script_name=="TRETS":
            split_obs=split_observations(self.observations,
                                         self.thres_time_twoobs)

            #TRETS parallelization with ray
            if self.parallelization:
                self.is_ray = True
                ray.init()
                parallelization_TRETS=TRETS(parallelization=self.is_ray)

                futures_list=[]         
                for obs in split_obs:
                    futures=parallelization_TRETS.TRETS_algorithm().remote(
                                        E1=self.e_inf_flux,
                                        E2=self.e_sup_flux,
                                        e_reco=self.e_reco,
                                        e_true=self.e_true,
                                        on_region=self.on_region,
                                        sqrt_TS_LiMa_threshold=self.sqrt_TS_LiMa_threshold,
                                        sqrt_TS_flux_UL_threshold=self.sqrt_TS_flux_UL_threshold,
                                        print_check=self.print_check,
                                        thres_time_twoobs=self.thres_time_twoobs,
                                        bin_event=self.bin_event,
                                        observations=obs,
                                        bkg_maker_reflected=self.bkg_maker_reflected,
                                        best_fit_spec_model=self.sky_model
                    )
                    futures_list.append(futures)
            
                light_curve=[]
                TS_column=[]
                for futures in futures_list:
                    lc,TS_col=ray.get(futures)
                    light_curve.append(lc)
                    TS_column.append(TS_col)

                ray.shutdown()
                TS_column=vstack(TS_column)
                light_curve=vstack(light_curve)
                light_curve.meta.update({"TS-value":TS_column})
                light_curve=FluxPoints.from_table(light_curve, 
                                               reference_model=self.sky_model, 
                                               sed_type="flux", 
                                               format="lightcurve")

            #TRETS without prallelization            
            else:
                self.is_ray = False
  
                light_curve=[]
                TS_column=[]
                #TRETS_local=TRETS(
                algorithm_TRETS=TRETS(parallelization=self.is_ray)
                #light_curve,TS_column=TRETS(
                TRETS_local=algorithm_TRETS.TRETS_algorithm()
                light_curve,TS_column=TRETS_local(
                                        E1=self.e_inf_flux,
                                        E2=self.e_sup_flux,
                                        e_reco=self.e_reco,
                                        e_true=self.e_true,
                                        on_region=self.on_region,
                                        sqrt_TS_LiMa_threshold=self.sqrt_TS_LiMa_threshold,
                                        sqrt_TS_flux_UL_threshold=self.sqrt_TS_flux_UL_threshold,
                                        print_check=self.print_check,
                                        thres_time_twoobs=self.thres_time_twoobs,
                                        bin_event=self.bin_event,
                                        observations=self.observations,
                                        bkg_maker_reflected=self.bkg_maker_reflected,
                                        best_fit_spec_model=self.sky_model

                )
                               
                #light_curve,TS_column=TRETS_local()
                
                light_curve.meta.update({"TS-value":TS_column})
                light_curve=FluxPoints.from_table(light_curve, 
                                               reference_model=self.sky_model, 
                                               sed_type="flux", 
                                               format="lightcurve")

            # TRETS parallelization with multiprocessing  
            #(NO FUNCIONA, crec que pq ja utilitzo pool dins un altre cop)   
#             if __name__ == '__main__':
#                 import contextlib
#                 with contextlib.closing(Pool(processes=None)) as p:
#                     print(split_obs)
#                     light_curve=p.map(self.TRETS, split_obs)
#                     light_curve=vstack(light_curve)
#                     light_curve=LightCurve(light_curve)
#                 p.join()
    
        if self.script_name=="intrarun_lc":
            light_curve=intrarun_lightcurve(
                                    E1=self.e_inf_flux,
                                    E2=self.e_sup_flux,
                                    e_reco=self.e_reco,
                                    e_true=self.e_true,
                                    interval_subrun=self.interval_subrun,
                                    on_region=self.on_region,
                                    observations=self.observations,
                                    bkg_maker_reflected=self.bkg_maker_reflected,
                                    best_fit_spec_model=self.sky_model 
            )
            
        if self.script_name=="runwise_lc":
            light_curve=runwise_lightcurve(
                                    E1=self.e_inf_flux,
                                    E2=self.e_sup_flux,
                                    e_reco=self.e_reco,
                                    e_true=self.e_true,
                                    on_region=self.on_region,
                                    observations=self.observations,
                                    bkg_maker_reflected=self.bkg_maker_reflected,
                                    best_fit_spec_model=self.sky_model
            )
    
        print("Duration: ",int(time.time()-t_start)*u.s,)
        return light_curve

