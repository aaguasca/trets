#!/usr/bin/env python
# Licensed under a 3-clause BSD style license - see LICENSE

import time
import ray
from .utils import (
    split_observations,
    fraction_outside_interval,
variance_error_prop_calculation
)
from .methods.fixed import intrarun
from .methods.variable import TRETS
import astropy.units as u
from astropy.table import (
    vstack
)
from gammapy.estimators import (
    FluxPoints,
)

__all__=[
    "lightcurve_methods",
    "weight_fluxes"
]    

class lightcurve_methods:
    """
    Two ways to obtain a light curve:
    - Using a fixed statistical significance of the source (TRETS)
    - Using a fixed time interval (intrarun and runwise light curve)
    
    Three methods to obtain a light curve:
    - TRETS
    - intrarun
    - runwise
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
                "sig_threshold",              #Significance detection threshold
                "sqrt_TS_flux_UL_threshold",  #Flux point Fit significance threshold
                "print_check",                #Value that shows different values to check the script. 
                "thres_time_twoobs",          #Threshold time to consider two consecutive runs
                "bin_event",                  #Number of events added in each iteration.
                "observations",               #Observation object with the runs used to compute the light curve.
                "bkg_maker_reflected",        #Background maker to estimate the background.
                "sky_model",                  #Assumed Skymodel of the source
                "bool_bayesian",              #Boolean to use or not bayesian approach
                "parallelization"             #Boolean to use parallelization or not
                }
            
        if self.script_name=="intrarun":
            allowed_keys = {
                "e_inf_flux",                 #""
                "e_sup_flux",                 #""
                "e_reco",                     #""
                "e_true",                     #""
                "time_bin",                   #The number of time we want the subruns to have.
                "on_region",                  #""
                "observations",               #""
                "bkg_maker_reflected",        #""
                "sky_model"                   #""
                }
            
        if self.script_name=="runwise":
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
                                        sig_threshold=self.sig_threshold,
                                        sqrt_TS_flux_UL_threshold=self.sqrt_TS_flux_UL_threshold,
                                        print_check=self.print_check,
                                        thres_time_twoobs=self.thres_time_twoobs,
                                        bin_event=self.bin_event,
                                        observations=obs,
                                        bkg_maker_reflected=self.bkg_maker_reflected,
                                        best_fit_spec_model=self.sky_model,
                                        bool_bayesian=self.bool_bayesian
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
                                        sig_threshold=self.sig_threshold,
                                        sqrt_TS_flux_UL_threshold=self.sqrt_TS_flux_UL_threshold,
                                        print_check=self.print_check,
                                        thres_time_twoobs=self.thres_time_twoobs,
                                        bin_event=self.bin_event,
                                        observations=self.observations,
                                        bkg_maker_reflected=self.bkg_maker_reflected,
                                        best_fit_spec_model=self.sky_model,
                                        bool_bayesian=self.bool_bayesian
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
    
        if self.script_name=="intrarun":
            light_curve=intrarun(
                                    E1=self.e_inf_flux,
                                    E2=self.e_sup_flux,
                                    e_reco=self.e_reco,
                                    e_true=self.e_true,
                                    time_bin=self.time_bin,
                                    on_region=self.on_region,
                                    observations=self.observations,
                                    bkg_maker_reflected=self.bkg_maker_reflected,
                                    best_fit_spec_model=self.sky_model 
            )
            
        if self.script_name=="runwise":
            light_curve=intrarun(
                                    E1=self.e_inf_flux,
                                    E2=self.e_sup_flux,
                                    e_reco=self.e_reco,
                                    e_true=self.e_true,
                                    on_region=self.on_region,
                                    observations=self.observations,
                                    bkg_maker_reflected=self.bkg_maker_reflected,
                                    best_fit_spec_model=self.sky_model,
            )
    
        print("Duration: ",int(time.time()-t_start)*u.s,)
        return light_curve


def weight_fluxes(fluxes_to_weight,reference_fluxes):
    """
    Weight 
    
    Parameters
    ----------
    fluxes_to_weight:
        Table with fluxes to weight using the same time bin as the one is 
        use for the reference_fluxes
    reference_fluxes:
        Table with the fluxes we want to compare with fluxes_to_weight
    
    Returns
    -------
    list_averange_flux: list
        List with the weighted average fluxes from fluxes_to_weight.
    lst_std_flux: list
        List with the standard error of the weighted average fluxes.
    center_time: list
        List with the central time value of the time bin of the weighted fluxes.
    ref_flux: list
        List with the reference flux value.
    ref_errflux: list
         List with the reference flux error.
    ref_center_time: list
        List with the central time value of the time bin of the reference fluxes.
    """
    
    list_averange_flux=[]
    lst_std_flux=[]
    ref_flux=[]
    ref_center_time=[]
    ref_errflux=[]
    center_time=[]
    for flux in reference_fluxes:
        int_time=[]
        int_flux=[]
        int_errflux=[]
        time=[]
        weight_tinside=[]
        weight_ul=[]

        for w_flux in fluxes_to_weight:
            
            #flux to weight inside the time interval of the reference flux
            if round(w_flux["time_min"],9)>=round(flux["time_min"],9) and round(w_flux["time_max"],9)<=round(flux["time_max"],9):

                #TRETS flux
                if "sig-THD" in fluxes_to_weight.meta and "sig_detection" in fluxes_to_weight.colnames:
                    if w_flux["sig_detection"]>=fluxes_to_weight.meta["sig-THD"]:
                        int_flux.append(w_flux["flux"][0])
                        int_errflux.append(w_flux["flux_err"][0])
                        weight_ul.append(1)
                        
                    #UL
                    else:
                        int_flux.append(w_flux["flux_ul"][0])
                        int_errflux.append(0.5*w_flux["flux_ul"][0])
                        weight_ul.append(1+fluxes_to_weight.meta["sig-THD"]-w_flux["sig_detection"])

                #fixed time interval flux
                else:     
                    if w_flux["sqrt_ts"][0]>2:
                        int_flux.append(w_flux["flux"][0])
                        int_errflux.append(w_flux["flux_err"][0])
                        weight_ul.append(1)      
                    #UL    
                    else:
                        int_flux.append(w_flux["flux_ul"][0])
                        int_errflux.append(0.5*w_flux["flux_ul"][0])
                        weight_ul.append(1+4-w_flux["ts"][0])
                        
                int_time.append(w_flux["time_max"]-w_flux["time_min"])
                time.append(w_flux["time_max"])
                time.append(w_flux["time_min"])
                #the time interval is inside the reference flux time window
                weight_tinside.append(1)

            #starts inside the time bin of the reference flux but stops later than the reference flux
            if round(w_flux["time_min"],9)>=round(flux["time_min"],9) and round(w_flux["time_min"],9)<=round(flux["time_max"],9) and round(w_flux["time_max"],9)>round(flux["time_max"],9):
                t_out=fraction_outside_interval(x=[round(w_flux["time_min"],9),round(w_flux["time_max"],9)],
                                                xmin=round(flux["time_min"],9),
                                                xmax=round(flux["time_max"],9))

                #TRETS flux
                if "sig-THD" in fluxes_to_weight.meta and "sig_detection" in fluxes_to_weight.colnames:
                    if w_flux["sig_detection"]>=fluxes_to_weight.meta["sig-THD"]:
                        int_flux.append(w_flux["flux"][0])
                        int_errflux.append(w_flux["flux_err"][0])
                        weight_ul.append(1)  
                    #UL
                    else:
                        int_flux.append(w_flux["flux_ul"][0])
                        int_errflux.append(0.5*w_flux["flux_ul"][0])
                        weight_ul.append(1+fluxes_to_weight.meta["sig-THD"]-w_flux["sig_detection"])   
                        
                #fixed time interval flux
                else:            
                    if w_flux["sqrt_ts"][0]>2:
                        int_flux.append(w_flux["flux"][0])
                        int_errflux.append(w_flux["flux_err"][0])
                        weight_ul.append(1)
                    #UL
                    else:                               
                        int_flux.append(w_flux["flux_ul"][0])
                        int_errflux.append(0.5*w_flux["flux_ul"][0])
                        weight_ul.append(1+4-w_flux["ts"][0])  
                        
                int_time.append(w_flux["time_max"]-w_flux["time_min"])
                time.append(w_flux["time_max"])
                time.append(w_flux["time_min"])
                #not all the flux is inside the reference flux
                weight_tinside.append(1-t_out)                        


            #ends inside the time bin of the refence flux but starts before the reference flux
            if round(w_flux["time_max"],9)>=round(flux["time_min"],9) and round(w_flux["time_max"],9)<=round(flux["time_max"],9) and round(w_flux["time_min"],9)<round(flux["time_min"],9):
                t_out=fraction_outside_interval(x=[round(w_flux["time_min"],9),round(w_flux["time_max"],9)],
                                                xmin=round(flux["time_min"],9),
                                                xmax=round(flux["time_max"],9))

                #TRETS flux
                if "sig-THD" in fluxes_to_weight.meta and "sig_detection" in fluxes_to_weight.colnames:
                    if w_flux["sig_detection"]>=fluxes_to_weight.meta["sig-THD"]:
                        int_flux.append(w_flux["flux"][0])
                        int_errflux.append(w_flux["flux_err"][0])
                        weight_ul.append(1)   
                    #UL
                    else:
                        int_flux.append(w_flux["flux_ul"][0])
                        int_errflux.append(0.5*w_flux["flux_ul"][0])
                        weight_ul.append(1+fluxes_to_weight.meta["sig-THD"]-w_flux["sig_detection"])
                        
                #fixed time interval flux
                else:
                    if w_flux["sqrt_ts"][0]>2:
                        int_flux.append(w_flux["flux"][0])
                        int_errflux.append(w_flux["flux_err"][0])
                        weight_ul.append(1)
                    #UL
                    else:    
                        int_flux.append(w_flux["flux_ul"][0])
                        int_errflux.append(0.5*w_flux["flux_ul"][0])
                        weight_ul.append(1+4-w_flux["ts"][0])
                        
                int_time.append(w_flux["time_max"]-w_flux["time_min"])
                time.append(w_flux["time_max"])
                time.append(w_flux["time_min"])
                #not all the flux is inside the reference flux
                weight_tinside.append(1-t_out)                        

        int_time=np.array(int_time)
        int_flux=np.array(int_flux)

        #if we have one or more fluxes in the subrun obs
        if len(int_time)!=0:
            #weighted as the 1/variance
#            weights=(1/np.array(int_errflux)**2)
            weights=(1/np.array(int_errflux)**2)*(np.array(weight_tinside)/np.array(weight_ul))

            flux_weighted_av=np.average(int_flux, weights=weights)
            if len(int_time)>1:

                #variance of the distribution of the weighted fluxes
                #variance=np.average((int_flux-flux_weighted_av)**2, weights=weights)
                #variance using error propagation of the weighted average
                variance=variance_error_prop_calculation(int_errflux,weights)

            else:
                variance=np.array(int_errflux[0])**2
            list_averange_flux.append(flux_weighted_av)
            lst_std_flux.append(variance**0.5)

            center_time.append((flux["time_min"]+flux["time_max"])/2)

            ref_flux.append(flux["flux"][0])
            ref_errflux.append(flux["flux_err"][0])
            ref_center_time.append((flux["time_min"]+flux["time_max"])/2)
            
        else:
            print("no fluxes in time bin [{},{}] MJD".format(flux["time_min"],flux["time_min"]))
            
    return list_averange_flux, lst_std_flux, center_time, ref_flux, ref_errflux, ref_center_time

