#!/usr/bin/env python
# Licensed under a 3-clause BSD style license - see LICENSE

from ..utils import conditional_ray, get_intervals
from ..bayes import BayesianProbability
from astropy.time import (
    Time
)
from astropy.units import Quantity, Unit
from astropy.table import Column,vstack, Table

import ray
import numpy as np
import matplotlib.pyplot as plt
import copy
from gammapy.makers import (
    SpectrumDatasetMaker
)
from gammapy.maps import (
    RegionGeom
)
from gammapy.data import (
    ObservationFilter
)
from gammapy.datasets import (
    Datasets,
    SpectrumDataset,
)
from gammapy.estimators import (
    LightCurveEstimator,
)
from gammapy.stats import (
    WStatCountsStatistic,
)


__all__ = ['TRETS']


class TRETS:
    def __init__(self,parallelization):
        if parallelization:
            self.is_ray=True
#            ray.init()
        else:
            self.is_ray=False
            
    @conditional_ray('is_ray')
    def TRETS_algorithm(E1,
              E2,
              e_reco,
              e_true,
              on_region,
              sqrt_TS_LiMa_threshold,
              sqrt_TS_flux_UL_threshold,
              print_check,
              thres_time_twoobs,
              bin_event,
              observations,
              bkg_maker_reflected,
              best_fit_spec_model
    ):  
        """
        TRETS algorithm. Computes the light curve between energies [E1,E2] where in each integral flux, 
        the number of events in that time interval gives a detection significance of the source of 
        TS="sqrt_TS_LiMa_threshold"**2 and the fit statistics is higher than "sqrt_TS_flux_UL_threshold". 
        The flux point is computed assuming a certain Skymodel "best_fit_spec_model" (spectral model).
    
        parameters
        ----------
        E1: astopy.units
            Minimum energy bound used to compute the integral flux. It must be conside with one center
            in e_reco.
        E2: astopy.units
            Maximum energy bound used to compute the integral flux. It must be conside with one center
            in e_reco.       
        e_reco:
            Reconstructed energy axis used in the SpectrumDatasetOnOff object
        e_true:
            True energy axis used in the SpectrumDatasetOnOff object
        on_region:
            On region located in the source position, it must has the same size used in the IRFs.

        sqrt_TS_LiMa_threshold:
            Li & Ma significance threshold used to compute a integral flux.
        sqrt_TS_flux_UL_threshold:
            Fit significance threshold to save the integral flux as a flux point.
        print_check: integer
            Value that shows different values to check the script. 
                0 -> no values are displayed
                1 -> print for each flux: events and run.
                2 -> prints for each interation: events, time integral, ...

        thres_time_twoobs: astropy.Units
            Threshold time to consider two consecutive runs. Use events from several runs 
            to compute the integral flux.
        bin_event: Integer
            Number of events added in each iteration.
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

        TS_column: ~astropy.table.Table
            Detection TS of the source.

        """        
    
        geom=RegionGeom.create(region=on_region, axes=[e_reco])

        dataset_empty = SpectrumDataset.create( # create the dataset container object for the likelihood fitting of the spectrum on the ON region
            geom, energy_axis_true=e_true
        )
        dataset_maker = SpectrumDatasetMaker( #maker to produce data reduction in order to obtain spectrum
            containment_correction=False, selection=["counts", "exposure", "edisp"] # make this maps
        )        

        print("Run IDs considered in this execution: ",observations.ids)

        #list where the the flux is saved
        lc_list=[]
        # array to store TS for each time interval
        ts_array=[]
        #dataset container where I keep the counts from previous observations
        prev_dataset = Datasets() 
        keep_dataset_bool=False
        bool_pass=False
        lc=LightCurveEstimator(energy_edges=[E1,E2], reoptimize=False, n_sigma=1,
                               selection_optional='all', atol=Quantity(1e-6,"s"))

        for i,obs in zip(observations.ids,observations):

            n_event=0
            print( )
            print( )
            print("OBSERVATION ID: %i. Total number of events: %i" %(obs.obs_id,len(obs.events.table)))
            print("OBS gti Tstart (TT):",observations[0].gti.time_start.tt.iso)
            print("OBS gti Tstop (TT):",observations[0].gti.time_stop.tt.iso)
            print("first event time (TT)",(observations[0].gti.time_ref.tt+\
                                      Quantity(observations[0].events.table["TIME"][0], "second")).iso)
            print("last event time (TT)", (observations[0].gti.time_ref.tt+\
                                      Quantity(observations[0].events.table["TIME"][-1], "second")).iso)

            event_id=obs.events.table["EVENT_ID"].data
            event_time=obs.events.table["TIME"].data
            array_event_ID_sorted=np.sort(obs.events.table["EVENT_ID"].data)


            list_t=get_intervals(obs.events.table["TIME"].data,bin_event)
            time_array=obs.gti.time_ref.tt+Quantity(list_t,"s")
            arg_start_time=0



            for cont_t in range(len(time_array)-1):

                if bool_pass:
                    pass
                else:
                    arg_start_time=cont_t
                arg_stop_time=cont_t+1   

                #filter the observation
                filters=ObservationFilter(time_filter=Time([time_array[arg_start_time],time_array[arg_stop_time]]))
                #copy the observation because the filter deletes all events out of the ID range specified
                subobs=copy.deepcopy(obs)
                #apply the filter
                subobs.obs_filter = filters

                t_0=time_array[arg_start_time]
                t_end=time_array[arg_stop_time]          

                #data reduction
                dataset = dataset_maker.run(dataset_empty.copy(name=str(n_event)), subobs)
                dataset_on_off = bkg_maker_reflected.run(dataset, subobs)

                #we do not have data from previous runs
                if len(prev_dataset)==0:
                    datasets_ONOFF=dataset_on_off
                #we have data to analyse from the previous runs
                else:
                    #create a temporal Dataset container with the dataset_on_off of the previous night
                    temp_dataset=prev_dataset.copy()
                    print(len(temp_dataset))
                    print("num events prev dataset:", Datasets(temp_dataset).stack_reduce().counts_off.data.sum()+Datasets(temp_dataset).stack_reduce().counts.data.sum())
                    print("num events this dataset:", dataset_on_off.counts_off.data.sum()+dataset_on_off.counts.data.sum())
                    #append the new dataset_on_off 
                    temp_dataset.append(dataset_on_off)
                    #stack the observations          
                    datasets_ONOFF=Datasets(temp_dataset).stack_reduce()

                #compute acceptance in the On/Off region
                acc_off = datasets_ONOFF.acceptance_off.data.sum()
                acc_on = datasets_ONOFF.acceptance.data.sum()

                #compute the number of counts in the On/Off region
                n_on=datasets_ONOFF.counts.data.sum()
                n_off=datasets_ONOFF.counts_off.data.sum()

                #compute W statistics
                stat=WStatCountsStatistic(n_on=n_on, n_off=n_off, alpha=acc_on/acc_off)
                ts_bin=stat.ts
                ts_sum=ts_bin
                sqrt_ts_LiMa=np.sqrt(ts_sum)
                #total TS obtained from the sum of the different bins (ts_sum is without considering bins)
                stat_sum_3bins_dataset=datasets_ONOFF.stat_sum()

                #!!!!!!!!CHECKS!!!!!!!!
                if print_check==2:
                    #també es pot comprovat sumant el temps "s" de la taula amb gti de la dataset
                    print( )
                    print( )
                    print(arg_start_time," to ",arg_stop_time,", total args:", len(time_array)-1)
                    print("OBSERVATION ID:", obs.obs_id)
                    print(f"start time observation {i}, {obs.gti.time_start.tt.iso}")
                    print(f"stop time observation {i}, {obs.gti.time_stop.tt.iso}")
                    print("arg in array_event_ID_sorted first event", np.argwhere(event_time==list_t[arg_start_time])[0,0])
                    print("arg in array_event_ID_sorted last event", np.argwhere(event_time==list_t[arg_stop_time])[0,0])
                    print("time first event", t_0.tt.iso)
                    print("-check gti start subobs-", subobs.gti.time_start.tt.iso)
                    print("time last event", t_end.tt.iso)
                    print("-check gti end subobs-", subobs.gti.time_stop.tt.iso)
                    print("dataset_on_off gti Tstart:",dataset_on_off.gti.time_start.tt.iso)
                    print("dataset_on_off gti Tstop:",dataset_on_off.gti.time_stop.tt.iso)
                    print("datasets_ONOFF gti Tstart:",datasets_ONOFF.gti.time_start.tt.iso)
                    print("datasets_ONOFF gti Tstop:",datasets_ONOFF.gti.time_stop.tt.iso)
                    print("dataset gti Tstart:",dataset.gti.time_start.tt.iso)
                    print("dataset gti Tstop:",dataset.gti.time_stop.tt.iso)                
                    print("first event ID:",event_id[np.argwhere(event_time==list_t[arg_start_time])][0,0])
                    print("last event ID:", event_id[np.argwhere(event_time==list_t[arg_stop_time])][0,0])
                    print("num events subobs:",len(subobs.events.table["EVENT_ID"]))
                    print("num events dataset:", datasets_ONOFF.counts_off.data.sum()+datasets_ONOFF.counts.data.sum())
                    print("non, noff,        TS,             sqrt TS")
                    print(n_on,n_off,ts_sum,sqrt_ts_LiMa)
                    print("counts On region:",datasets_ONOFF.counts.data.reshape(-1))
                    print("counts Off region:",datasets_ONOFF.background.data.reshape(-1))
                    print("excess counts:",datasets_ONOFF.excess.data.reshape(-1))    
                    print("stat_sum TS bins", stat_sum_3bins_dataset)


    #######################  NO passa el thd de sig ###################
                if n_on<10 or n_off<10 or sqrt_ts_LiMa<sqrt_TS_LiMa_threshold:# loop until the bin satisfy these conditions

                    bool_pass=True

                    if arg_stop_time==len(time_array)-1:

                        # there is another observation in the observation container
                        if np.argwhere(np.array(observations.ids)==i)[0,0]<len(observations)-1:

                            # time between two observations lower than the threshold
                            if (observations[int(np.argwhere(np.array(observations.ids)==i)[0,0]+1)].gti.time_start.tt - \
                                obs.gti.time_stop.tt).to(thres_time_twoobs.unit) < thres_time_twoobs:

                                save_dataset_on_off=dataset_on_off.copy(name="counts left obsid %s" %(i))
                                #safe the dataset_on_off to keep it for the next observation loop
                                prev_dataset.append(save_dataset_on_off)
                                prev_events=len(subobs.events.table["EVENT_ID"])

                                sqrt_ts_LiMa_ul=sqrt_ts_LiMa
                                keep_dataset_bool=True

                                if print_check==2:
                                    print("+++++++++ Append dataset from obs %s in the dataset container +++++++++" %(i))


                            else:# time between two observations higher than the threshold
                                sqrt_ts_LiMa_ul=sqrt_ts_LiMa
                                keep_dataset_bool=False

                                #delete the memory of any dataset_on_off saved previously for the future run
                                prev_dataset = Datasets()

                        else:# is the last observation
                            sqrt_ts_LiMa_ul=sqrt_ts_LiMa
                            keep_dataset_bool=False      


                        #compute UL
                        #comupte flux upper limit when the next observaiton starts later than the threshold
                        if keep_dataset_bool==False: 
                            print("&&&&&&&&&&&&&&&&&&&&&&&&&")
                            print("LAST TIME INTERVAL! n_on<10 or n_off<10 or sqrt_ts_LiMa<sqrt_TS_LiMa_threshold and NO keep_dataset_bool")
                            print("&&&&&&&&&&&&&&&&&&&&&&&&&")
                            datasets_ONOFF.models=best_fit_spec_model#set the model we use to estimate the light curve

                            #estimate the light curve of the bin
                            lc_f=lc.run(copy.deepcopy(datasets_ONOFF))
                            lc_f_table=lc_f.to_table(sed_type="flux", format="lightcurve")

                            #Consider this point as UL
                            lc_f_table["is_ul"].data[0]=np.ones(shape=np.shape(lc_f_table["is_ul"].data[0]),dtype=bool)  

                            lc_list.append(lc_f_table)

                            if print_check==1 or print_check==2:
                                print("---------------------------------------------------------------------------")
                                print("events' bin: %.0f to %.0f.  sqrt(TS)_Li&Ma=%.1f -> UL!" %(np.argwhere(event_time==list_t[arg_start_time])[0,0],
                                                                                                 np.argwhere(event_time==list_t[arg_stop_time])[0,0],
                                                                                                 sqrt_ts_LiMa_ul))
                                print("---------------------------------------------------------------------------")

                            #!!!!!!!!CHECKS!!!!!!!!
                            if print_check==2:
                                print("time_min",Time(lc_f_table["time_min"][0],scale="tt",format="mjd").tt.iso)
                                print("time_max",Time(lc_f_table["time_max"][0],scale="tt",format="mjd").tt.iso)
                                print("dataset_on_off Tstart:",dataset_on_off.gti.time_start.tt.iso)
                                print("dataset_on_off Tstop:",dataset_on_off.gti.time_stop.tt.iso)
                                print("datasets_ONOFF Tstart:",datasets_ONOFF.gti.time_start.tt.iso)
                                print("datasets_ONOFF Tstop:",datasets_ONOFF.gti.time_stop.tt.iso)
                                print("---------------------------------------------------------------------------")
                            ts_array.append(ts_sum)
                        if print_check == 1 or print_check == 2:
                            print( )                    



    ################## SI passa el thd de sig ####################
                if n_off>10 and n_on>10 and sqrt_ts_LiMa>sqrt_TS_LiMa_threshold: #comupte flux

                    datasets_ONOFF.models=best_fit_spec_model#set the model we use to estimate the light curve

                    #estimate the light curve of the bin
                    lc_f=lc.run(copy.deepcopy(datasets_ONOFF))
                    lc_f_table=lc_f.to_table(sed_type="flux", format="lightcurve")

                    sqrt_ts_flux=lc_f_table["sqrt_ts"][0,0]

                    # CHECK SQRT TS OF THE FLUX IS HIGHER THAN THE THRESHOLD
                    ######
                    #if the flux point has a sqrt_TS>sqrt_TS_flux_UL_thd, we accept, if not, we keep adding events
                    if sqrt_ts_flux<sqrt_TS_flux_UL_threshold:

                        bool_pass=True

                        if arg_stop_time==len(time_array)-1:
                            bool_last_even=False
                            # there is another observation in the observation container
                            if np.argwhere(np.array(observations.ids)==i)[0,0]<len(observations)-1:

                                # time between two observations lower than the threshold
                                if (observations[int(np.argwhere(np.array(observations.ids)==i)[0,0]+1)].gti.time_start.tt - \
                                    obs.gti.time_stop.tt).to(thres_time_twoobs.unit) < thres_time_twoobs:

                                    save_dataset_on_off=dataset_on_off.copy(name="counts left obsid %s" %(i))
                                    #safe the dataset_on_off to keep it for the next observation loop
                                    prev_dataset.append(save_dataset_on_off)
                                    prev_events=len(subobs.events.table["EVENT_ID"])

                                    sqrt_ts_flux_ul=sqrt_ts_flux
                                    sqrt_ts_LiMa_ul=sqrt_ts_LiMa
                                    keep_dataset_bool=True

                                    if print_check==2:
                                        print("+++++++++ Append dataset from obs %s in the dataset container +++++++++" %(i))


                                else:# time between two observations higher than the threshold
                                    sqrt_ts_flux_ul=sqrt_ts_flux
                                    sqrt_ts_LiMa_ul=sqrt_ts_LiMa
                                    keep_dataset_bool=False

                                    #delete the memory of any dataset_on_off saved previously for the future run
                                    prev_dataset = Datasets()

                            else:# is the last observation
                                sqrt_ts_flux_ul=sqrt_ts_flux
                                sqrt_ts_LiMa_ul=sqrt_ts_LiMa
                                keep_dataset_bool=False


                            #compute flux UL
                            if keep_dataset_bool==False:    
                                print("&&&&&&&&&&&&&&&&&&&&&&&&&")
                                print("LAST TIME INTERVAL! n_on>10, n_off>10, sqrt_ts_LiMa>sqrt_TS_LiMa_threshold but sqrt_ts_flux<sqrt_TS_flux_UL_threshold NO keep_dataset_bool")                            
                                print("&&&&&&&&&&&&&&&&&&&&&&&&&")

                                #consider this point as UL
                                lc_f_table["is_ul"].data[0]=np.ones(shape=np.shape(lc_f_table["is_ul"].data[0]),dtype=bool)

                                lc_list.append(lc_f_table)

                                if print_check==1 or print_check==2:
                                    print("---------------------------------------------------------------------------")
                                    print("events' bin: %.0f to %.0f.  sqrt(TS)_Li&Ma=%.1f"+\
                                          ", sqrt(TS)_flux=%.1f -> UL!" %(np.argwhere(event_time==list_t[arg_start_time])[0,0],
                                                                          np.argwhere(event_time==list_t[arg_stop_time])[0,0],
                                                                          sqrt_ts_LiMa_ul,sqrt_ts_flux_ul))
                                    print("---------------------------------------------------------------------------")

                                #!!!!!!!!CHECKS!!!!!!!!
                                if print_check==2:
                                    print("time_min",Time(lc_f_table["time_min"][0],scale="tt",format="mjd").tt.iso)
                                    print("time_max",Time(lc_f_table["time_max"][0],scale="tt",format="mjd").tt.iso)
                                    print("dataset_on_off Tstart:",dataset_on_off.gti.time_start.tt.iso)
                                    print("dataset_on_off Tstop:",dataset_on_off.gti.time_stop.tt.iso)
                                    print("datasets_ONOFF Tstart:",datasets_ONOFF.gti.time_start.tt.iso)
                                    print("datasets_ONOFF Tstop:",datasets_ONOFF.gti.time_stop.tt.iso)
                                    print("---------------------------------------------------------------------------")
                                ts_array.append(ts_sum)                            


                    #we safe the flux point
                    else:
                        bool_pass=False
                        print("&&&&&&&&&&&&&&&&&&&&&&&&&")
                        print("LAST TIME INTERVAL! n_on>10, n_off>10, sqrt_ts_LiMa>sqrt_TS_LiMa_threshold, sqrt_ts_flux>sqrt_TS_flux_UL_threshold")
                        print("&&&&&&&&&&&&&&&&&&&&&&&&&")

                        #not an upper limit
                        lc_f_table["is_ul"].data[0]=np.zeros(shape=np.shape(lc_f_table["is_ul"].data[0]),dtype=bool)

                        lc_list.append(lc_f.to_table(sed_type="flux", format="lightcurve"))

                        if keep_dataset_bool==True:#primer cop que calculo flux tenint en compte obs anterior
                            keep_dataset_bool=False
                            #as we have succesfully calculed the flux, delete the previous counts of the dataset container
                            prev_dataset = Datasets()
                            if print_check==1 or print_check==2:
                                print("---------------------------------------------------------------------------")
                                print("events' bin: %i to %i with %i events from previous obs. sqrt(TS)_Li&Ma=%.1f, sqrt(TS)_flux=%.1f" \
                                                  %(first_event,n_event,prev_events,sqrt_ts_LiMa,sqrt_ts_flux))

                        else:
                            if print_check==1 or print_check==2:
                                print("---------------------------------------------------------------------------")                            
                                print("events' bin: %.0f to %.0f. sqrt(TS)_Li&Ma=%.1f, sqrt(TS)_flux=%.1f" %(np.argwhere(event_time==list_t[arg_start_time])[0,0],
                                                                                                             np.argwhere(event_time==list_t[arg_stop_time])[0,0],
                                                                                                             sqrt_ts_LiMa,sqrt_ts_flux))
                        #!!!!!!!!CHECKS!!!!!!!!
                        if print_check==2:
                            print("---------------------------------------------------------------------------")
                            print("time_min",Time(lc_f_table["time_min"][0],scale="tt",format="mjd").tt.iso)
                            print("time_max",Time(lc_f_table["time_max"][0],scale="tt",format="mjd").tt.iso)
                            print("dataset_on_off Tstart:",dataset_on_off.gti.time_start.tt.iso)
                            print("dataset_on_off Tstop:",dataset_on_off.gti.time_stop.tt.iso)
                            print("datasets_ONOFF Tstart:",datasets_ONOFF.gti.time_start.tt.iso)
                            print("datasets_ONOFF Tstop:",datasets_ONOFF.gti.time_stop.tt.iso)
                            print("---------------------------------------------------------------------------")
                        ts_array.append(ts_sum)



        #stack al the flux point tables into a single table  
        light_curve=vstack(lc_list)

        #add the Li&Ma TS detection value
        TS_column=Table(data=[ts_array], names=["Li&Ma_TS_detection"], dtype=[np.float32])
        light_curve.meta.update({"TS-thd":sqrt_TS_LiMa_threshold**2})
        return light_curve.copy(),TS_column.copy()

