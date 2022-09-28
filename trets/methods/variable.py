#!/usr/bin/env python
# Licensed under a 3-clause BSD style license - see LICENSE

from ..utils import conditional_ray
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
        bool_last_event=False
        lc=LightCurveEstimator(energy_edges=[E1,E2], reoptimize=False, n_sigma=1,
                               selection_optional='all', atol=Quantity(1e-6,"s"))

        for i,obs in zip(observations.ids,observations):

            n_event=0
            print("OBSERVATION ID: %i. Total number of events: %i" %(obs.obs_id,len(obs.events.table)))
            event_id=obs.events.table["EVENT_ID"]
            array_event_ID_sorted=np.sort(obs.events.table["EVENT_ID"].data)


            while n_event<(len(event_id)-1): #loop that is repeated while we do not arrive to the last event

                first_event=n_event
                n_event=n_event+bin_event

                #print("primer while",(len(event_id)-1)-n_event,bool_last_event)

                if (len(event_id)-1)-n_event<0:#check if the last event in the bin exceed the max # of events in the run
                    n_event=len(event_id)-1 #if so, use the last one
                    bool_last_event=True

                #parameters of the statistic 
                n_off = 0 # counts in the off regions
                n_on = 0 # counts in the on region
                sqrt_ts_LiMa = 0 # square of the TS       

                #filter: filter using the ID of the event (which is sorted by time). This is faster than
                # subobs.select_time even replacing the GTI                
                event_filter = {'type': 'custom', 'opts': dict(parameter='EVENT_ID',\
                                band=(array_event_ID_sorted[first_event], array_event_ID_sorted[n_event]))}
                filters=ObservationFilter(event_filters=[event_filter])

                #copy the observation because the filter deletes all events out of the ID range specified
                subobs=copy.deepcopy(obs)
                #apply the filter
                subobs.obs_filter = filters

                #obtain the time at which the first and last event were recorded. 
                #This is done bc the filtered obs still has the gti of the whole observation
                    #obtain the position of the events in the table
                arg_first_event=np.argwhere(obs.events.table["EVENT_ID"]\
                                            ==array_event_ID_sorted[first_event])
                arg_last_event=np.argwhere(obs.events.table["EVENT_ID"]==array_event_ID_sorted[n_event])

                #obtain the time of the event (time in "s" wrt the ref. time)
                t0=obs.events.table["TIME"][arg_first_event[0,0]]
                tend=obs.events.table["TIME"][arg_last_event[0,0]]

                #copy the gti object and replace the start and stop time. If not, the GIT of the dataset
                #is not correct
                new_gti=subobs.gti.copy()
                new_gti.table["START"]=t0
                new_gti.table["STOP"]=tend
                subobs.__dict__["__gti_hdu"]=new_gti #replace the old gti for the new one
                subobs.__dict__["_gti"]=subobs.__dict__["__gti_hdu"]# create a new parameter in the dict named "_git"
                                                                    # this is done bc "__gti_hdu" parameter
                                                                # search the git in a HDU file, but my new gti is not 
                                                                    # in a HDU file, it is in the code itself
                del subobs.__dict__["__gti_hdu"]#delete the parameter "__git_hdu"
                if print_check==2:
                    #obtain the time in mjd
                    t_0=obs.gti.time_ref+Quantity(t0, "second")
                    t_end=obs.gti.time_ref+Quantity(tend, "second")

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

                sign=np.sign(n_on-n_off*(acc_on/acc_off))
                sqrt_ts_LiMa=sign*np.sqrt(ts_sum)


                #total TS obtained from the sum of the different bins (ts_sum is without considering bins)
                stat_sum_3bins_dataset=datasets_ONOFF.stat_sum()

                #!!!!!!!!CHECKS!!!!!!!!
                if print_check==2:
                    #també es pot comprovat sumant el temps "s" de la taula amb gti de la dataset
                    print( )
                    print( )
                    print("OBSERVATION ID:", obs.obs_id)
                    print(f"start time observation {i}, {obs.gti.time_start.iso}")
                    print("arg in array_event_ID_sorted first event", arg_first_event[0,0])
                    print("arg in array_event_ID_sorted last event",arg_last_event[0,0])
                    print("time first event", Quantity(t0, "second"),"or", t_0.iso)
                    print("-check gti start subobs-", subobs.gti.time_start.iso)
                    print("time last event", Quantity(tend, "second"),"or", t_end.iso)
                    print("-check gti end subobs-", subobs.gti.time_stop.iso)
                    print("dataset_on_off gti Tstart:",dataset_on_off.gti.time_start.iso)
                    print("dataset_on_off gti Tstop:",dataset_on_off.gti.time_stop.iso)
                    print("datasets_ONOFF gti Tstart:",datasets_ONOFF.gti.time_start.iso)
                    print("datasets_ONOFF gti Tstop:",datasets_ONOFF.gti.time_stop.iso)
                    print("dataset gti Tstart:",dataset.gti.time_start.iso)
                    print("dataset gti Tstop:",dataset.gti.time_stop.iso)                
                    print("first event ID:",array_event_ID_sorted[first_event])
                    print("last event ID:", array_event_ID_sorted[n_event])
                    print("num events subobs:",len(subobs.events.table["EVENT_ID"]))
                    print("non, noff,        TS,             sqrt TS")
                    print(n_on,n_off,ts_sum,sqrt_ts_LiMa)
                    print("counts On region:",datasets_ONOFF.counts.data.reshape(-1))
                    print("counts Off region:",datasets_ONOFF.background.data.reshape(-1))
                    print("excess counts:",datasets_ONOFF.excess.data.reshape(-1))    
                    print("stat_sum TS bins", stat_sum_3bins_dataset)

                while n_on<10 or n_off<10 or sqrt_ts_LiMa<sqrt_TS_LiMa_threshold:# loop until the bin satisfy these conditions

                    #increase the number of events in the loop
                    n_event=n_event+bin_event

                    # if the event in the end of the bin exceed the # of events in the run 
                    if (len(event_id)-1)-n_event<0: 

                        #if we did not use until the last event, we now yes
                        if bool_last_event==False:
                            n_event=len(event_id)-1
                            #print("last")
                            bool_last_event=True

                        #if we already have, end the loop
                        else:
                            bool_last_event=False
                            # there is another observation in the observation container
                            if np.argwhere(np.array(observations.ids)==i)[0,0]<len(observations)-1:

                                # time between two observations lower than the threshold
                                if (observations[int(np.argwhere(np.array(observations.ids)==i)[0,0]+1)].gti.time_start - \
                                    obs.gti.time_stop).to(thres_time_twoobs.unit) < thres_time_twoobs:

                                    save_dataset_on_off=dataset_on_off.copy(name="counts left obsid %s" %(i))
                                    #safe the dataset_on_off to keep it for the next observation loop
                                    prev_dataset.append(save_dataset_on_off)
                                    prev_events=n_event-bin_event-first_event

                                    n_on=99999 # the run and we do not satisfy the conditions we manually  
                                    n_off=99999 # change them, and compute a flux upper limit
                                    sqrt_ts_LiMa_ul=sqrt_ts_LiMa
                                    sqrt_ts_LiMa=99999
                                    keep_dataset_bool=True

                                    if print_check==2:
                                        print("+++++++++ Append dataset from obs %s in the dataset container +++++++++" %(i))


                                else:# time between two observations higher than the threshold
                                    n_on=99999 # the run and we do not satisfy the conditions we manually  
                                    n_off=99999 # change them, and compute a flux upper limit
                                    sqrt_ts_LiMa_ul=sqrt_ts_LiMa
                                    sqrt_ts_LiMa=99999
                                    keep_dataset_bool=False

                                    #delete the memory of any dataset_on_off saved (the prev. dataset is not used)
                                    prev_dataset = Datasets()

                            else:# is the last observation
                                n_on=99999 # the run and we do not satisfy the conditions we manually  
                                n_off=99999 # change them, and compute a flux upper limit
                                sqrt_ts_LiMa_ul=sqrt_ts_LiMa
                                sqrt_ts_LiMa=99999
                                keep_dataset_bool=False

                    else: # if the event in the end of the bin is equal or lower than the # of event in the run
                        pass

                    if (len(event_id)-1)-n_event>=0: 

                        #print("segon while",(len(event_id)-1)-n_event,bool_last_event)
                        #filter: filter using the ID of the event (which is sorted by time)
                        event_filter = {'type': 'custom', 'opts': dict(parameter='EVENT_ID',\
                                        band=(array_event_ID_sorted[first_event], array_event_ID_sorted[n_event]))}

                        filters=ObservationFilter(event_filters=[event_filter])


                        #copy the observation because the filter deletes all events out of the ID range specified
                        subobs=copy.deepcopy(obs)

                        #apply the filter
                        subobs.obs_filter = filters

                        #obtain the time at which the first and last event were recorded. This is done bc the filtered obs
                        #still has the gti of the whole observation
                        #obtain the position of the events in the table, first event
                        arg_first_event=np.argwhere(obs.events.table["EVENT_ID"]\
                                                    ==array_event_ID_sorted[first_event])
                        #last event
                        arg_last_event=np.argwhere(obs.events.table["EVENT_ID"]\
                                                   ==array_event_ID_sorted[n_event])

                        #obtain the time of the event (time in s wrt the ref. time)
                        t0=obs.events.table["TIME"][arg_first_event[0,0]]
                        tend=obs.events.table["TIME"][arg_last_event[0,0]]

                        #copy the gti object and replace the start and stop time
                        new_gti=subobs.gti.copy()
                        new_gti.table["START"]=t0
                        new_gti.table["STOP"]=tend
                        subobs.__dict__["__gti_hdu"]=new_gti
                        subobs.__dict__["_gti"]=subobs.__dict__["__gti_hdu"]# create a new parameter in the dict 
                                                                # named "_git" this is done bc "__gti_hdu" parameter
                                                                # search the git in a HDU file, but my new gti is not 
                                                                    # in a HDU file, it is in the code itself
                        del subobs.__dict__["__gti_hdu"]#delete the parameter "__git_hdu"

                        if print_check==2:
                            #obtain the time in mjd
                            t_0=obs.gti.time_ref+Quantity(t0, "second")
                            t_end=obs.gti.time_ref+Quantity(tend, "second")


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
                        sign=np.sign(n_on-n_off*(acc_on/acc_off))
                        sqrt_ts_LiMa=sign*np.sqrt(ts_sum)
                        #total TS obtained from the sum of the different bins (ts_sum is without considering bins)
                        stat_sum_3bins_dataset=datasets_ONOFF.stat_sum()

                        #!!!!!!!!CHECKS!!!!!!!!
                        if print_check==2:
                            #també es pot comprovat sumant el temps "s" de la taula amb gti de la dataset
                            print( )
                            print( )
                            print("OBSERVATION ID:", obs.obs_id)
                            print(f"start time observation obs_id {i}, {obs.gti.time_start.iso}")
                            print("arg in array_event_ID_sorted first event", arg_first_event[0,0])
                            print("arg in array_event_ID_sorted last event",arg_last_event[0,0])
                            print("time first event", Quantity(t0, "second"),"or", t_0.iso)
                            print("-check gti start subobs", subobs.gti.time_start.iso)
                            print("time last event", Quantity(tend, "second"),"or", t_end.iso)
                            print("-check gti end subobs", subobs.gti.time_stop.iso)
                            print("dataset_on_off gti Tstart:",dataset_on_off.gti.time_start.iso)
                            print("dataset_on_off gti Tstop:",dataset_on_off.gti.time_stop.iso)
                            print("datasets_ONOFF gti Tstart:",datasets_ONOFF.gti.time_start.iso)
                            print("datasets_ONOFF gti Tstop:",datasets_ONOFF.gti.time_stop.iso)
                            print("dataset gti Tstart:",dataset.gti.time_start.iso)
                            print("dataset gti Tstop:",dataset.gti.time_stop.iso)                
                            print("first event ID:",array_event_ID_sorted[first_event])
                            print("last event ID:", array_event_ID_sorted[n_event])
                            print("num events subobs:",len(subobs.events.table["EVENT_ID"]))
                            print("non, noff,       TS,            sqrt TS")
                            print(n_on,n_off,ts_sum,sqrt_ts_LiMa)
                            print("counts On region:",datasets_ONOFF.counts.data.reshape(-1))
                            print("counts Off region:",datasets_ONOFF.background.data.reshape(-1))
                            print("excess counts:",datasets_ONOFF.excess.data.reshape(-1)) 
                            print("stat_sum TS bins", stat_sum_3bins_dataset)

                if n_off!=99999 and n_on!=99999 and sqrt_ts_LiMa!=99999: #comupte flux

                    datasets_ONOFF.models=best_fit_spec_model#set the model we use to estimate the light curve

                    #estimate the light curve of the bin
                    lc_f=lc.run(copy.deepcopy(datasets_ONOFF))
                    sqrt_ts_flux=lc_f.to_table(sed_type="flux", format="lightcurve")["sqrt_ts"][0,0]

        # CHECK SQRT TS OF THE FLUX IS HIGHER THAN THE THRESHOLD
        ######
                    #if the flux point has a sqrt_TS>sqrt_TS_flux_UL_thd, we accept, if not, we keep adding events
                    while sqrt_ts_flux<sqrt_TS_flux_UL_threshold or sqrt_ts_LiMa<sqrt_TS_LiMa_threshold:

                        #increase the number of events in the loop
                        n_event=n_event+bin_event

                        # if the event in the end of the bin exceed the # of total events
                        if (len(event_id)-1)-n_event<0: 

                            #if we did not use until the last event, we now yes
                            if bool_last_event==False:
                                n_event=len(event_id)-1
                                bool_last_event=True

                            #if we already have, end the loop
                            else:
                                bool_last_event=False
                                # there is another observation in the observation container
                                if np.argwhere(np.array(observations.ids)==i)[0,0]<len(observations)-1:

                                    # time between two observations lower than the threshold
                                    if (observations[int(np.argwhere(np.array(observations.ids)==i)[0,0]+1)].gti.time_start - \
                                        obs.gti.time_stop).to(thres_time_twoobs.unit) < thres_time_twoobs:

                                        save_dataset_on_off=dataset_on_off.copy(name="counts left obsid %s" %(i))
                                        #safe the dataset_on_off to keep it for the next observation loop
                                        prev_dataset.append(save_dataset_on_off)
                                        prev_events=n_event-bin_event-first_event

                                        n_on=99999 # the run and we do not satisfy the conditions we manually  
                                        n_off=99999 # change them, and compute a flux upper limit
                                        sqrt_ts_flux_ul=sqrt_ts_flux
                                        sqrt_ts_flux=99999
                                        sqrt_ts_LiMa_ul=sqrt_ts_LiMa
                                        sqrt_ts_LiMa=99999
                                        keep_dataset_bool=True

                                        if print_check==2:
                                            print("+++++++++ Append dataset from obs %s in the dataset container +++++++++" %(i))


                                    else:# time between two observations higher than the threshold
                                        n_on=99999 # the run and we do not satisfy the conditions we manually  
                                        n_off=99999 # change them, and compute a flux upper limit
                                        sqrt_ts_flux_ul=sqrt_ts_flux
                                        sqrt_ts_flux=99999
                                        sqrt_ts_LiMa_ul=sqrt_ts_LiMa
                                        sqrt_ts_LiMa=99999
                                        keep_dataset_bool=False

                                        #delete the memory of any dataset_on_off saved (the prev. dataset is not used)
                                        prev_dataset = Datasets()

                                else:# is the last observation
                                    n_on=99999 # the run and we do not satisfy the conditions we manually  
                                    n_off=99999 # change them, and compute a flux upper limit
                                    sqrt_ts_flux_ul=sqrt_ts_flux
                                    sqrt_ts_flux=99999
                                    sqrt_ts_LiMa_ul=sqrt_ts_LiMa
                                    sqrt_ts_LiMa=99999
                                    keep_dataset_bool=False

                        else: # if the event in the end of the bin is equal or lower than the # of event in the run
                            pass

                        if (len(event_id)-1)-n_event>=0: 
                            #filter: filter using the ID of the event (which is sorted by time)
                            event_filter = {'type': 'custom', 'opts': dict(parameter='EVENT_ID',\
                                            band=(array_event_ID_sorted[first_event], array_event_ID_sorted[n_event]))}

                            filters=ObservationFilter(event_filters=[event_filter])


                            #copy the observation because the filter deletes all events out of the ID range specified
                            subobs=copy.deepcopy(obs)

                            #apply the filter
                            subobs.obs_filter = filters

                            #obtain the time at which the first and last event were recorded. This is done bc the filtered obs
                            #still has the gti of the whole observation
                            #obtain the position of the events in the table, first event
                            arg_first_event=np.argwhere(obs.events.table["EVENT_ID"]\
                                                        ==array_event_ID_sorted[first_event])
                            #last event
                            arg_last_event=np.argwhere(obs.events.table["EVENT_ID"]\
                                                       ==array_event_ID_sorted[n_event])

                            #obtain the time of the event (time in s wrt the ref. time)
                            t0=obs.events.table["TIME"][arg_first_event[0,0]]
                            tend=obs.events.table["TIME"][arg_last_event[0,0]]

                            #copy the gti object and replace the start and stop time
                            new_gti=subobs.gti.copy()
                            new_gti.table["START"]=t0
                            new_gti.table["STOP"]=tend
                            subobs.__dict__["__gti_hdu"]=new_gti
                            subobs.__dict__["_gti"]=subobs.__dict__["__gti_hdu"]# create a new parameter in the dict 
                                                                    # named "_git" this is done bc "__gti_hdu" parameter
                                                                    # search the git in a HDU file, but my new gti is not 
                                                                        # in a HDU file, it is in the code itself
                            del subobs.__dict__["__gti_hdu"]#delete the parameter "__git_hdu"

                            if print_check==2:
                                #obtain the time in mjd
                                t_0=obs.gti.time_ref+Quantity(t0, "second")
                                t_end=obs.gti.time_ref+Quantity(tend, "second")

                            #data reduction
                            dataset = dataset_maker.run(dataset_empty.copy(name=str(n_event)), subobs)
                            dataset_on_off = bkg_maker_reflected.run(dataset, subobs)

                            #we do not have data from previous runs
                            if len(prev_dataset)==0:
                                datasets_ONOFF=dataset_on_off
                            #we have data to analyse from the previous runs
                            else:
                                #create a temp Dataset container with the dataset_on_off of the previous night
                                temp_dataset=prev_dataset.copy()
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

                            #estimate the light curve of the bin
                            datasets_ONOFF.models=best_fit_spec_model
                            lc_f=lc.run(datasets_ONOFF) 
                            sqrt_ts_flux=lc_f.to_table(sed_type="flux", format="lightcurve")["sqrt_ts"][0,0]

                            #!!!!!!!!CHECKS!!!!!!!!
                            if print_check==2:
                                #també es pot comprovat sumant el temps "s" de la taula amb gti de la dataset
                                print( )
                                print( )
                                print("OBSERVATION ID:", obs.obs_id)
                                print(f"start time observation obs_id {i}, {obs.gti.time_start.iso}")
                                print("arg in array_event_ID_sorted first event", arg_first_event[0,0])
                                print("arg in array_event_ID_sorted last event",arg_last_event[0,0])
                                print("time first event", Quantity(t0, "second"),"or", t_0.iso)
                                print("-check gti start subobs", subobs.gti.time_start.iso)
                                print("time last event", Quantity(tend, "second"),"or", t_end.iso)
                                print("-check gti end subobs", subobs.gti.time_stop.iso)
                                print("dataset_on_off gti Tstart:",dataset_on_off.gti.time_start.iso)
                                print("dataset_on_off gti Tstop:",dataset_on_off.gti.time_stop.iso)
                                print("datasets_ONOFF gti Tstart:",datasets_ONOFF.gti.time_start.iso)
                                print("datasets_ONOFF gti Tstop:",datasets_ONOFF.gti.time_stop.iso)
                                print("dataset gti Tstart:",dataset.gti.time_start.iso)
                                print("dataset gti Tstop:",dataset.gti.time_stop.iso)                
                                print("first event ID:",array_event_ID_sorted[first_event])
                                print("last event ID:", array_event_ID_sorted[n_event])
                                print("num events subobs:",len(subobs.events.table["EVENT_ID"]))
                                print("non, noff,       TS,            sqrt TS")
                                print(n_on,n_off,ts_sum,sqrt_ts_LiMa)
                                print("counts On region:",datasets_ONOFF.counts.data.reshape(-1))
                                print("counts Off region:",datasets_ONOFF.background.data.reshape(-1))
                                print("excess counts:",datasets_ONOFF.excess.data.reshape(-1))
                                print("stat_sum TS bins", stat_sum_3bins_dataset)
                                print("TS of the flux point:",sqrt_ts_flux)

                    #we safe the flux point
                    if n_off!=99999 and n_on!=99999 and sqrt_ts_LiMa!=99999 and sqrt_ts_flux!=99999:
                        #use the same format as the TS column, not an upper limit
                        lc_f.to_table(sed_type="flux", format="lightcurve")["is_ul"] = Column(lc_f.to_table(sed_type="flux", format="lightcurve")["ts"]*0, dtype=bool)

                        #hi ha un problema amb el atol de la classe. Poso maualment l'interval de temps (no se si aquest
                        #error amb l'interval de temps afecta al flux... comprovar-ho)
                        lc_f.to_table(sed_type="flux", format="lightcurve")["time_min"]=datasets_ONOFF.gti.time_intervals[0][0].mjd#time start
                        #time end (-1 en l'arg pel cas que hi hagi més d'un time interval 
                        #(when we have the gti from the previous safed obs))
                        lc_f.to_table(sed_type="flux", format="lightcurve")["time_max"]=datasets_ONOFF.gti.time_intervals[-1][1].mjd

                        lc_list.append(lc_f.to_table(sed_type="flux", format="lightcurve"))

                        if keep_dataset_bool==True:#primer cop que calculo flux tenint en compte obs anterior
                            keep_dataset_bool=False
                            #as we have succesfully calculed the flux, delete the previous counts of the dataset container
                            prev_dataset = Datasets()
                            if print_check==1:
                                print("---------------------------------------------------------------------------")
                                print("events' bin: %i to %i with %i events from previous obs. sqrt(TS)_Li&Ma=%.1f, sqrt(TS)_flux=%.1f" \
                                                  %(first_event,n_event,prev_events,sqrt_ts_LiMa,sqrt_ts_flux))

                        else:
                            if print_check==1:
                                print("---------------------------------------------------------------------------")                            
                                print("events' bin: %.0f to %.0f. sqrt(TS)_Li&Ma=%.1f, sqrt(TS)_flux=%.1f" %(first_event,n_event,sqrt_ts_LiMa,sqrt_ts_flux))
                        #!!!!!!!!CHECKS!!!!!!!!
                        if print_check==2:
                            print("---------------------------------------------------------------------------")
                            print("time_min",Time(lc_f.to_table(sed_type="flux", format="lightcurve")["time_min"][0],format="mjd").iso)
                            print("time_max",Time(lc_f.to_table(sed_type="flux", format="lightcurve")["time_max"][0],format="mjd").iso)
                            print("dataset_on_off Tstart:",dataset_on_off.gti.time_start.iso)
                            print("dataset_on_off Tstop:",dataset_on_off.gti.time_stop.iso)
                            print("datasets_ONOFF Tstart:",datasets_ONOFF.gti.time_start.iso)
                            print("datasets_ONOFF Tstop:",datasets_ONOFF.gti.time_stop.iso)
                            print("---------------------------------------------------------------------------")
                        ts_array.append(ts_sum)

                    #comupte flux upper limit when the next observation starts later than the threshold
                    #and we could not reached high value fo
                    elif n_off==99999 and n_on==99999 and sqrt_ts_LiMa==99999 and sqrt_ts_flux==99999 and keep_dataset_bool==False: 
                        datasets_ONOFF.models=best_fit_spec_model#set the model we use to estimate the light curve

                        #use the same format as the TS column, consider this point as UL
                        lc_f.to_table(sed_type="flux", format="lightcurve")["is_ul"] = Column(lc_f.to_table(sed_type="flux", format="lightcurve")["ts"]*1, dtype=bool)

                        #hi ha un problema amb el atol de la classe. Poso maualment l'interval de temps (no se si aquest
                        #error amb l'interval de temps afecta al flux... comprovar-ho)
                        lc_f.to_table(sed_type="flux", format="lightcurve")["time_min"]=datasets_ONOFF.gti.time_intervals[0][0].mjd#time start
                        #time end (-1 en l'arg pel cas que hi hagi més d'un time interval 
                        #(when we have the gti from the previous safed obs))
                        lc_f.to_table(sed_type="flux", format="lightcurve")["time_max"]=datasets_ONOFF.gti.time_intervals[-1][1].mjd

                        lc_list.append(lc_f.to_table(sed_type="flux", format="lightcurve"))

                        if print_check==1:
                            print("---------------------------------------------------------------------------")
                            print("events' bin: %.0f to %.0f.  sqrt(TS)_Li&Ma=%.1f"+\
                                  ", sqrt(TS)_flux=%.1f -> UL!" %(first_event,n_event,sqrt_ts_LiMa_ul,sqrt_ts_flux_ul))
                            print("---------------------------------------------------------------------------")

                        #!!!!!!!!CHECKS!!!!!!!!
                        if print_check==2:
                            print("time_min",Time(lc_f.to_table(sed_type="flux", format="lightcurve")["time_min"][0],format="mjd").iso)
                            print("time_max",Time(lc_f.to_table(sed_type="flux", format="lightcurve")["time_max"][0],format="mjd").iso)
                            print("dataset_on_off Tstart:",dataset_on_off.gti.time_start.iso)
                            print("dataset_on_off Tstop:",dataset_on_off.gti.time_stop.iso)
                            print("datasets_ONOFF Tstart:",datasets_ONOFF.gti.time_start.iso)
                            print("datasets_ONOFF Tstop:",datasets_ONOFF.gti.time_stop.iso)
                            print("---------------------------------------------------------------------------")
                        ts_array.append(ts_sum)


                #comupte flux upper limit when the next observaiton starts later than the threshold
                elif n_off==99999 and n_on==99999 and sqrt_ts_LiMa==99999 and keep_dataset_bool==False: 

                    datasets_ONOFF.models=best_fit_spec_model#set the model we use to estimate the light curve

                    #estimate the light curve of the bin
                    lc_f=lc.run(copy.deepcopy(datasets_ONOFF))

                    #use the same format as the TS column, consider this point as UL
                    lc_f.to_table(sed_type="flux", format="lightcurve")["is_ul"] = Column(lc_f.to_table(sed_type="flux", format="lightcurve")["ts"]*1, dtype=bool)

                    #hi ha un problema amb el atol de la classe. Poso maualment l'interval de temps (no se si aquest
                    #error amb l'interval de temps afecta al flux... comprovar-ho)
                    lc_f.to_table(sed_type="flux", format="lightcurve")["time_min"]=datasets_ONOFF.gti.time_intervals[0][0].mjd#time start
                    #time end (-1 en l'arg pel cas que hi hagi més d'un time interval 
                    #(when we have the gti from the previous safed obs))
                    lc_f.to_table(sed_type="flux", format="lightcurve")["time_max"]=datasets_ONOFF.gti.time_intervals[-1][1].mjd

                    lc_list.append(lc_f.to_table(sed_type="flux", format="lightcurve"))

                    if print_check==1:
                        print("---------------------------------------------------------------------------")
                        print("events' bin: %.0f to %.0f.  sqrt(TS)_Li&Ma=%.1f -> UL!" %(first_event,n_event,sqrt_ts_LiMa_ul))
                        print("---------------------------------------------------------------------------")

                    #!!!!!!!!CHECKS!!!!!!!!
                    if print_check==2:
                        print("time_min",Time(lc_f.to_table(sed_type="flux", format="lightcurve")["time_min"][0],format="mjd").iso)
                        print("time_max",Time(lc_f.to_table(sed_type="flux", format="lightcurve")["time_max"][0],format="mjd").iso)
                        print("dataset_on_off Tstart:",dataset_on_off.gti.time_start.iso)
                        print("dataset_on_off Tstop:",dataset_on_off.gti.time_stop.iso)
                        print("datasets_ONOFF Tstart:",datasets_ONOFF.gti.time_start.iso)
                        print("datasets_ONOFF Tstop:",datasets_ONOFF.gti.time_stop.iso)
                        print("---------------------------------------------------------------------------")
                    ts_array.append(ts_sum)

            if print_check == 1 or print_check == 2:
                print( )

        #stack al the flux point tables into a single table  
        light_curve=vstack(lc_list)

        #add the Li&Ma TS detection value
        TS_column=Table(data=[ts_array], names=["Li&Ma_TS_detection"], dtype=[np.float32])
    #         light_curve=hstack([light_curve,TS_column])
        light_curve.meta.update({"TS-thd":sqrt_TS_LiMa_threshold**2})
        return light_curve.copy(),TS_column.copy()

