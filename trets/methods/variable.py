#!/usr/bin/env python
# Licensed under a 3-clause BSD style license - see LICENSE

from trets.utils import conditional_ray, get_intervals
from trets.bayes import BayesianProbability
from astropy.time import Time
import astropy.units as u
from astropy.units import Quantity
from astropy.table import vstack, Table

import numpy as np
import copy
from gammapy.makers import SpectrumDatasetMaker
from gammapy.maps import RegionGeom
from gammapy.data import ObservationFilter
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

__all__ = ["TRETS"]


class TRETS:
    def __init__(self, parallelization, bool_eventbin_iterate):

        self.bool_eventbin_iterate = bool_eventbin_iterate

        if parallelization:
            self.is_ray = True
        else:
            self.is_ray = False

    def iterate_timewise(self, time_bin, event_time_array, obs_time_ref):
        """
        Compute the time edges to filter the observation to run
        TRETS using a fixed time interval of time_bin.

        Parameters
        ----------
        time_bin: astropy.Quantity
            Time bin iteration.
        event_time_array: `numpy.ndarray`
            Numpy array with the trigger time of the events in the observation. The time
            interval is given in seconds w.r.t. the reference time of the observation.
        obs_time_ref: `astropy.time.core.Time`
            Reference time of the observation.
        Returns
        -------
        time_array: `astropy.time.core.Time`
            Array time with the edge of each interval.
        """

        if not time_bin.unit.is_equivalent(u.s):
            raise ValueError("Specify the interval of time using time quantity unit.")

        event_tini = event_time_array[0] * u.s
        event_tstop = event_time_array[-1] * u.s

        dt = event_tstop - event_tini
        list_t = np.linspace(
            0, dt.to_value("s"), int(np.ceil(dt.to_value("s") / time_bin.to_value("s")))
        )

        # TODO: investigate why using obs.gti.time_start.tt.mjd and removing event_time_array[0] I loose events
        time_array = obs_time_ref.tt + Quantity(event_tini.to_value("s") + list_t, "s")
        method_name = "time-bin-method"
        print(f"Using {method_name}")

        return time_array, time_bin, method_name

    def iterate_eventwise(self, event_bin, event_time_array, obs_time_ref):
        """
        Compute the time edges to filter the observation to run
        TRETS using a fixed number of events of event_bin=Non+Noff.

        Parameters
        ----------
        event_bin: astropy.Quantity
            Number of events (in all the observation) added in each bin iteration.
        event_time_array: `numpy.ndarray`
            Numpy array with the trigger time of the events in the observation. The time
            interval is given in seconds w.r.t. the reference time of the observation.
        obs_time_ref: `astropy.time.core.Time`
            Reference time of the observation.
        Returns
        -------
        time_array: `astropy.time.core.Time`
            Array time with the edge of each interval.

        """
        if not event_bin.unit.is_unity():
            raise ValueError("Specify the interval of events using dimensionless unit.")

        print(int(event_bin.to_value("")))
        list_t = get_intervals(event_time_array, int(event_bin))
        time_array = obs_time_ref + Quantity(list_t, "s")
        method_name = "event-bin-method"
        print(f"Using {method_name}")

        return time_array, event_bin, method_name

    def produce_intervals_method(self):
        if self.bool_eventbin_iterate:
            return self.iterate_eventwise
        else:
            return self.iterate_timewise

    @conditional_ray("is_ray")
    def TRETS_algorithm(
        self,
        is_DL4,
        E1,
        E2,
        e_reco,
        e_true,
        on_region,
        sig_threshold,
        sqrt_TS_flux_UL_threshold,
        print_check,
        thres_time_twoobs,
        bin_iterate,
        observations,
        bkg_maker_reflected,
        best_fit_spec_model,
        bool_bayesian=True,
    ):
        """
        TRETS algorithm. Computes the light curve between energies [E1,E2] where in each integral flux,
        the number of events in that time interval gives a detection significance of the source of
        TS="sig_threshold"**2 and the fit statistics is higher than "sqrt_TS_flux_UL_threshold".
        The flux point is computed assuming a certain SkyModel "best_fit_spec_model" (spectral model).

        Parameters
        ----------
        E1: astropy.units
            Minimum energy bound used to compute the integral flux. It must be considered with one center
            in e_reco.
        E2: astropy.units
            Maximum energy bound used to compute the integral flux. It must be considered with one center
            in e_reco.
        e_reco:
            Reconstructed energy axis used in the SpectrumDatasetOnOff object
        e_true:
            True energy axis used in the SpectrumDatasetOnOff object
        on_region:
            On region located in the source position, it must have the same size used in the IRFs.
        sig_threshold:
            Significance threshold used to compute an integral flux.
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
        bin_iterate: astropy.Quantity
            Quantity added in each iteration. Time or number of events.
        observations: `gammapy.data.Observations`
            Observation object with the runs used to compute the light curve.
        bkg_maker_reflected:
            Background maker to estimate the background.
        best_fit_spec_model:
            Assumed SkyModel of the source. Only spectral model.
        bool_bayesian: boolean
            If true, use bayesian approach to compute the detection significance

        Returns
        -------
        lc_subrun: LightCurve
            Light curve object.
        TS_column: ~astropy.table.Table
            Detection TS of the source.

        """

        if not is_DL4:
            print("Running TRETS at DL3 level (real data)")
            geom = RegionGeom.create(region=on_region, axes=[e_reco])

            dataset_empty = SpectrumDataset.create(geom, energy_axis_true=e_true)
            dataset_maker = SpectrumDatasetMaker(
                containment_correction=False, selection=["counts", "exposure", "edisp"]
            )
            observations_id = observations.ids

        else:
            print("Running TRETS at DL4 level (simulations or binned real data)")
            print(
                "DL3 specifications {e_reco, e_true, on_region, time_bin, bkg_maker_reflected} will not be "
                "considered... Using the Dataset objects specifications!"
            )
            observations_id = observations.names

        print(" ")
        print("Run IDs considered in this execution: ", observations_id)

        # list where the flux is saved
        lc_list = []
        # array to store TS for each time interval
        sig_array = []
        # dataset container where I keep the counts from previous observations
        prev_dataset = Datasets()
        keep_dataset_bool = False
        # If True, we take another start time of the dataset (Dataset passed conditions). Else
        # keep the same start time
        bool_pass = False
        lc = LightCurveEstimator(
            energy_edges=[E1, E2],
            reoptimize=False,
            n_sigma=1,
            n_sigma_ul=2,
            selection_optional=["all"],
            atol=Quantity(1e-6, "s"),
        )

        if bool_bayesian:
            n_on_thd = 0
            n_off_thd = 0
        else:
            n_on_thd = 10
            n_off_thd = 10

        for id, obs in zip(observations_id, observations):

            if print_check != 0:
                print(" ")
                print(" ")
                if not is_DL4:
                    print(
                        f"OBSERVATION ID: {id}. Total number of events: {len(obs.events.table)}"
                    )
                    print("OBS gti Tstart (TT):", observations[0].gti.time_start.tt.iso)
                    print("OBS gti Tstop (TT):", observations[0].gti.time_stop.tt.iso)
                    print(
                        "first event time (TT)",
                        (
                            observations[0].gti.time_ref.tt
                            + Quantity(
                                observations[0].events.table["TIME"][0], "second"
                            )
                        ).iso,
                    )
                    print(
                        "last event time (TT)",
                        (
                            observations[0].gti.time_ref.tt
                            + Quantity(
                                observations[0].events.table["TIME"][-1], "second"
                            )
                        ).iso,
                    )
                    event_id = obs.events.table["EVENT_ID"].data
                else:
                    print(f"OBSERVATION ID: {id}.")

            if not is_DL4:
                event_time = obs.events.table["TIME"].data
                interval_method = self.produce_intervals_method()
                time_array, bin_iterate, method_name = interval_method(
                    bin_iterate,
                    event_time_array=event_time,
                    obs_time_ref=obs.gti.time_ref.tt,
                )

            # for DL4 data, the events in the dataset do not have trigger time info
            else:
                # dummpy time_array
                time_array = np.array([0, 1], dtype=float)

            if print_check != 0:
                if not is_DL4:
                    if bin_iterate.unit.is_unity():
                        print(
                            "Time bin width used",
                            ((time_array[1].mjd - time_array[0].mjd) * u.d).to("s"),
                        )
                    else:
                        print(
                            "Time bin width used",
                            ((time_array[1].mjd - time_array[0].mjd) * u.d).to(
                                bin_iterate.unit
                            ),
                        )
                else:
                    print(
                        "Time bin width used", obs.gti.time_delta[0].to_value("s"), " s"
                    )
                print("Total number of time bins edges:", len(time_array))

            arg_start_time = 0
            for cont_t in range(len(time_array) - 1):

                if bool_pass:
                    pass
                else:
                    arg_start_time = cont_t
                arg_stop_time = cont_t + 1

                if not is_DL4:
                    # filter the observation
                    filters = ObservationFilter(
                        time_filter=Time(
                            [time_array[arg_start_time], time_array[arg_stop_time]]
                        )
                    )

                    # copy the observation because the filter deletes all events out of the ID range specified
                    subobs = copy.deepcopy(obs)
                    # apply the filter
                    subobs.obs_filter = filters

                    t_0 = time_array[arg_start_time]
                    t_end = time_array[arg_stop_time]

                    # data reduction
                    dataset = dataset_maker.run(
                        dataset_empty.copy(name=str(arg_start_time)), subobs
                    )
                    dataset_on_off = bkg_maker_reflected.run(dataset, subobs)

                # no need to filter a dataset. Copy the dataset and consider all events
                else:
                    dataset_on_off = obs.copy()

                # we do not have data from previous runs
                if len(prev_dataset) == 0:
                    datasets_ONOFF = dataset_on_off
                # we have data to analyse from the previous runs
                else:
                    # create a temporal Dataset container with the dataset_on_off of the previous night
                    temp_dataset = prev_dataset.copy()

                    if print_check != 0:
                        print(len(temp_dataset))
                        print(
                            "num events prev dataset:",
                            Datasets(temp_dataset).stack_reduce().counts_off.data.sum()
                            + Datasets(temp_dataset).stack_reduce().counts.data.sum(),
                        )
                        print(
                            "num events this dataset:",
                            dataset_on_off.counts_off.data.sum()
                            + dataset_on_off.counts.data.sum(),
                        )

                    # append the new dataset_on_off
                    temp_dataset.append(dataset_on_off)
                    # stack the observations
                    datasets_ONOFF = Datasets(temp_dataset).stack_reduce()

                # compute acceptance in the On/Off region
                acc_off = datasets_ONOFF.acceptance_off.data.sum()
                acc_on = datasets_ONOFF.acceptance.data.sum()

                # compute the number of counts in the On/Off region
                n_on = datasets_ONOFF.counts.data.sum()
                n_off = datasets_ONOFF.counts_off.data.sum()

                if bool_bayesian:
                    # too high value to compute factorials with enough precision
                    if (n_on > 110 and n_off > 110) or (n_on > 15 and n_off > 110):
                        # compute W statistics
                        stat = WStatCountsStatistic(
                            n_on=n_on, n_off=n_off, alpha=acc_on / acc_off
                        )
                        sig = stat.sqrt_ts
                        # total TS obtained from the sum of the different bins
                        stat_sum_3bins_dataset = datasets_ONOFF.stat_sum()
                    else:
                        # bayesian inference
                        bayes = BayesianProbability(
                            n_on=n_on, n_off=n_off, alpha=acc_on / acc_off
                        )
                        sig = bayes.detection_significance()
                        stat_sum_3bins_dataset = sig
                else:

                    # compute W statistics
                    stat = WStatCountsStatistic(
                        n_on=n_on, n_off=n_off, alpha=acc_on / acc_off
                    )
                    sig = stat.sqrt_ts
                    # total TS obtained from the sum of the different bins
                    stat_sum_3bins_dataset = datasets_ONOFF.stat_sum()

                # !!!!!!!!CHECKS!!!!!!!!
                if print_check == 2:
                    print(" ")
                    print(" ")
                    print(
                        arg_start_time,
                        " to ",
                        arg_stop_time,
                        ", total args:",
                        len(time_array) - 1,
                    )
                    print("OBSERVATION ID:", id)
                    if not is_DL4:
                        # també es pot comprovat sumant el temps "s" de la taula amb gti de la dataset
                        print(
                            f"start time observation {id}, {obs.gti.time_start.tt.iso}"
                        )
                        print(f"stop time observation {id}, {obs.gti.time_stop.tt.iso}")
                        print(
                            "arg first subobs event in observation",
                            np.argwhere(
                                event_time == subobs.events.table["TIME"].data[0]
                            )[0, 0],
                        )
                        print(
                            "arg first subobs event in observation",
                            np.argwhere(
                                event_time == subobs.events.table["TIME"].data[-1]
                            )[0, 0],
                        )
                        print("time first event", t_0.tt.iso)
                        print("-check gti start subobs-", subobs.gti.time_start.tt.iso)
                        print("time last event", t_end.tt.iso)
                        print("-check gti end subobs-", subobs.gti.time_stop.tt.iso)
                        print("dataset gti Tstart:", dataset.gti.time_start.tt.iso)
                        print("dataset gti Tstop:", dataset.gti.time_stop.tt.iso)
                    print(
                        "dataset_on_off gti Tstart:",
                        dataset_on_off.gti.time_start.tt.iso,
                    )
                    print(
                        "dataset_on_off gti Tstop:", dataset_on_off.gti.time_stop.tt.iso
                    )
                    print(
                        "datasets_ONOFF gti Tstart:",
                        datasets_ONOFF.gti.time_start.tt.iso,
                    )
                    print(
                        "datasets_ONOFF gti Tstop:", datasets_ONOFF.gti.time_stop.tt.iso
                    )

                    if not is_DL4:
                        print(
                            "first event ID:",
                            event_id[
                                np.argwhere(
                                    event_id == subobs.events.table["EVENT_ID"].data[0]
                                )
                            ][0, 0],
                        )
                        print(
                            "last event ID:",
                            event_id[
                                np.argwhere(
                                    event_id == subobs.events.table["EVENT_ID"].data[-1]
                                )
                            ][0, 0],
                        )
                        print(
                            "num events subobs:", len(subobs.events.table["EVENT_ID"])
                        )
                    print(
                        "num events dataset:",
                        datasets_ONOFF.counts_off.data.sum()
                        + datasets_ONOFF.counts.data.sum(),
                    )

                    print("######################")
                    print("non     noff     sig")
                    print(f"{n_on:1.1f}     {n_off:1.1f}      {sig:1.3f}")
                    print("######################")
                    print(datasets_ONOFF.info_dict()["sqrt_ts"])
                    print("counts On region:", datasets_ONOFF.counts.data.reshape(-1))
                    print(
                        "counts Off region:", datasets_ONOFF.counts_off.data.reshape(-1)
                    )
                    print("excess counts:", datasets_ONOFF.excess.data.reshape(-1))
                    print("stat_sum TS bins", stat_sum_3bins_dataset)

                # ######################  NO passa el thd de sig ###################
                # loop until the bin satisfy these conditions
                if n_on < n_on_thd or n_off < n_off_thd or sig < sig_threshold:
                    bool_pass = True

                    # last bin
                    if arg_stop_time == len(time_array) - 1:

                        # there is another observation in the observation container
                        if (
                            np.argwhere(np.array(observations_id) == id)[0, 0]
                            < len(observations) - 1
                        ):

                            # time between the two observations is lower than the threshold
                            if (
                                observations[
                                    int(
                                        np.argwhere(np.array(observations_id) == id)[
                                            0, 0
                                        ]
                                        + 1
                                    )
                                ].gti.time_start.tt
                                - obs.gti.time_stop.tt
                            ).to(thres_time_twoobs.unit) < thres_time_twoobs:

                                save_dataset_on_off = dataset_on_off.copy(
                                    name="counts left obsid %s" % id
                                )
                                # safe the dataset_on_off to keep it for the next observation loop
                                prev_dataset.append(save_dataset_on_off)
                                prev_events = (
                                    Datasets(prev_dataset)
                                    .stack_reduce()
                                    .counts_off.data.sum()
                                    + Datasets(prev_dataset)
                                    .stack_reduce()
                                    .counts.data.sum()
                                )
                                sig_ul = sig
                                keep_dataset_bool = True

                                if print_check == 2:
                                    print(
                                        f"+++++++++ Append dataset from obs {id} in the dataset container +++++++++"
                                    )

                            # time between two observations higher than the threshold
                            else:
                                sig_ul = sig
                                keep_dataset_bool = False

                                # delete the memory of any dataset_on_off saved previously for the future run
                                prev_dataset = Datasets()

                        # is the last observation
                        else:
                            sig_ul = sig
                            keep_dataset_bool = False

                        # compute UL
                        # compute flux upper limit if the next observation starts later than the threshold or last run
                        if keep_dataset_bool is False:

                            # set the model we use to estimate the light curve
                            datasets_ONOFF.models = best_fit_spec_model

                            # estimate the light curve of the bin
                            lc_f = lc.run(copy.deepcopy(datasets_ONOFF))
                            lc_f_table = lc_f.to_table(
                                sed_type="flux", format="lightcurve"
                            )

                            # Consider this point as UL
                            lc_f_table["is_ul"].data[0] = np.ones(
                                shape=np.shape(lc_f_table["is_ul"].data[0]), dtype=bool
                            )

                            lc_list.append(lc_f_table)

                            if print_check == 1 or print_check == 2:
                                print(
                                    "---------------------------------------------------------------------------"
                                )
                                if not is_DL4:
                                    print(
                                        "events' bin: %.0f to %.0f.  Sig=%.1f -> UL!"
                                        % (
                                            np.argwhere(
                                                event_time
                                                == subobs.events.table["TIME"].data[0]
                                            )[0, 0],
                                            np.argwhere(
                                                event_time
                                                == subobs.events.table["TIME"].data[-1]
                                            )[0, 0],
                                            sig_ul,
                                        )
                                    )
                                else:
                                    print(
                                        "events' in binned dataset: 0 to 1.  Sig=%.3f -> UL!"
                                        % (sig_ul)
                                    )
                                print(
                                    "---------------------------------------------------------------------------"
                                )

                            # !!!!!!!!CHECKS!!!!!!!!
                            if print_check == 2:
                                print(
                                    "time_min",
                                    Time(
                                        lc_f_table["time_min"][0],
                                        scale="tt",
                                        format="mjd",
                                    ).tt.iso,
                                )
                                print(
                                    "time_max",
                                    Time(
                                        lc_f_table["time_max"][0],
                                        scale="tt",
                                        format="mjd",
                                    ).tt.iso,
                                )
                                print(
                                    "dataset_on_off Tstart:",
                                    dataset_on_off.gti.time_start.tt.iso,
                                )
                                print(
                                    "dataset_on_off Tstop:",
                                    dataset_on_off.gti.time_stop.tt.iso,
                                )
                                print(
                                    "datasets_ONOFF Tstart:",
                                    datasets_ONOFF.gti.time_start.tt.iso,
                                )
                                print(
                                    "datasets_ONOFF Tstop:",
                                    datasets_ONOFF.gti.time_stop.tt.iso,
                                )
                                print(
                                    "---------------------------------------------------------------------------"
                                )
                            sig_array.append(sig)
                        if print_check == 1 or print_check == 2:
                            print(" ")

                # ################# SI passa el thd de sig ####################
                if (
                    n_off > n_off_thd and n_on > n_on_thd and sig > sig_threshold
                ):  # compute flux

                    datasets_ONOFF.models = best_fit_spec_model  # set the model we use to estimate the light curve

                    # estimate the light curve of the bin
                    lc_f = lc.run(copy.deepcopy(datasets_ONOFF))
                    lc_f_table = lc_f.to_table(sed_type="flux", format="lightcurve")

                    sqrt_ts_flux = lc_f_table["sqrt_ts"][0, 0]

                    # CHECK SQRT TS OF THE FLUX IS HIGHER THAN THE THRESHOLD
                    ######
                    # if the flux point has a sqrt_TS>sqrt_TS_flux_UL_thd, we accept, if not, we keep adding events
                    if sqrt_ts_flux < sqrt_TS_flux_UL_threshold:

                        bool_pass = True

                        if arg_stop_time == len(time_array) - 1:

                            # there is another observation in the observation container
                            if (
                                np.argwhere(np.array(observations_id) == id)[0, 0]
                                < len(observations) - 1
                            ):

                                # time between two observations lower than the threshold
                                start_time_next_run = observations[
                                    int(
                                        np.argwhere(np.array(observations_id) == id)[
                                            0, 0
                                        ]
                                        + 1
                                    )
                                ].gti.time_start.tt
                                time_diff = (
                                    start_time_next_run - obs.gti.time_stop.tt
                                ).to(thres_time_twoobs.unit)
                                if time_diff < thres_time_twoobs:

                                    save_dataset_on_off = dataset_on_off.copy(
                                        name="counts left obsid %s" % id
                                    )
                                    # safe the dataset_on_off to keep it for the next observation loop
                                    prev_dataset.append(save_dataset_on_off)

                                    prev_events = (
                                        Datasets(prev_dataset)
                                        .stack_reduce()
                                        .counts_off.data.sum()
                                        + Datasets(prev_dataset)
                                        .stack_reduce()
                                        .counts.data.sum()
                                    )

                                    sqrt_ts_flux_ul = sqrt_ts_flux
                                    sig_ul = sig
                                    keep_dataset_bool = True

                                    if print_check == 2:
                                        print(
                                            f"+++++++++ Append dataset from obs {id} in the dataset container +++++++++"
                                        )

                                else:  # time between two observations higher than the threshold
                                    sqrt_ts_flux_ul = sqrt_ts_flux
                                    sig_ul = sig
                                    keep_dataset_bool = False

                                    # delete the memory of any dataset_on_off saved previously for the future run
                                    prev_dataset = Datasets()

                            else:  # is the last observation
                                sqrt_ts_flux_ul = sqrt_ts_flux
                                sig_ul = sig
                                keep_dataset_bool = False

                            # compute flux UL
                            if keep_dataset_bool is False:

                                # consider this point as UL
                                lc_f_table["is_ul"].data[0] = np.ones(
                                    shape=np.shape(lc_f_table["is_ul"].data[0]),
                                    dtype=bool,
                                )

                                lc_list.append(lc_f_table)

                                if print_check == 1 or print_check == 2:
                                    print(
                                        "---------------------------------------------------------------------------"
                                    )
                                    if not is_DL4:
                                        print(
                                            "events' bin: %.0f to %.0f. detect_sig=%.3f, sqrt(TS)_flux=%.3f ->UL!"
                                            % (
                                                np.argwhere(
                                                    event_time
                                                    == subobs.events.table["TIME"].data[
                                                        0
                                                    ]
                                                )[0, 0],
                                                np.argwhere(
                                                    event_time
                                                    == subobs.events.table["TIME"].data[
                                                        -1
                                                    ]
                                                )[0, 0],
                                                sig_ul,
                                                sqrt_ts_flux_ul,
                                            )
                                        )
                                    else:
                                        print(
                                            "events' bin: 0 to 1. detect_sig=%.3f, sqrt(TS)_flux=%.3f ->UL!"
                                            % (sig_ul, sqrt_ts_flux_ul)
                                        )
                                    print(
                                        "---------------------------------------------------------------------------"
                                    )

                                # !!!!!!!!CHECKS!!!!!!!!
                                if print_check == 2:
                                    print(
                                        "time_min",
                                        Time(
                                            lc_f_table["time_min"][0],
                                            scale="tt",
                                            format="mjd",
                                        ).tt.iso,
                                    )
                                    print(
                                        "time_max",
                                        Time(
                                            lc_f_table["time_max"][0],
                                            scale="tt",
                                            format="mjd",
                                        ).tt.iso,
                                    )
                                    print(
                                        "dataset_on_off Tstart:",
                                        dataset_on_off.gti.time_start.tt.iso,
                                    )
                                    print(
                                        "dataset_on_off Tstop:",
                                        dataset_on_off.gti.time_stop.tt.iso,
                                    )
                                    print(
                                        "datasets_ONOFF Tstart:",
                                        datasets_ONOFF.gti.time_start.tt.iso,
                                    )
                                    print(
                                        "datasets_ONOFF Tstop:",
                                        datasets_ONOFF.gti.time_stop.tt.iso,
                                    )
                                    print(
                                        "---------------------------------------------------------------------------"
                                    )
                                sig_array.append(sig)

                    # we save the flux point
                    else:
                        bool_pass = False

                        # not an upper limit
                        lc_f_table["is_ul"].data[0] = np.zeros(
                            shape=np.shape(lc_f_table["is_ul"].data[0]), dtype=bool
                        )

                        lc_list.append(
                            lc_f.to_table(sed_type="flux", format="lightcurve")
                        )

                        if (
                            keep_dataset_bool is True
                        ):  # primer cop que calculo flux tenint en compte obs anterior
                            keep_dataset_bool = False
                            # as we have succesfully calculed the flux,
                            # delete the previous counts of the dataset container
                            prev_dataset = Datasets()
                            if print_check == 1 or print_check == 2:
                                print(
                                    "---------------------------------------------------------------------------"
                                )
                                if not is_DL4:
                                    index_start = np.argwhere(
                                        event_time
                                        == subobs.events.table["TIME"].data[0]
                                    )[0, 0]
                                    index_stop = np.argwhere(
                                        event_time
                                        == subobs.events.table["TIME"].data[-1]
                                    )[0, 0]
                                    print(
                                        f"events' bin: {index_start} to {index_stop} with {prev_events} events"
                                        f"from previous obs. detect_sig={sig:1.3f}, sqrt(TS)_flux={sqrt_ts_flux:1.3f}"
                                    )
                                else:
                                    print(
                                        f"events' bin: 0 to 1 with {prev_events} events from previous obs. "
                                        f"detect_sig={sig:1.3f}, sqrt(TS)_flux={sqrt_ts_flux:1.3f}"
                                    )
                        else:
                            if print_check == 1 or print_check == 2:
                                print(
                                    "---------------------------------------------------------------------------"
                                )
                                if not is_DL4:
                                    print(
                                        "events' bin: %.0f to %.0f. detect_sig=%.3f, sqrt(TS)_flux=%.3f"
                                        % (
                                            np.argwhere(
                                                event_time
                                                == subobs.events.table["TIME"].data[0]
                                            )[0, 0],
                                            np.argwhere(
                                                event_time
                                                == subobs.events.table["TIME"].data[-1]
                                            )[0, 0],
                                            sig,
                                            sqrt_ts_flux,
                                        )
                                    )
                                else:
                                    print(
                                        "events' bin: 0 to 1. detect_sig=%.3f, sqrt(TS)_flux=%.3f"
                                        % (sig, sqrt_ts_flux)
                                    )
                        # !!!!!!!!CHECKS!!!!!!!!
                        if print_check == 2:
                            print(
                                "---------------------------------------------------------------------------"
                            )
                            print(
                                "time_min",
                                Time(
                                    lc_f_table["time_min"][0], scale="tt", format="mjd"
                                ).tt.iso,
                            )
                            print(
                                "time_max",
                                Time(
                                    lc_f_table["time_max"][0], scale="tt", format="mjd"
                                ).tt.iso,
                            )
                            print(
                                "dataset_on_off Tstart:",
                                dataset_on_off.gti.time_start.tt.iso,
                            )
                            print(
                                "dataset_on_off Tstop:",
                                dataset_on_off.gti.time_stop.tt.iso,
                            )
                            print(
                                "datasets_ONOFF Tstart:",
                                datasets_ONOFF.gti.time_start.tt.iso,
                            )
                            print(
                                "datasets_ONOFF Tstop:",
                                datasets_ONOFF.gti.time_stop.tt.iso,
                            )
                            print(
                                "---------------------------------------------------------------------------"
                            )
                        sig_array.append(sig)

        # stack al the flux point tables into a single table
        light_curve = vstack(lc_list)

        # add the detection significance value
        sig_column = Table(
            data=[sig_array], names=["sig_detection"], dtype=[np.float32]
        )
        light_curve.meta.update({"sig-thd": sig_threshold})
        light_curve.meta.update({"sig-flux-thd": sqrt_TS_flux_UL_threshold})
        if not is_DL4:
            if method_name == "time-bin-method":
                light_curve.meta.update({method_name: bin_iterate.to("s")})
            else:
                light_curve.meta.update({method_name: bin_iterate})
        else:
            light_curve.meta.update({"time-bin-method": obs.gti.time_delta[0].to("s")})

        return light_curve.copy(), sig_column.copy()
