import sys
import os
import mne
import json
import numpy as np
import pandas as pd
import matplotlib
# matplotlib mode
matplotlib.use('QtAgg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from utils_visgam.visgam_paths_local import datafolders, cfg_dir, wdir, figure_dir_testretest
from utils_visgam.visgam_tfr import correct_baseline, invert_tfr
from utils_visgam.visgam_funcs import loadcfg, mean_sem, cohen_d, rank_biserial
from utils_visgam.visgam_sf import pairwise_component_match_with_plots
from utils_visgam.visgam_fit_gaussians import fit_peaks_tfr, peak_info, get_matched_peaks_status_closest_peak


##### Analysis configurations
freqbands       = ['alpha', 'gamma']
sensor_configs  = ['10_10', '10_20']
### ----- Two split-half strategies
split_types     = ['chronological', 'oddeven']
fig_folder      = figure_dir_testretest

### ----- Compare GED
is_compare_ged = False  # Set to True to compare GED components across Test and ReTest
is_matched_ged = True
##### Select subfolder within Results folder to save results
Results_Folder  = os.path.join('Results', 'SNR_controls', 'Revision_IN')
##### Run analysis for each frequency band and sensor configuration
for freqband in freqbands:
    for sensor_cfg in sensor_configs:
        print(f"\n{'='*60}")
        print(f"Starting analysis for {freqband} band with {sensor_cfg} sensor configuration.")
        print(f"{'='*60}\n")
        peaks_file = pd.read_csv(
            os.path.join(wdir, Results_Folder, f'results_{freqband}', f'{freqband}_peaks {sensor_cfg}.csv'),
            index_col=0)
        ##### Try-finally
        try:
            ### ----- Load one Demographic File per split type (kept separate so the
            ### ----- two split strategies never overwrite each other's columns)
            demog_files = {
                st: pd.read_csv(os.path.join(wdir, "DemographicFile.csv"), index_col=0)
                for st in split_types
            }

            ### ----- Obtain config files
            cfg         = loadcfg(cfg_dir)
            cf_ged      = cfg['ged_' + freqband]  # Spatial filter config
            cf_ged_dir  = cfg['ged_' + freqband + '_folders_' + sensor_cfg]  # [..._10_20]
            cf_peak     = cfg['gaussians_' + freqband]

            ### ----- Peak Finding Arguments
            spf_args    = dict(
                fit_range           =cf_peak['fit_range'],      # (fmin, fmax) Hz
                peak_range          =cf_peak['fit_range'],      # (fmin, fmax) Hz for valid peaks (should be within fit_range)
                time_range          =cf_peak['time_range'],     # (tmin, tmax)
                min_sep             =cf_peak['min_sep'],        # Minimum separation between peaks
                sdgaussian_range    =cf_peak['sd_range'],       # Range of widths for Gaussian fit in Hz
                edge_margin         =cf_peak['edge_margin'],    # Exclude peaks within margin of frequency edges
                smooth_sigma        =cf_peak['freq_smooth'],    # Smoothing kernel width in Hz
                dprime_threshold    =cf_peak['dprime'],         # Separation for 2 peak model
                change_slope        =cf_peak['slope_change'],   # Slope changes in the fit
                is_fit_negative     =cf_peak['fit_negative'],   # Whether to fit the negative of the spectrum (for alpha/beta)
                plot                =True,
                downweight_edges    =cf_peak['downweight_edges']
            )

            ### ----- Number of datasets
            n_subjects      = len(datafolders)
            print('Total sample: ', n_subjects)

            ### ----- Start Loop
            for nsubject in range(n_subjects):
                ### ----- Current dataset
                print('Current Subject:', datafolders[nsubject]['id'])

                ### ----- Subject ID
                subjid          = datafolders[nsubject]['id']
                id_demog        = subjid[8:]
                issz            = demog_files[split_types[0]].loc[id_demog]['SZ_EEG_INFO']
                #verbose         = spf_args.get('verbose', False)  # Get verbose flag from config

                ### ----- Define datafolders
                ### ----- EEG
                ##### Re-direct to preprocessed folder
                datafoldereeg_base = datafolders[nsubject]['eeg']
                if Results_Folder == os.path.join('Results', 'SNR_controls', 'Revision_IN'):
                    datafoldereeg_base = os.path.join(datafoldereeg_base, 'revision')
                datafoldereeg   = os.path.join(datafoldereeg_base, cf_ged_dir['results_folder'])
                eeg_tfr         = os.path.join(datafoldereeg, cf_ged_dir['tfr_filename'])
                eeg_evals       = np.load(os.path.join(datafoldereeg, cf_ged_dir['evals_filename']))
                ### ----- OPM
                datafolderopm   = os.path.join(datafolders[nsubject]['opm'], cf_ged_dir['results_folder'])
                opm_tfr         = os.path.join(datafolderopm, cf_ged_dir['tfr_filename'])
                opm_evals       = np.load(os.path.join(datafolderopm, cf_ged_dir['evals_filename']))

                ### ----- Print Eigenvalues
                print(f'EEG Eigenvalues: {eeg_evals}')
                print(f'OPM Eigenvalues: {opm_evals}')

                ### ----- Check eigenvalues higher than 1
                n_eigenvalues_eeg = np.sum(eeg_evals > 1)
                n_eigenvalues_opm = np.sum(opm_evals > 1)
                print(f'Number of EEG eigenvalues > 1: {n_eigenvalues_eeg}')
                print(f'Number of OPM eigenvalues > 1: {n_eigenvalues_opm}')

                if n_eigenvalues_eeg >= 3:
                    n_components_eeg = 3
                else:
                    n_components_eeg = n_eigenvalues_eeg
                    print(f"Warning: Only {n_components_eeg} EEG eigenvalues > 1. Adjusting number of components to {n_components_eeg}.")
                if n_eigenvalues_opm >= 3:
                    n_components_opm = 3
                else:
                    n_components_opm = n_eigenvalues_opm
                    print(f"Warning: Only {n_components_opm} OPM eigenvalues > 1. Adjusting number of components to {n_components_opm}.")

                ### ----- Load TFRs
                tfreeg      = mne.time_frequency.read_tfrs(eeg_tfr)
                tfropm      = mne.time_frequency.read_tfrs(opm_tfr)

                if not is_compare_ged:
                    if is_matched_ged:
                        ### ----- Select best-matched GED component based on previous analysis
                        matched_eeg_component = peaks_file.loc[id_demog]['ged_eeg']
                        matched_opm_component = peaks_file.loc[id_demog]['ged_opm']
                        tfreeg      = tfreeg.copy().pick(matched_eeg_component)
                        tfropm      = tfropm.copy().pick(matched_opm_component)
                    else:
                    ### ----- Select 'GED_1'
                        tfreeg      = tfreeg.copy().pick('GED_1')
                        tfropm      = tfropm.copy().pick('GED_1')


                ### ----- Number of trials available for each modality
                n_trials_eeg    = tfreeg.data.shape[0]
                n_trials_opm    = tfropm.data.shape[0]

                ### ----- Compute both split-half strategies from the same loaded data
                for split_type in split_types:

                    demog_file  = demog_files[split_type]

                    ### ----- Divide data into Test and Retest halves
                    if split_type == 'chronological':
                        ### ----- First half of trials (in recording order) vs second half
                        cut_eeg         = int(np.fix(n_trials_eeg/2))
                        idx_eeg_t       = np.arange(0, cut_eeg)
                        idx_eeg_rt      = np.arange(cut_eeg, n_trials_eeg)
                        cut_opm         = int(np.fix(n_trials_opm/2))
                        idx_opm_t       = np.arange(0, cut_opm)
                        idx_opm_rt      = np.arange(cut_opm, n_trials_opm)
                    elif split_type == 'oddeven':
                        ### ----- Interleaved odd/even trials (time-order neutral)
                        idx_eeg_t       = np.arange(0, n_trials_eeg, 2)
                        idx_eeg_rt      = np.arange(1, n_trials_eeg, 2)
                        idx_opm_t       = np.arange(0, n_trials_opm, 2)
                        idx_opm_rt      = np.arange(1, n_trials_opm, 2)
                    else:
                        raise ValueError(f"Unknown split_type: {split_type}")

                    ### ----- EEG
                    ### ----- Test
                    tfreeg_t    = mne.time_frequency.AverageTFRArray(
                        info=tfreeg.info, data=tfreeg.data[idx_eeg_t, :, :, :].mean(axis=0), times=tfreeg.times,
                        freqs=tfreeg.freqs, method=tfreeg.method)
                    ### ----- ReTest
                    tfreeg_rt   = mne.time_frequency.AverageTFRArray(
                        info=tfreeg.info, data=tfreeg.data[idx_eeg_rt, :, :, :].mean(axis=0), times=tfreeg.times,
                        freqs=tfreeg.freqs, method=tfreeg.method)

                    ### ----- OPM
                    ### ----- Test
                    tfropm_t    = mne.time_frequency.AverageTFRArray(
                        info=tfropm.info, data=tfropm.data[idx_opm_t, :, :, :].mean(axis=0), times=tfropm.times,
                        freqs=tfropm.freqs, method=tfropm.method)
                    ### ----- ReTest
                    tfropm_rt   = mne.time_frequency.AverageTFRArray(
                        info=tfropm.info, data=tfropm.data[idx_opm_rt, :, :, :].mean(axis=0), times=tfropm.times,
                        freqs=tfropm.freqs, method=tfropm.method)

                    ### ----- Correct baseline and return dB
                    ### ----- Test
                    avg_eeg_t       = tfreeg_t.apply_baseline(cf_peak['baseline'], mode='logratio')
                    avg_eeg_t.data  = avg_eeg_t.data * 10

                    avg_opm_t       = tfropm_t.apply_baseline(cf_peak['baseline'], mode='logratio')
                    avg_opm_t.data  = avg_opm_t.data * 10

                    ### ----- ReTest
                    avg_eeg_rt      = tfreeg_rt.apply_baseline(cf_peak['baseline'], mode='logratio')
                    avg_eeg_rt.data = avg_eeg_rt.data * 10
                    avg_opm_rt      = tfropm_rt.apply_baseline(cf_peak['baseline'], mode='logratio')
                    avg_opm_rt.data = avg_opm_rt.data * 10

                    
                    ### ----- Multiply -1 to TFR for alpha/beta
                    if cf_peak['fit_negative']:
                        avg_eeg_t   = invert_tfr(avg_eeg_t)
                        avg_opm_t   = invert_tfr(avg_opm_t)
                        avg_eeg_rt  = invert_tfr(avg_eeg_rt)
                        avg_opm_rt  = invert_tfr(avg_opm_rt)

                    if is_compare_ged:
                        ### ----- Correlate GED components and return best match within modality
                        ### ----- Prioritize analyzing the first ged component as if carries more explained variance
                        out_eeg     = pairwise_component_match_with_plots(avg_eeg_t, avg_eeg_rt,
                            fmin=cf_peak['freq_range'][0], fmax=cf_peak['freq_range'][1],
                            tmin=cf_peak['time_range'][0], tmax=cf_peak['time_range'][1],
                            n_eeg=3, n_opm=3,
                            #plot_overviews=True, plot_best_match=True, show_corr=True, prefer_first_if_r00_ge=0.5)
                            plot_overviews=False, plot_best_match=False, show_corr=False, prefer_first_if_r00_ge=0.5)

                        out_opm     = pairwise_component_match_with_plots(avg_opm_t, avg_opm_rt,
                            fmin=cf_peak['freq_range'][0], fmax=cf_peak['freq_range'][1],
                            tmin=cf_peak['time_range'][0], tmax=cf_peak['time_range'][1],
                            n_eeg=3, n_opm=3,
                            #plot_overviews=True, plot_best_match=True, show_corr=True, prefer_first_if_r00_ge=0.5)
                            plot_overviews=False, plot_best_match=False, show_corr=False, prefer_first_if_r00_ge=0.5)

                        ### ----- Best match
                        ### ----- The function is designed to get peaks EEG vs OPM. Here Test is EEG and ReTest is OPM
                        ### ----- EEG
                        out_eeg["best_eeg_name"], out_eeg["best_opm_name"], out_eeg["best_r"]
                        eeg_fit_t   = avg_eeg_t.copy().pick(out_eeg["best_eeg_name"])
                        eeg_fit_rt  = avg_eeg_rt.copy().pick(out_eeg["best_opm_name"])

                        ### ----- OPM
                        out_opm["best_eeg_name"], out_opm["best_opm_name"], out_opm["best_r"]
                        opm_fit_t   = avg_opm_t.copy().pick(out_opm["best_eeg_name"])
                        opm_fit_rt  = avg_opm_rt.copy().pick(out_opm["best_opm_name"])

                    else:
                        if is_matched_ged:
                            eeg_fit_t   = avg_eeg_t.copy().pick(matched_eeg_component)
                            eeg_fit_rt  = avg_eeg_rt.copy().pick(matched_eeg_component)
                            opm_fit_t   = avg_opm_t.copy().pick(matched_opm_component)
                            opm_fit_rt  = avg_opm_rt.copy().pick(matched_opm_component)
                        else:
                            eeg_fit_t   = avg_eeg_t.copy().pick('GED_1')
                            eeg_fit_rt  = avg_eeg_rt.copy().pick('GED_1')
                            opm_fit_t   = avg_opm_t.copy().pick('GED_1')
                            opm_fit_rt  = avg_opm_rt.copy().pick('GED_1')


                    ### ----- Fit Gaussians to the spectra of the best-matched components
                    eeg_peaks_t, fig_eeg_t      = fit_peaks_tfr(eeg_fit_t, **spf_args)
                    eeg_peaks_rt, fig_eeg_rt    = fit_peaks_tfr(eeg_fit_rt, **spf_args)
                    if fig_eeg_t is not None:
                        fig_eeg_t.savefig(os.path.join(fig_folder,
                                                    f"{subjid}_test_gaussian_fits_{sensor_cfg}_{freqband}_{split_type}.png"), dpi=300)
                        #fig_eeg.show()
                        plt.close(fig_eeg_t)
                    if fig_eeg_rt is not None:
                        fig_eeg_rt.savefig(os.path.join(fig_folder,
                                                    f"{subjid}_retest_gaussian_fits_{sensor_cfg}_{freqband}_{split_type}.png"), dpi=300)
                        #fig_eeg_rt.show()
                        plt.close(fig_eeg_rt)

                    opm_peaks_t, fig_opm_t      = fit_peaks_tfr(opm_fit_t, **spf_args)
                    opm_peaks_rt, fig_opm_rt    = fit_peaks_tfr(opm_fit_rt, **spf_args)

                    # Save OPM peaks figure
                    if fig_opm_t is not None:
                        fig_opm_t.savefig(os.path.join(fig_folder,
                                                    f"{subjid}_test_gaussian_fits_{sensor_cfg}_{freqband}_{split_type}_opm.png"), dpi=300)
                        #fig_opm_t.show()
                        plt.close(fig_opm_t)
                    if fig_opm_rt is not None:
                        fig_opm_rt.savefig(os.path.join(fig_folder,
                                                    f"{subjid}_retest_gaussian_fits_{sensor_cfg}_{freqband}_{split_type}_opm.png"), dpi=300)
                        #fig_opm_rt.show()
                        plt.close(fig_opm_rt)

                    ### ----- Extract peak info
                    peaks_eeg_info_t    = peak_info(eeg_peaks_t)
                    peaks_eeg_info_rt   = peak_info(eeg_peaks_rt)
                    peaks_opm_info_t    = peak_info(opm_peaks_t)
                    peaks_opm_info_rt   = peak_info(opm_peaks_rt)

                    ### ----- Cross-modality peak matching (same as original)
                    pf_eegdata_t, pf_eegdata_rt, cons_  = get_matched_peaks_status_closest_peak(peaks_eeg_info_t, peaks_eeg_info_rt,
                                                            target_freq=cf_peak['target_freq'],
                                                            tolerance=cf_peak['tolerance'])
                    pf_opmdata_t, pf_opmdata_rt, cons_  = get_matched_peaks_status_closest_peak(peaks_opm_info_t, peaks_opm_info_rt,
                                                            target_freq=cf_peak['target_freq'],
                                                            tolerance=cf_peak['tolerance'])

                    ### ----- Fill in demog file with matched component peaks
                    ### ----- Power at Peak Frequency (Matched Components)
                    demog_file.loc[id_demog, 'pp_opm_test']     = pf_opmdata_t['peak_model_db']
                    demog_file.loc[id_demog, 'pp_opm_retest']   = pf_opmdata_rt['peak_model_db']
                    demog_file.loc[id_demog, 'pp_eeg_test']     = pf_eegdata_t['peak_model_db']
                    demog_file.loc[id_demog, 'pp_eeg_retest']   = pf_eegdata_rt['peak_model_db']
                    ### ----- Peak Frequency (Matched Components)
                    demog_file.loc[id_demog, 'pf_opm_test']     = pf_opmdata_t['peak_freq_fit']
                    demog_file.loc[id_demog, 'pf_opm_retest']   = pf_opmdata_rt['peak_freq_fit']
                    demog_file.loc[id_demog, 'pf_eeg_test']     = pf_eegdata_t['peak_freq_fit']
                    demog_file.loc[id_demog, 'pf_eeg_retest']   = pf_eegdata_rt['peak_freq_fit']
                    ### ----- Power over the polynomial fit
                    demog_file.loc[id_demog, 'amp_opm_test']    = pf_opmdata_t['amp_fit']
                    demog_file.loc[id_demog, 'amp_opm_retest']  = pf_opmdata_rt['amp_fit']
                    demog_file.loc[id_demog, 'amp_eeg_test']    = pf_eegdata_t['amp_fit']
                    demog_file.loc[id_demog, 'amp_eeg_retest']  = pf_eegdata_rt['amp_fit']

                    ### ----- Number of Trials
                    demog_file.loc[id_demog, 'CommonTrials_Test']   = tfreeg_t.nave
                    demog_file.loc[id_demog, 'CommonTrials_ReTest'] = tfreeg_rt.nave


            print(f"\n{'='*60}")
            print(f"Processing complete. {n_subjects} subjects analyzed.")
            print(f"{'='*60}\n")

            ### ----- Check and create directories if needed
            results_freqband   = os.path.join(wdir, Results_Folder, 'results_' + freqband)
            os.makedirs(results_freqband, exist_ok=True)

            ### ----- Save one CSV per split-half strategy
            for split_type in split_types:
                results_filename = freqband + '_trt_' + split_type + ' ' + sensor_cfg + '.csv'
                demog_files[split_type].to_csv(os.path.join(results_freqband, results_filename))
                print(f"Results saved to: {os.path.join(results_freqband, results_filename)}")

        finally:
            print('ok')