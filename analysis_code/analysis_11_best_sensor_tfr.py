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
from utils_visgam.visgam_paths_local import datafolders, cfg_dir, wdir, figure_dir_gaussian_best_sensor
from utils_visgam.visgam_tfr import correct_baseline, invert_tfr
from utils_visgam.visgam_funcs import loadcfg, mean_sem, cohen_d, rank_biserial
from utils_visgam.visgam_sf import pairwise_component_match_with_plots, pool_eeg_10, pool_eeg_20, pool_opm_10
from utils_visgam.visgam_fit_gaussians import fit_peaks_tfr, peak_info, get_matched_peaks_status_closest_peak, extract_peak_info


##### Analysis configurations
freqbands       = ['alpha', 'gamma']
sensor_configs  = ['10_10']
Results_Folder  = os.path.join('Results', 'SNR_controls', 'Revision_IN')


##### Run analysis for each frequency band and sensor configuration
for freqband in freqbands:
    for sensor_cfg in sensor_configs:
        print(f"\n{'='*60}")
        print(f"Starting analysis for {freqband} band with {sensor_cfg} sensor configuration.")
        print(f"{'='*60}\n")    

        ##### Set directory for logs
        logs_dir    = os.path.join(wdir, Results_Folder,  'analysis_logs', 'best_sensor_analysis_' + freqband + '_' + sensor_cfg)
        os.makedirs(logs_dir, exist_ok=True)    
        logfile     = open(os.path.join(logs_dir, 'peak_analysis_' + freqband + '_' + sensor_cfg + '.log'), 'w')
        old_stdout = sys.stdout
        sys.stdout  = logfile

        ##### Try-finally 
        try:
            ### ----- Load Demographic File
            demog_file  = pd.read_csv(os.path.join(wdir, "DemographicFile.csv"), index_col=0)
            ### ----- Store Figures in local folder with the Peak Fittings
            fig_folder  = figure_dir_gaussian_best_sensor

            ### ----- Obtain config files
            cfg         = loadcfg(cfg_dir)
            cf_ged      = cfg['ged_' + freqband]  # Spatial filter config 
            cf_peak     = cfg['gaussians_' + freqband] 
            cf_folder   = 'TFR_' + freqband + '_allsensors'
            
            ### ----- Save config for gaussian fitting
            with open(os.path.join(logs_dir, 'cf_peak.json'), 'w') as f:
                json.dump(cf_peak, f, indent=4)

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

            ### ----- Results filename
            results_filename = freqband + '_best_peaks ' + sensor_cfg + '.csv'

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
                issz            = demog_file.loc[id_demog]['SZ_EEG_INFO']
                verbose         = spf_args.get('verbose', False)  # Get verbose flag from config

                ### ----- Define datafolders
                ### ----- EEG
                if Results_Folder == os.path.join('Results', 'SNR_controls', 'Revision_IN'):
                    datafoldereeg   = os.path.join(datafolders[nsubject]['eeg'], 'revision', cf_folder)
                eeg_tfr         = os.path.join(datafoldereeg, 'AverageTFR_Superlets.h5')
                ### ----- OPM
                datafolderopm   = os.path.join(datafolders[nsubject]['opm'], cf_folder)
                opm_tfr         = os.path.join(datafolderopm, 'AverageTFR_Superlets.h5')

                ### ----- load TFR data
                tfreeg          = mne.time_frequency.read_tfrs(eeg_tfr)
                tfropm          = mne.time_frequency.read_tfrs(opm_tfr)
                
                ### ----- Correct baseline
                ### ----- EEG
                chaneeg         = tfreeg.copy().apply_baseline(mode='logratio', baseline=cf_peak['baseline'])
                chaneeg.data    = chaneeg.data * 10 # Transform to dB 
                ### ----- OPM
                chanopm         = tfropm.copy().apply_baseline(mode='logratio', baseline=cf_peak['baseline'])
                chanopm.data    = chanopm.data * 10 # Transform to dB 
                chanopm_10      = chanopm.copy()

                ### ----- Select best EEG channel
                chaneeg_10      = chaneeg.copy().pick(pool_eeg_10)
                f_mask          = (chaneeg_10.freqs >= cf_peak['freq_range'][0]) & (chaneeg_10.freqs <= cf_peak['freq_range'][1])
                t_mask          = (chaneeg_10.times >= cf_peak['time_range'][0]) & (chaneeg_10.times <= cf_peak['time_range'][1])
                
                ### ----- EEG scores
                eeg_scores      = np.mean(chaneeg_10.data[:, f_mask, :][:, :, t_mask], axis=(1, 2))
                if freqband == 'alpha':
                    ##### ----- For alpha, we want the minimum (strongest desynchronization)
                    best_eeg_name   = chaneeg_10.ch_names[np.argmin(eeg_scores)]
                else:
                    best_eeg_name   = chaneeg_10.ch_names[np.argmax(eeg_scores)]
                
                ### ----- Final MNE TFR object for EEG
                eeg_fit         = chaneeg_10.copy().pick(best_eeg_name)

                ### ----- Select best OPM channel
                f_mask_opm      = (chanopm_10.freqs >= cf_peak['freq_range'][0]) & (chanopm_10.freqs <= cf_peak['freq_range'][1])
                t_mask_opm      = (chanopm_10.times >= cf_peak['time_range'][0]) & (chanopm_10.times <= cf_peak['time_range'][1])
                ### ----- OPM scores
                opm_scores      = np.mean(chanopm_10.data[:, f_mask_opm, :][:, :, t_mask_opm], axis=(1, 2))
                if freqband == 'alpha':
                    ##### ----- For alpha, we want the minimum (strongest desynchronization)
                    best_opm_name   = chanopm_10.ch_names[np.argmin(opm_scores)]
                else:                    
                    best_opm_name   = chanopm_10.ch_names[np.argmax(opm_scores)]
                
                ### ----- Final MNE TFR object for OPM
                opm_fit     = chanopm_10.copy().pick(best_opm_name)
                ### ----- Print EEG /OPM best channels
                print(f"Subj {subjid} | Best EEG: {best_eeg_name} | Best OPM: {best_opm_name}")

                ### ----- Multiply -1 to TFR for alpha/beta
                if cf_peak['fit_negative']:  
                    eeg_fit = invert_tfr(eeg_fit)
                    opm_fit = invert_tfr(opm_fit)

                ### ----- Fit Gaussians to the spectra of the best-matched components
                eeg_peaks, fig_eeg    = fit_peaks_tfr(eeg_fit, **spf_args)
                if fig_eeg is not None:
                    fig_eeg.savefig(os.path.join(fig_folder, 
                                                f"{subjid}_gaussian_fits_best_{sensor_cfg}_{freqband}_eeg.png"), dpi=300)
                    #fig_eeg.show()
                    plt.close(fig_eeg)
                    
                opm_peaks, fig_opm    = fit_peaks_tfr(opm_fit, **spf_args)
                # Save OPM peaks figure
                if fig_opm is not None:
                    fig_opm.savefig(os.path.join(fig_folder, 
                                                f"{subjid}_gaussian_fits_best_{sensor_cfg}_{freqband}_opm.png"), dpi=300)
                    #fig_opm.show()
                    plt.close(fig_opm)

                ### ----- Extract peak info
                peaks_eeg_info  = peak_info(eeg_peaks)
                peaks_opm_info  = peak_info(opm_peaks)

                ### ----- Cross-modality peak matching (same as original)
                pf_eegdata, pf_opmdata, cons_  = get_matched_peaks_status_closest_peak(peaks_eeg_info, peaks_opm_info, 
                                                        target_freq=cf_peak['target_freq'], 
                                                        tolerance=cf_peak['tolerance'])

                ### ----- Fill in demog file with matched component peaks
                ### ----- Power at Peak Frequency (Matched Components)
                demog_file.loc[id_demog, 'pp_opm'] = pf_opmdata['peak_model_db']
                demog_file.loc[id_demog, 'pp_eeg'] = pf_eegdata['peak_model_db']
                ### ----- Peak Frequency (Matched Components)
                demog_file.loc[id_demog, 'pf_opm'] = pf_opmdata['peak_freq_fit']
                demog_file.loc[id_demog, 'pf_eeg'] = pf_eegdata['peak_freq_fit']
                ### ----- Power over the polynomial fit
                demog_file.loc[id_demog, 'amp_opm'] = pf_opmdata['amp_fit']
                demog_file.loc[id_demog, 'amp_eeg'] = pf_eegdata['amp_fit']
                ### ----- Component labels
                demog_file.loc[id_demog, 'best_opm'] = best_opm_name
                demog_file.loc[id_demog, 'best_eeg'] = best_eeg_name
                    
                ### ----- Number of Trials
                demog_file.loc[id_demog, 'CommonTrials'] = tfreeg.nave  

            print(f"\n{'='*60}")
            print(f"Processing complete. {n_subjects} subjects analyzed.")
            print(f"{'='*60}\n")

            ### ----- Check and create directories if needed
            results_freqband   = os.path.join(wdir, Results_Folder, 'results_' + freqband)
            os.makedirs(results_freqband, exist_ok=True)

            ### ----- Save File with Peak Power and Frequency in Results
            demog_file.to_csv(os.path.join(results_freqband, results_filename))

        finally:
            sys.stdout = old_stdout
            logfile.close()
