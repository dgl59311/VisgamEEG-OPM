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
from utils_visgam.visgam_paths_local import datafolders, cfg_dir, wdir, figure_dir_peakfinder
from utils_visgam.visgam_tfr import correct_baseline, invert_tfr
from utils_visgam.visgam_funcs import loadcfg, mean_sem, cohen_d, rank_biserial
from utils_visgam.visgam_sf import pairwise_component_match_with_plots
from utils_visgam.visgam_fit_gaussians import peak_finder_tfr, extract_peak_info_peakfinder, get_matched_peaks_status_closest_peak


##### Analysis configurations
freqbands       = ['gamma', 'alpha']
sensor_configs  = ['10_10', '10_20']

##### Results folder
Results_Folder  = os.path.join('Results', 'SNR_controls', 'Revision_IN')
##### 
log_            = True

##### Run analysis for each frequency band and sensor configuration
for freqband in freqbands:
    for sensor_cfg in sensor_configs:
        print(f"\n{'='*60}")
        print(f"Starting analysis for {freqband} band with {sensor_cfg} sensor configuration.")
        print(f"{'='*60}\n")    

        ##### Set directory for logs
        logs_dir    = os.path.join(wdir, Results_Folder,  'analysis_logs', 'analysis_' + freqband + '_' + sensor_cfg)
        if log_:
            os.makedirs(logs_dir, exist_ok=True)    
            logfile     = open(os.path.join(logs_dir, 'peak_finder_analysis_' + freqband + '_' + sensor_cfg + '.log'), 'w')
            old_stdout = sys.stdout
            sys.stdout  = logfile

        ##### Try-finally 
        try:
            ### ----- Load Demographic File
            demog_file  = pd.read_csv(os.path.join(wdir, "DemographicFile.csv"), index_col=0)
            ### ----- Store Figures in local folder with the Peak Fittings
            fig_folder  = figure_dir_peakfinder

            ### ----- Obtain config files
            cfg         = loadcfg(cfg_dir)
            cf_ged      = cfg['ged_' + freqband]  # Spatial filter config 
            cf_ged_dir  = cfg['ged_' + freqband + '_folders_' + sensor_cfg]  # [..._10_20]
            cf_peak     = cfg['gaussians_' + freqband] 

            ### ----- Save config for gaussian fitting
            if log_:
                with open(os.path.join(logs_dir, 'cf_peak_peak_finder.json'), 'w') as f:
                    json.dump(cf_peak, f, indent=4)

            ### ----- Peak Finding Arguments 
            ##### Use same arguments as in the Gaussian Fits
            spf_args    = dict(
                fit_range           =cf_peak['fit_range'],      # (fmin, fmax) Hz
                peak_range          =cf_peak['fit_range'],      # (fmin, fmax) Hz for valid peaks (should be within fit_range)
                time_range          =cf_peak['time_range'],     # (tmin, tmax) 
                min_sep             =cf_peak['min_sep'],        # Minimum separation between peaks
                sdgaussian_range    =cf_peak['sd_range'],       # Range of widths for Gaussian fit in Hz
                edge_margin         =cf_peak['edge_margin'],    # Exclude peaks within margin of frequency edges
                smooth_sigma        =cf_peak['freq_smooth'],    # Smoothing kernel width in Hz
                is_fit_negative     =cf_peak['fit_negative'],   # Whether to fit the negative of the spectrum (for alpha/beta)
                plot                =True
            )

            ### ----- Results filename
            results_filename = freqband + '_peak_finder_peaks ' + sensor_cfg + '.csv'

            ### ----- Number of datasets
            n_subjects      = len(datafolders)
            print('Total sample: ', n_subjects)

            ### ----- Allocate to save TFRs
            eeg_ctrl, eeg_pat               = [], []
            opm_ctrl, opm_pat               = [], []
            eeg_ctrl_nbl, opm_ctrl_nbl      = [], []
            eeg_pat_nbl, opm_pat_nbl        = [], []

            ### ----- Start Loop
            for nsubject in range(n_subjects):
            #for nsubject in range(1, 2):
                ### ----- Current dataset   
                print('Current Subject:', datafolders[nsubject]['id'])

                ### ----- Subject ID
                subjid          = datafolders[nsubject]['id']
                id_demog        = subjid[8:]
                issz            = demog_file.loc[id_demog]['SZ_EEG_INFO']

                if log_:
                    verbose         = spf_args.get('verbose', False)  # Get verbose flag from config

                ### ----- Define datafolders
                ### ----- EEG
                datafoldereeg   = os.path.join(datafolders[nsubject]['eeg'], cf_ged_dir['results_folder'])
                ##### Re-direct to preprocessed folder
                if Results_Folder == os.path.join('Results', 'SNR_controls', 'Revision_IN'):
                    datafoldereeg   = os.path.join(datafolders[nsubject]['eeg'], 'revision', cf_ged_dir['results_folder'])
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
                tfreeg          = mne.time_frequency.read_tfrs(eeg_tfr)
                tfropm          = mne.time_frequency.read_tfrs(opm_tfr) 

                ### ----- Correct baseline and return dB
                avg_eeg         = correct_baseline(tfreeg, baseline=cf_peak['baseline'], islogandbaseline=0)
                avg_opm         = correct_baseline(tfropm, baseline=cf_peak['baseline'], islogandbaseline=0) 

                ### ----- Multiply -1 to TFR for alpha/beta
                if cf_peak['fit_negative']:  
                    avg_eeg = invert_tfr(avg_eeg)
                    avg_opm = invert_tfr(avg_opm)   

                ### ----- Correlate GED components and return best match
                ### ----- Prioritize analysis of first GED component (explains most variance)
                out         = pairwise_component_match_with_plots(
                    avg_eeg=avg_eeg, avg_opm=avg_opm,
                    fmin=cf_peak['freq_range'][0], fmax=cf_peak['freq_range'][1],
                    tmin=cf_peak['time_range'][0], tmax=cf_peak['time_range'][1],
                    smooth_sigma=cf_peak['freq_smooth'], 
                    n_eeg=n_components_eeg, n_opm=n_components_opm, 
                    
                    #plot_overviews=True, plot_best_match=True, show_corr=True, 
                    plot_overviews=False, plot_best_match=False, show_corr=False, 
                    prefer_first_if_r00_ge=0.5)  # If first component has r >= this threshold, select it
                    
                ### ----- Best matched components
                eeg_fit         = avg_eeg.copy().pick(out["best_eeg_name"])
                opm_fit         = avg_opm.copy().pick(out["best_opm_name"])

                ### ----- Fit Gaussians to the spectra of the best-matched components
                eeg_peaks, fig_eeg    = peak_finder_tfr(eeg_fit, **spf_args)
                if fig_eeg is not None:
                    fig_eeg.savefig(os.path.join(fig_folder, 
                                                f"{subjid}_gaussian_fits_{sensor_cfg}_{freqband}.png"), dpi=300)
                    #fig_eeg.show()
                    plt.close(fig_eeg)
                    
                opm_peaks, fig_opm    = peak_finder_tfr(opm_fit, **spf_args)
                # Save OPM peaks figure
                if fig_opm is not None:
                    fig_opm.savefig(os.path.join(fig_folder, 
                                                f"{subjid}_gaussian_fits_{sensor_cfg}_{freqband}_opm.png"), dpi=300)
                    #fig_opm.show()
                    plt.close(fig_opm)

                ### ----- Extract peak info
                peaks_eeg_info  = extract_peak_info_peakfinder(eeg_peaks, min_sep=cf_peak['min_sep'])
                peaks_opm_info  = extract_peak_info_peakfinder(opm_peaks, min_sep=cf_peak['min_sep'])

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
                    
                ### ----- Number of Trials
                demog_file.loc[id_demog, 'CommonTrials'] = len(tfreeg)

            print(f"\n{'='*60}")
            print(f"Processing complete. {n_subjects} subjects analyzed.")
            print(f"{'='*60}\n")

            ### ----- Check and create directories if needed
            results_freqband   = os.path.join(wdir, Results_Folder, 'results_' + freqband)
            os.makedirs(results_freqband, exist_ok=True)

            ### ----- Save File with Peak Power and Frequency in Results
            demog_file.to_csv(os.path.join(results_freqband, results_filename))
            print(f"Results saved to: {os.path.join(results_freqband, results_filename)}")

        finally:
            if log_:
                sys.stdout = old_stdout
                logfile.close()
            