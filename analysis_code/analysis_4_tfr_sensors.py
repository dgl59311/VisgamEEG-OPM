# analysis_2_tfr_sensors_evoked
import os
import json
import mne
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import gc
from utils_visgam.visgam_paths_local import datafolders, local_folder, cfg_dir
from utils_visgam.visgam_sf import pool_eeg_10, pool_eeg_20, pool_opm_10, get_stim_event_id
from utils_visgam.aslt_64 import aslt as aslt_hf
from utils_visgam.aslt_64_lf import aslt as aslt_lf
from utils_visgam.visgam_tfr import epochs_subset_by_names_array, tfr_subset_by_names, common_trial_names
from utils_visgam.visgam_funcs import loadcfg

### ----- matplotlib mode
matplotlib.use('QtAgg')     

### ----- import analysis cfg
cfg             = loadcfg(cfg_dir)
cf_ged          = cfg['ged_gamma']  # Spatial filter config 
print('Current config: ', cf_ged)

### ----- load configurations for TFRs
cf_tfr_gamma    = cfg['tfr_gamma']        # Configuration for TFR gamma
cf_tfr_alpha    = cfg['tfr_alpha']        # Configuration for TFR alpha

### ----- number of subjects
n_subjects      = len(datafolders)
print('Total sample: ', n_subjects)

##### Select subfolder within Results folder to save results
Results_Folder  = os.path.join('Results', 'SNR_controls', 'Revision_IN')

#%matplotlib qt
### ----- Start Loop
for nsubject in range(n_subjects):
    ##### Subject ID
    subjid          = datafolders[nsubject]['id']

    ##### Define datafolders
    ### ----- EEG
    ##### Re-direct to preprocessed folder
    datafoldereeg   = datafolders[nsubject]['eeg']
    if Results_Folder == os.path.join('Results', 'SNR_controls', 'Revision_IN'):
        datafoldereeg   = os.path.join(datafoldereeg, 'revision') 
    eegfile         = os.path.join(datafoldereeg, 'preprocessed_3_ica_eeg.fif')     
    ### ----- OPM
    datafolderopm   = datafolders[nsubject]['opm']
    opmfile         = os.path.join(datafolderopm, 'preprocessed_3_ica.fif') 

    ##### Load datasets
    eegdata         = mne.io.read_raw(eegfile, preload=True)
    opmdata         = mne.io.read_raw(opmfile, preload=True)

    ### ----- Interpolate EEG data and re-reference eeg to the average
    ### ----- Drop VEOG and Mastoids
    drop_list   = ['VEOG', 'LM', 'RM']
    to_drop     = [ch for ch in drop_list if ch in eegdata.ch_names]
    if len(to_drop) > 0:
        eegdata.drop_channels(to_drop)

    ### ----- Interpolate EEG and re-reference
    ##### ----- Obtain TFRs for all sensors
    eegdata.interpolate_bads(reset_bads=True, mode='accurate')
    eegdata.set_eeg_reference(ref_channels='average', projection=False, ch_type='eeg') 

    ##### Remove bads from OPM data (no interpolation)
    opmdata         = opmdata.copy().drop_channels(opmdata.info['bads'])
    opmchannels     = [ch for ch in pool_opm_10 if ch in opmdata.ch_names]
    opmdata.pick(opmchannels)
    
    ##### Epoch data
    ### ----- Get stimulus onset events
    eeg_events      = get_stim_event_id(eegdata, event_name='STIMON')
    opm_events      = get_stim_event_id(opmdata, event_name='STIMON')
    ### ----- EEG
    X_eeg   = mne.Epochs(eegdata.copy().filter(l_freq=5, h_freq=None), event_id=eeg_events, tmin=cf_ged['tmin'], tmax=cf_ged['tmax'],
        baseline=None, proj=False, detrend=1, reject_by_annotation=True, preload=True)
    ### ----- OPM
    X_opm   = mne.Epochs(opmdata.copy().filter(l_freq=5, h_freq=None), event_id=opm_events, tmin=cf_ged['tmin'], tmax=cf_ged['tmax'],
        baseline=None, proj=False, detrend=1, reject_by_annotation=True, preload=True)

    ### ----- Get only common trials across modalities
    common_names    = common_trial_names(X_eeg, X_opm)
    X_eeg           = epochs_subset_by_names_array(X_eeg, common_names)    
    X_opm           = epochs_subset_by_names_array(X_opm, common_names)

    del eegdata, opmdata
    
    ### ----- Perform superlet analysis on single sensors
    modality = ['eeg', 'opm']
    for mod_ in range(len(modality)): 
        if mod_ == 0:
            sfreq           = X_eeg.info['sfreq']  # your sampling rate
            X_norm          = X_eeg.get_data()
            n_channels      = X_norm.shape[1]
            visgamepochs    = X_eeg.copy()
            datafolder      = datafoldereeg 
            folderlocal     = os.path.join(local_folder, 'EEG') 
        else:
            sfreq           = X_opm.info['sfreq']  # your sampling rate
            X_norm          = X_opm.get_data()
            n_channels      = X_norm.shape[1]
            visgamepochs    = X_opm.copy()
            datafolder      = datafolderopm 
            folderlocal     = os.path.join(local_folder, 'OPMMEG')

        ### ----- Do TFR analysis for lower and higher frequencies separately
        for tfr_roi in range(2):
            ### ----- For gamma band analysis
            if tfr_roi == 0:
                cf_tfr              = cf_tfr_gamma
                ##### ----- All-sensors run: separate output folder 
                results_dir         = os.path.join(datafolder, 'TFR_gamma_allsensors')
                local_results_dir   = os.path.join(folderlocal, 'TFR_HIGHFREQ_allsensors')
                aslt_kernel         = aslt_hf
                ### ----- Check and create directories if needed
                os.makedirs(results_dir, exist_ok=True)
                os.makedirs(local_results_dir, exist_ok=True)
            ### ----- For alpha band analysis
            if tfr_roi == 1:
                cf_tfr              = cf_tfr_alpha
                ##### ----- Same all-sensors separation as above
                results_dir         = os.path.join(datafolder, 'TFR_alpha_allsensors')
                local_results_dir   = os.path.join(folderlocal, 'TFR_LOWFREQ_allsensors')
                aslt_kernel         = aslt_lf
                ### ----- Check and create directories if needed
                os.makedirs(results_dir, exist_ok=True)
                os.makedirs(local_results_dir, exist_ok=True)
    

            ### ----- Parallel across channels
            foi         = np.arange(cf_tfr['foi_start'], cf_tfr['foi_stop'], cf_tfr['foi_step'])
            n_cycles    = cf_tfr['n_cycles'] 
            order       = cf_tfr['order'] 
            mult        = cf_tfr['mult'] 

            ### ----- Function wrapper: takes one channel and returns its TFR
            def run_superlet_channel(data_ch, sfreq, foi, n_cycles, order, mult):
                return aslt_kernel(data_ch, sfreq=sfreq, foi=foi, n_cycles=n_cycles, order=order, mult=mult)
            
            ### ----- Limit parallel jobs to reduce memory usage
            n_jobs = 8 
            tfrs = Parallel(n_jobs=n_jobs, verbose=1)(
                delayed(run_superlet_channel)(
                    X_norm[:, ch_idx, :], sfreq, foi, n_cycles, order, mult
                )
                for ch_idx in range(n_channels)
            )

            ### ----- Explicitly collect garbage after parallel computation
            gc.collect()

            ### ----- Stack and ensure float32 dtype
            tfrs_array = np.stack(tfrs, axis=1).astype(np.float32, copy=False)
            del tfrs
            gc.collect()

            power = tfrs_array
            del tfrs_array
            gc.collect()

            n_ep, n_ch, n_f, n_t = power.shape
            assert n_ep == len(visgamepochs)               # epochs after rejection
            assert n_ch == len(visgamepochs.ch_names)      # channels after dropping bads
            assert n_f  == len(foi)
            assert n_t  == len(visgamepochs.times)

            ### ----- Events must correspond to the kept epochs
            assert visgamepochs.events.shape[0] == n_ep

            ### ----- Create the EpochsTFR object
            tfr = mne.time_frequency.EpochsTFRArray(
                info        = visgamepochs.info,
                events      = visgamepochs.events,
                event_id    = visgamepochs.event_id,
                data        = power,                  # shape: (n_epochs, n_channels, n_freqs, n_times)
                times       = visgamepochs.times,
                freqs       = foi,
                method      = 'superlet',         # just a label
                comment     = 'custom_superlet', # optional comment
            )
            tfr.crop(tmin=cf_tfr['tmin'], tmax=cf_tfr['tmax'])

            ### ----- Save average 
            average_data = tfr.average()
            average_data.save(os.path.join(results_dir, 'AverageTFR_Superlets.h5'), overwrite=True)

            ### ----- Per-trial save, restricted to the 10-electrode/OPM pool
            ### ----- (not the full sensor montage)
            pool_names      = pool_eeg_10 if mod_ == 0 else pool_opm_10
            pool_ch_present = [ch for ch in pool_names if ch in tfr.ch_names]
            tfr_pool        = tfr.copy().pick(pool_ch_present)
            tfr_pool.decimate(3)
            tfr_pool.save(os.path.join(local_results_dir, subjid + '_EpochsTFR_Superlets_pool.h5'), overwrite=True)

            ### ----- Clean up memory
            del results_dir, cf_tfr, tfr, tfr_pool, power, average_data, local_results_dir
            gc.collect()

        del sfreq, datafolder, folderlocal, visgamepochs

