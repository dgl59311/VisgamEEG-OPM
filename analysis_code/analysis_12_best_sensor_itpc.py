# analysis_2_tfr_sensors_evoked
import os
import json
import mne
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
#from utils_visgam.visgam_paths_vk import datafolders, local_folder, wdir, cfg_dir
from utils_visgam.visgam_paths_local import datafolders, local_folder, wdir, cfg_dir
from utils_visgam.visgam_sf import pool_eeg_10, pool_eeg_20, pool_opm_10, get_stim_event_id
from utils_visgam.aslt_64 import aslt
from utils_visgam.visgam_tfr import epochs_subset_by_names_array, tfr_subset_by_names, common_trial_names
from utils_visgam.visgam_funcs import loadcfg
from mne.time_frequency import tfr_morlet

### ----- matplotlib mode
matplotlib.use('QtAgg')     

### ----- Load Demog File
demog_file  = pd.read_csv(os.path.join(wdir,"DemographicFile.csv"), index_col=0)

### ----- import analysis cfg
sensor_cfg  = '10_20'  
cfg         = loadcfg(cfg_dir)
cf_itpc     = cfg['itpc']           # Configuration for ITPC
cf_ged      = cfg['ged_itpc']      # Configuration for ITPC

### ----- number of subjects
n_subjects  = len(datafolders)
print('Total sample: ', n_subjects)

##### Allocate data
itpcs_eeg    = []
itpcs_opm_x, itpcs_opm_y  = [], []

##### Select subfolder within Results folder to save results
Results_Folder  = os.path.join('Results', 'SNR_controls', 'Revision_IN')

#%matplotlib qt
for nsubject in range(n_subjects):

    ### ----- Subject ID
    subjid          = datafolders[nsubject]['id']
    ### ----- Determine if is patient
    issz            = demog_file.loc[subjid[8:]]['SZ_EEG_INFO'] 

    ### ----- Analyze data for controls only
    if not issz:

        ### ----- Define datafolders
        ##### Re-direct to preprocessed folder
        datafoldereeg   = datafolders[nsubject]['eeg']
        if Results_Folder == os.path.join('Results', 'SNR_controls', 'Revision_IN'):
            datafoldereeg   = os.path.join(datafoldereeg, 'revision')
        eegfile         = os.path.join(datafoldereeg, 'preprocessed_3_ica_eeg.fif')

        datafolderopm   = datafolders[nsubject]['opm']
        opmfile         = os.path.join(datafolderopm, 'preprocessed_3_ica.fif') 

        ### ----- Load data
        eegdata         = mne.io.read_raw(eegfile, preload=True)
        opmdata         = mne.io.read_raw(opmfile, preload=True)

        ### ----- Interpolate EEG data and re-reference eeg to the average
        #### Interpolate EEG and re-reference
        ### ----- Drop VEOG and Mastoids
        drop_list   = ['VEOG', 'LM', 'RM']
        to_drop     = [ch for ch in drop_list if ch in eegdata.ch_names]
        if len(to_drop) > 0:
            eegdata.drop_channels(to_drop)

        eegdata.interpolate_bads(reset_bads=True, mode='accurate')
        eegdata.set_eeg_reference(ref_channels='average', projection=False, ch_type='eeg') 
        ### ----- Select EEG sensors for spatial filter 
        eegchannels = [ch for ch in pool_eeg_10 if ch in eegdata.ch_names]
        eegdata.pick(eegchannels)

        ### ----- Remove bads from OPM data (no interpolation)
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

        ##### Get only the trials used for the spatial filter
        common_names    = common_trial_names(X_eeg, X_opm)
        X_eeg           = epochs_subset_by_names_array(X_eeg, common_names)
        X_opm           = epochs_subset_by_names_array(X_opm, common_names)

        ### ----- Perform ITPC
        foi             = np.arange(cf_itpc['foi_start'], cf_itpc['foi_stop'] + cf_itpc['foi_step'], cf_itpc['foi_step'])
        n_cycles        = np.linspace(cf_itpc['min_cycles'], cf_itpc['max_cycles'], len(foi))
        ### ----- EEG
        morleteeg       = tfr_morlet(X_eeg, freqs=foi, n_cycles=n_cycles, use_fft=True,
                                    return_itc=True, average=True, decim=cf_itpc['decim'], n_jobs=-1)
        ### ----- OPM
        morletopm       = tfr_morlet(X_opm, freqs=foi, n_cycles=n_cycles, use_fft=True,
                                    return_itc=True, average=True, decim=cf_itpc['decim'], n_jobs=-1)
        ### ----- Select ITPC
        morleteeg       = morleteeg[1]
        morletopm       = morletopm[1]

        ##### Select best channel for low itpc
        ### ----- EEG
        lf_eeg          = morleteeg.copy().crop(tmin=cf_itpc['lf_times'][0], tmax=cf_itpc['lf_times'][1], 
                                                fmin=cf_itpc['lf_freqs'][0], fmax=cf_itpc['lf_freqs'][1])
        best_lf_eeg     = np.max(lf_eeg.copy().data.mean(axis=(1, 2)))
        lf_eeg          = morleteeg.info['ch_names'][np.argmax(lf_eeg.data.mean(axis=(1, 2)))] 
        
        ### ----- OPM
        lf_opm          = morletopm.copy().crop(tmin=cf_itpc['lf_times'][0], tmax=cf_itpc['lf_times'][1], 
                                                fmin=cf_itpc['lf_freqs'][0], fmax=cf_itpc['lf_freqs'][1])
        best_lf_opm     = np.max(lf_opm.copy().data.mean(axis=(1, 2)))
        lf_opm          = morletopm.info['ch_names'][np.argmax(lf_opm.data.mean(axis=(1, 2)))] 


        ##### Select best channel for high itpc
        ### ----- EEG
        hf_eeg          = morleteeg.copy().crop(tmin=cf_itpc['hf_times'][0], tmax=cf_itpc['hf_times'][1], 
                                                fmin=cf_itpc['hf_freqs'][0], fmax=cf_itpc['hf_freqs'][1])
        best_hf_eeg     = np.max(hf_eeg.copy().data.mean(axis=(1, 2)))
        hf_eeg          = morleteeg.info['ch_names'][np.argmax(hf_eeg.data.mean(axis=(1, 2)))] 
        
        ### ----- OPM
        hf_opm          = morletopm.copy().crop(tmin=cf_itpc['hf_times'][0], tmax=cf_itpc['hf_times'][1], 
                                                fmin=cf_itpc['hf_freqs'][0], fmax=cf_itpc['hf_freqs'][1])
        best_hf_opm     = np.max(hf_opm.copy().data.mean(axis=(1, 2)))
        hf_opm          = morletopm.info['ch_names'][np.argmax(hf_opm.data.mean(axis=(1, 2)))] 

        ### ----- Fill in demog file
        id_demog    = subjid[8:]

        ### ----- Low ITPC
        demog_file.loc[id_demog, 'itpc_low_eeg']    = best_lf_eeg
        demog_file.loc[id_demog, 'CHAN_low_eeg']    = lf_eeg
        demog_file.loc[id_demog, 'itpc_low_opm']    = best_lf_opm
        demog_file.loc[id_demog, 'CHAN_low_opm']    = lf_opm

        ### ----- High ITPC
        demog_file.loc[id_demog, 'itpc_high_eeg']   = best_hf_eeg
        demog_file.loc[id_demog, 'CHAN_high_eeg']   = hf_eeg
        demog_file.loc[id_demog, 'itpc_high_opm']   = best_hf_opm
        demog_file.loc[id_demog, 'CHAN_high_opm']   = hf_opm

        ### ----- Common Trials
        demog_file.loc[id_demog, 'EEGTrials']       = morleteeg.nave
        demog_file.loc[id_demog, 'OPMTrials']       = morletopm.nave
        demog_file.loc[id_demog, 'CommonTrials']    = morleteeg.nave

demog_file.to_csv(os.path.join(wdir, Results_Folder, 'results_itpc', 'itpc_best_' + sensor_cfg + '.csv'))