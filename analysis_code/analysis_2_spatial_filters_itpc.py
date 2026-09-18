import os
import json
import sys
import mne
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
#from utils_visgam.visgam_paths_vk import datafolders, cfg_dir
from utils_visgam.visgam_paths_local import datafolders, cfg_dir, wdir
from utils_visgam.visgam_sf import pool_eeg_10, pool_opm_10, pool_eeg_20
from utils_visgam.visgam_sf import get_stim_event_id, ged_filter_evoked
from utils_visgam.visgam_tfr import epochs_subset_by_names_array, tfr_subset_by_names, common_trial_names
from utils_visgam.visgam_funcs import loadcfg
from mne.time_frequency import tfr_morlet

##### Run 
##### Sensor configurations
sensor_configs = ['10_10', '10_20']

##### Select subfolder within Results folder to save results
Results_Folder = os.path.join('Results', 'SNR_controls', 'Revision_IN')

for sensor_cfg in sensor_configs:
    ##### Set analysis parameters / Set .log file
    logs_dir    = os.path.join(wdir, Results_Folder,  'analysis_logs', 'analysis_itpc_' + sensor_cfg)
    os.makedirs(logs_dir, exist_ok=True)    
    logfile     = open(os.path.join(logs_dir, 'analysis_itpc_' + sensor_cfg + '.log'), 'w')
    old_stdout  = sys.stdout
    sys.stdout  = logfile
    
    try:
        ##### Import analysis cfg
        cfg         = loadcfg(cfg_dir)
        ### ----- Obtain config files/set for the current analysis
        cf_ged      = cfg['ged_itpc']  # Spatial filter config 
        cf_itpc     = cfg['itpc']      # Configuration for ITPC
        cf_ged_dir  = cfg['ged_itpc_folders_' + sensor_cfg]  # [..._10_20]

        ### ----- Set sensor pool to use for EEG
        if sensor_cfg   == '10_10':
            pool_eeg    = pool_eeg_10
        elif sensor_cfg == '10_20':
            pool_eeg    = pool_eeg_20   

        ##### Save cfg files 
        with open(os.path.join(logs_dir, 'cf_ged.json'), 'w') as f:
            json.dump(cf_ged, f, indent=4)
        with open(os.path.join(logs_dir, 'cf_itpc.json'), 'w') as f:
            json.dump(cf_itpc, f, indent=4)
        with open(os.path.join(logs_dir, 'cf_ged_dir.json'), 'w') as f:
            json.dump(cf_ged_dir, f, indent=4)


        ##### Number of datasets
        n_subjects  = len(datafolders)
        print('Total sample: ', n_subjects)

        #%matplotlib qt
        ### ----- Start Loop
        for nsubject in range(n_subjects):
            ### ----- Define datafolders
            ##### EEG
            datafoldereeg   = datafolders[nsubject]['eeg']
            ##### Re-direct to preprocessed folder
            if Results_Folder == os.path.join('Results', 'SNR_controls', 'Revision_IN'):
                datafoldereeg   = os.path.join(datafoldereeg, 'revision')
            eegfile         = os.path.join(datafoldereeg, 'preprocessed_3_ica_eeg.fif') 
            
            ##### OPM
            datafolderopm   = datafolders[nsubject]['opm']
            opmfile         = os.path.join(datafolderopm, 'preprocessed_3_ica.fif') 
            
            ### ----- Define resultsfolders
            results_eeg_dir = os.path.join(datafoldereeg, cf_ged_dir['results_folder'])
            results_opm_dir = os.path.join(datafolderopm, cf_ged_dir['results_folder'])
            ### ----- Check and create directories if needed
            os.makedirs(results_eeg_dir, exist_ok=True)
            os.makedirs(results_opm_dir, exist_ok=True)

            ##### Print current data set being analyzed
            print('Current Subject:', datafolders[nsubject]['id'])
            
            ### ----- Load data
            eegdata         = mne.io.read_raw(eegfile, preload=True)
            opmdata         = mne.io.read_raw(opmfile, preload=True)

            ### ----- Interpolate EEG data and re-reference eeg to the average
            ### ----- Drop VEOG and Mastoids
            drop_list       = ['VEOG', 'LM', 'RM']
            to_drop         = [ch for ch in drop_list if ch in eegdata.ch_names]
            if len(to_drop) > 0:
                eegdata.drop_channels(to_drop)

            eegdata.interpolate_bads(reset_bads=True, mode='accurate')
            eegdata.set_eeg_reference(ref_channels='average', projection=False, ch_type='eeg') 
            
            ### ----- Select EEG sensors for spatial filter 
            eegchannels     = [ch for ch in pool_eeg if ch in eegdata.ch_names]
            eegdata.pick(eegchannels)

            ### ----- Select OPM sensors for spatial filter 
            opmdata         = opmdata.copy().drop_channels(opmdata.info['bads'])
            opmchannels     = [ch for ch in pool_opm_10 if ch in opmdata.ch_names]
            opmdata.pick(opmchannels)

            ### ----- Save channels used for the spatial filter
            np.save(os.path.join(results_eeg_dir, 'eegsensors_sf.npy'), np.array(eegchannels))
            np.save(os.path.join(results_opm_dir, 'opmsensors_sf.npy'), np.array(opmchannels))

            ### ----- Get stimulus onset events
            eeg_events      = get_stim_event_id(eegdata, event_name='STIMON')
            opm_events      = get_stim_event_id(opmdata, event_name='STIMON')

            ### ----- Bandpass filter the data
            ##### EEG
            feegdata = eegdata.copy().filter(l_freq=cf_ged['lfreq'], h_freq=cf_ged['hfreq'], fir_design='firwin')
            ##### OPM
            fopmdata = opmdata.copy().filter(l_freq=cf_ged['lfreq'], h_freq=cf_ged['hfreq'], fir_design='firwin')

            ##### Epoch filtered data
            ### ----- EEG
            X_eeg   = mne.Epochs(feegdata, event_id=eeg_events, tmin=cf_ged['tmin'], tmax=cf_ged['tmax'],
                baseline=None, proj=False, detrend=1, reject_by_annotation=True, preload=True)
            ### ----- OPM
            X_opm   = mne.Epochs(fopmdata, event_id=opm_events, tmin=cf_ged['tmin'], tmax=cf_ged['tmax'],
                baseline=None, proj=False, detrend=1, reject_by_annotation=True, preload=True)
            
            ### ----- Get common trials
            common_names    = common_trial_names(X_eeg, X_opm)
            X_eeg           = epochs_subset_by_names_array(X_eeg, common_names)
            X_opm           = epochs_subset_by_names_array(X_opm, common_names)

            ### ----- Compute GED (fit on the 5-60 Hz band-pass filtered data)
            evals_eeg, w_eeg = ged_filter_evoked(X_eeg, signal_win=cf_ged['ged_epoch'], n_components=5)
            evals_opm, w_opm = ged_filter_evoked(X_opm, signal_win=cf_ged['ged_epoch'], n_components=5)

            ### ----- Prepare broadband data 
            X_eeg_bb        = mne.Epochs(eegdata.copy().filter(l_freq=5, h_freq=None),
                event_id=eeg_events, tmin=cf_ged['tmin'], tmax=cf_ged['tmax'],
                baseline=None, proj=False, detrend=1, reject_by_annotation=True, preload=True)
            X_opm_bb        = mne.Epochs(opmdata.copy().filter(l_freq=5, h_freq=None),
                event_id=opm_events, tmin=cf_ged['tmin'], tmax=cf_ged['tmax'],
                baseline=None, proj=False, detrend=1, reject_by_annotation=True, preload=True)

            common_names_bb     = common_trial_names(X_eeg_bb, X_opm_bb)
            X_eeg_bb            = epochs_subset_by_names_array(X_eeg_bb, common_names_bb)
            X_opm_bb            = epochs_subset_by_names_array(X_opm_bb, common_names_bb)

            ### ----- Apply Filters
            Y_eeg   = np.einsum('ecn,ck->ekn', X_eeg_bb.get_data(), w_eeg[:, :3])
            Y_opm   = np.einsum('ecn,ck->ekn', X_opm_bb.get_data(), w_opm[:, :3])

            ### ----- Save Eigenvalues
            ### ----- EEG
            np.save(os.path.join(results_eeg_dir, cf_ged_dir['evecs_filename']), w_eeg) 
            np.save(os.path.join(results_eeg_dir, cf_ged_dir['evals_filename']), evals_eeg)

            ### ----- OPM
            np.save(os.path.join(results_opm_dir, cf_ged_dir['evecs_filename']), w_opm) 
            np.save(os.path.join(results_opm_dir, cf_ged_dir['evals_filename']), evals_opm)   
            
            ### ----- Time-Frequency Analysis
            modality = ['eeg', 'opm']
            for mod_ in range(len(modality)): 
                if mod_ == 0:
                    ##### Get information to create arrays
                    visgamepochs    = X_eeg_bb.copy()
                    sfreq           = eegdata.info['sfreq']
                    picks           = 'eeg'
                    dataY           = Y_eeg
                    datafolder      = results_eeg_dir

                else:
                    ##### Get information to create arrays
                    visgamepochs    = X_opm_bb.copy()
                    sfreq           = opmdata.info['sfreq']
                    picks           = 'mag'
                    dataY           = Y_opm
                    datafolder      = results_opm_dir
            
                ##### Filter only present events
                present     = set(np.unique(visgamepochs.events[:, 2]))
                event_id_ok = {k: v for k, v in visgamepochs.event_id.items() if v in present}
            
                ##### Events must correspond to the kept epochs
                assert visgamepochs.events.shape[0] == len(visgamepochs) 

                ##### Create Information
                ch_type     = picks
                info_virt   = mne.create_info(ch_names=['ITPC_1', 'ITPC_2', 'ITPC_3'], sfreq=sfreq, ch_types=ch_type)

                ##### Create the EpochsTFR object
                epochs_arr = mne.EpochsArray(
                    info        = info_virt,
                    events      = visgamepochs.events,
                    event_id    = event_id_ok,
                    data        = dataY, 
                    tmin        = visgamepochs.tmin
                )
                ##### Create array of epochs and save average
                avgarray        = epochs_arr.copy().average()
                avgarray.save(os.path.join(datafolder, cf_ged_dir['filename_evoked']), overwrite=True)

                ### ----- Perform ITPC
                foi             = np.arange(cf_itpc['foi_start'], cf_itpc['foi_stop'] + cf_itpc['foi_step'], cf_itpc['foi_step'])
                n_cycles        = np.linspace(cf_itpc['min_cycles'], cf_itpc['max_cycles'], len(foi))
                morletdata      = tfr_morlet(epochs_arr, freqs=foi, n_cycles=n_cycles, use_fft=True,
                                            return_itc=True, average=True, decim=cf_itpc['decim'], n_jobs=-1)
                ### ----- Get ITPC
                morletdata      = morletdata[1]
                morletdata      = morletdata.copy().crop(tmin=-0.75, tmax=1)

                ### ----- Save Data
                morletdata.save(os.path.join(datafolder, cf_ged_dir['filename']), overwrite=True)

                ### ----- Clear modality data
                del visgamepochs, sfreq, picks, dataY, datafolder, morletdata
            
    finally:
        sys.stdout = old_stdout
        logfile.close()