import os
import sys
import mne
import json
import matplotlib
# matplotlib mode
matplotlib.use('QtAgg')     
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from utils_visgam.aslt_64 import aslt as aslt_hf
from utils_visgam.aslt_64_lf import aslt as aslt_lf
from utils_visgam.visgam_funcs import loadcfg
#from utils_visgam.visgam_paths_vk import datafolders, cfg_dir
from utils_visgam.visgam_paths_local import datafolders, cfg_dir, wdir
from utils_visgam.visgam_sf import pool_eeg_10, pool_opm_10, pool_eeg_20
from utils_visgam.visgam_sf import get_stim_event_id, ged_filter, sf_epoch_data_mark_ptp
from utils_visgam.visgam_tfr import epochs_subset_by_names_array, tfr_subset_by_names, common_trial_names, superlet_buffer
from joblib import Parallel, delayed

##### Run 
##### Frequency bands and sensor configurations
freqbands       = ['alpha', 'gamma']
sensor_configs  = ['10_10', '10_20']

##### Log
is_log          = True

##### Select subfolder within Results folder to save results
Results_Folder  = os.path.join('Results', 'SNR_controls', 'Revision_IN')

##### Run analysis for all frequency bands and sensor configurations
for freqband in freqbands:
    for sensor_cfg in sensor_configs:

        ##### ----- Select superlet kernel based on frequency range
        if freqband == 'gamma':
            aslt_kernel = aslt_hf
        else:
            aslt_kernel = aslt_lf # Import superlet with smaller buffer SD=3

        ##### Set .log file
        ##### ----- Create logs directory
        if is_log:
            logs_dir    = os.path.join(wdir, Results_Folder,  'analysis_logs', 'analysis_' + freqband + '_' + sensor_cfg)
            os.makedirs(logs_dir, exist_ok=True)    

        try:
            ##### Import analysis cfg
            cfg             = loadcfg(cfg_dir)
            ### ----- Obtain config files/set for the current analysis
            cf_ged          = cfg['ged_' + freqband]  # Spatial filter config 
            cf_tfr          = cfg['tfr_' + freqband]  # Configuration for TFR
            cf_ged_dir      = cfg['ged_' + freqband + '_folders_' + sensor_cfg]  # [..._10_20]

            ### ----- Set sensor pool to use for EEG
            if sensor_cfg   == '10_10':
                pool_eeg    = pool_eeg_10
            elif sensor_cfg == '10_20':
                pool_eeg    = pool_eeg_20   

            ##### If set to log, store cfg and create filepath
            if is_log:
                logfile     = open(os.path.join(logs_dir, 'analysis_' + freqband + '_' + sensor_cfg + '.log'), 'w')
                sys.stdout  = logfile

                ##### Save cfg files 
                with open(os.path.join(logs_dir, 'cf_ged.json'), 'w') as f:
                    json.dump(cf_ged, f, indent=4)
                with open(os.path.join(logs_dir, 'cf_tfr.json'), 'w') as f:
                    json.dump(cf_tfr, f, indent=4)
                with open(os.path.join(logs_dir, 'cf_ged_dir.json'), 'w') as f:
                    json.dump(cf_ged_dir, f, indent=4)

            ##### Print Number of datasets
            n_subjects      = len(datafolders)
            print('Total sample: ', n_subjects)

            ##### Start loop
            for nsubject in range(n_subjects):
                
                ### ----- Current dataset   
                print('Current Subject:', datafolders[nsubject]['id'])

                ### ----- Define datafolders
                ##### EEG
                datafoldereeg   = datafolders[nsubject]['eeg']
                ##### Re-direct to preprocessed folder
                ##### ----- EEG data will be read from the /revision subfolder
                if Results_Folder == os.path.join('Results', 'SNR_controls', 'Revision_IN'):
                    print('Redirecting to preprocessed folder for EEG data')
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
                
                ### ----- Load data
                eegdata         = mne.io.read_raw(eegfile, preload=True)
                opmdata         = mne.io.read_raw(opmfile, preload=True)

                ### ----- Interpolate EEG data and re-reference eeg to the average
                ### ----- Drop VEOG and Mastoids
                drop_list   = ['VEOG', 'LM', 'RM']
                to_drop     = [ch for ch in drop_list if ch in eegdata.ch_names]
                if len(to_drop) > 0:
                    eegdata.drop_channels(to_drop)

                eegdata.interpolate_bads(reset_bads=True, mode='accurate')
                eegdata.set_eeg_reference(ref_channels='average', projection=False, ch_type='eeg') 

                ### ----- Select EEG sensors for spatial filter 
                eegchannels = [ch for ch in pool_eeg if ch in eegdata.ch_names]
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

                ### ----- Filter data, mark bad segments and epoch data
                ### ----- EEG
                eeggamma, eegepochs, eeg_outliers = sf_epoch_data_mark_ptp(prepdata=eegdata, prepdataevents=eeg_events,
                    epoch_tmin=cf_ged['tmin'], epoch_tmax=cf_ged['tmax'], 
                    filt_lf=cf_ged['lfreq'], filt_hf=cf_ged['hfreq'], ptp=cf_ged['ptp'], k_channels=2, datatype='eeg')
                ##### Save outliers per sensor
                eeg_outliers.to_csv(os.path.join(results_eeg_dir, 'removed_epochs_per_sensor.csv'))

                ### ----- OPM
                opmgamma, opmepochs, opm_outliers = sf_epoch_data_mark_ptp(prepdata=opmdata, prepdataevents=opm_events,
                    epoch_tmin=cf_ged['tmin'], epoch_tmax=cf_ged['tmax'], 
                    filt_lf=cf_ged['lfreq'], filt_hf=cf_ged['hfreq'], ptp=cf_ged['ptp'], k_channels=2, datatype='mag')
                ##### Save outliers per sensor
                opm_outliers.to_csv(os.path.join(results_opm_dir, 'removed_epochs_per_sensor.csv'))
                #finally:
                #    print('OK')
                ### ----- Get common trials after preprocessing
                common_names        = common_trial_names(eegepochs, opmepochs)

                ### ----- Create epochs array only with common trials
                epochseeg_common    = epochs_subset_by_names_array(eegepochs, common_names)
                epochsopm_common    = epochs_subset_by_names_array(opmepochs, common_names)

                ### ----- Compute Spatial Filters using the same epochs for EEG and OPM
                ### ----- EEG
                rank_eeg            = len(epochseeg_common.info['ch_names']) 
                evals_eeg, w_eeg    = ged_filter(epochseeg_common, signal=cf_ged['epoch_signal'], 
                                            noise=cf_ged['epoch_noise'], n_components=rank_eeg)
                ### ----- OPM
                rank_opm            = len(epochsopm_common.info['ch_names']) 
                evals_opm, w_opm    = ged_filter(epochsopm_common, signal=cf_ged['epoch_signal'], 
                                            noise=cf_ged['epoch_noise'], n_components=rank_opm)

                ### ----- Save eigenvalues/vectors
                ### ----- EEG
                np.save(os.path.join(results_eeg_dir, cf_ged_dir['evecs_filename']), w_eeg) 
                np.save(os.path.join(results_eeg_dir, cf_ged_dir['evals_filename']), evals_eeg)   
                
                ### ----- OPM
                np.save(os.path.join(results_opm_dir, cf_ged_dir['evecs_filename']), w_opm) 
                np.save(os.path.join(results_opm_dir, cf_ged_dir['evals_filename']), evals_opm)     
                
                ### ----- Prepare broadband data and apply spatial filter
                ### ----- Add annotations for bad noise and signal
                eegdata.set_annotations(eeggamma.annotations.copy())
                opmdata.set_annotations(opmgamma.annotations.copy())

                ### ----- Epoch broadband data
                ### ----- EEG
                X_eeg   = mne.Epochs(eegdata.copy().filter(l_freq=5, h_freq=None), 
                    event_id=eeg_events, tmin=cf_ged['tmin'], tmax=cf_ged['tmax'],
                    baseline=None, proj=False, detrend=1, reject_by_annotation=True, preload=True)
                ### ----- OPM
                X_opm   = mne.Epochs(opmdata.copy().filter(l_freq=5, h_freq=None), 
                    event_id=opm_events, tmin=cf_ged['tmin'], tmax=cf_ged['tmax'],
                    baseline=None, proj=False, detrend=1, reject_by_annotation=True, preload=True)

                ### ----- Get common trials
                del common_names
                common_names    = common_trial_names(X_eeg, X_opm)

                ### ----- Select subset of common trials
                X_eeg_final     = X_eeg[common_names]
                X_opm_final     = X_opm[common_names]

                ### ----- Apply Filters
                max_channels    = 3
                Y_eeg           = np.einsum('ecn,ck->ekn', X_eeg_final.get_data(), w_eeg[:, :max_channels])
                Y_opm           = np.einsum('ecn,ck->ekn', X_opm_final.get_data(), w_opm[:, :max_channels])

                ### ----- TFR analysis
                modality    = ['eeg', 'opm']
                ### ----- Process data for EEG and OPM
                for mod_ in range(len(modality)): 
                    if mod_ == 0: # For EEG data
                        visgamepochs    = X_eeg_final.copy()
                        sfreq           = eegdata.info['sfreq']
                        picks           = 'eeg'
                        dataY           = Y_eeg
                        datafolder      = results_eeg_dir
                        nchannels       = max_channels
                    else:
                        visgamepochs    = X_opm_final.copy()
                        sfreq           = opmdata.info['sfreq']
                        picks           = 'mag'
                        dataY           = Y_opm
                        datafolder      = results_opm_dir
                        nchannels       = max_channels

                    ### ----- Parallel across channels
                    foi         = np.arange(cf_tfr['foi_start'], cf_tfr['foi_stop'], cf_tfr['foi_step'])
                    n_cycles    = cf_tfr['n_cycles'] 
                    order       = cf_tfr['order'] 
                    mult        = cf_tfr['mult'] 
                    n_channels  = nchannels # GED components 

                    ### ----- Function wrapper: takes one channel and returns its TFR
                    def run_superlet_channel(data_ch, sfreq, foi, n_cycles, order, mult):
                        return aslt_kernel(data_ch, sfreq=sfreq, foi=foi, n_cycles=n_cycles, order=order, mult=mult)
                    
                    tfrs = Parallel(n_jobs=8, verbose=1)(
                        delayed(run_superlet_channel)(
                            dataY[:, ch_idx, :], sfreq, foi, n_cycles, order, mult
                        )
                        for ch_idx in range(n_channels)
                    )

                    ### ----- Stack into final result: shape (n_trials, n_channels, n_freqs, n_times)
                    tfrs_array = np.stack(tfrs, axis=1)
                    del tfrs
                    
                    ### ----- Reduce size of tfr array
                    power      = tfrs_array.astype(np.float32, copy=False)
                    del tfrs_array

                    ### ----- Control that the dimensionality of the array is correct
                    n_ep, n_ch, n_f, n_t = power.shape
                    assert n_ep == len(visgamepochs)               
                    assert n_ch == n_channels                     
                    assert n_f  == len(foi)
                    assert n_t  == len(visgamepochs.times)

                    ### ----- Events must correspond to the kept epochs
                    assert visgamepochs.events.shape[0] == n_ep

                    ### ----- Filter only present events
                    present     = set(np.unique(visgamepochs.events[:, 2]))
                    event_id_ok = {k: v for k, v in visgamepochs.event_id.items() if v in present}

                    ### ----- Create Info For TFR containing GED channels
                    ch_type     = picks
                    info_virt   = mne.create_info(ch_names=['GED_1', 'GED_2', 'GED_3'], sfreq=sfreq, ch_types=ch_type)

                    ### ----- Create the EpochsTFR object with the superlet output
                    tfr = mne.time_frequency.EpochsTFRArray(
                        info        = info_virt,
                        events      = visgamepochs.events,
                        event_id    = visgamepochs.event_id,
                        data        = power,                 
                        times       = visgamepochs.times,
                        freqs       = foi,
                        method      = 'superlet',        
                        comment     = 'custom_superlet', 
                    )
                    ### ----- Prepare TFR before saving: Crop and decimate 
                    tfr.crop(tmin=cf_tfr['tmin'], tmax=cf_tfr['tmax'])
                    tfr.decimate(cf_tfr['decim'])
                    
                    ### ----- Create the EpochsTFR object with the GED virtual channel
                    epochs_arr = mne.EpochsArray(
                        info        = info_virt,
                        events      = visgamepochs.events,
                        event_id    = event_id_ok,
                        data        = dataY, # shape: (n_epochs, n_channels, n_times)
                        tmin        = visgamepochs.tmin
                    )

                    ### ----- Save data
                    tfr.save(os.path.join(datafolder, cf_ged_dir['tfr_filename']), overwrite=True)
                    epochs_arr.save(os.path.join(datafolder, cf_ged_dir['epoch_filename']), overwrite=True)

                    ### ----- Store Results using Wavelets
                    n_cycles    = 3  # simple fixed choice
                    morlet_tfr  = mne.time_frequency.tfr_morlet(
                        epochs_arr, freqs=foi, n_cycles=n_cycles, use_fft=True, return_itc=False, 
                        average=True, decim=2, n_jobs=-1, verbose=True)       
                    morlet_tfr.save(os.path.join(datafolder, 'morlet_' + cf_ged_dir['tfr_filename']), overwrite=True)

                    ### ----- Clear all modality data
                    del visgamepochs, sfreq, picks, dataY, datafolder, tfr, epochs_arr, morlet_tfr
        finally:
            if is_log :
                sys.stdout = sys.__stdout__  # Reset to default after analysis
                logfile.close()


