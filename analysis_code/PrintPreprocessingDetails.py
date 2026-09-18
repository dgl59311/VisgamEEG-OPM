import os
import mne
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter
from utils_visgam.visgam_paths_local import datafolders, wdir, daten

##### Load Demog File
demog_file      = pd.read_csv(os.path.join(wdir, "DemographicFile.csv"), index_col=0)
trialinfo_dir   = os.path.join(daten, 'BEHAVIOR')

##### Name of Preprocessed files
final_eeg       = 'preprocessed_3_ica_eeg.fif'
final_opm       = 'preprocessed_3_ica.fif'

##### Output folder for EEG data
eegoutputfolder = 'revision' # initial_submission_IN

#### Name for ICA files
ica_eeg         = 'ica_solution_eeg.fif'
ica_opm         = 'ica_solution.fif'
##### ----- Start loop and print preprocessing details
for nsubject in range(49):
    
    # Define datafolders
    foldereeg   = datafolders[nsubject]['eeg']
    folderopm   = datafolders[nsubject]['opm']
    subjid      = datafolders[nsubject]['id']
    print('Current Subject:', subjid)
    
    subjectid   = subjid.replace('subject_', '')

    # EEG
    # load preprocessed data files
    eegdata         = mne.io.read_raw(os.path.join(foldereeg, eegoutputfolder, final_eeg), preload=True)
    totaleegchans   = len(eegdata.copy().pick_types(eeg=True, exclude=[]).ch_names)


    # check valid trials
    _, event_id_all = mne.events_from_annotations(eegdata)
    stim_event_id   = {k: v for k, v in event_id_all.items() if 'STIMON' in k}
    eeg_epochs      = mne.Epochs(eegdata, event_id=stim_event_id, tmin=-0.75, tmax=1.0,
                                    baseline=None, proj=False, picks=None, detrend=1, 
                                    reject_by_annotation=True, preload=True)
    
    ### ----- Identify all muscle-related bad trials
    muscle_dropped_idx_eeg = [
        i for i, reasons in enumerate(eeg_epochs.drop_log)
        if any('muscle' in r.lower() for r in reasons)
    ]
    print(f"Trials dropped due to a muscle-related annotation: {len(muscle_dropped_idx_eeg)}")
    print(muscle_dropped_idx_eeg)

    ### ----- Identify trials marked by the automatic algorithm as BAD_muscle 
    muscle_algo_idx_eeg = [
        i for i, reasons in enumerate(eeg_epochs.drop_log)
        if 'BAD_muscle' in reasons
    ]
    print(f"Trials rejected by the muscle-detection algorithm: {len(muscle_algo_idx_eeg)}")

    # Load ICA solution
    eegicadata      = mne.preprocessing.read_ica(os.path.join(foldereeg, eegoutputfolder, ica_eeg))

    # Channels Removed
    eeg_bad_channels    = len(eegdata.info['bads'])
    # Trials removed
    eeg_trials_removed  = 240 - len(eeg_epochs)
    # ICA components removed
    eeg_ica_removed     = len(eegicadata.exclude)


    # OPM
    # load preprocessed data files
    opmdata         = mne.io.read_raw(os.path.join(folderopm, final_opm), preload=True)
    opm_chans       = opmdata.copy().pick_types(meg=True, exclude=[]).ch_names
    bases           = [ch[:-1] for ch in opm_chans]
    # Count channels with 2 axes
    counts          = Counter(bases)
    n_two_axis      = sum(c == 2 for c in counts.values())
    n_one_axis      = sum(c == 1 for c in counts.values())

    totalopmchans   = n_two_axis
    totalopmtwoaxes = len(opm_chans)

    # check valid trials
    _, event_id_all = mne.events_from_annotations(opmdata)
    stim_event_id   = {k: v for k, v in event_id_all.items() if 'STIMON' in k}
    opm_epochs      = mne.Epochs(opmdata, event_id=stim_event_id, tmin=-0.75, tmax=1.0,
                                baseline=None, proj=False, picks=None, detrend=1, 
                                reject_by_annotation=True, preload=True)

    ### ----- Identify all muscle-related bad trials
    muscle_dropped_idx_opm = [
        i for i, reasons in enumerate(opm_epochs.drop_log)
        if any('muscle' in r.lower() for r in reasons)
    ]
    print(f"Trials dropped due to a muscle-related annotation: {len(muscle_dropped_idx_opm)}")

    ### ----- Identify trials marked by the automatic algorithm as BAD_muscle 
    muscle_algo_idx_opm = [
        i for i, reasons in enumerate(opm_epochs.drop_log)
        if 'BAD_muscle' in reasons
    ]
    print(f"Trials rejected by the muscle-detection algorithm: {len(muscle_algo_idx_opm)}")

    # Load ICA solution
    opmicadata          = mne.preprocessing.read_ica(os.path.join(folderopm, ica_opm))

    # Channels Removed
    opm_bad_channels    = len(opmdata.info['bads'])
    # Trials removed
    opm_trials_removed  = 240 - len(opm_epochs)
    # ICA components removed
    opm_ica_removed     = len(opmicadata.exclude)


    # Get behavioral data
    fpath       = os.path.join(trialinfo_dir, subjectid + '_trial_info.csv')
    df          = pd.read_csv(fpath, sep=';')  # adjust sep if needed
    # Mean RT for CODE==96
    rt_96       = df.loc[df['CODE'] == 96, 'RT'].astype(float).dropna()
    mean_rt_96  = rt_96.mean() if not rt_96.empty else np.nan
    # Accuracy: trials with CODE in {96,120} / total
    acc         = df['CODE'].isin([96, 120]).sum() / len(df) if len(df) > 0 else np.nan

    # Write data
    demog_file.loc[subjectid, 'RT']                         = mean_rt_96
    demog_file.loc[subjectid, 'Acc']                        = acc
    # Preprocessing
    demog_file.loc[subjectid, 'RemovedTrialsEEG']           = eeg_trials_removed
    demog_file.loc[subjectid, 'RemovedTrialsOPM']           = opm_trials_removed
    
    demog_file.loc[subjectid, 'AvailableTrialsEEG']         = 240-eeg_trials_removed
    demog_file.loc[subjectid, 'AvailableTrialsOPM']         = 240-opm_trials_removed

    # Annotate muscle
    demog_file.loc[subjectid, 'AnnotateMuscleTrialsEEG']    = len(muscle_algo_idx_eeg)
    demog_file.loc[subjectid, 'AnnotateMuscleTrialsOPM']    = len(muscle_algo_idx_opm)

    # All muscle trials removed
    demog_file.loc[subjectid, 'AllMuscleTrialsEEG']         = len(muscle_dropped_idx_eeg)
    demog_file.loc[subjectid, 'AllMuscleTrialsOPM']         = len(muscle_dropped_idx_opm)

    demog_file.loc[subjectid, 'RemovedICAEEG']              = eeg_ica_removed
    demog_file.loc[subjectid, 'RemovedICAOPM']              = opm_ica_removed
    
    demog_file.loc[subjectid, 'RemovedChanEEG']             = eeg_bad_channels
    demog_file.loc[subjectid, 'RemovedChanOPM']             = opm_bad_channels

    demog_file.loc[subjectid, 'TotalChanEEG']               = totaleegchans
    demog_file.loc[subjectid, 'TotalChanOPM']               = totalopmchans

    demog_file.loc[subjectid, 'TotalOPMTwoAxes']            = totalopmtwoaxes

    ##### ----- MAD-removed trials
    mad_filename    = 'removed_epochs_per_sensor.csv' 
    eeg_gamma_10_10 = os.path.join(foldereeg, eegoutputfolder, 'spatial_filter_gamma')
    opm_gamma_10_10 = os.path.join(folderopm, 'spatial_filter_gamma')

    eeg_gamma_10_20 = os.path.join(foldereeg, eegoutputfolder, 'spatial_filter_gamma_20')
    opm_gamma_10_20 = os.path.join(folderopm, 'spatial_filter_gamma_20')

    eeg_alpha_10_10 = os.path.join(foldereeg, eegoutputfolder, 'spatial_filter_alpha')
    opm_alpha_10_10 = os.path.join(folderopm, 'spatial_filter_alpha')

    eeg_alpha_10_20 = os.path.join(foldereeg, eegoutputfolder, 'spatial_filter_alpha_20')
    opm_alpha_10_20 = os.path.join(folderopm, 'spatial_filter_alpha_20')

    ##### ----- Write in csv
    ##### EEG
    ##### ----- Gamma
    mad_10_10_eeg   = pd.read_csv(os.path.join(eeg_gamma_10_10, mad_filename))
    demog_file.loc[subjectid, 'MAD_EEG_Gamma_10_10'] = mad_10_10_eeg['total_trials_removed'].median()
    mad_10_20_eeg   = pd.read_csv(os.path.join(eeg_gamma_10_20, mad_filename))
    demog_file.loc[subjectid, 'MAD_EEG_Gamma_10_20'] = mad_10_20_eeg['total_trials_removed'].median()
    ##### ----- Alpha
    mad_10_10_eeg   = pd.read_csv(os.path.join(eeg_alpha_10_10, mad_filename))
    demog_file.loc[subjectid, 'MAD_EEG_Alpha_10_10'] = mad_10_10_eeg['total_trials_removed'].median()
    mad_10_20_eeg   = pd.read_csv(os.path.join(eeg_alpha_10_20, mad_filename))
    demog_file.loc[subjectid, 'MAD_EEG_Alpha_10_20'] = mad_10_20_eeg['total_trials_removed'].median()

    ##### OPM
    ##### ----- Gamma
    mad_10_10_opm   = pd.read_csv(os.path.join(opm_gamma_10_10, mad_filename))
    demog_file.loc[subjectid, 'MAD_OPM_Gamma_10_10'] = mad_10_10_opm['total_trials_removed'].median()
    mad_10_20_opm   = pd.read_csv(os.path.join(opm_gamma_10_20, mad_filename))
    demog_file.loc[subjectid, 'MAD_OPM_Gamma_10_20'] = mad_10_20_opm['total_trials_removed'].median()
    ##### ----- Alpha
    mad_10_10_opm   = pd.read_csv(os.path.join(opm_alpha_10_10, mad_filename))
    demog_file.loc[subjectid, 'MAD_OPM_Alpha_10_10'] = mad_10_10_opm['total_trials_removed'].median()
    mad_10_20_opm   = pd.read_csv(os.path.join(opm_alpha_10_20, mad_filename))
    demog_file.loc[subjectid, 'MAD_OPM_Alpha_10_20'] = mad_10_20_opm['total_trials_removed'].median()


    ##### ----- Check available trials after preprocessing and MAD trial rejection
    gamma_10_10 = mne.read_epochs(os.path.join(foldereeg, eegoutputfolder, 'spatial_filter_gamma', 'sf_epochs_gamma.h5'), preload=True)
    gamma_10_20 = mne.read_epochs(os.path.join(foldereeg, eegoutputfolder, 'spatial_filter_gamma_20', 'sf_epochs_gamma_20.h5'), preload=True)

    alpha_10_10 = mne.read_epochs(os.path.join(foldereeg, eegoutputfolder, 'spatial_filter_alpha', 'sf_epochs_alpha.h5'), preload=True)
    alpha_10_20 = mne.read_epochs(os.path.join(foldereeg, eegoutputfolder, 'spatial_filter_alpha_20', 'sf_epochs_alpha_20.h5'), preload=True)

    ##### ----- Save GED-analyzed trial number
    demog_file.loc[subjectid, 'GED_Gamma_10_10']            = len(gamma_10_10)
    demog_file.loc[subjectid, 'GED_Gamma_10_20']            = len(gamma_10_20)
    demog_file.loc[subjectid, 'GED_Alpha_10_10']            = len(alpha_10_10)   
    demog_file.loc[subjectid, 'GED_Alpha_10_20']            = len(alpha_10_20)


# Save Filled In Demog File
demog_file.to_csv(os.path.join(wdir, 'Results', 'SNR_controls', 'Revision_IN', 'PreprocessingInfo.csv'))

##### Print values
import numpy as np
import pandas as pd
import os 
from utils_visgam.visgam_paths_local import datafolders, wdir, daten

demog_file = pd.read_csv(os.path.join(wdir, 'Results', 'SNR_controls', 'Revision_IN', 'PreprocessingInfo.csv'))
demog_file[demog_file['SZ_EEG_INFO'] == 0]['MAD_EEG_Gamma_10_10'].values
demog_file[demog_file['SZ_EEG_INFO'] == 0]['MAD_OPM_Gamma_10_10'].values

demog_file[demog_file['SZ_EEG_INFO'] == 0]['GED_Gamma_10_10'].values

##### Create table for manuscript
controldata         = demog_file[demog_file['SZ_EEG_INFO'] == 0]
cols_               = ['AvailableTrialsEEG', 'AvailableTrialsOPM', 'RemovedChanEEG', 'RemovedChanOPM', 
         'AnnotateMuscleTrialsEEG', 'AnnotateMuscleTrialsOPM', 'RemovedICAEEG', 'RemovedICAOPM']
preprocessinginfo   = controldata[cols_]

##### Print mean et sd for manuscript
print(preprocessinginfo.mean())
print(preprocessinginfo.std())

##### ----- Print Retained Trials
print('mean EEG',   100*((preprocessinginfo['AvailableTrialsEEG']/240).mean()))
print('std EEG',    100*((preprocessinginfo['AvailableTrialsEEG']/240).std()))

print('mean OPM',   100*((preprocessinginfo['AvailableTrialsOPM']/240).mean()))
print('std OPM',    100*((preprocessinginfo['AvailableTrialsOPM']/240).std()))

##### ----- Print mean/SD of trials actually used in each GED analysis
##### ----- (controls only) - these are the final, matched EEG-OPM trial
##### ----- counts per band/sensor-config, i.e. what's left after all
##### ----- rejections (annotation-based, ICA, and per-sensor MAD).
ged_cols = {
    'Gamma 10_10': 'GED_Gamma_10_10',
    'Gamma 10_20': 'GED_Gamma_10_20',
    'Alpha 10_10': 'GED_Alpha_10_10',
    'Alpha 10_20': 'GED_Alpha_10_20',
}
for label, col in ged_cols.items():
    vals = controldata[col]
    print(f'{label}: Overlapping trials mean ({vals.mean():.2f}) sd ({vals.std():.2f})')


##### ----- Print common trials for ITPC analysis
itpcpath = os.path.join(wdir, 'Results', 'SNR_controls', 'Revision_IN', 'results_itpc', 'itpc 10_10.csv')
itpcdata = pd.read_csv(itpcpath, index_col=0)
itpcdata = itpcdata[itpcdata['SZ_EEG_INFO'] == 0]
print('mean trials',   itpcdata['CommonTrials'].mean())
print('sd trials',   itpcdata['CommonTrials'].std(ddof=1))



##### ----- Print Information About superlets
from utils_visgam.visgam_tfr import superlet_buffer

##### Check Width of Superlets
### ----- Check for Higher Frequencies
for freq in np.arange(25, 105, 5):
    buffer = superlet_buffer(freq, sd_=6, sfreq=500, n_cycles=4, order_start=4, 
                            order_stop=20, foi_start=25, foi_stop=100, foi_step=0.5)

### ----- Check for Lower Frequencies
for freq in np.arange(7, 35, 0.5):
    buffer = superlet_buffer(freq, sd_=3, sfreq=500, n_cycles=3, order_start=1, 
                             order_stop=1, foi_start=7, foi_stop=35, foi_step=0.5)
    