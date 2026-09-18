# analysis_2_tfr_sensors_evoked
import os
import json
import mne
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mne.time_frequency import tfr_morlet
#from utils_visgam.visgam_paths_vk import datafolders, local_folder, wdir, cfg_dir
from utils_visgam.visgam_paths_local import datafolders, local_folder, wdir, cfg_dir
from utils_visgam.visgam_sf import pool_eeg_10, pool_eeg_20, pool_opm_10, get_stim_event_id
from utils_visgam.aslt_64 import aslt
from utils_visgam.visgam_tfr import epochs_subset_by_names_array, tfr_subset_by_names, common_trial_names
from utils_visgam.visgam_funcs import mean_sem, loadcfg, bootstrap_ci, mean_sd
from scipy.stats import zscore

### ----- matplotlib mode
matplotlib.use('QtAgg')     

### ----- Load Demog File
demog_file      = pd.read_csv(os.path.join(wdir,"DemographicFile.csv"), index_col=0)

### ----- Import analysis cfg
cfg             = loadcfg(cfg_dir) 
cf_ged          = cfg['ged_gamma']     
cf_peaks_gamma  = cfg['gaussians_gamma']   

### ----- Print Number of subjects
n_subjects      = len(datafolders)
print('Total sample: ', n_subjects)

### ----- Allocate data
evoked_eeg      = []
evoked_opm_x, evoked_opm_y    = [], []

### ----- Concatenate ids
ids             = []
eeg_traces      = []

##### Select subfolder within Results folder to save results
Results_Folder  = os.path.join('Results', 'SNR_controls', 'Revision_IN')
#%matplotlib qt
for nsubject in range(n_subjects):

    ##### Subject ID
    subjid          = datafolders[nsubject]['id']
    ##### Determine if is patient
    issz            = demog_file.loc[subjid[8:]]['SZ_EEG_INFO'] 

    ##### Analyze data for controls 
    if not issz:

        ### ----- Define datafolders
        ##### EEG
        ##### Re-direct to preprocessed folder
        datafoldereeg   = datafolders[nsubject]['eeg']
        if Results_Folder == os.path.join('Results', 'SNR_controls', 'Revision_IN'):
            datafoldereeg   = os.path.join(datafoldereeg, 'revision')
        eegfile         = os.path.join(datafoldereeg, 'preprocessed_3_ica_eeg.fif')
        ##### OPM
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
        eegchannels = [ch for ch in pool_eeg_10 if ch in eegdata.ch_names]
        eegdata.pick(eegchannels)

        ##### Remove bads from OPM data (no interpolation)
        opmdata         = opmdata.copy().drop_channels(opmdata.info['bads'])
        opmchannels     = [ch for ch in pool_opm_10 if ch in opmdata.ch_names]
        opmdata.pick(opmchannels)
        
        ##### Epoch data
        ### ----- Get stimulus onset events
        eeg_events  = get_stim_event_id(eegdata, event_name='STIMON')
        opm_events  = get_stim_event_id(opmdata, event_name='STIMON')

        ##### EEG
        eegdata = eegdata.copy().filter(l_freq=5, h_freq=None)
        X_eeg   = mne.Epochs(eegdata, event_id=eeg_events, tmin=cf_ged['tmin'], tmax=cf_ged['tmax'],
            baseline=None, proj=False, detrend=1, reject_by_annotation=True, preload=True)
        
        ##### OPM
        opmdata = opmdata.copy().filter(l_freq=5, h_freq=None)
        X_opm   = mne.Epochs(opmdata, event_id=opm_events, tmin=cf_ged['tmin'], tmax=cf_ged['tmax'],
            baseline=None, proj=False, detrend=1, reject_by_annotation=True, preload=True)

        ##### Get only the common trials after preprocessing
        common_names    = common_trial_names(X_eeg, X_opm)
        X_eeg           = epochs_subset_by_names_array(X_eeg, common_names)
        X_opm           = epochs_subset_by_names_array(X_opm, common_names)

        ##### Correct baseline and average across trials
        ### ----- EEG
        avg_eeg         = X_eeg.copy().apply_baseline(baseline=cf_peaks_gamma['baseline']).average() 
        ### ----- OPM
        avg_opm         = X_opm.copy().apply_baseline(baseline=cf_peaks_gamma['baseline']).average() 

        ##### Separate axes for OPM
        picks_mag       = mne.pick_types(avg_opm.info, meg=True, eeg=False, eog=False, stim=False, misc=False)
        opm_x_chs       = [avg_opm.ch_names[i] for i in picks_mag if avg_opm.ch_names[i].lower().endswith("x")]
        opm_y_chs       = [avg_opm.ch_names[i] for i in picks_mag if avg_opm.ch_names[i].lower().endswith("y")]
        avg_x           = avg_opm.copy().pick(opm_x_chs)
        avg_y           = avg_opm.copy().pick(opm_y_chs)

        ##### Compute GFP: Take STD across sensors
        evoked_eeg.append(avg_eeg.get_data().std(axis=0)) 
        evoked_opm_x.append(avg_x.get_data().std(axis=0)) 
        evoked_opm_y.append(avg_y.get_data().std(axis=0)) 

        ##### Compute EEG traces
        eeg_traces.append(avg_eeg.data.mean(0))

        ids.append(subjid)

### ----- Evoked EEG
### ----- Transform data to microvolts (original units in V) 
### ----- Transform OPM data to fT (original units in Tesla)
eeg_ev  = np.array(evoked_eeg) * 1e6 
opmx_ev = np.array(evoked_opm_x) * 1e15 
opmy_ev = np.array(evoked_opm_y) * 1e15

##### Time vector
times   = avg_eeg.times 

##### Calculate Mean and SEM across subjects
sem = True
if sem:
    avg_eeg_gfp, sem_eeg_gfp            = mean_sem(eeg_ev) 
    lerror_eeg_gfp, uerror_eeg_gfp      = sem_eeg_gfp, sem_eeg_gfp

    avg_opmx_gfp, sem_opmx_gfp          = mean_sem(opmx_ev) 
    lerror_opmx_gfp, uerror_opmx_gfp    = sem_opmx_gfp, sem_opmx_gfp

    avg_opmy_gfp, sem_opmy_gfp          = mean_sem(opmy_ev) 
    lerror_opmy_gfp, uerror_opmy_gfp    = sem_opmy_gfp, sem_opmy_gfp
else:
    avg_eeg_gfp, lerror_eeg_gfp, uerror_eeg_gfp     = bootstrap_ci(eeg_ev)
    avg_opmx_gfp, lerror_opmx_gfp, uerror_opmx_gfp  = bootstrap_ci(opmx_ev) 
    avg_opmy_gfp, lerror_opmy_gfp, uerror_opmy_gfp  = bootstrap_ci(opmy_ev) 
    

##### Get Max Value Per Subject in the Time Window of Interest
time_window = (0, 0.55)
eeg_max     = np.array([eeg_ev[i][(times >= time_window[0]) & (times <= time_window[1])].max() for i in range(len(eeg_ev))])
opmx_max    = np.array([opmx_ev[i][(times >= time_window[0]) & (times <= time_window[1])].max() for i in range(len(opmx_ev))])
opmy_max    = np.array([opmy_ev[i][(times >= time_window[0]) & (times <= time_window[1])].max() for i in range(len(opmy_ev))])

##### Get Time Point where the value is maximum in the time window of interest
eeg_peak_time   = np.array([times[(times >= time_window[0]) & (times <= time_window[1])][eeg_ev[i][(times >= time_window[0]) & (times <= time_window[1])].argmax()] for i in range(len(eeg_ev))])
opmx_peak_time  = np.array([times[(times >= time_window[0]) & (times <= time_window[1])][opmx_ev[i][(times >= time_window[0]) & (times <= time_window[1])].argmax()] for i in range(len(opmx_ev))])
opmy_peak_time  = np.array([times[(times >= time_window[0]) & (times <= time_window[1])][opmy_ev[i][(times >= time_window[0]) & (times <= time_window[1])].argmax()] for i in range(len(opmy_ev))])

##### Print Mean and SD of the time points where the peaks are located
print('Mean Peak Time EEG: ', eeg_peak_time.mean().round(3))
print('SD Peak Time EEG: ', eeg_peak_time.std().round(3))
print('Mean Peak Time OPM X: ', opmx_peak_time.mean().round(3))
print('SD Peak Time OPM X: ', opmx_peak_time.std().round(3))
print('Mean Peak Time OPM Y: ', opmy_peak_time.mean().round(3))
print('SD Peak Time OPM Y: ', opmy_peak_time.std().round(3))


##### Get Plot Configurations
font_size = 8
plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': font_size,             # base font size
    'axes.titlesize': font_size,        # title
    'axes.labelsize': font_size,        # x/y labels
    'xtick.labelsize': font_size,       # x-tick labels
    'ytick.labelsize': font_size,       # y-tick labels
    'legend.fontsize': font_size,       # legend text
})

##### Create Figure
cm_to_inch = 1 / 2.54
fig = plt.figure(figsize=(9.1 * cm_to_inch, 2.8 * cm_to_inch))
ax1 = plt.subplot(1, 2, 1)
ax2 = plt.subplot(1, 2, 2)

##### EEG
ax1.plot(times, avg_eeg_gfp , color='slategray', linewidth=1.5)
ax1.fill_between(times, (avg_eeg_gfp - lerror_eeg_gfp), 
                 (avg_eeg_gfp + uerror_eeg_gfp), color='slategray', alpha=0.3)

##### OPM Plot
ax2.plot(times, avg_opmx_gfp , color='crimson', linewidth=1.5)
ax2.plot(times, avg_opmy_gfp , color="#008080"
, linewidth=1.5)

### ----- OPM X-axis
ax2.fill_between(times, (avg_opmx_gfp - lerror_opmx_gfp), 
                 (avg_opmx_gfp + uerror_opmx_gfp), color='crimson', alpha=0.2)
### ----- OPM Y-axis
ax2.fill_between(times, (avg_opmy_gfp - lerror_opmy_gfp), 
                 (avg_opmy_gfp + uerror_opmy_gfp), color="#008080", alpha=0.2)
##### Set Labels
for ax in (ax1, ax2):
    ax.set_xlabel("Time (s)")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlim([-0.25, 0.5])
    ax.set_xticks([-0.2, 0.0, 0.2, 0.4])
    ax.axvline(0.0, ls='--', lw=1, color='k', alpha=0.5)

ax1.set_ylabel("GFP (µV)")
ax2.set_ylabel("GFP (fT)")

##### Configure Limits
ax1.set_ylim([0, 2])
ax2.set_ylim([0, 200])

##### Add legend
ax1.legend(['EEG'], handlelength=1, bbox_to_anchor=(0.5, 1.6))
ax2.legend(['Z-axis', 'Y-axis'], ncol=2, handlelength=1.0, 
           loc='upper center', columnspacing=0.4, bbox_to_anchor=(0.5, 1.6))

plt.subplots_adjust(left=0.1, right=0.98, top=0.78, bottom=0.35,
                    wspace=0.5, hspace=0)
plt.show()
fig.savefig(os.path.join(wdir, Results_Folder, 'figures', 'Figure_2_3', 'EEG_OPM_GFP.png'), 
            dpi=300)
