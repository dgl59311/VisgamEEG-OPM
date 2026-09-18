# analysis_2_tfr_sensors_evoked
##### ----- Run Separately for each sensor configuration
##### ----- This Script Also Generates A Subplot for Figure 2_3
##### ----- This Script Also Prints Values To Be Reported 
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
from utils_visgam.visgam_tfr import epochs_subset_by_names_array, tfr_subset_by_names, common_trial_names
from utils_visgam.visgam_funcs import loadcfg
from mne.time_frequency import tfr_morlet

### ----- matplotlib mode
matplotlib.use('QtAgg')  

##### Set Main Analysis
   ### ----- Load Demog File
demog_file      = pd.read_csv(os.path.join(wdir,"DemographicFile.csv"), index_col=0)
### ----- import analysis cfg
cfg             = loadcfg(cfg_dir)
### ----- Obtain config files
cf_itpc         = cfg['itpc']           # Configuration for ITPC
cf_ged          = cfg['ged_itpc']      # Configuration for ITPC

##### Set analysis parameters
sensor_cfg      = '10_10' # 10_20 # Number of OPM/EEG sensors

### ----- Set sensor pool to use for EEG
if sensor_cfg   == '10_10':
    pool_eeg    = pool_eeg_10
elif sensor_cfg == '10_20':
    pool_eeg    = pool_eeg_20   

### ----- number of subjects
n_subjects      = len(datafolders)
print('Total sample: ', n_subjects)

##### Allocate data
eeg_ctrl_itpcs  = []
opmx_ctrl_itpcs, opmy_ctrl_itpcs = [], []
eeg_pat_itpcs   = []
opmx_pat_itpcs, opmy_pat_itpcs  = [], []

##### Select subfolder within Results folder to save results
Results_Folder  = os.path.join('Results', 'SNR_controls', 'Revision_IN')

#%matplotlib qt
for nsubject in range(n_subjects):

    ### ----- Subject ID
    subjid          = datafolders[nsubject]['id']
    ### ----- Determine if is patient
    issz            = demog_file.loc[subjid[8:]]['SZ_EEG_INFO'] 

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
    eegchannels = [ch for ch in pool_eeg if ch in eegdata.ch_names]
    eegdata.pick(eegchannels)

    ### ----- Remove bads from OPM data (no interpolation)
    opmdata         = opmdata.copy().drop_channels(opmdata.info['bads'])
    opmchannels     = [ch for ch in pool_opm_10 if ch in opmdata.ch_names]
    opmdata.pick(opmchannels)
    
    ##### Epoch data
    ### ----- Get stimulus onset events
    eeg_events  = get_stim_event_id(eegdata, event_name='STIMON')
    opm_events  = get_stim_event_id(opmdata, event_name='STIMON')

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
    
    ### ----- Create Average arrays
    info_avg    = mne.create_info(ch_names=['channel_average'], sfreq=morleteeg.sfreq, ch_types='eeg')
    tmpeeg      = np.nanmean(morleteeg.data, axis=0) 
    eegitpc     = mne.time_frequency.AverageTFRArray(
                    info=info_avg, data=tmpeeg[np.newaxis, :, :], times=morleteeg.times, 
                    freqs=morleteeg.freqs, method=morleteeg.method) 
    
    ### ----- OPM Split Sensor Axes
    picks_mag   = mne.pick_types(morletopm.info, meg=True, eeg=False, eog=False, stim=False, misc=False)
    opm_x_chs   = [morletopm.ch_names[i] for i in picks_mag if morletopm.ch_names[i].lower().endswith("x")]
    opm_y_chs   = [morletopm.ch_names[i] for i in picks_mag if morletopm.ch_names[i].lower().endswith("y")]
    itpcs_x     = morletopm.copy().pick(opm_x_chs)
    itpcs_y     = morletopm.copy().pick(opm_y_chs)
    info_avg    = mne.create_info(ch_names=['channel_average'], sfreq=morletopm.sfreq, ch_types='mag')
    tmpopmx     = np.nanmean(itpcs_x.data, axis=0) 
    tmpopmy     = np.nanmean(itpcs_y.data, axis=0) 

    ### ----- Create Average arrays
    opmxitpc    = mne.time_frequency.AverageTFRArray(
        info=info_avg, data=tmpopmx[np.newaxis, :, :], times=morletopm.times, 
        freqs=morletopm.freqs, method=morletopm.method) 
    
    opmyitpc    = mne.time_frequency.AverageTFRArray(
        info=info_avg, data=tmpopmy[np.newaxis, :, :], times=morletopm.times, 
        freqs=morletopm.freqs, method=morletopm.method) 
    
    if not issz:
        eeg_ctrl_itpcs.append(eegitpc)
        opmx_ctrl_itpcs.append(opmxitpc)
        opmy_ctrl_itpcs.append(opmyitpc)
    else:
        eeg_pat_itpcs.append(eegitpc)
        opmx_pat_itpcs.append(opmxitpc)
        opmy_pat_itpcs.append(opmyitpc)


##### Print average data for controls for paper
### ----- Separate high and low frequencies
### ----- High frequencies
higheegvalues   = []
highopmxvalues  = []
highopmyvalues  = []
for i in range(len(eeg_ctrl_itpcs)):
    ##### EEG
    tmpdata = eeg_ctrl_itpcs[i].copy().crop(tmin=cf_itpc['hf_times'][0], tmax=cf_itpc['hf_times'][1], 
                                            fmin=cf_itpc['hf_freqs'][0], fmax=cf_itpc['hf_freqs'][1]).data.mean()
    higheegvalues.append(tmpdata)
    ##### OPM X
    tmpdata = opmx_ctrl_itpcs[i].copy().crop(tmin=cf_itpc['hf_times'][0], tmax=cf_itpc['hf_times'][1], 
                                             fmin=cf_itpc['hf_freqs'][0], fmax=cf_itpc['hf_freqs'][1]).data.mean()
    highopmxvalues.append(tmpdata)
    ##### OPM Y
    tmpdata = opmy_ctrl_itpcs[i].copy().crop(tmin=cf_itpc['hf_times'][0], tmax=cf_itpc['hf_times'][1], 
                                             fmin=cf_itpc['hf_freqs'][0], fmax=cf_itpc['hf_freqs'][1]).data.mean()
    highopmyvalues.append(tmpdata)

eegvalues   = np.array(higheegvalues)
opmxvalues  = np.array(highopmxvalues)
opmyvalues  = np.array(highopmyvalues)

##### Print data for paper
print(f'\n--- High-frequency ITPC ({cf_itpc["hf_freqs"]} Hz, {cf_itpc["hf_times"]} s) ---')

print('Mean Pool EEG: ', eegvalues.mean().round(3))
print('SD Pool EEG: ', eegvalues.std().round(3))

print('Mean Pool x: ', opmxvalues.mean().round(3))
print('SD Pool x: ', opmxvalues.std().round(3))

print('Mean Pool y: ', opmyvalues.mean().round(3))
print('SD Pool y: ', opmyvalues.std().round(3))

### ----- Low frequencies
loweegvalues    = []
lowopmxvalues   = []
lowopmyvalues   = []
for i in range(len(eeg_ctrl_itpcs)):
    ##### EEG
    tmpdata = eeg_ctrl_itpcs[i].copy().crop(tmin=cf_itpc['lf_times'][0], tmax=cf_itpc['lf_times'][1], 
                                            fmin=cf_itpc['lf_freqs'][0], fmax=cf_itpc['lf_freqs'][1]).data.mean()
    loweegvalues.append(tmpdata)
    ##### OPM X
    tmpdata = opmx_ctrl_itpcs[i].copy().crop(tmin=cf_itpc['lf_times'][0], tmax=cf_itpc['lf_times'][1], 
                                             fmin=cf_itpc['lf_freqs'][0], fmax=cf_itpc['lf_freqs'][1]).data.mean()
    lowopmxvalues.append(tmpdata)
    ##### OPM Y
    tmpdata = opmy_ctrl_itpcs[i].copy().crop(tmin=cf_itpc['lf_times'][0], tmax=cf_itpc['lf_times'][1], 
                                             fmin=cf_itpc['lf_freqs'][0], fmax=cf_itpc['lf_freqs'][1]).data.mean()
    lowopmyvalues.append(tmpdata)

eegvalues   = np.array(loweegvalues)
opmxvalues  = np.array(lowopmxvalues)
opmyvalues  = np.array(lowopmyvalues)

##### Print data for paper
print(f'\n--- Low-frequency ITPC ({cf_itpc["lf_freqs"]} Hz, {cf_itpc["lf_times"]} s) ---')
print('Mean Pool EEG: ', eegvalues.mean().round(3))
print('SD Pool EEG: ', eegvalues.std().round(3))

print('Mean Pool x: ', opmxvalues.mean().round(3))
print('SD Pool x: ', opmxvalues.std().round(3))

print('Mean Pool y: ', opmyvalues.mean().round(3))
print('SD Pool y: ', opmyvalues.std().round(3))


##### Get Grand Average data for controls/patients
if sensor_cfg == '10_10':
    ga_eeg  = mne.grand_average(eeg_ctrl_itpcs)
    ga_opmx = mne.grand_average(opmx_ctrl_itpcs)
    ga_opmy = mne.grand_average(opmy_ctrl_itpcs)

    ##### Plot settings
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
    ##### Plot average TFRs
    lims_plot   = (0, 0.5)
    kwargs      = dict(tmin=-0.1, tmax=0.5, vlim=lims_plot, fmin=5, fmax=55, cmap='viridis', colorbar=False, show=False)  
    fig, axes   = plt.subplots(1, 3, figsize=(10/2.54, 3/2.54), sharey=True) 
    ax11, ax21, ax31 = axes[0], axes[1], axes[2]

    ##### Plot average TFRS
    ga_eeg.plot(axes=ax11, **kwargs)
    ga_opmx.plot(axes=ax21, **kwargs)
    ga_opmy.plot(axes=ax31, **kwargs)
    # Set labels
    xticks = [0.0, 0.2, 0.4]
    yticks = [10, 30, 50]
    for ax in (ax11, ax21, ax31):
        ax.set_xlabel('') 
        ax.set_xticks(xticks)

    ax11.set_ylabel('Frequency (Hz)', labelpad=6)  
    ax21.set_xlabel('Time (s)') 
    ax11.set_yticks(yticks)  
    ax21.set_ylabel('')  
    ax31.set_ylabel('')  
    plt.subplots_adjust(left=0.12, right=0.98, top=0.98, bottom=0.3,
                        wspace=0.1, hspace=0)
    #plt.show()
    ##### Save Figure
    fig.savefig(os.path.join(wdir, Results_Folder, 'figures', 'Figure_2_3', 'Average_ITPCs.png'), 
                dpi=300)

    ##### Plot Colorbar
    vmin, vmax = 0, 0.5
    cmap = "viridis"

    ##### Plot Colorbar
    ##### vertical, slim
    fig, ax = plt.subplots(figsize=(1.5/2.54, 3/2.54))
    ax.axis("off")

    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    cbar = fig.colorbar(sm, ax=ax, orientation="vertical", fraction=0.8, pad=0.0)
    cbar.set_label("ITPC")
    cbar.set_ticks([vmin, 0.25, vmax])
    outpath = os.path.join(wdir, Results_Folder, "figures", "Figure_2_3", "Colorbar_ITPC.png")
    fig.savefig(outpath, dpi=300, bbox_inches="tight", transparent=True)
    plt.close(fig)
