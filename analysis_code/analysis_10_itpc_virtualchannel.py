import os
import sys
import json
import mne
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mne.time_frequency import tfr_morlet
#from utils_visgam.visgam_paths_vk import datafolders, cfg_dir, wdir
from utils_visgam.visgam_paths_local import datafolders, local_folder, wdir, cfg_dir
from utils_visgam.visgam_sf import pool_eeg_10, pool_opm_10, get_stim_event_id
from utils_visgam.visgam_tfr import epochs_subset_by_names_array, tfr_subset_by_names, common_trial_names
from utils_visgam.visgam_funcs import mean_sem, loadcfg, align_polarity, mean_sd, bootstrap_ci
from scipy.stats import zscore
from utils_visgam.visgam_sf import pairwise_component_match_with_plots

##### Analysis configurations
sensor_configs      = ['10_10', '10_20']
sensor_cfg          = sensor_configs[0]  # Set to '10_10' or '10_20' as needed

##### Select subfolder within Results folder to save results
Results_Folder      = os.path.join('Results', 'SNR_controls', 'Revision_IN')

### ----- Results filename
results_filename    = 'itpc ' + sensor_cfg + '.csv'

print(f"\n{'='*60}")
print(f"Starting ITPC analysis with {sensor_cfg} sensor configuration.")
print(f"{'='*60}\n")    

##### Set directory for logs
logs_dir    = os.path.join(wdir, 'Results',  'analysis_logs', 'analysis_itpc_' + sensor_cfg)
os.makedirs(logs_dir, exist_ok=True)    
logfile     = open(os.path.join(logs_dir, 'match_ged_analysis_itpc_' + sensor_cfg + '.log'), 'w')
old_stdout  = sys.stdout
sys.stdout  = logfile

### ----- load Demog File
demog_file  = pd.read_csv(os.path.join(wdir,"DemographicFile.csv"), index_col=0)

### ----- import analysis cfg
cfg         = loadcfg(cfg_dir)
cf_itpc     = cfg['itpc']           # Configuration for ITPC
cf_ged      = cfg['ged_itpc']      # Configuration for ITPC
cf_ged_dir  = cfg['ged_itpc_folders_' + sensor_cfg]      # Configuration for ITPC

### ----- number of subjects
n_subjects  = len(datafolders)
print('Total sample: ', n_subjects)

### ----- Allocate data to concatenate
higheeg_itpc, highopm_itpc = [], []
loweeg_itpc, lowopm_itpc = [], []
loweegbest, lowopmbest = [], []
higheegbest, highopmbest = [], []

### ----- number of subjects
n_subjects  = len(datafolders)
print('Total sample: ', n_subjects)

### ----- Start loop
for nsubject in range(n_subjects):

    ### ----- Current dataset   
    print('Current Subject:', datafolders[nsubject]['id'])

    ### ----- Subject ID
    subjid          = datafolders[nsubject]['id']
    id_demog        = subjid[8:]
    issz            = demog_file.loc[id_demog]['SZ_EEG_INFO']

    ### ----- Define datafolders
    ### ----- EEG
    ##### Re-direct to preprocessed folder
    datafoldereeg_base = datafolders[nsubject]['eeg']
    if Results_Folder == os.path.join('Results', 'SNR_controls', 'Revision_IN'):
        datafoldereeg_base = os.path.join(datafoldereeg_base, 'revision')
    datafoldereeg   = os.path.join(datafoldereeg_base, cf_ged_dir['results_folder'])
    eeg_tfr         = os.path.join(datafoldereeg, cf_ged_dir['filename'])
    eeg_evals       = np.load(os.path.join(datafoldereeg, cf_ged_dir['evals_filename'])) 
    ### ----- OPM   
    datafolderopm   = os.path.join(datafolders[nsubject]['opm'], cf_ged_dir['results_folder'])
    opm_tfr         = os.path.join(datafolderopm, cf_ged_dir['filename']) 
    opm_evals       = np.load(os.path.join(datafolderopm, cf_ged_dir['evals_filename'])) 

    ### ----- Analyze data for controls 
    if not issz:
        
        ### ----- Load data/TFR
        itpceeg         = mne.time_frequency.read_tfrs(eeg_tfr)
        itpcopm         = mne.time_frequency.read_tfrs(opm_tfr)

        ### ----- Read Evoked Data 
        itpceveeg       = mne.read_evokeds(os.path.join(datafoldereeg, cf_ged_dir['filename_evoked']) )
        itpcevopm       = mne.read_evokeds(os.path.join(datafolderopm, cf_ged_dir['filename_evoked']) )
        
        ### ----- Get number of trials
        trialseeg       = itpceveeg[0].nave
        trialsopm       = itpcevopm[0].nave

        ### ----- ITPC component matching always considers the top 3 GED
        ### ----- components (unlike gamma/alpha, no eigenvalue>1 gate is
        ### ----- applied here - see note below on why that threshold doesn't
        ### ----- transfer to ged_filter_evoked's evoked-vs-single-trial ratio).
        n_components_eeg = 3
        n_components_opm = 3

        ### ----- Match components
        ### ----- Low frequency ITPC
        out_lf = pairwise_component_match_with_plots(avg_eeg=itpceeg, avg_opm=itpcopm,
            fmin=cf_itpc['lf_freqs'][0], fmax=cf_itpc['lf_freqs'][1],
            tmin=cf_itpc['lf_times'][0], tmax=cf_itpc['lf_times'][1],
            eeg_names=['ITPC_1', 'ITPC_2', 'ITPC_3'],
            opm_names=['ITPC_1', 'ITPC_2', 'ITPC_3'],
            n_eeg=n_components_eeg, n_opm=n_components_opm,
            #plot_overviews=True, plot_best_match=True, show_corr=True, prefer_first_if_r00_ge=0.5)
            plot_overviews=False, plot_best_match=False, show_corr=False, prefer_first_if_r00_ge=0.5)

        ### ----- High Frequency ITPC
        out_hf = pairwise_component_match_with_plots(avg_eeg=itpceeg, avg_opm=itpcopm,
            fmin=cf_itpc['hf_freqs'][0], fmax=cf_itpc['hf_freqs'][1],
            tmin=cf_itpc['hf_times'][0], tmax=cf_itpc['hf_times'][1],
            eeg_names=['ITPC_1', 'ITPC_2', 'ITPC_3'],
            opm_names=['ITPC_1', 'ITPC_2', 'ITPC_3'],
            n_eeg=n_components_eeg, n_opm=n_components_opm,
            #plot_overviews=True, plot_best_match=True, show_corr=True, prefer_first_if_r00_ge=0.5)
            plot_overviews=False, plot_best_match=False, show_corr=False, prefer_first_if_r00_ge=0.5)
        
        ##### Select best GED component
        ### ----- Low ITPC
        out_lf["best_eeg_name"], out_lf["best_opm_name"], out_lf["best_r"]
        lf_eeg     = itpceeg.copy().pick(out_lf["best_eeg_name"])
        lf_opm     = itpcopm.copy().pick(out_lf["best_opm_name"])
        ### ----- High ITPC
        out_hf["best_eeg_name"], out_hf["best_opm_name"], out_hf["best_r"]
        hf_eeg     = itpceeg.copy().pick(out_hf["best_eeg_name"])
        hf_opm     = itpcopm.copy().pick(out_hf["best_opm_name"])

        ### ----- Store Results
        ##### Low Band ROI (7-20 Hz)
        tmplfeeg = lf_eeg.copy().crop(tmin=cf_itpc['lf_times'][0], tmax=cf_itpc['lf_times'][1], 
                                      fmin=cf_itpc['lf_freqs'][0], fmax=cf_itpc['lf_freqs'][1])
        tmplfopm = lf_opm.copy().crop(tmin=cf_itpc['lf_times'][0], tmax=cf_itpc['lf_times'][1], 
                                      fmin=cf_itpc['lf_freqs'][0], fmax=cf_itpc['lf_freqs'][1])
        demog_file.loc[id_demog, 'itpc_low_eeg']    = tmplfeeg.data.mean()
        demog_file.loc[id_demog, 'GED_low_eeg']     = out_lf["best_eeg_name"]
        demog_file.loc[id_demog, 'itpc_low_opm']    = tmplfopm.data.mean()
        demog_file.loc[id_demog, 'GED_low_opm']     = out_lf["best_opm_name"]
        
        # High Band ROI (25-55 Hz)
        tmphfeeg = hf_eeg.copy().crop(tmin=cf_itpc['hf_times'][0], tmax=cf_itpc['hf_times'][1], 
                                      fmin=cf_itpc['hf_freqs'][0], fmax=cf_itpc['hf_freqs'][1])
        tmphfopm = hf_opm.copy().crop(tmin=cf_itpc['hf_times'][0], tmax=cf_itpc['hf_times'][1], 
                                      fmin=cf_itpc['hf_freqs'][0], fmax=cf_itpc['hf_freqs'][1])
        demog_file.loc[id_demog, 'itpc_high_eeg']   = tmphfeeg.data.mean()
        demog_file.loc[id_demog, 'GED_high_eeg']    = out_hf["best_eeg_name"]
        demog_file.loc[id_demog, 'itpc_high_opm']   = tmphfopm.data.mean()
        demog_file.loc[id_demog, 'GED_high_opm']    = out_hf["best_opm_name"]
        # Common Trials
        demog_file.loc[id_demog, 'EEGTrials'] = trialseeg
        demog_file.loc[id_demog, 'OPMTrials'] = trialsopm
        demog_file.loc[id_demog, 'CommonTrials'] = trialseeg

        ### -----Concatenate Data for Grand Averages
        info_avg_eeg = mne.create_info(ch_names=['channel_average'], sfreq=itpceeg.sfreq, ch_types='eeg')      
        info_avg_opm = mne.create_info(ch_names=['channel_average'], sfreq=itpcopm.sfreq, ch_types='mag')   

        ### ----- High ITPC
        ### ----- EEG
        high_eeg    = mne.time_frequency.AverageTFRArray(
            info=info_avg_eeg, data=hf_eeg.pick(out_hf["best_eeg_name"]).data, times=hf_eeg.times, 
            freqs=hf_eeg.freqs, method=hf_eeg.method) 
        ### ----- OPM
        high_opm = mne.time_frequency.AverageTFRArray(
            info=info_avg_opm, data=hf_opm.pick(out_hf["best_opm_name"]).data, times=hf_opm.times, 
            freqs=hf_opm.freqs, method=hf_opm.method) 

        ### ----- Low ITPC
        ### ----- EEG
        low_eeg    = mne.time_frequency.AverageTFRArray(
            info=info_avg_eeg, data=lf_eeg.pick(out_lf["best_eeg_name"]).data, times=lf_eeg.times, 
            freqs=lf_eeg.freqs, method=lf_eeg.method) 
        ### ----- OPM
        low_opm = mne.time_frequency.AverageTFRArray(
            info=info_avg_opm, data=lf_opm.pick(out_lf["best_opm_name"]).data, times=lf_opm.times, 
            freqs=lf_opm.freqs, method=lf_opm.method) 

        ### ----- Store TFRs
        higheeg_itpc.append(high_eeg)
        highopm_itpc.append(high_opm)

        loweeg_itpc.append(low_eeg)
        lowopm_itpc.append(low_opm)


##### Store CSV
results_itpc = os.path.join(wdir, Results_Folder, 'results_itpc')
os.makedirs(results_itpc, exist_ok=True)

demog_file.to_csv(os.path.join(results_itpc, results_filename))

##### Combine power spectra for plotting
def combine_tfr_low_high(tfr_low, tfr_high, f_split_low_max=22.5, f_split_high_min=23.0):
    """
    Combine two AverageTFR objects by frequency:
      - take freqs <= f_split_low_max from tfr_low
      - take freqs >= f_split_high_min from tfr_high
    Returns a new AverageTFRArray.
    """

    # --- sanity checks (minimal but important)
    if not np.allclose(tfr_low.times, tfr_high.times):
        raise ValueError("Times do not match between low and high TFR.")
    if tfr_low.ch_names != tfr_high.ch_names:
        raise ValueError("Channel lists do not match between low and high TFR.")

    # pick frequency indices
    f_low = np.asarray(tfr_low.freqs)
    f_high = np.asarray(tfr_high.freqs)

    idx_low = np.where(f_low <= f_split_low_max)[0]
    idx_high = np.where(f_high >= f_split_high_min)[0]

    if idx_low.size == 0:
        raise ValueError("No low frequencies found for the requested low range.")
    if idx_high.size == 0:
        raise ValueError("No high frequencies found for the requested high range.")

    # slice and concatenate (data: ch x freq x time)
    data_low = np.asarray(tfr_low.data)[:, idx_low, :]
    data_high = np.asarray(tfr_high.data)[:, idx_high, :]

    freqs_combined = np.concatenate([f_low[idx_low], f_high[idx_high]])
    data_combined = np.concatenate([data_low, data_high], axis=1)

    # build new AverageTFR
    tfr_combined = mne.time_frequency.AverageTFRArray(
        info=tfr_low.info.copy(),
        data=data_combined,
        times=np.asarray(tfr_low.times),
        freqs=freqs_combined,
        nave=getattr(tfr_low, "nave", None),
        method=getattr(tfr_low, "method", "unknown"),
    )
    return tfr_combined

### ----- Prepare ITPC spectra for plotting
##### Calculate Grand Average
GA_ctrl_eeg_high    = mne.grand_average(higheeg_itpc)
GA_ctrl_opm_high    = mne.grand_average(highopm_itpc)
GA_ctrl_eeg_low     = mne.grand_average(loweeg_itpc)
GA_ctrl_opm_low     = mne.grand_average(lowopm_itpc)

### ----- Combine Power Spectra
GA_ctrl_eeg         = combine_tfr_low_high(GA_ctrl_eeg_low, GA_ctrl_eeg_high)
GA_ctrl_opm         = combine_tfr_low_high(GA_ctrl_opm_low, GA_ctrl_opm_high)

##### Plot Grand Average TFRs
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

kwargs      = dict(tmin=-0.1, tmax=0.5, vlim=(0, 0.6), cmap='viridis', colorbar=False, show=False)  
fig, axes   = plt.subplots(2, 1, figsize=(9/2.54, 8.3/2.54), sharex=True) 
ax11        = axes[0]
ax21        = axes[1]

##### Draw TFRs
GA_ctrl_eeg.plot(axes=ax11, **kwargs)
GA_ctrl_opm.plot(axes=ax21, **kwargs)

##### Per-axes labels 
for ax in (ax11, ax21):
    ax.set_ylabel('Frequency (Hz)', labelpad=6)  
    ax.axvline(0.0, ls='-', lw=1, color='k', alpha=0.5)
    ax.axhline(22.5, ls='-', lw=2, color='white', alpha=0.7)
    ax.set_yticks([10, 30, 50])
    ax.set_yticklabels(['10', '30', '50'])

ax11.set_xlabel('')  
ax21.set_xlabel('Time (s)', labelpad=2)  
##### Add the text in data coordinates
ax11.text(-0.2, 63, 'EEG',
        ha='center', va='center', 
        fontsize=8, fontweight='bold')

ax21.text(-0.2, 63, 'OPM-MEG',
          ha='center', va='center',
          fontsize=9, fontweight='bold')

##### Increase spacing so labels don't overlap 
plt.subplots_adjust(left=0.23, right=0.8, top=0.93, bottom=0.125,
                    wspace=0, hspace=0.17)
##### Shared colorbar (right side) 
mappable    = ax11.images[0] if ax11.images else ax11.collections[0]
cax         = fig.add_axes([0.82, 0.33, 0.025, 0.40])  # [left, bottom, width, height]
cb          = fig.colorbar(mappable, cax=cax)
cb.set_label('ITPC')
cb.ax.yaxis.set_label_position('right')  # keep label visible (inside the cbar)
cb.ax.tick_params()
##### Set x-axis ticks in steps of 0.2 from -0.2 to 1.0
xticks = np.arange(-0.0, 0.5, 0.2)
ax21.set_xticks(xticks)
ax21.set_xticklabels([f'{x:.1f}' for x in xticks])


##### Draw rectangle marking ROIs
from matplotlib.patches import Rectangle
x0, x1 = 0.05, 0.2
y0, y1 = 25, 55

for ax in (ax11, ax21):
    rect = Rectangle(
        (x0, y0),            # bottom-left corner
        x1 - x0, y1 - y0,     # width, height
        fill=False, lw=1.5, ec='darkgrey'  # outline only
    )
    ax.add_patch(rect)

##### Draw rectangle
x0, x1 = 0.05, 0.4
y0, y1 = 7, 20

for ax in (ax11, ax21):
    rect = Rectangle(
        (x0, y0),            # bottom-left corner
        x1 - x0, y1 - y0,     # width, height
        fill=False, lw=1.5, ec='darkgrey'  # outline only
    )
    ax.add_patch(rect)


##### Save Figure
fig.savefig(os.path.join(wdir, Results_Folder, 'figures', 'Figure_7', 'Grand_Average_ITPC.png'), 
            dpi=300)

plt.show()

##### Plot single subject data
def plot_all_subjects_tfr(tfr_list, title="Subject ITPC"):
    # Create a figure with 5 rows and 6 columns
    fig, axes = plt.subplots(5, 6, figsize=(20, 15), sharex=True, sharey=True)
    axes = axes.flatten()
    
    for i in range(len(axes)):
        ax = axes[i]
        if i < len(tfr_list):
            # Plot the TFR onto the specific axis
            # picks=[0] because we already extracted the "Best_Comp"
            tfr_list[i].plot(picks=[0], axes=ax, colorbar=False, show=False)
            
            # Label with subject index or name
            ax.set_title(f"Sub {i+1}", fontsize=10)
            
            # Clean up labels to avoid clutter
            if i % 6 != 0: ax.set_ylabel('') # Only show Y label on leftmost plots
            if i < 24: ax.set_xlabel('')    # Only show X label on bottom plots
        else:
            # Hide empty subplots (the last 4)
            ax.axis('off')

    fig.suptitle(title, fontsize=20, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

# Usage:
plot_all_subjects_tfr(higheeg_itpc, title="EEG ITPC - All Subjects")
plot_all_subjects_tfr(highopm_itpc, title="OPM-MEG ITPC - All Subjects")

