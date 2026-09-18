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
from utils_visgam.visgam_tfr import epochs_subset_by_names_array, tfr_subset_by_names, common_trial_names
from utils_visgam.visgam_funcs import mean_sem, loadcfg, align_polarity, mean_sd, bootstrap_ci, align_modalities_by_corr
from scipy.stats import zscore
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.colors as mcolors

### ----- matplotlib mode
matplotlib.use('QtAgg')     

##### Results folder
Results_Folder  = os.path.join('Results', 'SNR_controls', 'Revision_IN')

### ----- load Demog File and results from Gaussian Fitting
demog_file  = pd.read_csv(os.path.join(wdir,"DemographicFile.csv"), index_col=0)
gamma_peaks = pd.read_csv(os.path.join(wdir, Results_Folder, 'results_gamma', 'gamma_peaks 10_10.csv'), index_col=0)

### ----- import analysis cfg
cfg             = loadcfg(cfg_dir) 
cf_ged          = cfg['ged_gamma']      
cf_peaks_gamma  = cfg['gaussians_gamma']   
cf_ged_dir      = cfg['ged_gamma_folders_10_10']     

### ----- number of subjects
n_subjects  = len(datafolders)
print('Total sample: ', n_subjects)

### ----- Concatenate virtual channel data
vc_eeg, vc_opm = [], []
### ----- Concatenate weights of GED
all_evecs_opm, all_evecs_eeg    = [], []
best_ged_eeg, best_ged_opm      = [], []
### ----- Concatenate subject id
ids = [] 
### ----- Concatenate TFR data
eeg_ctrl_nbl, opm_ctrl_nbl = [], []

#%matplotlib qt
for nsubject in range(n_subjects):

    ### ----- Subject ID
    subjid          = datafolders[nsubject]['id']
    ### ----- Determine if is patient
    issz            = demog_file.loc[subjid[8:]]['SZ_EEG_INFO'] 

    ### ----- Analyze data for controls only
    if not issz:
        
        ### ----- Define datafolders
        datafoldereeg   = datafolders[nsubject]['eeg']
        ##### Re-direct to preprocessed folder
        if Results_Folder == os.path.join('Results', 'SNR_controls', 'Revision_IN'):
            datafoldereeg   = os.path.join(datafoldereeg, 'revision')
        eeg_sf          = os.path.join(datafoldereeg, cf_ged_dir['results_folder'], 
                                       cf_ged_dir['epoch_filename']) 

        datafolderopm   = datafolders[nsubject]['opm']
        opm_sf          = os.path.join(datafolderopm, cf_ged_dir['results_folder'], 
                                       cf_ged_dir['epoch_filename']) 

        ### ----- Load epochs array from Spatial Filter
        eegsf           = mne.read_epochs(eeg_sf, preload=True)
        opmsf           = mne.read_epochs(opm_sf, preload=True)  

        ### ----- Interpolate EEG data and re-reference eeg to the average
        ### ----- Concatenate virtual channel data
        vc_eeg.append(eegsf.filter(l_freq=None, h_freq=None).average().data)
        vc_opm.append(opmsf.filter(l_freq=None, h_freq=None).average().data)

        ### ----- Load eigenvectors 
        evecs_eeg   = np.load(os.path.join(datafoldereeg, cf_ged_dir['results_folder'], cf_ged_dir['evecs_filename'])) 
        evecs_opm   = np.load(os.path.join(datafolderopm, cf_ged_dir['results_folder'], cf_ged_dir['evecs_filename'])) 

        ### ----- Load channels
        eegchans    = np.load(os.path.join(datafoldereeg, cf_ged_dir['results_folder'], 'eegsensors_sf.npy')).tolist()
        opmchans    = np.load(os.path.join(datafolderopm, cf_ged_dir['results_folder'], 'opmsensors_sf.npy')).tolist()

        ### ----- Load which eigencomponent was selected for the comparison
        best_ged    = gamma_peaks.loc[subjid[8:]]
        ged_eeg     = int(best_ged['ged_eeg'][-1]) - 1
        ged_opm     = int(best_ged['ged_opm'][-1]) - 1
        best_ged_eeg.append(ged_eeg)
        best_ged_opm.append(ged_opm)

        ### ----- Pool sensors
        row = {ch: np.nan for ch in pool_opm_10}
        row.update({ch: float(val) for ch, val in zip(opmchans, evecs_opm[:, ged_opm])})
        all_evecs_opm.append(row)
        del row

        ### ----- Pool EEG electrodes
        row = {ch: np.nan for ch in pool_eeg_10}
        row.update({ch: float(val) for ch, val in zip(eegchans, evecs_eeg[:, ged_eeg])})
        all_evecs_eeg.append(row)

        ### ----- Append subject names
        ids.append(subjid)

        ### ----- Load one full EEG dataset and get info (first control subject
        ### ----- actually processed, regardless of its position in datafolders)
        if len(ids) == 1:
            eegfile     = os.path.join(datafoldereeg, 'preprocessed_3_ica_eeg.fif')
            eegdata     = mne.io.read_raw(eegfile, preload=True)
            alleeginfo  = eegdata.info
            del eegdata

        ### ----- Load TFR data
        ### ----- EEG - reuse the already revision-redirected datafoldereeg
        ### ----- (do NOT rebuild it from datafolders[nsubject]['eeg'] here,
        ### ----- that would silently drop the redirect)
        eeg_tfr         = os.path.join(datafoldereeg, cf_ged_dir['results_folder'], cf_ged_dir['tfr_filename'])
        ### ----- OPM
        datafolderopm   = os.path.join(datafolders[nsubject]['opm'], cf_ged_dir['results_folder'])
        opm_tfr         = os.path.join(datafolderopm, cf_ged_dir['tfr_filename'])
        ### ----- Load TFRs
        tfreeg          = mne.time_frequency.read_tfrs(eeg_tfr)
        tfropm          = mne.time_frequency.read_tfrs(opm_tfr) 
        
        ### ----- Get Data for Grand Average Plots
        info_avg_eeg = mne.create_info(ch_names=['channel_average'], sfreq=tfreeg.sfreq, ch_types='eeg')      
        info_avg_opm = mne.create_info(ch_names=['channel_average'], sfreq=tfropm.sfreq, ch_types='mag')     
        ### ----- Superlet without baseline corrections
        ### ----- EEG
        tmptfreeg   = tfreeg.data[:, ged_eeg, :, :].mean(axis=0)
        eeg_avg_nbl = mne.time_frequency.AverageTFRArray(
            info=info_avg_eeg, data=tmptfreeg[np.newaxis, :, :], times=tfreeg.times, 
            freqs=tfreeg.freqs, method=tfreeg.method) 
        ### ----- OPM
        tmptfropm   = tfropm.data[:, ged_opm, :, :].mean(axis=0)
        opm_avg_nbl = mne.time_frequency.AverageTFRArray(
            info=info_avg_opm, data=tmptfropm[np.newaxis, :, :], times=tfropm.times, 
            freqs=tfropm.freqs, method=tfropm.method) 
        
        ##### Concatenate data across subjects
        eeg_ctrl_nbl.append(eeg_avg_nbl)
        opm_ctrl_nbl.append(opm_avg_nbl)


### ----- Plot PSD for Virtual Channels 
from matplotlib.lines import Line2D      
from scipy.ndimage import gaussian_filter1d

# Plot without  baseline correction for visualization
pre_handle  = Line2D([0], [0], color='k', linestyle='--', linewidth=1, label='Pre-stimulus')
post_handle = Line2D([0], [0], color='k', linestyle='-',  linewidth=1, label='Post-stimulus')

gamma_low, gamma_high   = 25, 90
scale_eeg, scale_opm    = 1, 1
eegpre, eegpost         = [], []
opmpre, opmpost         = [], []

base                            = cf_peaks_gamma['baseline']
tmin_sustained, tmax_sustained  = cf_peaks_gamma['time_range']

### ----- Same conversion as fit_peaks_tfr (analysis_3): gaussian_filter1d's
### ----- sigma is in bins, so a true 1Hz kernel needs smooth_sigma_hz / freq_step,
### ----- not a bare sigma=1 (which the original code used, giving only a
### ----- 0.5Hz-equivalent width on this axis's 0.5Hz-per-bin frequency grid).
smooth_sigma_hz      = cf_peaks_gamma['freq_smooth']
freq_step            = 0.5
smooth_sigma_bins    = smooth_sigma_hz / freq_step

for i in range(len(eeg_ctrl_nbl)):
    ##### EEG
    tmpeeg_pre  = eeg_ctrl_nbl[i].copy().crop(tmin=base[0], tmax=base[1], fmin=gamma_low, fmax=gamma_high)
    tmpeeg_pre  = np.mean(tmpeeg_pre.data * scale_eeg, 2)[0]
    ### ----- Smooth with a gaussian kernel at 1Hz as in the peak fitting procedure
    tmpeeg_pre  = gaussian_filter1d(tmpeeg_pre, sigma=smooth_sigma_bins)

    eegpre.append(tmpeeg_pre)
    tmpeeg_post = eeg_ctrl_nbl[i].copy().crop(tmin=tmin_sustained, tmax=tmax_sustained, fmin=gamma_low, fmax=gamma_high)
    tmpeeg_post = np.mean(tmpeeg_post.data * scale_eeg, 2)[0]
    tmpeeg_post = gaussian_filter1d(tmpeeg_post, sigma=smooth_sigma_bins)
    eegpost.append(tmpeeg_post)

    # OPM
    tmpopm_pre  = opm_ctrl_nbl[i].copy().crop(tmin=base[0], tmax=base[1], fmin=gamma_low, fmax=gamma_high)
    tmpopm_pre  = np.mean(tmpopm_pre.data * scale_opm, 2)[0]
    tmpopm_pre  = gaussian_filter1d(tmpopm_pre, sigma=smooth_sigma_bins)
    opmpre.append(tmpopm_pre)
    tmpopm_post = opm_ctrl_nbl[i].copy().crop(tmin=tmin_sustained, tmax=tmax_sustained, fmin=gamma_low, fmax=gamma_high)
    tmpopm_post = np.mean(tmpopm_post.data * scale_opm, 2)[0]
    tmpopm_post  = gaussian_filter1d(tmpopm_post, sigma=smooth_sigma_bins)
    opmpost.append(tmpopm_post)

eegpre, eegpost         = np.array(eegpre), np.array(eegpost)
opmpre, opmpost         = np.array(opmpre), np.array(opmpost)

### ----- Configuration for Plotting
cm_to_inch          = 1 / 2.54
width_cm, height_cm = 7, 3.5
font_size           = 8
x                   = np.arange(gamma_low, gamma_high + 0.5, 0.5)
plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': font_size,             # base font size
    'axes.titlesize': font_size,        # title
    'axes.labelsize': font_size,        # x/y labels
    'xtick.labelsize': font_size,       # x-tick labels
    'ytick.labelsize': font_size,       # y-tick labels
    'legend.fontsize': font_size,       # legend text
})

# Loop to create two separate figures
for sensor_type in ['EEG', 'OPM']:
    # 1. Setup data and styling
    if sensor_type == 'EEG':
        predata, postdata = eegpre, eegpost
        ylabel = r'Power (a.u)'
        color = 'slategray'
        fname = 'EEG_TFR_Power'
        y_lims = [0.2, 1] # Adjust based on your data range
    else:
        predata, postdata = opmpre, opmpost
        ylabel = r'Power (a.u)'
        color = 'crimson'
        fname = 'OPM_TFR_Power'
        y_lims = [0.2, 1]

    # 2. Calculate Mean and SD
    m_pre, sd_pre       = mean_sem(predata)
    m_post, sd_post     = mean_sem(postdata)

    # Fix for Log Scale: Ensure lower bounds never hit <= 0
    prel = np.maximum(m_pre - sd_pre, 1e-6)
    preu = m_pre + sd_pre
    postl = np.maximum(m_post - sd_post, 1e-6)
    postu = m_post + sd_post

    # 3. Create Figure
    fig = plt.figure(figsize=(width_cm * cm_to_inch, height_cm * cm_to_inch))
    # Manual axes to ensure labels aren't cropped: [left, bottom, width, height]
    ax = fig.add_axes([0.18, 0.28, 0.78, 0.65])

    # Plot Baseline (Dotted)
    ax.plot(x, m_pre, color=color, linestyle=':', linewidth=1, label='Baseline', alpha=0.7)
    ax.fill_between(x, prel, preu, color=color, alpha=0.4, edgecolor='none')

    # Plot Post-Stimulus (Solid)
    ax.plot(x, m_post, color=color, linestyle='-', linewidth=1, label='Post-Stimulus')
    ax.fill_between(x, postl, postu, color=color, alpha=0.7, edgecolor='none')

    # 4. Formatting
    #ax.set_yscale('log')
    #ax.set_xlim(5, 90)
    ax.set_ylim(y_lims)
    #ax.set_yscale('log')

    ax.set_xlabel("Frequency (Hz)", fontsize=font_size)
    ax.set_ylabel(ylabel, fontsize=font_size)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=font_size - 1)

    # Legend at top right
    ax.legend(loc='upper right', fontsize=font_size - 1, frameon=False, handlelength=1.5)

    # 5. Save and Show
    save_path = os.path.join(wdir, 'Results', 'SNR_controls', 'Revision_IN', 'figures', 'Figure_4_5', f'{fname}.png')
    fig.savefig(save_path, dpi=300, bbox_inches=None, transparent=True)
    plt.show()

### ----- Plot contributing sensors
### ----- Create DataFrames
df_opm      = pd.DataFrame(all_evecs_opm, index=ids)  
df_eeg      = pd.DataFrame(all_evecs_eeg, index=ids)  

### ----- Rank data
ranks_eeg   = df_eeg.abs().rank(axis=1, ascending=True, method="average")
ranks_opm   = df_opm.abs().rank(axis=1, ascending=True, method="average")

### ----- Normalize dynamically
### ----- We subtract the minimum rank (usually 1) and divide by the range (max - min)
eeg_rel_rank = (ranks_eeg.sub(ranks_eeg.min(axis=1), axis=0)).div(
    ranks_eeg.max(axis=1) - ranks_eeg.min(axis=1), axis=0
)

opm_rel_rank = (ranks_opm.sub(ranks_opm.min(axis=1), axis=0)).div(
    ranks_opm.max(axis=1) - ranks_opm.min(axis=1), axis=0
)

### ----- Define data
eeg_data    = np.nanmean(eeg_rel_rank[pool_eeg_10], axis=0)
### ----- Divide axes/set equivalent OPM positions compared to EEG sensors
map_x       = {'5LC':'04lbx', '6L':'06lx', '7L':'07lx', '8L':'08lx', '9L':'09lx',
                '5RC':'04rbx', '6R':'06rx', '7R':'07rx', '8R':'08rx', '9R':'09rx'}

map_y       = {'5LC':'04lby', '6L':'06ly', '7L':'07ly', '8L':'08ly', '9L':'09ly',
                '5RC':'04rby', '6R':'06ry', '7R':'07ry', '8R':'08ry', '9R':'09ry'}

eeg_order   = ['5LC','6L','7L','8L','9L','5RC','6R','7R','8R','9R']

df_opm_x    = pd.DataFrame({eeg: opm_rel_rank[opm] for eeg, opm in map_x.items()})[eeg_order]
df_opm_y    = pd.DataFrame({eeg: opm_rel_rank[opm] for eeg, opm in map_y.items()})[eeg_order]

opm_x_topo  = np.nanmean(df_opm_x, axis=0)
opm_y_topo  = np.nanmean(df_opm_y, axis=0)


### ----- Configuration plots
font_size = 8

### ----- Generate EEG topography
eeg_info = alleeginfo
### ----- average across subjects
picks    = mne.pick_channels(eeg_info.ch_names, include=pool_eeg_10)
info_sub = mne.pick_info(eeg_info.copy(), picks)   
### ----- --- Configuration ---
w_cm, h_cm = 2.5, 2.5 # Slightly smaller to pull them in
font_size = 8

plot_configs = [
    {'data': eeg_data,   'title': '',        'is_opm': False},
    {'data': opm_x_topo, 'title': 'Z-axis',    'is_opm': True},
    {'data': opm_y_topo, 'title': 'Y-axis',    'is_opm': True}
]

### ----- Total width adjusted to pull them closer ( * w_cm + small margin for colorbar)
fig, axes = plt.subplots(1, 3, figsize=(3*w_cm/2.54, h_cm/2.54))

### ----- Manual spacing control: wspace < 0 pulls them closer together
plt.subplots_adjust(left=0.05, right=0.85, wspace=0.1) 

for i, config in enumerate(plot_configs):
    ax = axes[i]
    temp_info = info_sub.copy()
    
    if config['is_opm']:
        mont = temp_info.get_montage()
        ch_pos_all = mont.get_positions()['ch_pos']
        pos = np.array([ch_pos_all[ch] for ch in pool_eeg_10])
        dx = 0.01
        pos_new = pos.copy()
        pos_new[:, 0] += np.where(pos[:, 0] < 0, +dx, -dx)
        mont_new = mne.channels.make_dig_montage(
            ch_pos={ch: p for ch, p in zip(pool_eeg_10, pos_new)},
            coord_frame="head"
        )
        temp_info.set_montage(mont_new)

    mne.viz.plot_sensors(temp_info, axes=ax, kind='topomap', 
                            sphere=(0, -0.035, 0, 0.1), show=False)
        
    ### ----- Remove head patches (optional, based on your previous preference)
    for patch in ax.patches:
        patch.set_visible(False)

    scatter = ax.collections[0]
    scatter.set_sizes([8])
    scatter.set_array(config['data'])
    scatter.set_cmap("turbo")
    scatter.set_clim(0.1, 0.9)
    scatter.set_linewidths(0.2)
    scatter.set_edgecolors('black')
    ax.set_title(config['title'], fontsize=font_size, pad=2)
    ax.set_axis_off()

fig.savefig(os.path.join(wdir, "Results", "SNR_controls", "Revision_IN", "figures", "Figure_4_5", "Heads_Only.png"),
            dpi=300, transparent=True, bbox_inches='tight')
#plt.show()


### ----- --- Generate the Horizontal Colorbar ---
### ----- Width is now larger than height (e.g., 2cm wide, 0.4cm tall)
fig_cbar, ax_cbar = plt.subplots(figsize=(1.8 / 2.54, 0.2 / 2.54)) 

### ----- Create the dummy mappable
sm = plt.cm.ScalarMappable(cmap="turbo", norm=plt.Normalize(vmin=0, vmax=1))

### ----- Set orientation to 'horizontal'
cbar = fig_cbar.colorbar(sm, cax=ax_cbar, orientation='horizontal')

### ----- Tick styling
cbar.set_ticks([0.1, 0.9])
cbar.set_ticklabels(['Low', 'High'])
cbar.ax.tick_params(size=0, labelsize=font_size - 2, pad=3)

### ----- Label styling - placed on top or bottom
### ----- For horizontal, 'label' usually looks best with a small pad
cbar.set_label('Contribution', fontsize=font_size, labelpad=3)

### ----- Match your plot's line style
cbar.outline.set_linewidth(0.2)

### ----- Save with bbox_inches='tight' to crop the empty figure space around it
save_path_cbar = os.path.join(wdir, "Results", "SNR_controls", "Revision_IN", "figures", "Figure_4_5", "Standalone_Colorbar_Horizontal.png")
fig_cbar.savefig(save_path_cbar, dpi=300, transparent=True, bbox_inches='tight')

#plt.show()

##### Plot time series for virtual channels
### ----- Time vector
times   =  eegsf.times
### ----- Plot Grand Average TFRs
font_size = 7
plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': font_size,             # base font size
    'axes.titlesize': font_size,        # title
    'axes.labelsize': font_size,        # x/y labels
    'xtick.labelsize': font_size,       # x-tick labels
    'ytick.labelsize': font_size,       # y-tick labels
    'legend.fontsize': font_size,       # legend text
})

### ----- Plot virtual channel data
### ----- Filter for the best GED component across subjects
best_vc_eeg = np.array([vc_eeg[i][best_ged_eeg[i], :] for i in range(len(vc_eeg))])
best_vc_opm = np.array([vc_opm[i][best_ged_opm[i], :] for i in range(len(vc_opm))])

plt.plot(times, best_vc_opm.std(axis=0))
plt.plot(times, best_vc_eeg.std(axis=0))
plt.show()

### ----- Plot virtual channels for both modalities
##### Match the sign for modalities in the window used for filter optimization
vc_eeg_arr, vc_opm_arr, _ = align_modalities_by_corr(
    best_vc_eeg, best_vc_opm, times, 0.15, 0.75, flip="eeg", desired_sign=1)  

### ----- Align across subjects
vc_eeg_arr = align_polarity(vc_eeg_arr, times, 0.15, 0.75, desired_sign=1, method='prc') 
vc_opm_arr = align_polarity(vc_opm_arr, times, 0.15, 0.75, desired_sign=1, method='prc') 


### ----- Get mean et sem
vc_eeg_arr              = np.array(vc_eeg_arr[0]) 
vc_opm_arr              = np.array(vc_opm_arr[0]) 
avg_vceeg, sem_vceeg    = mean_sem(vc_eeg_arr) 
avg_vcopm, sem_vcopm    = mean_sem(vc_opm_arr) 

##### Determine figure size
cm_to_inch              = 1 / 2.54
width_cm, height_cm     = 5, 3

##### Plot for EEG and OPM separately
for i in range(2):
    fig = plt.figure(figsize=(width_cm * cm_to_inch, height_cm * cm_to_inch))
    
    ###### Define the plot area manually: [left, bottom, width, height]
    # These values are percentages of the figure size (0 to 1)
    # 0.28 leaves enough room for "zscore" on the left
    # 0.22 leaves enough room for "Time (s)" on the bottom
    ax = fig.add_axes([0.28, 0.3, 0.65, 0.70]) 
    
    if i == 0:
        datavc, semvc = avg_vceeg, sem_vceeg
        leg_, color, fname = 'EEG', 'slategray', 'vceeg'
        bbox_to_anchor = (0.5, 0.35)
    else:
        datavc, semvc = avg_vcopm, sem_vcopm
        leg_, color, fname = 'OPM-MEG', 'crimson', 'vcopm'
        bbox_to_anchor = (0.36, 0.35)

    ax.plot(times, datavc, color=color, linewidth=1.5)
    ax.fill_between(times, (datavc - semvc), (datavc + semvc), color=color, alpha=0.3)
    
    ##### Simple formatting
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("zscore")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlim([-0.1, 0.75])
    ax.set_ylim([-3.1, 2.2])
    ax.set_xticks([0.0, 0.2, 0.4, 0.6]) 
    ax.set_yticks([-3, -1.5, 0, 1.5])
    
    ax.axvline(0.0, ls='--', lw=1, color='k', alpha=0.5, zorder=0)
    ax.axhline(0.0, ls='--', lw=1, color='k', alpha=0.5, zorder=0)
    
    ax.legend([leg_], handlelength=1, bbox_to_anchor=bbox_to_anchor, 
              fontsize=7, frameon=False)

    ##### Save Figure
    save_path = os.path.join(wdir, 'Results', 'SNR_controls', 'Revision_IN', 'figures', 'Figure_4_5', fname + '.png')
    fig.savefig(save_path, 
                dpi=300, 
                bbox_inches=None, 
                transparent=True)
    
    #plt.show() # Show it after saving so it doesn't mess with the file layout
    plt.close(fig)


##### Plot overlapping traces in one figure
cm_to_inch          = 1 / 2.54
width_cm, height_cm = 7, 3

### ----- ---- One figure, both traces
fig = plt.figure(figsize=(width_cm * cm_to_inch, height_cm * cm_to_inch))
ax = fig.add_axes([0.28, 0.3, 0.65, 0.70])

### ----- EEG
ax.plot(times, avg_vceeg, color='slategray', linewidth=1, label='EEG')
ax.fill_between(times, avg_vceeg - sem_vceeg, avg_vceeg + sem_vceeg,
                color='slategray', alpha=0.25)

### ----- OPM
ax.plot(times, avg_vcopm, color='crimson', linewidth=1, label='OPM-MEG', alpha=0.75)
ax.fill_between(times, avg_vcopm - sem_vcopm, avg_vcopm + sem_vcopm,
                color='crimson', alpha=0.25)

### ----- Formatting 
ax.set_xlabel("Time (s)")
ax.set_ylabel("zscore")
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xlim([-0.1, 0.75])
ax.set_ylim([-3.1, 2.2])
ax.set_xticks([0.0, 0.2, 0.4, 0.6]) 
ax.set_yticks([-3, -1.5, 0, 1.5])
ax.axvline(0.0, ls='--', lw=1, color='k', alpha=0.5, zorder=0)
ax.axhline(0.0, ls='--', lw=1, color='k', alpha=0.5, zorder=0)

ax.legend(handlelength=1, fontsize=7, frameon=False, loc='lower right')

### ----- Save
save_path = os.path.join(wdir, 'Results', 'SNR_controls', 'Revision_IN', 'figures', 'Figure_4_5', 'vceeg_opm.png')
fig.savefig(save_path, dpi=300, bbox_inches=None, transparent=True)

#plt.show()
plt.close(fig)





