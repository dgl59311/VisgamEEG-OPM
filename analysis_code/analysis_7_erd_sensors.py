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

#### ----- matplotlib mode
matplotlib.use('QtAgg')     

### ----- Load Demog File
demog_file      = pd.read_csv(os.path.join(wdir,"DemographicFile.csv"), index_col=0)

### ----- Import analysis cfg
cfg             = loadcfg(cfg_dir)
cf_peaks_alpha  = cfg['gaussians_alpha']  
cf_tfr_alpha    = cfg['tfr_alpha']        # Configuration for TFR

### ----- Print number of subjects
n_subjects      = len(datafolders)
print('Total sample: ', n_subjects)

### ----- Allocate
tfrs_eeg    = []
tfrs_opm_x, tfrs_opm_y  = [], []
subjects_id = []

##### Allocate for topomaps (per-channel spatial values, all sensors)
tfrlist_eeg, tfrlist_opm = [], []

##### Select subfolder within Results folder to save results
Results_Folder  = os.path.join('Results', 'SNR_controls', 'Revision_IN')

### ----- Start Loop
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
    datafolderopm   = datafolders[nsubject]['opm']

    ### ----- Load data
    ### ----- EEG
    eegfolderalpha  = os.path.join(datafoldereeg, 'TFR_alpha_allsensors')
    tfreeg          = mne.time_frequency.read_tfrs(os.path.join(eegfolderalpha, 'AverageTFR_Superlets.h5'))
    # OPM
    opmfolderalpha  = os.path.join(datafolderopm, 'TFR_alpha_allsensors')
    tfropm          = mne.time_frequency.read_tfrs(os.path.join(opmfolderalpha, 'AverageTFR_Superlets.h5'))

    ### ----- Analyze data for controls 
    if not issz:
        ### ----- Append subjects names
        subjects_id.append(subjid)
        ### ----- EEG
        ### ----- Correct Baseline
        bl_eeg_all      = tfreeg.copy().apply_baseline(mode='logratio', baseline=cf_peaks_alpha['baseline'])
        bl_eeg          = bl_eeg_all.copy().pick_channels(pool_eeg_10)
        ### ----- OPM
        bl_opm          = tfropm.copy().apply_baseline(mode='logratio', baseline=cf_peaks_alpha['baseline'])
        bl_opm          = bl_opm.copy() # OPM channels were selected before TFR computation
        opm_chs         = bl_opm.ch_names

        ### ----- Per-channel topodata (all EEG sensors, for topomaps)
        bl_eeg_topodata = bl_eeg_all.copy().crop(tmin=cf_peaks_alpha['time_range'][0], tmax=cf_peaks_alpha['time_range'][1],
                                      fmin=cf_peaks_alpha['freq_range'][0], fmax=cf_peaks_alpha['freq_range'][1]).data.mean(axis=(1, 2))
        eeg_chs         = bl_eeg_all.ch_names
        row_eeg         = {ch: np.nan for ch in pool_eeg_20}
        row_eeg.update({ch: float(val) for ch, val in zip(eeg_chs, bl_eeg_topodata)})
        tfrlist_eeg.append(row_eeg)

        ### ----- Per-channel topodata (OPM, for topomaps)
        bl_opm_topodata = bl_opm.copy().crop(tmin=cf_peaks_alpha['time_range'][0], tmax=cf_peaks_alpha['time_range'][1],
                                      fmin=cf_peaks_alpha['freq_range'][0], fmax=cf_peaks_alpha['freq_range'][1]).data.mean(axis=(1, 2))
        row             = {ch: np.nan for ch in pool_opm_10}
        row.update({ch: float(val) for ch, val in zip(opm_chs, bl_opm_topodata)})
        tfrlist_opm.append(row)

        ### ----- Append Average tfrs
        ### ----- EEG
        info_avg        = mne.create_info(ch_names=['channel_average'], sfreq=bl_eeg.sfreq, ch_types='eeg')
        tmpeeg          = 10*np.nanmean(bl_eeg.data, axis=0) 
        eegtfr          = mne.time_frequency.AverageTFRArray(
                        info=info_avg, data=tmpeeg[np.newaxis, :, :], times=bl_eeg.times, 
                        freqs=bl_eeg.freqs, method=bl_eeg.method) 
        ### ----- Append data
        tfrs_eeg.append(eegtfr)

        ### ----- OPM
        picks_mag   = mne.pick_types(bl_opm.info, meg=True, eeg=False, eog=False, stim=False, misc=False)
        opm_x_chs   = [bl_opm.ch_names[i] for i in picks_mag if bl_opm.ch_names[i].lower().endswith("x")]
        opm_y_chs   = [bl_opm.ch_names[i] for i in picks_mag if bl_opm.ch_names[i].lower().endswith("y")]
        tfr_x       = bl_opm.copy().pick(opm_x_chs)
        tfr_y       = bl_opm.copy().pick(opm_y_chs)
        info_avg    = mne.create_info(ch_names=['channel_average'], sfreq=bl_opm.sfreq, ch_types='mag')
        tmpopmx     = 10*np.nanmean(tfr_x.data, axis=0) 
        tmpopmy     = 10*np.nanmean(tfr_y.data, axis=0) 
        
        ### ----- Create Average Array
        opmxtfr     = mne.time_frequency.AverageTFRArray(
            info=info_avg, data=tmpopmx[np.newaxis, :, :], times=bl_opm.times, 
            freqs=bl_opm.freqs, method=bl_opm.method) 
        
        opmytfr     = mne.time_frequency.AverageTFRArray(
            info=info_avg, data=tmpopmy[np.newaxis, :, :], times=bl_opm.times, 
            freqs=bl_opm.freqs, method=bl_opm.method) 

        ### ----- Append data
        tfrs_opm_x.append(opmxtfr)
        tfrs_opm_y.append(opmytfr)

##### Create DataFrames (for topomaps)
df_opm  = pd.DataFrame(tfrlist_opm, index=subjects_id)
df_eeg  = pd.DataFrame(tfrlist_eeg, index=subjects_id)

##### To plot topomaps, match OPMs to the closest EEG sensor
map_x       = {'5LC':'04lbx', '6L':'06lx', '7L':'07lx', '8L':'08lx', '9L':'09lx',
                '5RC':'04rbx', '6R':'06rx', '7R':'07rx', '8R':'08rx', '9R':'09rx'}
map_y       = {'5LC':'04lby', '6L':'06ly', '7L':'07ly', '8L':'08ly', '9L':'09ly',
                '5RC':'04rby', '6R':'06ry', '7R':'07ry', '8R':'08ry', '9R':'09ry'}
eeg_order   = ['5LC','6L','7L','8L','9L','5RC','6R','7R','8R','9R']

df_opm_x    = pd.DataFrame({eeg: df_opm[opm] for eeg, opm in map_x.items()})[eeg_order]
df_opm_y    = pd.DataFrame({eeg: df_opm[opm] for eeg, opm in map_y.items()})[eeg_order]

##### Print control data for paper
eegvalues   = []
opmxvalues  = []
opmyvalues  = []

for i in range(len(subjects_id)):
    tmpdata = tfrs_eeg[i].copy().crop(tmin=cf_peaks_alpha['time_range'][0], tmax=cf_peaks_alpha['time_range'][1], 
                                      fmin=cf_peaks_alpha['freq_range'][0], fmax=cf_peaks_alpha['freq_range'][1]).data.mean()
    eegvalues.append(tmpdata)
    tmpdata = tfrs_opm_x[i].copy().crop(tmin=cf_peaks_alpha['time_range'][0], tmax=cf_peaks_alpha['time_range'][1], 
                                        fmin=cf_peaks_alpha['freq_range'][0], fmax=cf_peaks_alpha['freq_range'][1]).data.mean()
    opmxvalues.append(tmpdata)
    tmpdata = tfrs_opm_y[i].copy().crop(tmin=cf_peaks_alpha['time_range'][0], tmax=cf_peaks_alpha['time_range'][1], 
                                        fmin=cf_peaks_alpha['freq_range'][0], fmax=cf_peaks_alpha['freq_range'][1]).data.mean()
    opmyvalues.append(tmpdata)

##### ----- Make arrays and print data for paper
eegvalues   = np.array(eegvalues)
opmxvalues  = np.array(opmxvalues)
opmyvalues  = np.array(opmyvalues)

##### Print data for paper
print('Mean Pool EEG: ', eegvalues.mean().round(3))
print('SD Pool EEG: ', eegvalues.std().round(3))

print('Mean Pool x: ', opmxvalues.mean().round(3))
print('SD Pool x: ', opmxvalues.std().round(3))

print('Mean Pool y: ', opmyvalues.mean().round(3))
print('SD Pool y: ', opmyvalues.std().round(3))

##### Get Grand Averages for plotting
ga_eeg  = mne.grand_average(tfrs_eeg)
ga_opmx = mne.grand_average(tfrs_opm_x)
ga_opmy = mne.grand_average(tfrs_opm_y)

##### Configuration plots
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

##### Plot average TFRs for ERD
lims_plot   = 3
kwargs      = dict(tmin=-0.1, tmax=0.9, vlim=(-lims_plot, lims_plot), fmin=7, fmax=35, colorbar=False, show=False)  
fig, axes   = plt.subplots(1, 3, figsize=(10/2.54, 3/2.54), sharey=True) 
ax11, ax21, ax31 = axes[0], axes[1], axes[2]

##### Plot average TFRS
ga_eeg.plot(axes=ax11, **kwargs)
ga_opmx.plot(axes=ax21, **kwargs)
ga_opmy.plot(axes=ax31, **kwargs)

##### Set labels
xticks = [0.0, 0.4, 0.8]
yticks = [10, 20, 30]
for ax in (ax11, ax21, ax31):
    ax.set_xlabel('') 
    ax.set_xticks(xticks)

ax11.set_ylabel('Frequency (Hz)', labelpad=6)  
ax21.set_xlabel('Time (s)') 
ax11.set_yticks(yticks)  
ax21.set_ylabel('')  
ax31.set_ylabel('')  
#plt.show()
plt.subplots_adjust(left=0.12, right=0.98, top=0.98, bottom=0.3,
                    wspace=0.1, hspace=0)

# Change this to create the titles
#ax11.set_title("EEG", pad=18)
#ax21.set_title("OPM X-axis", pad=18)
#ax31.set_title("OPM Y-axis", pad=18)
#plt.subplots_adjust(left=0.12, right=0.98, top=0.7, bottom=0.3,
#                    wspace=0.1, hspace=0)

##### Save Figure
fig.savefig(os.path.join(wdir, Results_Folder, 'figures', 'Figure_2_3', 'Average_TFRs.png'), 
            dpi=300)

##### Plot Colorbar
vmin, vmax = -3, 3
cmap = "RdBu_r"

##### vertical, slim
fig, ax = plt.subplots(figsize=(1.5/2.54, 3/2.54))
ax.axis("off")

norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])

cbar = fig.colorbar(sm, ax=ax, orientation="vertical", fraction=0.8, pad=0.0)
cbar.set_label("Change (dB)")
cbar.set_ticks([vmin, 0, vmax])
outpath = os.path.join(wdir, Results_Folder, "figures", "Figure_2_3", "Colorbar_TFR_alpha.png")
fig.savefig(outpath, dpi=300, bbox_inches="tight", transparent=True)
plt.close(fig)


##### ===== Topomaps (same logic as analysis_6, for alpha) =====
kwargs_topo = dict(cmap="RdBu_r", vlim=(-3, 3), extrapolate="local",
                border=0, sphere=(0, -0.035, 0, 0.1), outlines="head", mask=np.ones(10, dtype=bool),
                mask_params = dict(marker='o', markerfacecolor='w', markeredgecolor='k',
                linewidth=0, markersize=1.5), ch_type="eeg", show=False)
w_cm, h_cm  = 3.33, 3.33

##### Generate EEG topography (pool_eeg_10)
eeg_info = bl_eeg_all.info
eeg_data = 10*np.nanmean(df_eeg[pool_eeg_10], axis=0)
picks    = mne.pick_channels(eeg_info.ch_names, include=pool_eeg_10)
info_sub = mne.pick_info(eeg_info.copy(), picks)

fig, ax = plt.subplots(figsize=(w_cm/2.54, h_cm/2.54))
im, cn = mne.viz.plot_topomap(
    eeg_data, info_sub, axes=ax, **kwargs_topo
)
ax.text(0.5, 0, "EEG", transform=ax.transAxes,
        ha="center", va="top")
fig.savefig(
    os.path.join(wdir, Results_Folder, "figures", "Figure_2_3", "TopomapEEG_alpha.png"),
    dpi=300,
    bbox_inches="tight", transparent=True)
plt.close(fig)


##### Generate EEG topography with all sensors (not just the pool_eeg_10 subset)
all_eeg_chs     = [ch for ch in df_eeg.columns if ch in eeg_info.ch_names]
eeg_data_all    = 10*np.nanmean(df_eeg[all_eeg_chs], axis=0)
picks_all       = mne.pick_channels(eeg_info.ch_names, include=all_eeg_chs)
info_sub_all    = mne.pick_info(eeg_info.copy(), picks_all)

kwargs_topo_all                 = dict(kwargs_topo)
kwargs_topo_all['sensors']      = False
kwargs_topo_all['mask']         = None
kwargs_topo_all['extrapolate']  = 'local'

fig, ax = plt.subplots(figsize=(w_cm/2.54, h_cm/2.54))
im, cn = mne.viz.plot_topomap(
    eeg_data_all, info_sub_all, axes=ax, **kwargs_topo_all
)

##### Lock the view limits (see analysis_6 for why)
xlim_fixed  = ax.get_xlim()
ylim_fixed  = ax.get_ylim()

from mne.viz.topomap import _get_pos_outlines
pos_2d, _   = _get_pos_outlines(info_sub_all, picks=None, sphere=kwargs_topo['sphere'])
is_pool     = np.array([ch in pool_eeg_10 for ch in all_eeg_chs])

mparams     = kwargs_topo['mask_params']
ax.plot(pos_2d[~is_pool, 0], pos_2d[~is_pool, 1], linestyle='None', alpha=0.3,
        zorder=3, **mparams)
ax.plot(pos_2d[is_pool, 0], pos_2d[is_pool, 1], linestyle='None',
        zorder=4, **mparams)

ax.set_xlim(xlim_fixed)
ax.set_ylim(ylim_fixed)

ax.text(0.5, 0, "EEG", transform=ax.transAxes,
        ha="center", va="top")
fig.savefig(
    os.path.join(wdir, Results_Folder, "figures", "Figure_2_3", "TopomapEEG_alpha_allsensors.png"),
    dpi=300,
    bbox_inches="tight", transparent=True)
plt.close(fig)


##### Plot Topomap OPM
opm_x_topo  = 10*df_opm_x.mean(axis=0, skipna=True).to_numpy()
opm_y_topo  = 10*df_opm_y.mean(axis=0, skipna=True).to_numpy()

mont        = eeg_info.get_montage()
ch_pos_all  = mont.get_positions()['ch_pos']
pos         = np.array([ch_pos_all[ch] for ch in pool_eeg_10])

dx              = 0.01
pos_new         = pos.copy()
pos_new[:, 0]   += np.where(pos[:, 0] < 0, +dx, -dx)

mont_new = mne.channels.make_dig_montage(
    ch_pos={ch: p for ch, p in zip(pool_eeg_10, pos_new)},
    coord_frame="head"
)
info_sub.set_montage(mont_new)

for opmax in ['x', 'y']:
    if opmax == 'x':
        opmtopodata = opm_x_topo
        titletext   = 'OPM Z-axis'
        figname     = 'Topomap_OPM_alpha_z.png'
    else:
        opmtopodata = opm_y_topo
        titletext   = 'OPM Y-axis'
        figname     = 'Topomap_OPM_alpha_y.png'

    fig, ax = plt.subplots(figsize=(w_cm/2.54, h_cm/2.54))
    ax.clear()

    im, cn = mne.viz.plot_topomap(opmtopodata, info_sub, axes=ax, **kwargs_topo)
    ax.text(0.5, 0, titletext, transform=ax.transAxes,
            ha="center", va="top")
    outpath = os.path.join(wdir, Results_Folder, "figures", "Figure_2_3", figname)
    fig.savefig(outpath, dpi=300, bbox_inches="tight", transparent=True)
    plt.close(fig)


##### Plot topomap colorbar
vmin, vmax  = -3, 3
cmap        = "RdBu_r"
w_cm, h_cm = 3, 3.0
fig, ax = plt.subplots(figsize=(w_cm/2.54, h_cm/2.54))
ax.axis("off")
norm        = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
sm          = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])

cbar        = fig.colorbar(sm, ax=ax, fraction=1.0, pad=0.0, orientation='horizontal')
cbar.set_label("Change (dB)")
cbar.set_ticks([vmin, 0, vmax])

outpath = os.path.join(wdir, Results_Folder, "figures", "Figure_2_3", "Topomap_colorbar_alpha.png")
fig.savefig(outpath, dpi=300, bbox_inches="tight", transparent=True)
plt.close(fig)

