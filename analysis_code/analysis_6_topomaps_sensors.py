##### Script to generate average sensor level values and Figures 
##### ----- This Scripts Also Analyzes Aperiodic Slopes 
import os
import json
import mne
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from utils_visgam.visgam_paths_local import datafolders, local_folder, wdir, cfg_dir
from utils_visgam.visgam_sf import pool_eeg_10, pool_eeg_20, pool_opm_10, get_stim_event_id
from utils_visgam.visgam_tfr import epochs_subset_by_names_array, tfr_subset_by_names, common_trial_names
from utils_visgam.visgam_funcs import loadcfg

### ----- matplotlib mode
matplotlib.use('QtAgg')     

### -----Load Demog File
demog_file      = pd.read_csv(os.path.join(wdir, "DemographicFile.csv"), index_col=0)

### ----- Import analysis cfg
cfg             = loadcfg(cfg_dir)
cf_peaks_gamma  = cfg['gaussians_gamma']   
cf_peaks_alpha  = cfg['gaussians_alpha']  
cf_tfr_gamma    = cfg['tfr_gamma']        # Configuration for TFR
cf_tfr_alpha    = cfg['tfr_alpha']        # Configuration for TFR

### ----- Print number of subjects
n_subjects  = len(datafolders)
print('Total sample: ', n_subjects)

### ----- Allocate lists
tfrlist_eeg, tfrlist_opm = [], []
tfrs_eeg, tfrs_opm_x, tfrs_opm_y  = [], [], []

### ----- Allocate lists for the grand-average frequency-domain "change in
### ----- power" spectra (EEG + OPM overlaid, per band) added at the end of
### ----- this script
spec_eeg_gamma, spec_opmx_gamma, spec_opmy_gamma = [], [], []
spec_eeg_alpha, spec_opmx_alpha, spec_opmy_alpha = [], [], []
freqs_gamma, freqs_alpha = None, None

##### Select subfolder within Results folder to save results
Results_Folder  = os.path.join('Results', 'SNR_controls', 'Revision_IN')

### ----- Concatenate Subjects IDs
subjects_id = []
for nsubject in range(n_subjects):

    ### ----- Subject ID
    subjid          = datafolders[nsubject]['id']
    ### ----- Determine if is patient
    issz            = demog_file.loc[subjid[8:]]['SZ_EEG_INFO'] 
    ### ----- Define datafolders
    datafoldereeg   = datafolders[nsubject]['eeg']
    if Results_Folder == os.path.join('Results', 'SNR_controls', 'Revision_IN'):
        datafoldereeg   = os.path.join(datafoldereeg, 'revision') 

    datafolderopm   = datafolders[nsubject]['opm']

    ### ----- Load TFR data
    ### ----- EEG
    eegfoldergamma  = os.path.join(datafoldereeg, 'TFR_gamma_allsensors')
    tfreeg          = mne.time_frequency.read_tfrs(os.path.join(eegfoldergamma, 'AverageTFR_Superlets.h5'))
    ### ----- OPM
    opmfoldergamma  = os.path.join(datafolderopm, 'TFR_gamma_allsensors')
    tfropm          = mne.time_frequency.read_tfrs(os.path.join(opmfoldergamma, 'AverageTFR_Superlets.h5'))

    ### ----- Load Alpha TFR data too (needed for the frequency-domain
    ### ----- "change in power" spectrum panel added at the end of this script)
    ### ----- EEG
    eegfolderalpha  = os.path.join(datafoldereeg, 'TFR_alpha_allsensors')
    tfreeg_alpha    = mne.time_frequency.read_tfrs(os.path.join(eegfolderalpha, 'AverageTFR_Superlets.h5'))
    ### ----- OPM
    opmfolderalpha  = os.path.join(datafolderopm, 'TFR_alpha_allsensors')
    tfropm_alpha    = mne.time_frequency.read_tfrs(os.path.join(opmfolderalpha, 'AverageTFR_Superlets.h5'))

    ### ----- Load an EEG dataset and get info
    if nsubject     == 0:
        eegfile     = os.path.join(datafoldereeg, 'preprocessed_3_ica_eeg.fif') 
        eegdata     = mne.io.read_raw(eegfile, preload=True)
        alleeginfo  = eegdata.info
        del eegdata 

    ### ----- Analyze data for controls only
    if not issz:
        ### ----- EEG
        ### ----- Correct Baseline
        bl_eeg          = tfreeg.copy().apply_baseline(mode='logratio', baseline=cf_peaks_gamma['baseline'])
        bl_eeg_topodata = bl_eeg.copy().crop(tmin=cf_peaks_gamma['time_range'][0], tmax=cf_peaks_gamma['time_range'][1], 
                                      fmin=cf_peaks_gamma['freq_range'][0], fmax=cf_peaks_gamma['freq_range'][1]).data.mean(axis=(1, 2))
        eeg_chs         = bl_eeg.ch_names
        row_eeg         = {ch: np.nan for ch in pool_eeg_20}
        row_eeg.update({ch: float(val) for ch, val in zip(eeg_chs, bl_eeg_topodata)})
        tfrlist_eeg.append(row_eeg)

        ### ----- OPM
        bl_opm          = tfropm.copy().apply_baseline(mode='logratio', baseline=cf_peaks_gamma['baseline'])
        bl_opm_topodata = bl_opm.copy().crop(tmin=cf_peaks_gamma['time_range'][0], tmax=cf_peaks_gamma['time_range'][1], 
                                      fmin=cf_peaks_gamma['freq_range'][0], fmax=cf_peaks_gamma['freq_range'][1]).data.mean(axis=(1, 2))
        opm_chs         = bl_opm.ch_names 
        ### ----- Store into a dict with NaNs for missing channels 
        row             = {ch: np.nan for ch in pool_opm_10}
        row.update({ch: float(val) for ch, val in zip(opm_chs, bl_opm_topodata)})
        tfrlist_opm.append(row)

        # Append subjects names
        subjects_id.append(subjid)

        ### ----- Append tfrs
        info_avg    = mne.create_info(ch_names=['channel_average'], sfreq=bl_eeg.sfreq, ch_types='eeg')
        ##### Select EEG sensors in pool_eeg_10
        picks_eeg   = mne.pick_channels(bl_eeg.ch_names, include=pool_eeg_10)
        tmpeeg      = 10*np.nanmean(bl_eeg.data[picks_eeg, :, :], axis=0) 
        eegtfr      = mne.time_frequency.AverageTFRArray(
                        info=info_avg, data=tmpeeg[np.newaxis, :, :], times=bl_eeg.times, 
                        freqs=bl_eeg.freqs, method=bl_eeg.method) 
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

        ##### Create AverageTFRArray objects for OPM X and Y separately
        opmxtfr     = mne.time_frequency.AverageTFRArray(
            info=info_avg, data=tmpopmx[np.newaxis, :, :], times=bl_opm.times, 
            freqs=bl_opm.freqs, method=bl_opm.method) 
        
        opmytfr     = mne.time_frequency.AverageTFRArray(
            info=info_avg, data=tmpopmy[np.newaxis, :, :], times=bl_opm.times, 
            freqs=bl_opm.freqs, method=bl_opm.method) 
        
        tfrs_opm_x.append(opmxtfr)
        tfrs_opm_y.append(opmytfr)

        ### ----- Frequency-domain "change in power" spectra (pool-averaged,
        ### ----- averaged over each band's own fit window, frequency axis
        ### ----- kept) 
        spec_g_eeg  = bl_eeg.copy().crop(tmin=cf_peaks_gamma['time_range'][0], tmax=cf_peaks_gamma['time_range'][1],
                                          fmin=cf_peaks_gamma['fit_range'][0], fmax=cf_peaks_gamma['fit_range'][1])
        spec_eeg_gamma.append(10*np.nanmean(spec_g_eeg.data[picks_eeg, :, :], axis=(0, 2)))
        freqs_gamma = spec_g_eeg.freqs

        spec_g_opm  = bl_opm.copy().crop(tmin=cf_peaks_gamma['time_range'][0], tmax=cf_peaks_gamma['time_range'][1],
                                          fmin=cf_peaks_gamma['fit_range'][0], fmax=cf_peaks_gamma['fit_range'][1])
        spec_opmx_gamma.append(10*np.nanmean(spec_g_opm.copy().pick(opm_x_chs).data, axis=(0, 2)))
        spec_opmy_gamma.append(10*np.nanmean(spec_g_opm.copy().pick(opm_y_chs).data, axis=(0, 2)))

        ### ----- Alpha 
        bl_eeg_alpha = tfreeg_alpha.copy().apply_baseline(mode='logratio', baseline=cf_peaks_alpha['baseline'])
        bl_opm_alpha = tfropm_alpha.copy().apply_baseline(mode='logratio', baseline=cf_peaks_alpha['baseline'])

        picks_eeg_a = mne.pick_channels(bl_eeg_alpha.ch_names, include=pool_eeg_10)
        spec_a_eeg  = bl_eeg_alpha.copy().crop(tmin=cf_peaks_alpha['time_range'][0], tmax=cf_peaks_alpha['time_range'][1],
                                                fmin=cf_peaks_alpha['fit_range'][0], fmax=cf_peaks_alpha['fit_range'][1])
        spec_eeg_alpha.append(10*np.nanmean(spec_a_eeg.data[picks_eeg_a, :, :], axis=(0, 2)))
        freqs_alpha = spec_a_eeg.freqs

        picks_mag_a = mne.pick_types(bl_opm_alpha.info, meg=True, eeg=False, eog=False, stim=False, misc=False)
        opm_x_chs_a = [bl_opm_alpha.ch_names[i] for i in picks_mag_a if bl_opm_alpha.ch_names[i].lower().endswith("x")]
        opm_y_chs_a = [bl_opm_alpha.ch_names[i] for i in picks_mag_a if bl_opm_alpha.ch_names[i].lower().endswith("y")]
        spec_a_opm  = bl_opm_alpha.copy().crop(tmin=cf_peaks_alpha['time_range'][0], tmax=cf_peaks_alpha['time_range'][1],
                                                fmin=cf_peaks_alpha['fit_range'][0], fmax=cf_peaks_alpha['fit_range'][1])
        spec_opmx_alpha.append(10*np.nanmean(spec_a_opm.copy().pick(opm_x_chs_a).data, axis=(0, 2)))
        spec_opmy_alpha.append(10*np.nanmean(spec_a_opm.copy().pick(opm_y_chs_a).data, axis=(0, 2)))

##### Create DataFrames
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

# Print data for paper
print('Mean Pool EEG: ', 10*df_eeg[pool_eeg_10].mean(axis=1).mean().round(3))
print('SD Pool EEG: ', 10*df_eeg[pool_eeg_10].mean(axis=1).std().round(3))

print('Mean Pool z: ', 10*df_opm_x.mean(axis=1).mean().round(3))
print('SD Pool z: ', 10*df_opm_x.mean(axis=1).std().round(3))

print('Mean Pool y: ', 10*df_opm_y.mean(axis=1).mean().round(3))
print('SD Pool y: ', 10*df_opm_y.mean(axis=1).std().round(3))

##### Get Grand Average TFR data 
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
kwargs      = dict(cmap="RdBu_r", vlim=(-1, 1), extrapolate="local", 
                border=0, sphere=(0, -0.035, 0, 0.1), outlines="head", mask=np.ones(10, dtype=bool),
                mask_params = dict(marker='o', markerfacecolor='w', markeredgecolor='k',
                linewidth=0, markersize=1.5), ch_type="eeg", show=False) 
w_cm, h_cm  = 3.33, 3.33

##### Generate EEG topography
eeg_info = bl_eeg.info
##### Average across subjects/make dB
eeg_data = 10*np.nanmean(df_eeg[pool_eeg_10], axis=0)
##### Get info file for the subset of channels to plot (pool_eeg_10)
picks    = mne.pick_channels(eeg_info.ch_names, include=pool_eeg_10)
info_sub = mne.pick_info(eeg_info.copy(), picks)   

##### Create Figure
fig, ax = plt.subplots(figsize=(w_cm/2.54, h_cm/2.54))
im, cn = mne.viz.plot_topomap(
    eeg_data, info_sub, axes=ax, **kwargs
)
ax.text(0.5, 0, "EEG", transform=ax.transAxes,
        ha="center", va="top")
#plt.show()
#fig.savefig(
#    os.path.join(wdir, Results_Folder, "figures", "Figure_2_3", "TopomapEEG.png"),
#    dpi=300,
#    bbox_inches="tight", transparent=True)
plt.close(fig)


##### Generate EEG topography with all sensors (not just the pool_eeg_10 subset)
##### Restrict to channels actually present in eeg_info, in df_eeg's column order
all_eeg_chs     = [ch for ch in df_eeg.columns if ch in eeg_info.ch_names]
eeg_data_all    = 10*np.nanmean(df_eeg[all_eeg_chs], axis=0)
##### Get info file for all sensors, same order as eeg_data_all
picks_all       = mne.pick_channels(eeg_info.ch_names, include=all_eeg_chs)
info_sub_all    = mne.pick_info(eeg_info.copy(), picks_all)

##### Same plotting kwargs, but suppress the built-in sensor markers/mask -
##### markers are drawn manually below so pool_eeg_10 vs the rest can have
##### distinct sizes/colors, and the pool_eeg_10 area can be highlighted.
kwargs_all                 = dict(kwargs)
kwargs_all['sensors']      = False
kwargs_all['mask']         = None
kwargs_all['extrapolate']  = 'local'

##### Create Figure
fig, ax = plt.subplots(figsize=(w_cm/2.54, h_cm/2.54))
im, cn = mne.viz.plot_topomap(
    eeg_data_all, info_sub_all, axes=ax, **kwargs_all
)

##### Lock the view limits plot_topomap just set - otherwise the manual
##### markers below (spanning a wider area than the local extrapolation
##### hull) trigger autoscale and the head ends up looking smaller.
xlim_fixed  = ax.get_xlim()
ylim_fixed  = ax.get_ylim()

##### Get the exact 2D sensor coordinates plot_topomap used internally,
##### so the manual markers line up with the data exactly.
from mne.viz.topomap import _get_pos_outlines
pos_2d, _   = _get_pos_outlines(info_sub_all, picks=None, sphere=kwargs['sphere'])
is_pool     = np.array([ch in pool_eeg_10 for ch in all_eeg_chs])

##### Same marker style as the pool_eeg_10 plot (mask_params) for both groups -
##### non-pool sensors are just dimmed via alpha.
mparams     = kwargs['mask_params']
ax.plot(pos_2d[~is_pool, 0], pos_2d[~is_pool, 1], linestyle='None', alpha=0.3,
        zorder=3, **mparams)
ax.plot(pos_2d[is_pool, 0], pos_2d[is_pool, 1], linestyle='None',
        zorder=4, **mparams)

##### Reapply the original framing
ax.set_xlim(xlim_fixed)
ax.set_ylim(ylim_fixed)

ax.text(0.5, 0, "EEG", transform=ax.transAxes,
        ha="center", va="top")
#plt.show()
#fig.savefig(
#    os.path.join(wdir, Results_Folder, "figures", "Figure_2_3", "TopomapEEG_allsensors.png"),
#    dpi=300,
#    bbox_inches="tight", transparent=True)
plt.close(fig)


##### Plot Topomap OPM
### ----- Divide axes/set equivalent OPM positions compared to EEG sensors
opm_x_topo  = 10*df_opm_x.mean(axis=0, skipna=True).to_numpy()  # shape (10,)
opm_y_topo  = 10*df_opm_y.mean(axis=0, skipna=True).to_numpy()  # shape (10,)

### ----- get 3D positions for these channels from the montage
mont        = eeg_info.get_montage()
ch_pos_all  = mont.get_positions()['ch_pos']
pos         = np.array([ch_pos_all[ch] for ch in pool_eeg_10])  # meters

### ----- For visualization move OPM sensors ~1 cm toward midline (x -> 0)
dx              = 0.01
pos_new         = pos.copy()
pos_new[:, 0]   += np.where(pos[:, 0] < 0, +dx, -dx)  # left->right, right->left

### ----- Build modified montage and attach to info_sub
mont_new = mne.channels.make_dig_montage(
    ch_pos={ch: p for ch, p in zip(pool_eeg_10, pos_new)},
    coord_frame="head"
)
info_sub.set_montage(mont_new)

##### Run this for X and Y axes
for opmax in ['x', 'y']:
    if opmax == 'x':
        opmtopodata = opm_x_topo
        titletext   = 'OPM Z-axis' 
        figname     = 'Topomap_OPM_z.png'
    else:
        opmtopodata = opm_y_topo
        titletext   = 'OPM Y-axis' 
        figname     = 'Topomap_OPM_y.png'

    fig, ax = plt.subplots(figsize=(w_cm/2.54, h_cm/2.54))
    ax.clear()  # optional, but safe

    im, cn = mne.viz.plot_topomap(opmtopodata, info_sub, axes=ax, **kwargs)
    ax.text(0.5, 0, titletext, transform=ax.transAxes,
            ha="center", va="top")
    outpath = os.path.join(wdir, Results_Folder, "figures", "Figure_2_3", figname)
    #fig.savefig(outpath, dpi=300, bbox_inches="tight", transparent=True)
    plt.close(fig)


##### Plot colorbar
vmin, vmax  = -1, 1
cmap        = "RdBu_r"
##### Create a figure and axis for the colorbar
w_cm, h_cm = 3, 3.0
fig, ax = plt.subplots(figsize=(w_cm/2.54, h_cm/2.54))
ax.axis("off")
norm        = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
sm          = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])  # required for some Matplotlib versions

cbar        = fig.colorbar(sm, ax=ax, fraction=1.0, pad=0.0, orientation='horizontal')
cbar.set_label("Change (dB)")          # optional
cbar.set_ticks([vmin, 0, vmax])     # optional

outpath = os.path.join(wdir, Results_Folder, "figures", "Figure_2_3", "Topomap_colorbar.png")
#fig.savefig(outpath, dpi=300, bbox_inches="tight", transparent=True)
plt.close(fig)


##### Plot average TFRs for ERS (gamma) - same layout as analysis_7's alpha ERD plot,
##### using the pool_eeg_10/pool_opm_10 grand averages already computed above (ga_eeg/ga_opmx/ga_opmy)
lims_plot   = 1
kwargs_tfr  = dict(tmin=-0.1, tmax=0.9, vlim=(-lims_plot, lims_plot),
                    fmin=cf_tfr_gamma['foi_start'], fmax=cf_tfr_gamma['foi_stop']-0.5,
                    colorbar=False, show=False)
fig, axes   = plt.subplots(1, 3, figsize=(10/2.54, 3/2.54), sharey=True)
ax11, ax21, ax31 = axes[0], axes[1], axes[2]

##### Plot average TFRs
ga_eeg.plot(axes=ax11, **kwargs_tfr)
ga_opmx.plot(axes=ax21, **kwargs_tfr)
ga_opmy.plot(axes=ax31, **kwargs_tfr)

##### Set labels
xticks = [0.0, 0.4, 0.8]
yticks = [30, 60, 90]
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

##### Save Figure
#fig.savefig(os.path.join(wdir, Results_Folder, "figures", "Figure_2_3", "Average_TFRs_gamma.png"),
#            dpi=300)
plt.close(fig)

##### Plot Colorbar
vmin, vmax = -lims_plot, lims_plot
cmap = "RdBu_r"
fig, ax = plt.subplots(figsize=(1.5/2.54, 3/2.54))
ax.axis("off")
norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, orientation="vertical", fraction=0.8, pad=0.0)
cbar.set_label("Change (dB)")
cbar.set_ticks([vmin, 0, vmax])
outpath = os.path.join(wdir, Results_Folder, "figures", "Figure_2_3", "Colorbar_TFR_gamma.png")
#fig.savefig(outpath, dpi=300, bbox_inches="tight", transparent=True)
plt.close(fig)


##### ----- Frequency-domain "change in power" spectrum, EEG + OPM overlaid
spec_eeg_gamma_arr  = np.array(spec_eeg_gamma)
spec_opmx_gamma_arr = np.array(spec_opmx_gamma)
spec_opmy_gamma_arr = np.array(spec_opmy_gamma)
spec_eeg_alpha_arr  = np.array(spec_eeg_alpha)
spec_opmx_alpha_arr = np.array(spec_opmx_alpha)
spec_opmy_alpha_arr = np.array(spec_opmy_alpha)

def _mean_sem(x):
    m   = np.nanmean(x, axis=0)
    sem = np.nanstd(x, axis=0) / np.sqrt(np.sum(~np.isnan(x), axis=0))
    return m, sem

eeg_g_m,  eeg_g_sem  = _mean_sem(spec_eeg_gamma_arr)
opmx_g_m, opmx_g_sem = _mean_sem(spec_opmx_gamma_arr)
opmy_g_m, opmy_g_sem = _mean_sem(spec_opmy_gamma_arr)
eeg_a_m,  eeg_a_sem  = _mean_sem(spec_eeg_alpha_arr)
opmx_a_m, opmx_a_sem = _mean_sem(spec_opmx_alpha_arr)
opmy_a_m, opmy_a_sem = _mean_sem(spec_opmy_alpha_arr)

cm_to_inch = 1 / 2.54
w_cm, h_cm = 5.14, 5.88

with plt.rc_context({
    'font.size': 7, 'axes.titlesize': 7, 'axes.labelsize': 7,
    'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 6.5,
}):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(w_cm * cm_to_inch, h_cm * cm_to_inch))

    ##### ----- Gamma panel
    ax1.plot(freqs_gamma, eeg_g_m, color='slategray', linewidth=1.2, label='EEG')
    ax1.fill_between(freqs_gamma, eeg_g_m - eeg_g_sem, eeg_g_m + eeg_g_sem, color='slategray', alpha=0.3)
    ax1.plot(freqs_gamma, opmx_g_m, color='crimson', linewidth=1.2, label='OPM Z-axis')
    ax1.fill_between(freqs_gamma, opmx_g_m - opmx_g_sem, opmx_g_m + opmx_g_sem, color='crimson', alpha=0.2)
    ax1.plot(freqs_gamma, opmy_g_m, color='#008080', linewidth=1.2, label='OPM Y-axis')
    ax1.fill_between(freqs_gamma, opmy_g_m - opmy_g_sem, opmy_g_m + opmy_g_sem, color='#008080', alpha=0.2)
    ax1.axvspan(cf_peaks_gamma['freq_range'][0], cf_peaks_gamma['freq_range'][1],
                color='tab:orange', alpha=0.08, zorder=0, linewidth=0)
    ax1.axhline(0.0, ls='--', lw=0.8, color='k', alpha=0.5)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.set_ylabel('Change (dB)')
    ax1.set_title('Gamma', pad=2)

    ##### ----- Alpha/beta panel
    ax2.plot(freqs_alpha, eeg_a_m, color='slategray', linewidth=1.2, label='EEG')
    ax2.fill_between(freqs_alpha, eeg_a_m - eeg_a_sem, eeg_a_m + eeg_a_sem, color='slategray', alpha=0.3)
    ax2.plot(freqs_alpha, opmx_a_m, color='crimson', linewidth=1.2, label='OPM Z-axis')
    ax2.fill_between(freqs_alpha, opmx_a_m - opmx_a_sem, opmx_a_m + opmx_a_sem, color='crimson', alpha=0.2)
    ax2.plot(freqs_alpha, opmy_a_m, color='#008080', linewidth=1.2, label='OPM Y-axis')
    ax2.fill_between(freqs_alpha, opmy_a_m - opmy_a_sem, opmy_a_m + opmy_a_sem, color='#008080', alpha=0.2)
    ax2.axvspan(cf_peaks_alpha['freq_range'][0], cf_peaks_alpha['freq_range'][1],
                color='tab:blue', alpha=0.08, zorder=0, linewidth=0)
    ax2.axhline(0.0, ls='--', lw=0.8, color='k', alpha=0.5)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.set_ylabel('Change (dB)')
    ax2.set_title('Alpha/Beta', pad=2)
    ax2.set_xlabel('Frequency (Hz)', labelpad=1)

    ##### ----- Single shared legend (same 3 series in both panels)
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 1.0), handlelength=1.2, columnspacing=0.7)

    plt.subplots_adjust(left=0.24, right=0.97, top=0.85, bottom=0.14, hspace=0.55)
    #fig.savefig(os.path.join(wdir, Results_Folder, "figures", "Figure_2_3",
    #                          "EEG_OPM_PowerChange_Spectrum.png"),
    #            dpi=300)
    plt.close(fig)


##### ----- PSD 1/f slope (aperiodic exponent) quantification via FOOOF
from specparam import SpectralModel

##### ----- Metric extraction
def _get_gof_error(fm):
    for get_gof, get_err in [
        (lambda: fm.results.get_metrics('gof'),        lambda: fm.results.get_metrics('error')),
        (lambda: fm.get_metrics('gof'),                lambda: fm.get_metrics('error')),
        (lambda: fm.get_metrics('gof', 'r_squared'),   lambda: fm.get_metrics('error', 'mae')),
        (lambda: fm.results.get_metrics('gof', 'r_squared'), lambda: fm.results.get_metrics('error', 'mae')),
        (lambda: fm.r_squared_,                        lambda: fm.error_),
    ]:
        try:
            return get_gof(), get_err()
        except Exception:
            continue
    raise AttributeError(
        "Could not extract goodness-of-fit / error metrics from this SpectralModel "
        "instance -- run `print([a for a in dir(fm) if not a.startswith('_')])` and "
        "`print(specparam.__version__)` after a fit to find the right accessor for "
        "this specparam version."
    )

##### ----- Set FOOOF Parameters
fooof_freq_range = [5, 90]
fooof_settings    = dict(
    peak_width_limits=[2, 12],
    max_n_peaks=6,
    peak_threshold=2.0,
    aperiodic_mode='fixed',
    verbose=False,
)

fooof_figdir = os.path.join(wdir, Results_Folder, "figures", "Figures_Supplementary", "FOOOF_fits")
os.makedirs(fooof_figdir, exist_ok=True)

fooof_records = []
fooof_fits_by_group = {}

for nsubject in range(n_subjects):

    sid             = datafolders[nsubject]['id']
    ##### ----- Fit Data For Controls Only
    issz_fooof      = demog_file.loc[sid[8:]]['SZ_EEG_INFO']
    if issz_fooof:
        continue

    datafoldereeg   = datafolders[nsubject]['eeg']
    if Results_Folder == os.path.join('Results', 'SNR_controls', 'Revision_IN'):
        datafoldereeg = os.path.join(datafoldereeg, 'revision')
    datafolderopm   = datafolders[nsubject]['opm']

    for modality, datafolder_raw in [('eeg', datafoldereeg), ('opm', datafolderopm)]:

        psd_path = os.path.join(datafolder_raw, 'psds', f'baseline_post_psd_{modality}.npz')
        if not os.path.exists(psd_path):
            print(f'  [FOOOF] Skipping {sid} {modality}: PSD file not found ({psd_path})')
            continue

        psd_data = np.load(psd_path)
        freqs    = psd_data['freqs']

        for period, period_label in [('pre', 'Baseline'), ('post', 'Post-stim')]:

            ##### ----- Pool-average PSD 
            curve = psd_data[period].mean(axis=0)

            fm = SpectralModel(**fooof_settings)
            fm.fit(freqs, curve, fooof_freq_range)

            exponent  = fm.get_params('aperiodic', 'exponent')
            offset    = fm.get_params('aperiodic', 'offset')
            r_squared, fit_error = _get_gof_error(fm)

            fooof_records.append(dict(
                subject=sid, modality=modality, period=period,
                exponent=exponent, offset=offset,
                r_squared=r_squared, fit_error=fit_error,
            ))

            ##### ----- Per-subject / modality / period diagnostic figure
            fig, ax = plt.subplots(figsize=(4, 3))
            fm.plot(plt_log=True, ax=ax, add_legend=True)
            ax.set_title(f'{sid} | {modality.upper()} | {period_label}\n'
                         f'exponent={exponent:.2f}, R²={r_squared:.3f}', fontsize=8)
            fig.tight_layout()
            #fig.savefig(os.path.join(fooof_figdir, f'{sid}_{modality}_{period}.png'), dpi=150)
            plt.close(fig)

            fooof_fits_by_group.setdefault((modality, period), []).append((sid, fm, exponent, r_squared))

    print(f'  [FOOOF] Done: {sid}')

##### ----- Grid summary
for (modality, period), entries in fooof_fits_by_group.items():
    period_label = 'Baseline' if period == 'pre' else 'Post-stim'
    n            = len(entries)
    ncols        = int(np.ceil(np.sqrt(n)))
    nrows        = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(2.2 * ncols, 2.0 * nrows))
    axes      = np.atleast_1d(axes).ravel()

    for i, (sid, fm, exponent, r_squared) in enumerate(entries):
        ax = axes[i]
        fm.plot(plt_log=True, ax=ax, add_legend=(i == 0))
        ax.set_title(f'{sid}\nexp={exponent:.2f}, R²={r_squared:.3f}', fontsize=6)
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.tick_params(labelsize=5)

    for j in range(n, len(axes)):
        axes[j].axis('off')

    fig.suptitle(f'{modality.upper()} - {period_label} - PSD slope fits (all subjects)', fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    #fig.savefig(os.path.join(fooof_figdir, f'_gridsummary_{modality}_{period}.png'), dpi=150)
    plt.close(fig)

##### ----- Group summary: per-subject exponents/R2 (full table) + mean +/- SD per
##### ----- modality/period 
fooof_df = pd.DataFrame(fooof_records)
fooof_df.to_csv(os.path.join(wdir, Results_Folder, 'results_other', "PSD_slope_fooof_persubject.csv"), index=False)

fooof_summary = fooof_df.groupby(['modality', 'period'])[['exponent', 'r_squared']].agg(['mean', 'std', 'count'])
fooof_summary.to_csv(os.path.join(wdir, Results_Folder, 'results_other', "PSD_slope_fooof_summary.csv"))
print('\n[FOOOF] PSD slope (aperiodic exponent) summary:')
print(fooof_summary)

##### =======================================================================
##### ----- Group-average APERIODIC-ONLY fit, linear frequency axis
##### ----- One figure per modality, baseline and post-stimulus combined on
##### ----- the same axes. Reuses the fm objects already fit and stored in
##### ----- fooof_fits_by_group above -- no refitting, no reloading.
##### =======================================================================
def _get_ap_fit(fm):
    """Aperiodic-only fitted curve (log10 power), handling both the newer
    specparam get_model() API and the older/internal FOOOF attribute."""
    for get_ap in [
        lambda: fm.get_model(component='aperiodic', space='log'),
        lambda: fm.results.get_model(component='aperiodic', space='log'),
        lambda: fm.get_model('aperiodic'),
        lambda: fm._ap_fit,
    ]:
        try:
            return get_ap()
        except Exception:
            continue
    raise AttributeError(
        "Could not extract the aperiodic-only fit from this SpectralModel instance -- "
        "check the get_model()/_ap_fit accessor for this specparam version."
    )

period_style    = {'pre': dict(ls=':', label='Baseline'), 'post': dict(ls='-', label='Post-stim')}
modality_color  = {'eeg': 'slategray', 'opm': 'crimson'}

fig, axes = plt.subplots(1, 2, figsize=(8, 3), sharey=False)

for ax, modality in zip(axes, ['eeg', 'opm']):

    for period in ['pre', 'post']:
        entries = fooof_fits_by_group.get((modality, period), [])
        if len(entries) == 0:
            continue

        all_ap, freqs_ref = [], None
        for sid, fm, exponent, r_squared in entries:
            freqs   = fm.freqs
            ap_fit  = _get_ap_fit(fm)
            if freqs_ref is None:
                freqs_ref = freqs
            elif len(freqs) != len(freqs_ref):
                print(f'  [FOOOF ap-fit] Skipping {sid} {modality} {period}: frequency grid mismatch')
                continue
            all_ap.append(ap_fit)

        all_ap  = np.array(all_ap)
        n_fits  = all_ap.shape[0]
        mean_ap = all_ap.mean(axis=0)
        sem_ap  = all_ap.std(axis=0, ddof=1) / np.sqrt(n_fits)

        style = period_style[period]
        ax.plot(freqs_ref, mean_ap, color=modality_color[modality], linewidth=1.4,
                ls=style['ls'], label=f"{style['label']}")
        #ax.plot(freqs_ref,  all_ap.T, color=modality_color[modality], linewidth=1,
        #        ls=style['ls'])
        ax.fill_between(freqs_ref, mean_ap - sem_ap, mean_ap + sem_ap,
                         color=modality_color[modality], alpha=0.15)

    ax.set_xscale('linear')
    ax.set_xlabel('Frequency (Hz)')
    if modality == 'opm':
        ax.set_title('OPM-MEG', fontsize=8)
    else:
        ax.set_title('EEG', fontsize=8)
    ax.legend(fontsize=7, frameon=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

axes[0].set_ylabel('log10(Power)')
fig.tight_layout()
#fig.show()
#fig.savefig(os.path.join(fooof_figdir, '_aperiodicfit_linearfreq_eeg_opm.png'), dpi=150)
plt.close(fig)

print('[FOOOF] Saved group-average aperiodic-only fit (linear frequency axis): eeg + opm')

#####----- Figure 3A
##### ===========================================================================
from matplotlib.lines import Line2D

MANUSCRIPT_PANEL_RCPARAMS = {
    'font.family':      'Arial',
    'font.size':         8,
    'axes.titlesize':    8,
    'axes.labelsize':    8,
    'xtick.labelsize':   6,
    'ytick.labelsize':   6,
    'legend.fontsize':   8,
}
PANEL_WIDTH_CM  = 7.5
PANEL_HEIGHT_CM = 3.5
FREQ_CUTOFF     = 150
GROUP_FILTER = 'controls'

out_dir = os.path.join(wdir, Results_Folder, "figures", "Figure_2_3")

def get_subject_folders(nsubject):
    """(sid, id_demog, datafoldereeg, datafolderopm) with the same 'revision'
    redirect for EEG used throughout the rest of the pipeline."""
    sid           = datafolders[nsubject]['id']
    id_demog      = sid[8:]
    datafoldereeg = datafolders[nsubject]['eeg']
    if Results_Folder == os.path.join('Results', 'SNR_controls', 'Revision_IN'):
        datafoldereeg = os.path.join(datafoldereeg, 'revision')
    datafolderopm = datafolders[nsubject]['opm']
    return sid, id_demog, datafoldereeg, datafolderopm


def subject_in_group(id_demog):
    """Group membership from the master demographic file (SZ_EEG_INFO)."""
    if id_demog not in demog_file.index:
        return False
    sz = demog_file.loc[id_demog, 'SZ_EEG_INFO']
    if pd.isna(sz):
        return False
    if GROUP_FILTER == 'controls':
        return sz == 0
    elif GROUP_FILTER == 'patients':
        return sz == 1
    return True

def plot_manuscript_psd_panel(spread='sem'):
    assert spread in ('sem', 'iqr')
    cm_to_in = 1 / 2.54
    style = {
        'eeg': dict(color='slategray', label='EEG'),
        'opm': dict(color='crimson',   label='OPM-MEG'),
    }

    curves = {}   # curves[modality] = dict(freqs, subj_pre (n_subj, n_freq), subj_post, n)
    for modality in ['eeg', 'opm']:
        all_pre, all_post, freqs = [], [], None
        for nsubject in range(len(datafolders)):
            sid, id_demog, datafoldereeg, datafolderopm = get_subject_folders(nsubject)
            if not subject_in_group(id_demog):
                continue

            datafolder = datafoldereeg if modality == 'eeg' else datafolderopm
            #fpath = os.path.join(datafolder, 'psds', f'baseline_post_psd_{modality}.npz')
            fpath = os.path.join(datafolder, 'psds', f'extended_baseline_post_psd_{modality}.npz')
            if not os.path.exists(fpath):
                continue

            npz   = np.load(fpath)
            freqs = npz['freqs']
            ##### ----- Average over channels in linear power, then convert to dB
            all_pre.append(10 * np.log10(npz['pre'].mean(axis=0)))
            all_post.append(10 * np.log10(npz['post'].mean(axis=0)))

        if len(all_pre) == 0:
            continue

        freqs  = np.asarray(freqs)
        f_mask = freqs <= FREQ_CUTOFF
        curves[modality] = dict(
            freqs=freqs[f_mask],
            subj_pre=np.array(all_pre)[:, f_mask],
            subj_post=np.array(all_post)[:, f_mask],
            n=len(all_pre),
        )

    cf_alpha_range = tuple(cfg['gaussians_alpha']['freq_range'])
    cf_gamma_range = tuple(cfg['gaussians_gamma']['freq_range'])

    with plt.rc_context(MANUSCRIPT_PANEL_RCPARAMS):
        fig, axes = plt.subplots(1, 2, figsize=(PANEL_WIDTH_CM * cm_to_in, PANEL_HEIGHT_CM * cm_to_in))

        for ax, (modality, s) in zip(axes, style.items()):
            if modality not in curves:
                ax.set_visible(False)
                continue
            c = curves[modality]
            f = c['freqs']

            ##### ----- Alpha/beta and gamma band background shading, behind
            ##### ----- everything (zorder=0), same freq_range used for the
            ##### ----- Gaussian peak fitting elsewhere in the pipeline
            ax.axvspan(*cf_alpha_range, color='tab:blue',   alpha=0.08, zorder=0, linewidth=0)
            ax.axvspan(*cf_gamma_range, color='tab:orange', alpha=0.07, zorder=0, linewidth=0)

            ##### ----- Baseline: dotted, thinner, lighter fill. Post-stim: solid,
            ##### ----- thicker, darker fill (post-stim is the condition of
            ##### ----- interest, so it's the visually emphasized curve in both
            ##### ----- line and shading).
            if spread == 'iqr':
                pre_center  = np.median(c['subj_pre'],  axis=0)
                pre_lo,  pre_hi  = np.percentile(c['subj_pre'],  [25, 75], axis=0)
                post_center = np.median(c['subj_post'], axis=0)
                post_lo, post_hi = np.percentile(c['subj_post'], [25, 75], axis=0)
            else:
                pre_center  = c['subj_pre'].mean(axis=0)
                pre_sem     = c['subj_pre'].std(axis=0, ddof=1) / np.sqrt(c['subj_pre'].shape[0])
                pre_lo, pre_hi = pre_center - pre_sem, pre_center + pre_sem
                post_center = c['subj_post'].mean(axis=0)
                post_sem    = c['subj_post'].std(axis=0, ddof=1) / np.sqrt(c['subj_post'].shape[0])
                post_lo, post_hi = post_center - post_sem, post_center + post_sem

            ax.fill_between(f, pre_lo, pre_hi, color=s['color'], alpha=0.15, linewidth=0, zorder=1)
            ax.plot(f, pre_center, color=s['color'], linestyle=':', linewidth=0.7, zorder=3)
            ax.fill_between(f, post_lo, post_hi, color=s['color'], alpha=0.40, linewidth=0, zorder=2)
            ax.plot(f, post_center, color=s['color'], linestyle='-', linewidth=1.1, zorder=3)

            ax.set_xscale('log')
            ax.set_xlim(right=FREQ_CUTOFF)
            ax.set_xlabel('Frequency (Hz)')
            ax.set_title(s['label'])
            ##### ----- Force exactly the same NUMBER of ticks in both panels
            ##### ----- (4), but round the positions to whole dB so labels don't
            ##### ----- show ugly decimals -- LinearLocator alone gives evenly
            ##### ----- spaced but non-round values (e.g. -107.76).
            ymin, ymax = ax.get_ylim()
            tick_positions = np.round(np.linspace(ymin, ymax, 4)).astype(int)
            ax.set_yticks(tick_positions)

        axes[0].set_ylabel('Power (dB)')

        ##### ----- Shared legend above the panels (linestyle only) instead of an
        ##### ----- in-axes legend that collided with the data
        legend_handles = [
            Line2D([0], [0], color='k', linestyle=':', linewidth=0.7, label='Baseline'),
            Line2D([0], [0], color='k', linestyle='-', linewidth=1.1, label='Post-stim'),
        ]
        fig.legend(handles=legend_handles, loc='upper center', ncol=2, frameon=False,
                   bbox_to_anchor=(0.5, 1.15), handlelength=1.5, columnspacing=1.0)

        fig.tight_layout(pad=0.3)
        #fig.show()
        fpath_out = os.path.join(out_dir, f'manuscript_panel_raw_psd_baseline_post_{GROUP_FILTER}_{spread}.png')
        fig.savefig(fpath_out, dpi=600, bbox_inches='tight')
        plt.close(fig)
        print(f'Saved: manuscript_panel_raw_psd_baseline_post_{GROUP_FILTER}_{spread}.png '
              f'({PANEL_WIDTH_CM}x{PANEL_HEIGHT_CM} cm, EEG n={curves.get("eeg", {}).get("n")}, '
              f'OPM n={curves.get("opm", {}).get("n")})')

plot_manuscript_psd_panel(spread='sem')