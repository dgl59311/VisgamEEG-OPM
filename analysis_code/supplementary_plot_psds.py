import os
import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import wilcoxon
from utils_visgam.visgam_paths_local import datafolders, cfg_dir, wdir
from utils_visgam.visgam_funcs import loadcfg

##### ----- Load configuration File 
cfg             = loadcfg(cfg_dir)

##### ----- Which subjects to include: 'controls' | 'patients' | 'all'
GROUP_FILTER    = 'controls'

##### ----- Define results folder
Results_Folder  = os.path.join('Results', 'SNR_controls', 'Revision_IN')
revision_dir    = os.path.join(wdir, Results_Folder)
out_dir         = os.path.join(revision_dir, 'figures', 'Figures_Supplementary')
os.makedirs(out_dir, exist_ok=True)

demog_file = pd.read_csv(os.path.join(wdir, 'DemographicFile.csv'), index_col=0)

combos = [
    ('gamma', '10_10'),
    ('alpha', '10_10'),
]


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



##### ===========================================================================

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

SUPP_WIDTH_CM  = 18
SUPP_HEIGHT_CM = 16
GED_FREQBAND   = 'gamma'
GED_SENSOR_CFG = '10_10'


def _collect_raw_curves():
    """Per-subject baseline/post curves from the raw, pre-GED sensor PSD --
    same source as plot_manuscript_psd_panel."""
    curves = {}
    for modality in ['eeg', 'opm']:
        all_pre, all_post, freqs = [], [], None
        for nsubject in range(len(datafolders)):
            sid, id_demog, datafoldereeg, datafolderopm = get_subject_folders(nsubject)
            if not subject_in_group(id_demog):
                continue
            datafolder = datafoldereeg if modality == 'eeg' else datafolderopm
            fpath = os.path.join(datafolder, 'psds', f'baseline_post_psd_{modality}.npz')
            if not os.path.exists(fpath):
                continue
            npz   = np.load(fpath)
            freqs = npz['freqs']
            all_pre.append(10 * np.log10(npz['pre'].mean(axis=0)))
            all_post.append(10 * np.log10(npz['post'].mean(axis=0)))
        if len(all_pre) == 0:
            continue
        freqs  = np.asarray(freqs)
        f_mask = freqs <= FREQ_CUTOFF
        curves[modality] = dict(freqs=freqs[f_mask],
                                 subj_pre=np.array(all_pre)[:, f_mask],
                                 subj_post=np.array(all_post)[:, f_mask],
                                 n=len(all_pre))
    return curves


def _collect_ged_curves(freqband, sensor_cfg):
    """Per-subject baseline/post curves from the GED-filtered, matched-
    component signal -- same source/logic as plot_ged_psd_group."""
    cf_ged_dir = cfg['ged_' + freqband + '_folders_' + sensor_cfg]
    peaks_csv  = os.path.join(revision_dir, 'results_' + freqband, f'{freqband}_peaks {sensor_cfg}.csv')
    if not os.path.exists(peaks_csv):
        print(f'  [_collect_ged_curves] missing {peaks_csv}')
        return {}
    peaks_df = pd.read_csv(peaks_csv, index_col=0)

    curves = {}
    for modality in ['eeg', 'opm']:
        comp_col = 'ged_eeg' if modality == 'eeg' else 'ged_opm'
        all_pre, all_post, freqs = [], [], None
        for nsubject in range(len(datafolders)):
            sid, id_demog, datafoldereeg, datafolderopm = get_subject_folders(nsubject)
            if not subject_in_group(id_demog):
                continue
            if id_demog not in peaks_df.index:
                continue
            row = peaks_df.loc[id_demog]
            if comp_col not in row or pd.isna(row[comp_col]):
                continue
            comp = row[comp_col]

            results_dir = os.path.join(datafoldereeg if modality == 'eeg' else datafolderopm,
                                        cf_ged_dir['results_folder'])
            fpath = os.path.join(results_dir, cf_ged_dir['epoch_filename'])
            if not os.path.exists(fpath):
                continue

            epochs_ged = mne.read_epochs(fpath, preload=True, verbose=False)
            if comp not in epochs_ged.ch_names:
                continue
            epochs_ged.pick(comp)

            pre_      = epochs_ged.copy().crop(tmin=-0.75, tmax=0.0)
            post_     = epochs_ged.copy().crop(tmin=0.0, tmax=0.75)
            n_per_seg = pre_.get_data().shape[-1]
            psd_pre   = pre_.compute_psd(method='welch', fmin=5, fmax=150,
                                          n_per_seg=n_per_seg, n_fft=n_per_seg, verbose=False)
            psd_post  = post_.compute_psd(method='welch', fmin=5, fmax=150,
                                           n_per_seg=n_per_seg, n_fft=n_per_seg, verbose=False)

            freqs = psd_pre.freqs
            all_pre.append(10 * np.log10(psd_pre.get_data()[:, 0, :].mean(axis=0)))
            all_post.append(10 * np.log10(psd_post.get_data()[:, 0, :].mean(axis=0)))

        if len(all_pre) == 0:
            continue
        freqs  = np.asarray(freqs)
        f_mask = freqs <= FREQ_CUTOFF
        curves[modality] = dict(freqs=freqs[f_mask],
                                 subj_pre=np.array(all_pre)[:, f_mask],
                                 subj_post=np.array(all_post)[:, f_mask],
                                 n=len(all_pre))
    return curves


def plot_supplementary_pre_post_ged(freqband=GED_FREQBAND, sensor_cfg=GED_SENSOR_CFG):
    cm_to_in = 1 / 2.54
    style = {
        'eeg': dict(color='slategray', label='EEG'),
        'opm': dict(color='crimson',   label='OPM-MEG'),
    }
    cf_alpha_range = tuple(cfg['gaussians_alpha']['freq_range'])
    cf_gamma_range = tuple(cfg['gaussians_gamma']['freq_range'])

    raw_curves = _collect_raw_curves()
    ged_curves = _collect_ged_curves(freqband, sensor_cfg)

    def _draw_panel(ax, c, s):
        f = c['freqs']

        ax.axvspan(*cf_alpha_range, color='tab:blue',   alpha=0.08, zorder=0, linewidth=0)
        ax.axvspan(*cf_gamma_range, color='tab:orange', alpha=0.07, zorder=0, linewidth=0)

        ##### ----- Individual-subject lines (thin, semi-transparent), slightly
        ##### ----- thicker than before, plus a robust median +/- IQR summary
        ##### ----- on top -- median/IQR instead of mean/SEM throughout, since
        ##### ----- IQR is robust to the few outlier subjects noted earlier and
        ##### ----- doesn't assume normality the way mean/SEM does.
        for row in c['subj_pre']:
            ax.plot(f, row, color=s['color'], linestyle=':', linewidth=0.6, alpha=0.15, zorder=1)
        for row in c['subj_post']:
            ax.plot(f, row, color=s['color'], linestyle='-', linewidth=0.6, alpha=0.15, zorder=1)

        pre_lo,  pre_hi  = np.percentile(c['subj_pre'],  [25, 75], axis=0)
        post_lo, post_hi = np.percentile(c['subj_post'], [25, 75], axis=0)
        ax.fill_between(f, pre_lo, pre_hi,   color=s['color'], alpha=0.12, linewidth=0, zorder=2)
        ax.fill_between(f, post_lo, post_hi, color=s['color'], alpha=0.22, linewidth=0, zorder=2)

        ax.plot(f, np.median(c['subj_pre'],  axis=0), color=s['color'], linestyle=':', linewidth=1.0, zorder=3)
        ax.plot(f, np.median(c['subj_post'], axis=0), color=s['color'], linestyle='-', linewidth=1.6, zorder=3)

    for xscale in ['log', 'linear']:
        with plt.rc_context(MANUSCRIPT_PANEL_RCPARAMS):
            fig, axes = plt.subplots(2, 2, figsize=(SUPP_WIDTH_CM * cm_to_in, SUPP_HEIGHT_CM * cm_to_in))

            for col, modality in enumerate(['eeg', 'opm']):
                s = style[modality]
                if modality in raw_curves:
                    _draw_panel(axes[0, col], raw_curves[modality], s)
                else:
                    axes[0, col].set_visible(False)
                axes[0, col].set_title(f"{s['label']} - before GED")

                if modality in ged_curves:
                    _draw_panel(axes[1, col], ged_curves[modality], s)
                else:
                    axes[1, col].set_visible(False)
                axes[1, col].set_title(f"{s['label']} - after GED ({freqband} {sensor_cfg})")

            for ax in axes.ravel():
                if xscale == 'log':
                    ax.set_xscale('log')
                    ax.set_xlim(right=FREQ_CUTOFF)   # left bound: data's natural ~5 Hz floor
                else:
                    ax.set_xlim(0, FREQ_CUTOFF)
                ax.set_xlabel('Frequency (Hz)')

            axes[0, 0].set_ylabel('Power (dB)')
            axes[1, 0].set_ylabel('Power (dB)')

            legend_handles = [
                Line2D([0], [0], color='k', linestyle=':', linewidth=1.0, label='Baseline'),
                Line2D([0], [0], color='k', linestyle='-', linewidth=1.6, label='Post-stim'),
            ]
            fig.legend(handles=legend_handles, loc='upper center', ncol=2, frameon=False,
                       bbox_to_anchor=(0.5, 1.03), handlelength=1.5, columnspacing=1.0)

            #fig.suptitle(f'Sensor-level PSD before vs. after GED ({freqband} {sensor_cfg}, {GROUP_FILTER})',
            #             y=1.07, fontsize=10)
            fig.tight_layout(pad=0.6)
            suffix = 'logx' if xscale == 'log' else 'linx'
            fpath_out = os.path.join(out_dir,
                        f'supp_pre_post_ged_{freqband}_{sensor_cfg}_{GROUP_FILTER}_{suffix}.png')
            fig.savefig(fpath_out, dpi=400, bbox_inches='tight')
            plt.close(fig)
            print(f'Saved: supp_pre_post_ged_{freqband}_{sensor_cfg}_{GROUP_FILTER}_{suffix}.png')


for freqband, sensor_cfg in combos:
    plot_supplementary_pre_post_ged(freqband=freqband, sensor_cfg=sensor_cfg)
