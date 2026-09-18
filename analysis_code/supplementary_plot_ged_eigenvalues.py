import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
from utils_visgam.visgam_paths_local import datafolders, local_folder, wdir, cfg_dir
from utils_visgam.visgam_funcs import loadcfg
import pandas as pd

##### ----- Use the revision folder for EEG eigenvalues
use_revision_eeg    = True

##### ----- Number of eigenvalues used in the final analysis
max_channels        = 3

##### ----- All combinations to plot
sensor_label_map    = {
    '10_10': '10 EEG sensors \n *10 OPMs (z + y)',
    '10_20': '20 EEG sensors \n *10 OPMs (z + y)',
}
combos              = [
    ('gamma', '10_10'),
    ('gamma', '10_20'),
    ('alpha', '10_10'),
    ('alpha', '10_20'),
]

##### ----- Load configuration files
cfg             = loadcfg(cfg_dir)
### ----- Load Demog File
demog_file      = pd.read_csv(os.path.join(wdir,"DemographicFile.csv"), index_col=0)

##### ----- Data Loader
def load_evals(freqband, sensor_cfg, demogdata, isctrl):
    cf_ged_dir      = cfg['ged_' + freqband + '_folders_' + sensor_cfg]
    eeg_evals_list  = []
    opm_evals_list  = []
    subj_ids        = []

    for d in datafolders:
        sid             = d['id']
        datafoldereeg   = d['eeg']
        datafolderopm   = d['opm']

        ##### Check demographic file for ctrl/patient status
        if isctrl:
            demog_key = sid[8:]
            if demog_key not in demogdata.index:
                print(f'Skipping {sid}: not found in demographics file')
                continue
            is_patient = int(demogdata.loc[demog_key, 'SZ_EEG_INFO']) == 1
            if is_patient:
                continue

        ##### ----- Revision folder for EEG eigenvalues
        datafoldereeg   = os.path.join(datafoldereeg, 'revision')
        ##### ----- Identify the results folder
        results_eeg_dir = os.path.join(datafoldereeg, cf_ged_dir['results_folder'])
        results_opm_dir = os.path.join(datafolderopm, cf_ged_dir['results_folder'])
        ##### ----- Identify .npy files storing the eigenvalues
        eeg_evals_path  = os.path.join(results_eeg_dir, cf_ged_dir['evals_filename'])
        opm_evals_path  = os.path.join(results_opm_dir, cf_ged_dir['evals_filename'])

        if not (os.path.exists(eeg_evals_path) and os.path.exists(opm_evals_path)):
            print(f'Files not found for subject {sid}')
            continue

        ##### ----- Append files for plotting 
        eeg_evals_list.append(np.load(eeg_evals_path))
        opm_evals_list.append(np.load(opm_evals_path))
        subj_ids.append(sid)

    print(f'[{freqband}, {sensor_cfg}] loaded eigenvalues for {len(subj_ids)} subjects')

    ##### ----- Report min/max number of available OPM channels (both axes) across subjects
    opm_n_channels  = [len(a) for a in opm_evals_list]
    if opm_n_channels:
        print(f'[{freqband}, {sensor_cfg}] OPM channels available (both axes) across subjects: '
              f'min={min(opm_n_channels)}, max={max(opm_n_channels)}')

    return eeg_evals_list, opm_evals_list, subj_ids


##### Add NaN for datasets of different rank, because OPMs were not interpolated
def pad_to_matrix(list_of_arrays):
    max_len     = max(len(a) for a in list_of_arrays)
    mat         = np.full((len(list_of_arrays), max_len), np.nan)
    for i, a in enumerate(list_of_arrays):
        mat[i, :len(a)] = a
    return mat


##### ----- Colors -----------------------------------------------------------
color_individual    = '0.75'            # light gray for individual subject lines
color_above1        = '#d62728'     # red-ish for markers where eigenvalue > 1
color_below1        = '#1f77b4'     # blue-ish for markers where eigenvalue <= 1
color_mean          = 'k'
shade_used_color    = 'gold'

def plot_panel(ax, mat, show_xlabel, show_title, title):
    idx     = np.arange(1, mat.shape[1] + 1)
    mean_   = np.nanmean(mat, axis=0)
    sem_    = np.nanstd(mat, axis=0) / np.sqrt(np.sum(~np.isnan(mat), axis=0))

    ##### ----- Shade the first max_channels eigenvalues 
    ax.axvspan(0.5, max_channels + 0.5, color=shade_used_color, alpha=0.15, zorder=0,
               label=f'Used in analysis (first {max_channels})')

    ##### ----- Individual subject lines 
    for row in mat:
        ax.plot(idx, row, color=color_individual, alpha=0.5, linewidth=0.6, zorder=1)

    ##### ----- Individual subject points, color-coded by threshold (>1 vs <=1)
    flat_idx    = np.tile(idx, mat.shape[0])
    flat_val    = mat.flatten()
    above       = flat_val > 1
    ax.scatter(flat_idx[above], flat_val[above], s=6, color=color_above1, alpha=0.6, zorder=2, label='Eigenvalue > 1')
    ax.scatter(flat_idx[~above & ~np.isnan(flat_val)], flat_val[~above & ~np.isnan(flat_val)],
               s=6, color=color_below1, alpha=0.4, zorder=2, label='Eigenvalue ≤ 1')

    ##### ----- Average line on top
    ax.plot(idx, mean_, color=color_mean, linewidth=1.8, zorder=3, label='Mean across subjects')
    ax.fill_between(idx, mean_ - sem_, mean_ + sem_, color=color_mean, alpha=0.15, zorder=3)

    ##### ----- Reference line at eigenvalue = 1
    ax.axhline(1, color='gray', linestyle='--', linewidth=0.8, zorder=1)

    if show_xlabel:
        ax.set_xlabel('Eigenvalue rank', fontsize=8)
    ax.set_yscale('log')
    ax.yaxis.set_minor_formatter(NullFormatter())
    if show_title:
        ax.set_title(title, fontsize=9)
    ax.set_xlim(0.5, idx.max() + 0.5)
    ax.set_xticks(np.arange(2, 22, 2))
    ax.tick_params(labelsize=7)


##### ----- Build figure
cm              = 1 / 2.54
fig_width_cm    = 17
fig_height_cm   = 15   # well under the 18 cm cap, adjust if needed
fig_ged, axes   = plt.subplots(len(combos), 2, figsize=(fig_width_cm * cm, fig_height_cm * cm), sharex='col')

for row, (freqband, sensor_cfg) in enumerate(combos):
    print(freqband, sensor_cfg)
    #### ----- Plot only for controls
    eeg_list, opm_list, subj_list = load_evals(freqband, sensor_cfg, demog_file, isctrl=True)
    print([opm_list[ii].shape for ii in range(26)])
    eeg_mat             = pad_to_matrix(eeg_list)
    opm_mat             = pad_to_matrix(opm_list)
    print(subj_list)

    is_last_row = (row == len(combos) - 1)
    is_first_row = (row == 0)

    plot_panel(axes[row, 0], eeg_mat, show_xlabel=is_last_row, show_title=is_first_row, title='EEG')
    plot_panel(axes[row, 1], opm_mat, show_xlabel=is_last_row, show_title=is_first_row, title='OPM-MEG')

    ##### ----- Row label (band + sensor config)
    row_label = f'{freqband.capitalize()}\n{sensor_label_map[sensor_cfg]}'
    bbox = axes[row, 0].get_position()
    fig_ged.text(0.05, (bbox.y0 + bbox.y1) / 2, row_label,
                 rotation=90, ha='center', va='center', fontsize=6.5)
    axes[row, 0].set_ylabel('Eigenvalue (GED)', fontsize=8)

fig_ged.subplots_adjust(left=0.2, right=0.98, top=0.88, bottom=0.08, hspace=0.5, wspace=0.4)

##### ----- Single shared legend
handles, labels = axes[0, 0].get_legend_handles_labels()
fig_ged.legend(handles, labels, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 0.995), fontsize=7.5, frameon=False)

print(f'Figure size: {fig_width_cm} x {fig_height_cm} cm')
##### ----- Save Figure
figure_dir = r'E:\ongoing\Visgam\Results\SNR_controls\Revision_IN\figures\Figures_Supplementary\GED_eigenvalues.png'
os.makedirs(os.path.dirname(figure_dir), exist_ok=True)
fig_ged.savefig(figure_dir, dpi=300)
plt.show()

