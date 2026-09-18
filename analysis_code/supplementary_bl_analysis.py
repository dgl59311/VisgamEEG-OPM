import os
import mne
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon, pearsonr, ttest_1samp
from utils_visgam.visgam_paths_local import datafolders, wdir, cfg_dir, local_folder
from utils_visgam.visgam_funcs import loadcfg

### ----- matplotlib mode
matplotlib.use('QtAgg')

##### ----- Load Demog File (controls only)
demog_file      = pd.read_csv(os.path.join(wdir, "DemographicFile.csv"), index_col=0)

##### ----- Import analysis cfg
cfg             = loadcfg(cfg_dir)
cf_peaks_gamma  = cfg['gaussians_gamma']
cf_peaks_alpha  = cfg['gaussians_alpha']

band_specs = [
    ('gamma', cf_peaks_gamma['freq_range'], {
        'pre':  (cf_peaks_gamma['baseline'],   'Baseline'),
        'post': (cf_peaks_gamma['time_range'], 'Post-stim'),
    }),
    ('alpha', cf_peaks_alpha['freq_range'], {
        'pre':  (cf_peaks_alpha['baseline'],   'Baseline'),
        'post': (cf_peaks_alpha['time_range'], 'Post-stim'),
    }),
]

##### ----- Print number of subjects
n_subjects      = len(datafolders)
print('Total sample: ', n_subjects)

##### Select subfolder within Results folder to save/load
Results_Folder  = os.path.join('Results', 'SNR_controls', 'Revision_IN')
outdir          = os.path.join(wdir, Results_Folder, 'figures', 'Figures_Supplementary')
os.makedirs(outdir, exist_ok=True)

##### ----- Define local folder paths for TFR data
LOCAL_TFR_SUBFOLDER     = {'gamma': 'TFR_HIGHFREQ_allsensors', 'alpha': 'TFR_LOWFREQ_allsensors'}
LOCAL_DEVICE_FOLDER     = {'eeg': 'EEG', 'opm': 'OPMMEG'}

##### ----- Help functions
def _load_pool_tfr(sid, modality, band_name):
    ##### ----- Load the saved per-trial pool TFR 
    folder  = os.path.join(local_folder, LOCAL_DEVICE_FOLDER[modality], LOCAL_TFR_SUBFOLDER[band_name])
    path    = os.path.join(folder, f'{sid}_EpochsTFR_Superlets_pool.h5')
    if not os.path.exists(path):
        return None
    tfr = mne.time_frequency.read_tfrs(path)
    return tfr, tfr.data, tfr.times, tfr.freqs


def _window_band_data(power, times, freqs, win, freq_range, return_db=False):
    tmask               = (times >= win[0]) & (times <= win[1])
    fmask               = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
    chan_trial_power    = power[:, :, fmask, :][:, :, :, tmask].mean(axis=(2, 3))   # (n_trials, n_channels)

    if return_db:
        return 10*np.log10(chan_trial_power.mean(1))

    return chan_trial_power.mean(1)


def _paired_dz(a, b):
    diff = np.asarray(a) - np.asarray(b)
    return diff.mean() / diff.std(ddof=1)


records         = []
corr_records    = []

for nsubject in range(n_subjects):

    sid     = datafolders[nsubject]['id']
    issz    = demog_file.loc[sid[8:]]['SZ_EEG_INFO']
    if issz:
        continue    # controls only

    print('Current Subject:', sid)

    for band_name, freq_range, periods in band_specs:

        ##### ----- Load the saved per-trial pool TFRs 
        eeg_loaded = _load_pool_tfr(sid, 'eeg', band_name)
        opm_loaded = _load_pool_tfr(sid, 'opm', band_name)
        if eeg_loaded is None or opm_loaded is None:
            print(f'  Skipping {sid}, {band_name}: pool TFR file(s) not found '
                  f'(has analysis_4_tfr_sensors.py been (re)run for this subject?)')
            continue
        eeg_tfr, eeg_power, eeg_times, freqs = eeg_loaded
        opm_tfr, opm_power, opm_times, _     = opm_loaded
        print(f'  {band_name}: EEG n_trials={eeg_power.shape[0]}, OPM n_trials={opm_power.shape[0]}')

        for period_key, (win, period_label) in periods.items():

            ##### ----- without dB
            eeg_trial_data = _window_band_data(eeg_power, eeg_times, freqs, win, freq_range, return_db=True)
            opm_trial_data = _window_band_data(opm_power, opm_times, freqs, win, freq_range, return_db=True)

            records.append(dict(
                subject=sid, modality='eeg', band=band_name, period=period_key,
                period_label=period_label, n_trials=len(eeg_trial_data),
                mean_power=eeg_trial_data.mean(), sd_db=eeg_trial_data.std(ddof=1),
            ))
            records.append(dict(
                subject=sid, modality='opm', band=band_name, period=period_key,
                period_label=period_label, n_trials=len(opm_trial_data),
                mean_power=opm_trial_data.mean(), sd_db=opm_trial_data.std(ddof=1),
            ))

            ##### ----- Trial-by-trial covariation between EEG's and OPM's
            ##### ----- baseline/signal fluctuations, for the SAME trials
            r, p = pearsonr(eeg_trial_data, opm_trial_data)
            corr_records.append(dict(
                subject=sid, band=band_name, period=period_key, period_label=period_label,
                n_trials=len(eeg_trial_data), r=r, p=p,
            ))

##### ----- Save per-subject records
df      = pd.DataFrame(records)
df.to_csv(os.path.join(wdir, Results_Folder, 'results_other', 'baseline_std.csv'), index=False)

##### ----- Group summary: mean +/- SD of the per-subject stability index,
##### ----- per device/band/period
summary = df.groupby(['modality', 'band', 'period'])['sd_db'].agg(['mean', 'std', 'count'])
print('\nStability (SD of per-trial dB deviations) summary:')
print(summary)

stats_records = []
##### ----- (1) EEG vs OPM, within each band/period
for band_name, _, periods in band_specs:
    for period_key in periods:
        sub         = df[(df['band'] == band_name) & (df['period'] == period_key)]
        wide        = sub.pivot(index='subject', columns='modality', values='sd_db').dropna()
        diff        = wide['eeg'] - wide['opm']
        stat, pval  = ttest_1samp(diff, 0)
        dz          = _paired_dz(wide['eeg'], wide['opm'])
        stats_records.append(dict(
            comparison='eeg_vs_opm', band=band_name, period=period_key, n=len(wide), t=stat, p=pval, dz=dz,
            group_a='eeg', a_mean=wide['eeg'].mean(), a_sd=wide['eeg'].std(),
            group_b='opm', b_mean=wide['opm'].mean(), b_sd=wide['opm'].std(),
        ))
        print(f'\n[{band_name}, {period_key}] EEG vs OPM stability, paired t-test: '
            f't={stat:.2f}, p={pval:.4f}, dz={dz:.2f} (n={len(wide)})')


stats_df = pd.DataFrame(stats_records)
stats_df.to_csv(os.path.join(wdir, Results_Folder, 'results_other', 'stats_baseline_summary.csv'), index=False)

##### ----- Save per-subject trial-by-trial EEG-OPM correlations
corr_df         = pd.DataFrame(corr_records)
corr_df.to_csv(os.path.join(wdir, Results_Folder, 'results_other', 'baseline_eeg_opm_correlations.csv'), index=False)

##### ----- Print summary of trial-by-trial EEG-OPM correlations
summary_corr    = corr_df.groupby(['band', 'period'])['r'].agg(['mean', 'std', 'count'])
print(summary_corr)

##### ----- Figure: paired per-subject stability, one row per period
##### ----- (baseline / post-stim)
font_size = 8
plt.rcParams.update({
    'font.family': 'Arial', 'font.size': font_size, 'axes.titlesize': font_size,
    'axes.labelsize': font_size, 'xtick.labelsize': font_size, 'ytick.labelsize': font_size,
})

period_keys = list(band_specs[0][2].keys())   # ['pre', 'post']
fig, axes = plt.subplots(len(period_keys), len(band_specs),
                          figsize=(3.2 * len(band_specs), 3.0 * len(period_keys)), sharey='col')

for row, period_key in enumerate(period_keys):
    for col, (band_name, _, periods) in enumerate(band_specs):
        ax          = np.atleast_2d(axes)[row, col]
        period_label = periods[period_key][1]
        wide = df[(df['band'] == band_name) & (df['period'] == period_key)] \
            .pivot(index='subject', columns='modality', values='sd_db').dropna()
        for _, r in wide.iterrows():
            ax.plot([0, 1], [r['eeg'], r['opm']], color='gray', alpha=0.4, linewidth=0.8, zorder=1)
        ax.scatter(np.zeros(len(wide)), wide['eeg'], color='slategray', zorder=2, label='EEG')
        ax.scatter(np.ones(len(wide)), wide['opm'], color='crimson', zorder=2, label='OPM-MEG')
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['EEG', 'OPM-MEG'])
        ax.set_xlim(-0.4, 1.4)
        if col == 0:
            ax.set_ylabel(f'{period_label}\nSD (dB)')
        if row == 0:
            ax.set_title(band_name.capitalize())
        ax.spines['top'].set_visible(False)

for col, (band_name, _, _) in enumerate(band_specs):
    col_vals = df.loc[df['band'] == band_name, 'sd_db']
    pad      = 0.05 * (col_vals.max() - col_vals.min())
    for row in range(len(period_keys)):
        ax_i = np.atleast_2d(axes)[row, col]
        ax_i.set_ylim(col_vals.min() - pad, col_vals.max() + pad)
        ax_i.spines['right'].set_visible(False)

fig.tight_layout()
fig.savefig(os.path.join(outdir, 'baseline_cleanliness_paired.png'), dpi=300)
plt.close(fig)

##### ----- Figure: distribution of per-subject trial-by-trial EEG-OPM
##### ----- correlations
fig, axes = plt.subplots(1, len(band_specs), figsize=(3.2 * len(band_specs), 3.2), sharey=True)
for ax, (band_name, _, periods) in zip(np.atleast_1d(axes), band_specs):
    for i, (period_key, (_, period_label)) in enumerate(periods.items()):
        sub = corr_df[(corr_df['band'] == band_name) & (corr_df['period'] == period_key)]
        jitter = (np.random.default_rng(0).random(len(sub)) - 0.5) * 0.2
        color = 'slategray' if period_key == 'pre' else 'darkorange'
        ax.scatter(np.full(len(sub), i) + jitter, sub['r'], color=color, alpha=0.7, label=period_label)
    ax.axhline(0.0, ls='--', lw=1, color='k', alpha=0.5)
    ax.set_xticks(range(len(periods)))
    ax.set_xticklabels([lab for _, lab in periods.values()])
    ax.set_title(band_name.capitalize())
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
np.atleast_1d(axes)[0].set_ylabel('Trial-by-trial EEG-OPM r')

fig.tight_layout()
fig.savefig(os.path.join(outdir, 'eeg_opm_trial_correlations.png'), dpi=300)
plt.close(fig)

print('\nDone. Outputs saved to:', os.path.join(wdir, Results_Folder))
