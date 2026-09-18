import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr, wilcoxon
from utils_visgam.visgam_funcs import cohen_d

##### ----- Script to compare peak fitting results
revision        = r'E:\ongoing\Visgam\Results\SNR_controls\Revision_IN'
sensor_cfg      = '10_10'
freqbands       = ['alpha', 'gamma']

required_cols   = ['pf_eeg', 'pf_opm', 'pp_eeg', 'pp_opm']


##### ----- Controls only ------------------------------------------------------
def controls_only(df, name):
    if 'SZ_EEG_INFO' not in df.columns:
        raise ValueError(f"'{name}' has no SZ_EEG_INFO column to filter controls.")
    sz_info     = df['SZ_EEG_INFO']
    n_missing   = int(sz_info.isna().sum())
    n_patient   = int((sz_info == 1).sum())
    is_control  = (sz_info == 0)
    ctrl        = df.loc[is_control].copy()
    print(f'{name:16s}: {len(df)} total -> {len(ctrl)} controls, {n_patient} patients, '
          f'{n_missing} missing SZ_EEG_INFO (excluded)')
    return ctrl


##### ----- Correlation: peak frequency and peak power, per modality -----------
def report_corr(x, y, label):
    r_p, p_p = pearsonr(x, y)
    r_s, p_s = spearmanr(x, y)
    bias     = float(np.mean(y - x))
    print(f'  {label:24s}: n={len(x):2d}  Pearson r={r_p:+.3f} (p={p_p:.4g})  '
          f'Spearman rho={r_s:+.3f} (p={p_s:.4g})  mean(peak_finder - gaussian)={bias:+.3f}')
    return dict(label=label, n=len(x), r_pearson=r_p, p_pearson=p_p,
                r_spearman=r_s, p_spearman=p_s, bias=bias)


merged_by_band  = {}
corr_by_band    = {}

for freqband in freqbands:
    print(f"\n{'='*70}")
    print(f"Band: {freqband}  (sensor config: {sensor_cfg})")
    print(f"{'='*70}")

    filename_gaussian   = f'{freqband}_peaks {sensor_cfg}.csv'
    filename_peaks      = f'{freqband}_peak_finder_peaks {sensor_cfg}.csv'
    path_gaussian       = os.path.join(revision, 'results_' + freqband, filename_gaussian)
    path_peaks          = os.path.join(revision, 'results_' + freqband, filename_peaks)

    for _p, _name in [(path_gaussian, f'{freqband} Gaussian-fit results'),
                       (path_peaks, f'{freqband} Peak-finder results')]:
        if not os.path.exists(_p):
            raise FileNotFoundError(f"{_name} not found at: {_p}")

    gaussians   = pd.read_csv(path_gaussian, index_col=0)
    peaks       = pd.read_csv(path_peaks, index_col=0)

    ##### ----- Sanity check: SZ_EEG_INFO should agree between the two files for any
    ##### ----- subject present in both (both were saved from the same DemographicFile.csv)
    common_all  = sorted(set(gaussians.index) & set(peaks.index))
    mismatch    = [sid for sid in common_all
                   if gaussians.loc[sid, 'SZ_EEG_INFO'] != peaks.loc[sid, 'SZ_EEG_INFO']]
    if mismatch:
        print(f'WARNING: SZ_EEG_INFO disagrees between the two files for {len(mismatch)} '
              f'subjects: {mismatch}')

    print('----- Control filtering -----')
    gaussians_ctrl  = controls_only(gaussians, 'gaussians')
    peaks_ctrl      = controls_only(peaks, 'peak_finder')

    ##### ----- Keep only control subjects with usable (non-NaN) peak data in BOTH methods
    gaussians_valid = gaussians_ctrl.dropna(subset=required_cols)
    peaks_valid     = peaks_ctrl.dropna(subset=required_cols)

    common_ids      = sorted(set(gaussians_valid.index) & set(peaks_valid.index))
    dropped_gauss   = sorted(set(gaussians_ctrl.index) - set(common_ids))
    dropped_peak    = sorted(set(peaks_ctrl.index) - set(common_ids))

    print(f'Control subjects with usable data in both methods: {len(common_ids)}')
    if dropped_gauss:
        print(f'  In Gaussian-fit controls but not matched (missing/NaN or absent in peak-finder): {dropped_gauss}')
    if dropped_peak:
        print(f'  In peak-finder controls but not matched (missing/NaN or absent in Gaussian-fit): {dropped_peak}')

    g = gaussians_valid.loc[common_ids]
    p = peaks_valid.loc[common_ids]

    ##### ----- Merge into one frame with explicit method suffixes -----------------
    merged = g[required_cols].add_suffix('_gauss').join(p[required_cols].add_suffix('_peak'))
    merged_by_band[freqband] = merged

    print(f'\n----- Peak frequency: Gaussian fit vs. peak-finder ({freqband}, controls only) -----')
    corr_results = []
    corr_results.append(report_corr(merged['pf_eeg_gauss'], merged['pf_eeg_peak'], 'Peak frequency (EEG)'))
    corr_results.append(report_corr(merged['pf_opm_gauss'], merged['pf_opm_peak'], 'Peak frequency (OPM)'))

    print(f'\n----- Peak power: Gaussian fit vs. peak-finder ({freqband}, controls only) -----')
    corr_results.append(report_corr(merged['pp_eeg_gauss'], merged['pp_eeg_peak'], 'Peak power (EEG)'))
    corr_results.append(report_corr(merged['pp_opm_gauss'], merged['pp_opm_peak'], 'Peak power (OPM)'))

    corr_by_band[freqband] = pd.DataFrame(corr_results)

    ##### ----- Effect size: EEG vs OPM (paired dz), computed separately per method
    ##### ----- Sign convention matches analysis_3_fit_gaussians.py: diff = EEG - OPM
    print(f'\n----- Effect size (EEG vs OPM, paired dz, {freqband}, controls only) -----')
    dz_gauss = cohen_d(merged['pp_eeg_gauss'], merged['pp_opm_gauss'], paired=True)
    dz_peak  = cohen_d(merged['pp_eeg_peak'],  merged['pp_opm_peak'],  paired=True)
    print(f"  Cohen's dz (Gaussian fit): {dz_gauss:.3f}   (n={len(merged)})")
    print(f"  Cohen's dz (Peak-finder):  {dz_peak:.3f}   (n={len(merged)})")
    print(f"  Difference (peak_finder - gaussian): {dz_peak - dz_gauss:+.3f}")

    ##### ----- Is the EEG-OPM advantage itself different between methods, subject by subject
    diff_gauss  = merged['pp_eeg_gauss'] - merged['pp_opm_gauss']
    diff_peak   = merged['pp_eeg_peak']  - merged['pp_opm_peak']

    stat, p_wil     = wilcoxon(diff_gauss, diff_peak)
    r_diff, p_diff  = pearsonr(diff_gauss, diff_peak)
    print(f"\n  Per-subject EEG-OPM difference score, Gaussian vs peak-finder:")
    print(f"    Wilcoxon signed-rank (paired difference-of-differences): stat={stat:.3f}, p={p_wil:.4g}")
    print(f"    Correlation between the two methods' per-subject difference score: "
          f"r={r_diff:+.3f} (p={p_diff:.4g})")

    ##### ----- Bootstrap 95% CI for the difference in dz between methods
    ##### ----- (paired resampling of subjects, so both dz's are recomputed from the
    ##### ----- same resampled subject set on each iteration)
    rng     = np.random.default_rng(0)
    n_boot  = 5000
    n_subj  = len(merged)
    eeg_g, opm_g = merged['pp_eeg_gauss'].to_numpy(), merged['pp_opm_gauss'].to_numpy()
    eeg_p, opm_p = merged['pp_eeg_peak'].to_numpy(),  merged['pp_opm_peak'].to_numpy()

    boot_diff = np.empty(n_boot)
    for b in range(n_boot):
        idx         = rng.integers(0, n_subj, size=n_subj)
        d_gauss_b   = eeg_g[idx] - opm_g[idx]
        d_peak_b    = eeg_p[idx] - opm_p[idx]
        dz_gauss_b  = np.mean(d_gauss_b) / (np.std(d_gauss_b, ddof=1) + 1e-12)
        dz_peak_b   = np.mean(d_peak_b) / (np.std(d_peak_b, ddof=1) + 1e-12)
        boot_diff[b] = dz_peak_b - dz_gauss_b

    ci_lo, ci_hi = np.percentile(boot_diff, [2.5, 97.5])
    excludes_zero = (ci_lo > 0) or (ci_hi < 0)
    print(f"\n  Bootstrap 95% CI for (dz_peak_finder - dz_gaussian): [{ci_lo:.3f}, {ci_hi:.3f}]  "
          f"{'-> excludes 0, methods differ' if excludes_zero else '-> includes 0, no evidence of a difference'}")

    ##### ----- Save the merged comparison table for transparency (one file per band,
    ##### ----- since EEG/OPM peak values are not directly comparable across bands)
    out_csv = os.path.join(revision, 'results_' + freqband, f'peak_finder_vs_gaussian_controls_comparison_{freqband}.csv')
    merged.to_csv(out_csv)
    print(f'\nSaved merged comparison table to: {out_csv}')


##### ----- Single combined figure: 2 rows (frequency on top, power below) x
##### ----- 4 columns (alpha-EEG, alpha-OPM, gamma-EEG, gamma-OPM). -------------
##### ----- Axis limits are shared within each (row, band) pair - i.e. alpha's
##### ----- EEG/OPM columns share one scale, gamma's EEG/OPM columns share a
##### ----- separate scale, since alpha and gamma values are not comparable.
row_specs = [
    ('pf', 'Peak frequency (Hz)'),
    ('pp', 'Peak power (dB)'),
]
col_specs = [
    ('alpha', 'EEG'), ('alpha', 'OPM'), ('gamma', 'EEG'), ('gamma', 'OPM')
]

cm              = 1 / 2.54
fig_width_cm    = 17.5   # stays under the 18 cm cap
fig_height_cm   = 9.5    # stays under the 18 cm cap
fig, axes       = plt.subplots(len(row_specs), len(col_specs),
                                figsize=(fig_width_cm * cm, fig_height_cm * cm))
###### ----- Define labels for plotting
band_label_map     = {'alpha': 'Alpha/beta', 'gamma': 'Gamma'}
modality_label_map = {'EEG': 'EEG', 'OPM': 'OPM-MEG'}

for row, (metric, metric_label) in enumerate(row_specs):
    for band in ['alpha', 'gamma']:
        merged  = merged_by_band[band]
        x_eeg, y_eeg = merged[f'{metric}_eeg_gauss'], merged[f'{metric}_eeg_peak']
        x_opm, y_opm = merged[f'{metric}_opm_gauss'], merged[f'{metric}_opm_peak']

        ##### ----- Shared axis limits across EEG/OPM for this (row, band)
        all_vals    = pd.concat([x_eeg, y_eeg, x_opm, y_opm])
        lims        = [all_vals.min(), all_vals.max()]
        pad         = 0.05 * (lims[1] - lims[0] + 1e-9)
        lims        = [lims[0] - pad, lims[1] + pad]

        col_pair = (0, 1) if band == 'alpha' else (2, 3)
        for col, (x, y, modality) in zip(col_pair, [(x_eeg, y_eeg, 'EEG'), (x_opm, y_opm, 'OPM')]):
            ax = axes[row, col]
            ax.scatter(x, y, s=10, alpha=0.75, color='tab:blue', edgecolor='k', linewidth=0.3, zorder=2)
            ax.plot(lims, lims, ls='--', color='gray', linewidth=0.7, zorder=1)
            r_s, _ = spearmanr(x, y)
            ax.set_title(f'{band_label_map[band]} {modality_label_map[modality]}\nρ = {r_s:.3f}', fontsize=6.5)
            ax.set_xlim(lims)
            ax.set_ylim(lims)
            ax.set_aspect('equal', adjustable='box')
            ax.tick_params(labelsize=5.5)

            ticks = ax.get_xticks()
            ticks = ticks[(ticks >= lims[0]) & (ticks <= lims[1])]
            ax.set_xticks(ticks)
            ax.set_yticks(ticks)


            if row == len(row_specs) - 1:
                ax.set_xlabel('Gaussian fit', fontsize=6.5)
            if col == 0:
                ax.set_ylabel('Peak-finder', fontsize=6.5)

    ##### ----- Row label (frequency vs power) in the left margin
    bbox = axes[row, 0].get_position()
    fig.text(0.02, (bbox.y0 + bbox.y1) / 2, metric_label,
              rotation=90, ha='center', va='center', fontsize=7.5)

fig.subplots_adjust(left=0.09, right=0.98, top=0.90, bottom=0.10, wspace=0.4, hspace=0.55)

print(f'\nFigure size: {fig_width_cm} x {fig_height_cm} cm')
figure_dir = r'E:\ongoing\Visgam\Results\SNR_controls\Revision_IN\figures\Figures_Supplementary\peak_finder_vs_gaussian.png'
os.makedirs(os.path.dirname(figure_dir), exist_ok=True)
fig.savefig(figure_dir, dpi=300)
plt.show()

print(f'Saved figure to: {figure_dir}')
