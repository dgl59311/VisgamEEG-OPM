# Helpers for spatial filter analysis
import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from scipy.ndimage import gaussian_filter1d
from utils_visgam.visgam_preprocessing import match_orig_time

#### ----- Pool of sensors for OPM and EEG
pool_eeg_10     = ['5LC', '6L', '7L', '8L', '9L', '5RC', '6R', '7R', '8R', '9R']

pool_eeg_20     = ['5LC', '6L', '7L', '8L', '9L', '5RC', '6R', '7R', '8R', '9R', 
                   '6Z', '7Z', '8Z', '9Z', '4LD','4LC', '5LB', '4RD', '4RC', '5RB']

pool_eeg_30     = ['5LC', '6L', '7L', '8L', '9L', '5RC', '6R', '7R', '8R', '9R', 
                    '6Z', '7Z', '8Z', '9Z', '4LD','4LC', '5LB', '4RD', '4RC', '5RB',
                    '5L', '5R', '5Z', '3LA', '3RA', '4RB', '4LB', '2RA', '2LA', '4Z']

pool_opm_10     = ['04lbx', '04lby', '04rbx', '04rby', '06lx', '06ly', 
    '06rx', '06ry', '07lx', '07ly','07rx', '07ry','08rx', '08ry',
    '08lx', '08ly','09rx', '09ry','09lx', '09ly']


### ----- Get stimulus onset events
def get_stim_event_id(prepdata, event_name='STIMON'):
    events_full, event_id_all   = mne.events_from_annotations(prepdata)
    stim_event_id               = {k: v for k, v in event_id_all.items() if event_name in k}
    return stim_event_id

### ----- Mark segments with high peak to peak that can affect the covariance calculation
def mark_bad_ptp_segments_MAD(rawdata, eventinfo, tmin_rej, tmax_rej, dur_mark, k_channels, eeg, annot_name='BAD_MAD'):

    ### ----- Epoch data to find outliers        
    epochs_for_rej = mne.Epochs(rawdata.copy(), event_id=eventinfo, tmin=tmin_rej, tmax=tmax_rej,
        baseline=None, proj=False, picks=None, detrend=1, reject_by_annotation=True, preload=True)

    ### ----- Calculate peak to peak amplitudes
    data            = epochs_for_rej.get_data()
    ptp             = data.max(axis=2) - data.min(axis=2)

    ### ----- Robust outlier thresholds using median ± k * MAD (per channel)
    med             = np.median(ptp, axis=0)
    mad             = np.median(np.abs(ptp - med), axis=0)

    ### ----- Convert MAD to a robust SD estimate (Gaussian consistency factor)
    robust_sd       = 1.4826 * (mad)

    ### ----- Determine Thresholds
    k               = 3.5
    high_thr        = med + k * robust_sd
    low_thr         = med - k * robust_sd

    ### ----- Mark trials
    extreme_mask    = (ptp > high_thr)  # (n_epochs, n_channels)

    ### ----- Reject trials that are outliers in at least k_channels
    bad_trials      = np.where(extreme_mask.sum(axis=1) >= k_channels)[0]
    bad_trials      = np.unique(bad_trials)

    ### ----- Print how many outlier trials were found
    print('Trials exceeding MAD: ', len(bad_trials))

    ### ----- Mark in raw
    bad_event_onsets    = epochs_for_rej.events[bad_trials, 0] / epochs_for_rej.info['sfreq']
    onsets              = bad_event_onsets + tmin_rej

    ### ----- Mark bad segment
    durations           = [dur_mark] * len(onsets)
    descriptions        = [annot_name] * len(onsets)

    if eeg:
        print('EEG data')
        annots = mne.Annotations(onset=onsets, duration=durations, description=descriptions)
        annotations_user_fixed = match_orig_time(annots, rawdata.annotations.orig_time)
    else:
        annotations_user_fixed = mne.Annotations(onset=onsets, duration=durations, description=descriptions)
        
    rawdata.set_annotations(rawdata.annotations + annotations_user_fixed)
    
    ### ----- Count outliers per sensor
    extreme_counts = extreme_mask.sum(axis=0)  # shape (n_channels,)
    ch_names = epochs_for_rej.ch_names

    return rawdata, dict(zip(ch_names, extreme_counts)), bad_trials


### ----- Filter within the range of interest and mark bad trials based on peak to peak amplitudes
def sf_epoch_data_mark_ptp(prepdata, prepdataevents,
                  epoch_tmin, epoch_tmax, filt_lf, filt_hf, ptp, k_channels, 
                  datatype, skip_annotations=False):

    ### ----- Determine data type
    is_eeg = (datatype == 'eeg')
    
    ### ----- Get PTP settings
    if ptp is not None:
        if (not isinstance(ptp, (list, tuple))) or len(ptp) != 3:
            raise ValueError(f"'ptp' must be a 3-element list like [tmin, tmax, annotlen], got: {ptp}")
        ptp_tmin, ptp_tmax, ptp_annotlen = ptp
    else:
        ptp_tmin = ptp_tmax = ptp_annotlen = None
    
    ### ----- Determine if skip bad segments during filtering
    skip_ = ('edge', 'bad_acq_skip') # default
    if skip_annotations:
        skip_ = ("BAD",)

    ### ----- Filter data if at least one bound is provided
    filtdata = prepdata.copy()
    if (filt_lf is not None) or (filt_hf is not None):
        filtdata = filtdata.filter(l_freq=filt_lf, h_freq=filt_hf,
            fir_design="firwin", phase="zero-double",
            skip_by_annotation=skip_)

    # Mark bad PTP segments only if ALL PTP params are provided
    if (ptp_tmin is not None) and (ptp_tmax is not None) and (ptp_annotlen is not None):

        ### ----- Mark Baseline
        bltmin = ptp_tmin
        bltmax = ptp_tmin + ptp_annotlen
        print('Baseline ptp window: ', [bltmin, bltmax])
        filtdata, count_bl, bad_trials_pre = mark_bad_ptp_segments_MAD(
            filtdata, prepdataevents,
            tmin_rej=bltmin, tmax_rej=bltmax,
            dur_mark=ptp_annotlen, k_channels=k_channels, eeg=is_eeg, annot_name='BAD_pre')
        
        ### ----- Mark Post
        sigtmin = ptp_tmax - ptp_annotlen
        sigtmax = ptp_tmax
        print('Post ptp window: ', [sigtmin, sigtmax])
        filtdata, count_post, bad_trials_post = mark_bad_ptp_segments_MAD(
            filtdata, prepdataevents,
            tmin_rej=sigtmin, tmax_rej=sigtmax,
            dur_mark=ptp_annotlen, k_channels=k_channels, eeg=is_eeg, annot_name='BAD_post')
        
    ### ----- Create epochs from filtered data
    epochdata = mne.Epochs(
        filtdata, event_id=prepdataevents,
        tmin=epoch_tmin, tmax=epoch_tmax,
        baseline=None, proj=False, detrend=1,
        reject_by_annotation=True, preload=True
    )

    if (ptp_tmin is not None) and (ptp_tmax is not None) and (ptp_annotlen is not None):
        ### ----- Create dataframe with marked outliers outlier
        n_total_removed = len(set(bad_trials_pre) | set(bad_trials_post))

        df_outliers = (pd.DataFrame({"BAD_post": count_post, "BAD_pre": count_bl})
                        .fillna(0)
                        .astype(int)
                        .sort_index())
        df_outliers['total_trials_removed'] = n_total_removed

    else:
        df_outliers = None
        
    return filtdata, epochdata, df_outliers

### ----- Filter for ITPC
def ged_filter_evoked(epochdata, signal_win, n_components=3):
    ### ----- Set window for analysis
    tmin, tmax      = signal_win

    ### ----- total (single-trial) covariance in the window (shrinkage for stability)
    C_total         = mne.compute_covariance(epochdata, tmin=tmin, tmax=tmax,
                                     method="oas", rank="info")
    B               = C_total.data

    ### ----- Covariance of the evoked signal i
    evoked          = epochdata.average()
    t0, t1          = evoked.time_as_index([tmin, tmax])
    X               = evoked.data[:, t0:t1]  # (n_channels, n_times)
    A               = (X @ X.T) / max(1, X.shape[1])

    ### ----- GED
    evals, evecs    = eigh(A, B)
    idx             = np.argsort(evals)[::-1][:n_components]
    evals_top       = evals[idx]
    W               = evecs[:, idx]

    ### ----- Normalize so w.T @ B @ w == 1 (same normalization you use elsewhere)
    for k in range(W.shape[1]):
        denom = np.sqrt(W[:, k].T @ B @ W[:, k] + 1e-12)
        W[:, k] /= denom

    return evals_top, W

### ----- Calculate GED spatial filters
def ged_filter(epochdata, signal, noise, n_components=3):
    # signal/noise = (tmin, tmax)

    # post-stimulus (“signal”) covariance
    tmin_signal, tmax_signal = signal
    C_sig = mne.compute_covariance(
        epochdata, tmin=tmin_signal, tmax=tmax_signal,
        method='oas', rank='info'
    )

    # baseline (“noise”) covariance
    tmin_base, tmax_base = noise
    C_base = mne.compute_covariance(
        epochdata, tmin=tmin_base, tmax=tmax_base,
        method='oas', rank='info'
    )

    A = C_sig.data
    B = C_base.data

    # Generalized eigen-decomposition (evals are ascending)
    evals, evecs = eigh(A, B)

    # Indices of the top n_components eigenvalues (largest first)
    idx = np.argsort(evals)[::-1][:n_components]

    # Select top eigenvalues + corresponding eigenvectors
    evals_top = evals[idx]
    W = evecs[:, idx]  # (n_channels, n_components)

    # Normalize each spatial filter so that w.T @ B @ w == 1
    for k in range(W.shape[1]):
        denom = np.sqrt(W[:, k].T @ B @ W[:, k] + 1e-12)
        W[:, k] = W[:, k] / denom

    return evals_top, W


### ----- Match GED components across modalities
def _component_spectrum(avg_obj, fmin=30, fmax=80, tmin=0.25, tmax=0.75):
    freqs = np.asarray(avg_obj.freqs)
    times = np.asarray(avg_obj.times)
    data  = np.asarray(avg_obj.data)  # (n_comp, n_freq, n_time)
    if data.ndim != 3:
        raise ValueError(f"Expected avg_obj.data to be 3D (comp x freqs x times), got {data.shape}")

    fmask = (freqs >= fmin) & (freqs <= fmax)
    tmask = (times >= tmin) & (times <= tmax)
    if not np.any(fmask):
        raise ValueError(f"No frequencies in [{fmin}, {fmax}] Hz.")
    if not np.any(tmask):
        raise ValueError(f"No times in [{tmin}, {tmax}] s.")

    spec = data[:, fmask, :][:, :, tmask].mean(axis=2)  # mean over time
    return freqs[fmask], spec

### ----- Zscore data
def _zscore_rows(X, eps=1e-12):
    mu = X.mean(axis=1, keepdims=True)
    sd = np.maximum(X.std(axis=1, keepdims=True), eps)  # ddof=0 (np default)
    return (X - mu) / sd

### ----- Plot modality overview (TFR + spectrum)
def _plot_modality_overview(avg_obj, names_sel, n_sel, title,
                            fmin, fmax, tmin, tmax,
                            vlim=2.0, cmap="RdBu_r"):
    """
    2 x n_sel figure:
      top row: TFR imshow for each component (full freq range of avg_obj)
      bottom row: spectrum used for matching (mean over tmin..tmax, restricted to fmin..fmax)
    """
    freqs = np.asarray(avg_obj.freqs)
    times = np.asarray(avg_obj.times)
    data  = np.asarray(avg_obj.data)[:n_sel]  # (n_sel, n_freq, n_time)

    ### ----- Get only relevant freqs and spectrum for analysis
    fre_sel, spec_sel_all   = _component_spectrum(avg_obj, fmin, fmax, tmin, tmax)
    spec_sel                = spec_sel_all[:n_sel]

    fig, axes = plt.subplots(2, n_sel, figsize=(3.2*n_sel, 6))
    if n_sel == 1:
        axes = np.array([[axes[0]], [axes[1]]])

    last_im = None
    for c in range(n_sel):
        ax0 = axes[0, c]
        last_im = ax0.imshow(
            data[c], origin="lower", aspect="auto",
            extent=[times[0], times[-1], freqs[0], freqs[-1]],
            vmin=-vlim, vmax=vlim, cmap=cmap
        )
        ax0.set_title(names_sel[c])
        ax0.axvline(tmin, linestyle="--", linewidth=1)
        ax0.axvline(tmax, linestyle="--", linewidth=1)
        ax0.axhspan(fmin, fmax, alpha=0.15)
        ax0.set_xlabel("Time (s)")
        if c == 0:
            ax0.set_ylabel("Frequency (Hz)")

        ax1 = axes[1, c]
        ax1.plot(fre_sel, spec_sel[c])
        ax1.set_xlim(fmin, fmax)
        ax1.set_xlabel("Frequency (Hz)")
        if c == 0:
            ax1.set_ylabel("ΔPower")
        ax1.set_title(f"Mean {tmin:.2f}–{tmax:.2f}s")

    fig.suptitle(title, y=1.02, fontsize=14)
    cbar = fig.colorbar(last_im, ax=axes[0, :].ravel().tolist(), fraction=0.015, pad=0.02)
    cbar.set_label("TFR value (Δ)")
    plt.tight_layout()
    plt.show()

### ----- Compare spectra of GED components
def pairwise_component_match_with_plots(
    avg_eeg, avg_opm,
    fmin=35.0, fmax=70.0,
    tmin=0.25, tmax=0.75,
    n_eeg=3, n_opm=3, smooth_sigma=0,
    eeg_names=None, opm_names=None,
    zscore_within_band=True,
    prefer_first_if_r00_ge=0.5,     # tie-break rule
    vlim=2.0, cmap="RdBu_r",
    plot_overviews=True,
    plot_best_match=True,
    show_corr=True
):
    # spectra for matching
    fre_eeg, spec_eeg_all = _component_spectrum(avg_eeg, fmin, fmax, tmin, tmax)
    fre_opm, spec_opm_all = _component_spectrum(avg_opm, fmin, fmax, tmin, tmax)

    # no interpolation: enforce same freq grid
    if not (len(fre_eeg) == len(fre_opm) and np.allclose(fre_eeg, fre_opm)):
        raise ValueError("EEG and OPM frequency grids differ; set them identical or re-enable interpolation.")
    fre_used = fre_eeg

    if eeg_names is None:
        eeg_names = [f"GED_{i+1}" for i in range(spec_eeg_all.shape[0])]
    if opm_names is None:
        opm_names = [f"GED_{j+1}" for j in range(spec_opm_all.shape[0])]

    ### ----- Select only available components
    n_eeg = min(n_eeg, spec_eeg_all.shape[0])
    n_opm = min(n_opm, spec_opm_all.shape[0])

    spec_eeg        = spec_eeg_all[:n_eeg].copy()
    spec_opm        = spec_opm_all[:n_opm].copy()
    eeg_names_sel   = eeg_names[:n_eeg]
    opm_names_sel   = opm_names[:n_opm]

    ### ----- Make a copy of spectra for plotting
    spec_eeg_plot   = spec_eeg.copy()
    spec_opm_plot   = spec_opm.copy()

    ### ----- Smooth spectra
    if smooth_sigma and smooth_sigma > 0:
        freq_step           = fre_used[1] - fre_used[0]
        smooth_sigma_bins   = smooth_sigma / freq_step
        spec_eeg_plot       = gaussian_filter1d(spec_eeg_plot, sigma=smooth_sigma_bins, axis=1)
        spec_opm_plot       = gaussian_filter1d(spec_opm_plot, sigma=smooth_sigma_bins, axis=1)
        spec_eeg            = gaussian_filter1d(spec_eeg, sigma=smooth_sigma_bins, axis=1)
        spec_opm            = gaussian_filter1d(spec_opm, sigma=smooth_sigma_bins, axis=1)


    ### ----- Compute correlation matrix=
    if zscore_within_band:
        A = _zscore_rows(spec_eeg)
        B = _zscore_rows(spec_opm)
    else:
        ### ----- no zscore: use raw power levels 
        A = spec_eeg
        B = spec_opm
    ### ----- Compute Pearson correlation
    R   = np.corrcoef(A, B)[:n_eeg, n_eeg:]

    # selection rule
    if (n_eeg >= 1) and (n_opm >= 1) and (R[0, 0] >= prefer_first_if_r00_ge):
        i_best, j_best = 0, 0
        reason = f"forced GED_1 because R[0,0]={R[0,0]:.3f} ≥ {prefer_first_if_r00_ge}"
    else:
        i_best, j_best = np.unravel_index(np.nanargmax(R), R.shape)
        reason = "argmax correlation"

    best_eeg_name   = eeg_names_sel[i_best]
    best_opm_name   = opm_names_sel[j_best]
    best_r          = float(R[i_best, j_best])

    ### ----- Figure 1: correlation matrix
    if show_corr:
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(R, vmin=-1, vmax=1, aspect="equal", cmap="RdBu_r")
        ax.set_title(f"EEG vs OPM component correlation")
        ax.set_xlabel("OPM GED component")
        ax.set_ylabel("EEG GED component")
        ax.set_xticks(np.arange(n_opm))
        ax.set_yticks(np.arange(n_eeg))
        ax.set_xticklabels(opm_names_sel, rotation=45, ha="right")
        ax.set_yticklabels(eeg_names_sel)
        for i in range(n_eeg):
            for j in range(n_opm):
                ax.text(j, i, f"{R[i, j]:.2f}", ha="center", va="center", fontsize=9)
        ax.plot([j_best], [i_best], marker="s", markersize=14, markerfacecolor="none",
                markeredgewidth=2)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label("Correlation")
        plt.tight_layout()
        plt.show()
        print(f"Selected: {best_eeg_name} ↔ {best_opm_name} (r={best_r:.3f}); {reason}")

    ### ----- Figure 2 & 3: modality overviews (imshow + spectrum)
    if plot_overviews:
        _plot_modality_overview(avg_eeg, eeg_names_sel, n_eeg,
                                "EEG GED components (TFR + spectrum in window)",
                                fmin, fmax, tmin, tmax, vlim=vlim, cmap=cmap)
        _plot_modality_overview(avg_opm, opm_names_sel, n_opm,
                                "OPM GED components (TFR + spectrum in window)",
                                fmin, fmax, tmin, tmax, vlim=vlim, cmap=cmap)

    # Figure 4: best-match TFR side-by-side + spectra overlay
    if plot_best_match:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(fre_used, spec_eeg_plot[i_best], label=f"EEG {best_eeg_name}")
        ax.plot(fre_used, spec_opm_plot[j_best], label=f"OPM {best_opm_name}")
        ax.set_xlim(fmin, fmax)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("ΔPower")
        ax.set_title(f"Best-match spectra in window (selection r={best_r:.3f})")
        ax.legend()
        plt.tight_layout()
        plt.show()

        freqs_eeg = np.asarray(avg_eeg.freqs); times_eeg = np.asarray(avg_eeg.times)
        freqs_opm = np.asarray(avg_opm.freqs); times_opm = np.asarray(avg_opm.times)
        data_eeg = np.asarray(avg_eeg.data)[i_best]
        data_opm = np.asarray(avg_opm.data)[j_best]

        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        im0 = axes[0].imshow(
            data_eeg, origin="lower", aspect="auto",
            extent=[times_eeg[0], times_eeg[-1], freqs_eeg[0], freqs_eeg[-1]],
            vmin=-vlim, vmax=vlim, cmap=cmap
        )
        axes[0].set_title(f"EEG {best_eeg_name}")
        axes[0].set_xlabel("Time (s)")
        axes[0].set_ylabel("Frequency (Hz)")
        axes[0].axvline(tmin, linestyle="--", linewidth=1)
        axes[0].axvline(tmax, linestyle="--", linewidth=1)
        axes[0].axhspan(fmin, fmax, alpha=0.15)

        im1 = axes[1].imshow(
            data_opm, origin="lower", aspect="auto",
            extent=[times_opm[0], times_opm[-1], freqs_opm[0], freqs_opm[-1]],
            vmin=-vlim, vmax=vlim, cmap=cmap
        )
        axes[1].set_title(f"OPM {best_opm_name}")
        axes[1].set_xlabel("Time (s)")
        axes[1].set_ylabel("Frequency (Hz)")
        axes[1].axvline(tmin, linestyle="--", linewidth=1)
        axes[1].axvline(tmax, linestyle="--", linewidth=1)
        axes[1].axhspan(fmin, fmax, alpha=0.15)

        fig.suptitle(f"Best-matched components TFR (shared vlim ±{vlim})", y=1.02)
        fig.colorbar(im1, ax=axes.ravel().tolist(), fraction=0.03, pad=0.02).set_label("TFR value (Δ)")
        plt.tight_layout()
        plt.show()

    return {
        "R": R,
        "best_eeg_name": best_eeg_name,
        "best_opm_name": best_opm_name,
        "best_eeg_index": i_best,
        "best_opm_index": j_best,
        "best_r": best_r,
        "selection_reason": reason,
        "eeg_names": eeg_names_sel,
        "opm_names": opm_names_sel,
        "freqs_used": fre_used,
    }

### ----- Take dataframe and select the best component based on peak frequency
def select_best_component_by_pf_retest(df, modalities=("eeg", "opm"), components=(0, 1, 2)):
    df = df.copy()

    for mod in modalities:
        # Compute absolute test-retest PF differences for each component
        diff_cols = []
        for comp in components:
            col = f"{mod}_pfdiff_component_{comp}"
            df[col] = (
                df[f"pf_{mod}_test_component_{comp}"] -
                df[f"pf_{mod}_retest_component_{comp}"]
            ).abs()
            diff_cols.append(col)

        # Best component per subject
        diff_mat    = df[diff_cols].to_numpy(dtype=float)

        # default: component with smallest PF difference
        best_idx    = np.nanargmin(diff_mat, axis=1)
        best_comp   = np.array(components)[best_idx]
        best_diff   = np.nanmin(diff_mat, axis=1)

        # override: keep component 0 if its test-retest difference is <= 5 Hz
        comp0_diff  = df[f"{mod}_pfdiff_component_0"].to_numpy(dtype=float)
        use_comp0   = comp0_diff <= 5

        best_comp[use_comp0] = 0
        best_diff[use_comp0] = comp0_diff[use_comp0]

        df[f"best_{mod}_component"] = best_comp
        df[f"best_{mod}_pfdiff"] = best_diff

        # Copy selected values into new columns
        for measure in ["pf", "pp", "amp"]:
            for split in ["test", "retest"]:
                out_col = f"{measure}_{mod}_{split}"
                df[out_col] = np.nan

                for comp in components:
                    mask = df[f"best_{mod}_component"] == comp
                    in_col = f"{measure}_{mod}_{split}_component_{comp}"
                    df.loc[mask, out_col] = df.loc[mask, in_col]

        # Optional: remove temporary diff columns
        df = df.drop(columns=diff_cols)

    return df

