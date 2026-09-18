import numpy as np
import matplotlib.pyplot as plt
import mne
from scipy.stats import median_abs_deviation
import os
import pandas as pd
from scipy.signal import hilbert
from mne import create_info, io
from scipy.ndimage import uniform_filter1d
import json
from scipy.stats import zscore
from scipy.stats import mannwhitneyu

# --- Load config files for the analysis
def loadcfg(cfgdir):
    with open(cfgdir, "r") as f:
        cfg = json.load(f)
    return cfg


# --- Compute mean and SEM ---
def mean_sem(indata):
    ### ----- Dimensions have to be SUBJECTS x VARIABLE
    indata_mean = indata.mean(axis=0)
    indata_sem  = indata.std(axis=0) / np.sqrt(indata.shape[0])
    return indata_mean, indata_sem


# --- Compute mean and SD 
def mean_sd(indata):
    ### ----- Dimensions have to be SUBJECTS x VARIABLE
    indata_mean = indata.mean(axis=0)
    indata_std  = indata.std(axis=0) 
    return indata_mean, indata_std

# --- Compute Cohen's d for independent or paired samples
def cohen_d(group1, group2, paired=False):
    if paired:
        diff = np.array(group1) - np.array(group2)
        return np.mean(diff) / np.std(diff, ddof=1) 
    else:        
        mean1, mean2 = np.mean(group1), np.mean(group2)
        std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
        pooled_std = np.sqrt((std1**2 + std2**2) / 2)
        return (mean1 - mean2) / pooled_std

# --- Compute rank biserial correlation for independent samples
def rank_biserial(group1, group2):
    n1, n2 = len(group1), len(group2)
    U, _ = mannwhitneyu(group1, group2, alternative="two-sided")
    rbc = 1 - 2 * U / (n1 * n2)
    return rbc


# --- Normalize data between -1 and 1 
def minmax_normalization(indata):
    ### ----- Normalize data from -1 to 1
    ### ----- Get data: shape = (n_trials, n_channels, n_times)
    X = indata
    ### ----- Initialize array for normalized data
    X_norm = np.empty_like(X)

    ### ----- Loop through trials and channels
    for i_trial in range(X.shape[0]):
        for i_chan in range(X.shape[1]):
            x = X[i_trial, i_chan, :]
            x_min = np.min(x)
            x_max = np.max(x)
            if x_max != x_min:
                ##### Normalize to [0, 1] first, then scale to [-1, +1]
                x_norm = 2 * (x - x_min) / (x_max - x_min) - 1
            else:
                x_norm = np.zeros_like(x)  # flat signal → map to 0
            X_norm[i_trial, i_chan, :] = x_norm

    return X_norm


# --- Compute bootstrap errorbars 
def bootstrap_ci(data, n_resamples=5000, ci=95):
    ### ----- Take data shape
    n_subj, n_feat = data.shape
    ### ----- Preallocate bootstrap means
    boot_means = np.zeros((n_resamples, n_feat))
    ##### Bootstrap loop
    for b in range(n_resamples):
        ### ----- Sample subjects with replacement
        idx = np.random.randint(0, n_subj, size=n_subj)  
        boot_means[b] = data[idx].mean(axis=0)

    ### ----- Compute mean and CI
    mean_ts = data.mean(axis=0)
    alpha = (100 - ci) / 2
    lower = np.percentile(boot_means, alpha, axis=0)
    upper = np.percentile(boot_means, 100 - alpha, axis=0)

    return mean_ts, lower, upper


### --- Align timeseries polarity for virtual channel data 
def align_polarity(
    data, 
    times, 
    tmin, 
    tmax, 
    desired_sign=1, 
    zscore_mode="global", 
    method='peak' # or mean
):
    ### ----- Select time indices
    idx = np.where((times >= tmin) & (times <= tmax))[0]
    if len(idx) == 0:
        raise ValueError("No time points found in the specified window.")

    ### ----- Compute mean amplitude in the window for each subject
    if method == 'mean':
        window_mean = data[:, idx].mean(axis=1)
    elif method == 'peak':
        max_value = data[:, idx].max(axis=1)
        min_value = data[:, idx].min(axis=1)
        ##### If max_value is larger than min value then peak is positive, otherwise negative
        window_mean = min_value + max_value 
    elif method == 'prc':
        win = data[:, idx]
        max_value   = np.percentile(win, 97.5, axis=1)
        min_value   = np.percentile(win, 2.5, axis=1)
        window_mean = np.abs(max_value) - np.abs(min_value)
    else:
        raise ValueError("method must be 'mean', 'peak', or 'prc'")

    ### ----- Determine sign correction
    flip_vector = np.where(np.sign(window_mean) == desired_sign, 1, -1)

    ### ----- Apply correction
    data_aligned = data * flip_vector[:, np.newaxis]

    ### ----- Optional z-scoring
    if zscore_mode == "subject":
        data_out = zscore(data_aligned, axis=1)
    elif zscore_mode == "global":
        data_out = zscore(data_aligned, axis=None)
    else:
        data_out = data_aligned

    return data_out, flip_vector


### ----- Align timeseries for virtual channels to match modalities
def align_modalities_by_corr(
    vc_eeg, vc_opm, times, tmin, tmax,
    flip="opm",          # "opm" or "eeg"
    desired_sign=1,      # +1 to enforce positive corr
):

    vc_eeg  = np.asarray(vc_eeg)
    vc_opm  = np.asarray(vc_opm)
    times   = np.asarray(times)

    if vc_eeg.shape != vc_opm.shape:
        raise ValueError("vc_eeg and vc_opm must have the same shape.")
    if vc_eeg.ndim != 2:
        raise ValueError("Inputs must be (n_subjects, n_times).")

    idx = np.where((times >= tmin) & (times <= tmax))[0]
    if idx.size == 0:
        raise ValueError("No time points found in the specified window.")

    X           = vc_eeg[:, idx]
    Y           = vc_opm[:, idx]
    vc_eeg_out  = vc_eeg.copy()
    vc_opm_out  = vc_opm.copy()

    ### ----- Correlate data using Pearson to get the sign
    ### ----- Flip subject by subject
    for i in range(X.shape[0]):
        r = np.corrcoef(X[i], Y[i])[0, 1]
        if np.sign(r) != desired_sign:
            if flip == "opm":
                Y[i] *= -1
                vc_eeg_out[i] *= -1
            elif flip == "eeg":
                X[i] *= -1
                vc_opm_out[i] *= -1
            else:
                raise ValueError("flip must be 'opm' or 'eeg'.")
    return vc_eeg_out, vc_opm_out, times

import numpy as np
import mne


def correct_bl_trigger(raw, t=-0.75, annot_id='BL', target_prefix='STIMON', verbose=True):
    """
    Reposition annotations named `annot_id` (e.g. 'BL') to sit at a fixed
    offset `t` (in seconds) relative to the next matching annotation whose
    description starts with `target_prefix` (e.g. 'STIMON').

    Parameters
    ----------
    raw : mne.io.Raw
        Raw object containing the annotations to correct.
    t : float
        Desired offset in seconds relative to the target trigger.
        Negative values place the annotation before the target
        (e.g. t=-0.75 means 0.75s before STIMON).
    annot_id : str
        Exact description string of the annotation to reposition (e.g. 'BL').
    target_prefix : str
        Prefix used to identify the reference/target annotations
        (e.g. 'STIMON' matches 'STIMON_001', 'STIMON_002', ...).
    verbose : bool
        Print a summary of changes and warnings.

    Returns
    -------
    raw : mne.io.Raw
        The same Raw object with corrected annotations (modified in place
        and returned for convenience).
    """
    onsets = raw.annotations.onset.copy()
    durations = raw.annotations.duration.copy()
    descriptions = np.array(raw.annotations.description, dtype=object)

    src_idx = np.where(descriptions == annot_id)[0]
    tgt_idx = np.where(np.char.startswith(descriptions.astype(str), target_prefix))[0]
    tgt_onsets = onsets[tgt_idx]

    if len(src_idx) == 0:
        raise ValueError(f"No annotations found with description '{annot_id}'")
    if len(tgt_idx) == 0:
        raise ValueError(f"No annotations found with prefix '{target_prefix}'")

    new_onsets = onsets.copy()
    n_fixed = 0
    n_skipped = 0

    for i in src_idx:
        src_time = onsets[i]
        diffs = tgt_onsets - src_time

        # find nearest target occurring after the source annotation
        future = diffs[diffs > 0]
        if len(future) == 0:
            if verbose:
                print(f"Warning: no '{target_prefix}' found after "
                      f"'{annot_id}' at {src_time:.3f}s — skipped")
            n_skipped += 1
            continue

        matched_tgt_time = src_time + future.min()
        new_onsets[i] = matched_tgt_time + t
        n_fixed += 1

    new_annot = mne.Annotations(
        onset=new_onsets,
        duration=durations,
        description=descriptions,
        orig_time=raw.annotations.orig_time
    )
    raw.set_annotations(new_annot)

    if verbose:
        print(f"'{annot_id}' corrected: {n_fixed} fixed, {n_skipped} skipped "
              f"(no matching '{target_prefix}' found after them)")

    return raw