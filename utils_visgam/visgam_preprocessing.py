# Helpers for preprocessing
import os
import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mne import create_info, io
from scipy.ndimage import uniform_filter1d
from scipy.stats import median_abs_deviation


def detect_power_outliers(psd, threshold=3.0, db_cutoff=40.0, bins=25, axis_label=''):
    """
    Detects power outliers in dB from an MNE PSD object.
    
    Parameters:
    - psd: mne.time_frequency.EpochsPSD or RawPSD (e.g. output of compute_psd)
    - threshold: z-score threshold for robust MAD-based outlier detection
    - db_cutoff: for illustration. Not used for channel rejection.
    - bins: number of histogram bins
    - axis_label: optional title/label for plot
    
    Returns:
    - List of outlier channel names
    """
    psd_data    = psd.get_data()  # shape: [n_channels, n_freqs]
    psd_db      = 10 * np.log10(psd_data * 1e30)  # Convert to dB (ref 1 fT^2/Hz)
    avg_power   = np.mean(psd_db, axis=1)

    # Compute robust z-score
    median  = np.median(avg_power)
    mad     = median_abs_deviation(avg_power, scale='normal')
    zscores = (avg_power - median) / mad

    # Outlier indices
    outlier_idx = np.where((np.abs(zscores) > threshold))[0]
    outlier_chs = [psd.ch_names[i] for i in outlier_idx]

    # Plot
    plt.figure(figsize=(6, 4))
    plt.hist(avg_power, bins=bins, color='lightgray', edgecolor='black')
    plt.axvline(median, color='blue', linestyle='-', label='Median')
    plt.axvline(median + 3*mad, color='blue', linestyle='--', label='±3 MAD')
    plt.axvline(median - 3*mad, color='blue', linestyle='--')
    plt.axvline(db_cutoff, color='black', linestyle='--', linewidth=0.5, label=f'{db_cutoff} dB')
    for idx in outlier_idx:
        plt.axvline(avg_power[idx], color='red', linestyle='-', linewidth=1)
    plt.xlabel('Power (dB) [fT²/Hz]')
    plt.ylabel('# Sensors')
    plt.title(f'{axis_label}: Power outliers')
    plt.legend()
    plt.tight_layout()
    plt.show()

    return outlier_chs


def detect_power_outliers_eeg(psd, threshold=3.0, db_cutoff=10.0, bins=25, axis_label=''):
    """
    Detects power outliers in dB µV²/Hz from an EEG PSD object.

    Parameters:
    - psd: mne.time_frequency.EpochsPSD or RawPSD (output of mne.compute_psd)
    - threshold: z-score threshold for robust MAD-based outlier detection
    - db_cutoff: for illustration. Not used for channel rejection. µV²/Hz
    - bins: histogram bins
    - axis_label: label for plot title

    Returns:
    - List of outlier EEG channel names
    """
    psd_data    = psd.get_data()  # shape: [n_channels, n_freqs]
    psd_db      = 10 * np.log10(psd_data * 1e12)  # Convert to dB µV²/Hz
    avg_power   = np.mean(psd_db, axis=1)

    median      = np.median(avg_power)
    mad         = median_abs_deviation(avg_power, scale='normal')
    zscores     = (avg_power - median) / mad

    outlier_idx = np.where(np.abs(zscores) > threshold)[0]
    outlier_chs = [psd.ch_names[i] for i in outlier_idx]

    plt.figure(figsize=(6, 4))
    plt.hist(avg_power, bins=bins, color='lightgray', edgecolor='black')
    plt.axvline(median, color='blue', linestyle='-', label='Median')
    plt.axvline(median + 3*mad, color='blue', linestyle='--', label='±3 MAD')
    plt.axvline(median - 3*mad, color='blue', linestyle='--')
    plt.axvline(db_cutoff, color='black', linestyle='--', linewidth=0.5, label=f'{db_cutoff} dB')
    for idx in outlier_idx:
        plt.axvline(avg_power[idx], color='red', linestyle='-', linewidth=1)
    plt.xlabel('Power (dB) [µV²/Hz]')
    plt.ylabel('# Channels')
    plt.title(f'{axis_label}: Power Outliers')
    plt.legend()
    plt.tight_layout()
    plt.show()

    return outlier_chs, avg_power[outlier_idx]


def rename_copied_files_to_match_vhdr(folder):
    # Find copied .vhdr
    vhdr_file = [f for f in os.listdir(folder) if f.lower().endswith('.vhdr')][0]
    vhdr_path = os.path.join(folder, vhdr_file)

    # Read expected filenames from inside .vhdr
    with open(vhdr_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    expected_eeg = next(line for line in lines if line.startswith('DataFile=')).split('=')[1].strip()
    expected_vmrk = next(line for line in lines if line.startswith('MarkerFile=')).split('=')[1].strip()

    # Rename EEG + VMRK
    for ext, expected_name in [('.eeg', expected_eeg), ('.vmrk', expected_vmrk)]:
        actual = [f for f in os.listdir(folder) if f.lower().endswith(ext)][0]
        if actual != expected_name:
            os.rename(os.path.join(folder, actual), os.path.join(folder, expected_name))
            print(f"Renamed {actual} → {expected_name}")

    # Return the unchanged vhdr_path (since we don’t rename it)
    return vhdr_path


def extract_rt_table(events, event_id, stim_label, sfreq, rawdst, subjectid, max_window_s=4.5):
    """
    Extract trial-level reaction times for a speed change detection paradigm.

    Parameters:
    - events: MNE event array
    - event_id: dict mapping annotation labels to event codes
    - stim_label: e.g. 'Stimulus/s48'
    - sfreq: sampling frequency
    - rawdst: folder where to save the CSV
    - subjectid: string identifier of the subject
    - max_window_s: maximum trial duration from stimulus onset (default 4.5s)
    """

    stim_code   = event_id.get(stim_label)
    s72_code    = event_id.get('Stimulus/s72')
    s96_code    = event_id.get('Stimulus/s96')
    s120_code   = event_id.get('Stimulus/s120')

    if stim_code is None or s72_code is None or s96_code is None or s120_code is None:
        print(f"Skipping subject {subjectid}: missing one or more required event codes.")
        return

    stim_indices = np.where(events[:, 2] == stim_code)[0]
    stim_samples = events[:, 0]

    trial_ids, codes, rts = [], [], []

    for i, stim_idx in enumerate(stim_indices):
        stim_onset = stim_samples[stim_idx]
        trial_ids.append(f"STIMON_{i+1:03d}")
        window_end = stim_onset + int(max_window_s * sfreq)

        # Look for s72 within [stim_onset, window_end]
        s72_idx = next(
            (j for j in range(stim_idx + 1, len(events))
             if stim_samples[j] <= window_end and events[j, 2] == s72_code),
            None
        )

        # If s72 is found, look for s96 between s72 and window_end
        if s72_idx is not None:
            s72_sample = stim_samples[s72_idx]
            s96_idx = next(
                (j for j in range(s72_idx + 1, len(events))
                 if stim_samples[j] <= window_end and events[j, 2] == s96_code),
                None
            )
            if s96_idx is not None:
                s96_sample = stim_samples[s96_idx]
                rt = (s96_sample - s72_sample) / sfreq
                codes.append(96)
                rts.append(f"{rt:.3f}")  # Use dot as decimal separator
                continue  # done with this trial

        # If no s72/s96 found, check for s120 (correct rejection) in window
        s120_idx = next(
            (j for j in range(stim_idx + 1, len(events))
             if stim_samples[j] <= window_end and events[j, 2] == s120_code),
            None
        )

        if s120_idx is not None:
            codes.append(120)
            rts.append(np.nan)
        else:
            codes.append(0)
            rts.append(np.nan)

    # Create dataframe and format properly
    df = pd.DataFrame({
        'TrialID': trial_ids,
        'CODE': codes,
        'RT': rts
    })

    outpath = os.path.join(rawdst, f"{subjectid}_trial_info.csv")
    df.to_csv(outpath, index=False, sep=';', encoding='utf-8')
    print(f"Reaction time table written to: {outpath}")


def match_orig_time(annotations, target_orig_time):
    # Include recording metadata
    return mne.Annotations(
        onset=annotations.onset,
        duration=annotations.duration,
        description=annotations.description,
        orig_time=target_orig_time
    )


def add_gfp_and_scaled_rms_as_misc(inputmne, picktype, rms_window_sec=0.2, match_percentile=99):
    """
    Add GFP and RMS (scaled to match GFP units) as misc channels for inspection.

    Parameters
    ----------
    inputmne : mne.io.Raw
        Bandpass-filtered raw object (e.g. 110–140 Hz).
    picktype : str or list
        Channel type(s) or names (e.g., 'eeg').
    rms_window_sec : float
        Window size in seconds for RMS smoothing (default 0.2s).
    match_percentile : float
        Percentile level used to match RMS and GFP scale (default 99).

    Returns
    -------
    raw_augmented : mne.io.Raw
        Raw with added misc channels: GFP and scaled RMS.
    """
    sfreq   = inputmne.info['sfreq']
    data    = inputmne.get_data(picks=picktype)

    # GFP (µV): std across channels at each time point
    gfp     = data.std(axis=0, keepdims=True)

    # RMS envelope across channels (with smoothing)
    window_samples  = int(sfreq * rms_window_sec)
    squared         = data ** 2
    smoothed        = uniform_filter1d(squared, size=window_samples, axis=1)
    rms             = np.sqrt(smoothed)
    total_rms       = rms.sum(axis=0, keepdims=True)

    # --- Scale RMS to match GFP units ---
    gfp_ref             = np.percentile(gfp, match_percentile)
    rms_ref             = np.percentile(total_rms, match_percentile)
    scale_factor        = gfp_ref / rms_ref if rms_ref != 0 else 1.0
    total_rms_scaled    = total_rms * scale_factor

    # Create info and RawArray objects
    info_gfp            = create_info(['GFP'], sfreq, ch_types='misc')
    info_rms            = create_info(['RMS_TotalPower_scaled'], sfreq, ch_types='emg')

    raw_gfp             = io.RawArray(gfp, info_gfp)
    raw_rms             = io.RawArray(total_rms_scaled, info_rms)

    # Copy metadata
    for raw_misc in [raw_gfp, raw_rms]:
        raw_misc.info._unlocked = True
        raw_misc.info['highpass'] = inputmne.info['highpass']
        raw_misc.info['lowpass'] = inputmne.info['lowpass']
        raw_misc.set_meas_date(inputmne.info['meas_date'])  # ensure alignment
        raw_misc.set_annotations(inputmne.annotations.copy())  # optional but good
        raw_misc.info._unlocked = False

    raw_augmented = inputmne.copy().add_channels([raw_gfp, raw_rms])
    return raw_augmented


def plot_ica_for_opm(icaout, infofile):
    topos = icaout.get_components()
    ica_ch_names = infofile['ch_names'] 
    ica_ch_types = ['eeg'] * len(ica_ch_names) # set as EEG only for plotting
    info = mne.create_info(ch_names=ica_ch_names, sfreq=250, ch_types=ica_ch_types)
    montage = mne.channels.read_dig_fif('Opm_montage.fif')
    info.set_montage(montage)
    # Get channel positions from montage
    chs = [info.get_montage().get_positions()['ch_pos'][ch] for ch in ica_ch_names]
    pos = np.array([[xyz[0], xyz[1]] for xyz in chs])  # only x, y

    # Plot in subplots
    n_components = topos.shape[1]
    n_cols = 5
    n_rows = int(np.ceil(n_components / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.5*n_cols, 2.5*n_rows))
    axes = axes.flatten()

    for i in range(n_components):
        mne.viz.plot_topomap(topos[:, i], pos=pos, axes=axes[i], show=False)
        axes[i].set_title(f'IC {i}', fontsize=10)

    # Hide unused subplots
    for j in range(n_components, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()
