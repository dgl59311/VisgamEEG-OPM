import numpy as np
import mne
from scipy.ndimage import gaussian_filter1d
from scipy.linalg import eigh
from joblib import Parallel, delayed
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, peak_widths
from utils_visgam.visgam_preprocessing import match_orig_time


### ----- Identify retained trials
def kept_trial_names(tfr):
    # Find names of events retained after rejecting trials
    code_to_name = {v: k for k, v in tfr.event_id.items()}
    return {code_to_name[c] for c in tfr.events[:, 2] if c in code_to_name}


### ----- Find common trials between two TFRs
def common_trial_names(array1, array2):
    #### Get trials
    kept_1 = kept_trial_names(array1)
    kept_2 = kept_trial_names(array2)

    #### Find common trials 
    common_names = sorted(kept_1 & kept_2)
    print(f"Common trials: n={len(common_names)}")
    return common_names


### ----- Provide trial names and return subsetted TFRs or Epochs with only those trials
def tfr_subset_by_names(tfr, names):
    
    #### Keep only names present in this TFR
    names = [n for n in names if n in tfr.event_id]
    if not names:
        raise ValueError("None of the requested names are in tfr.event_id")

    #### Map names -> codes and find matching epoch indices (in original order)
    codes = {n: tfr.event_id[n] for n in names}
    keep_idx = np.where(np.isin(tfr.events[:, 2], list(codes.values())))[0]
    if keep_idx.size == 0:
        raise ValueError("No epochs matched the requested names")

    #### Prune event_id to codes actually present
    present_codes   = set(tfr.events[keep_idx, 2])
    event_id_new    = {n: c for n, c in codes.items() if c in present_codes}
    metadata_sel    = tfr.metadata.iloc[keep_idx] if tfr.metadata is not None else None

    return mne.time_frequency.EpochsTFRArray(
        info=tfr.info.copy(),
        data=tfr.data[keep_idx],
        times=tfr.times.copy(),
        freqs=tfr.freqs.copy(),
        events=tfr.events[keep_idx],
        event_id=event_id_new,
        metadata=metadata_sel,
        method=tfr.method
    )


### ----- Same as above but for Epochs 
def epochs_subset_by_names_array(epochs, names):
    ##### Keep only names that exist in this epochs object
    names = [n for n in names if n in epochs.event_id]
    if not names:
        raise ValueError("None of the requested names are in epochs.event_id")

    ##### Map names -> codes and find matching epoch indices (in original order)
    codes = {n: epochs.event_id[n] for n in names}
    keep_idx = np.where(np.isin(epochs.events[:, 2], list(codes.values())))[0]
    if keep_idx.size == 0:
        raise ValueError("No epochs matched the requested names")

    ##### Use MNE's own indexing to build the subset; this keeps all metadata,
    return epochs[keep_idx]


### ----- Function to correct for baseline
def correct_baseline(tfrdata, baseline, islogandbaseline=False):
    #### Correct baseline
    if islogandbaseline:
        avg_tfr = mne.time_frequency.AverageTFRArray(
            info=tfrdata.info, data=np.log10(tfrdata.data).mean(axis=0), times=tfrdata.times, 
            freqs=tfrdata.freqs, method=tfrdata.method).apply_baseline(baseline, mode='mean').crop(tmin=-0.75, tmax=1)  
    else:
        avg_tfr = tfrdata.average(method='mean').apply_baseline(baseline, mode='logratio').crop(tmin=-0.75, tmax=1)  
        avg_tfr.data *= 10 # Convert to dB
    return avg_tfr
    
### ----- Function to invert TFR in order to find negative peaks as positive
def invert_tfr(tfrdata):
    inverted_data = -tfrdata.data
    return mne.time_frequency.AverageTFRArray(
        info=tfrdata.info, data=inverted_data, times=tfrdata.times, 
        freqs=tfrdata.freqs, method=tfrdata.method)

### ----- Function to convert superlet parameters to buffer size
def superlet_buffer(freq, sd_, sfreq, foi_start, foi_stop, foi_step,
                     n_cycles=3, order_start=5, order_stop=20):
    """
    Calculate the buffer (padding) required for a superlet at a given frequency,
    exactly matching aslt_64.py / aslt_64_lf.py's order interpolation logic
    (index-based np.fix over the same foi grid, not a standalone formula).
    """
    ### ----- Rebuild the same foi grid and order list that aslt() builds internally
    foi         = np.arange(foi_start, foi_stop, foi_step)
    n_freqs     = len(foi)
    order_ls    = np.fix(np.linspace(order_start, order_stop, n_freqs)).astype(int)

    ### ----- Find this frequency's position in that grid
    idx     = np.argmin(np.abs(foi - freq))
    order   = int(order_ls[idx])

    n_cycles_max    = n_cycles + (order - 1)
    sd              = (n_cycles_max / 2) * (1.0 / freq) / 2.5
    wl              = int(2 * np.floor(np.fix(sd_ * sd * sfreq) / 2) + 1)
    buffer_samples  = int(np.fix(wl / 2))
    buffer_sec      = buffer_samples / sfreq
    print(f"Frequency: {freq} Hz, Order: {order}, n_cycles_max: {n_cycles_max}, sfreq: {sfreq}, wl: {wl}, buffer: {buffer_samples} samples ({buffer_sec:.4f} s)")
    return buffer_sec, buffer_samples