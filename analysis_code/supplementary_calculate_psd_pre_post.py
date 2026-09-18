import os
import mne
import numpy as np
from utils_visgam.visgam_paths_local import datafolders
from utils_visgam.visgam_sf import pool_eeg_10, pool_opm_10, get_stim_event_id

##### ----- Set folder for results
Results_Folder = os.path.join('Results', 'SNR_controls', 'Revision_IN')

n_subjects = len(datafolders)
print('Total sample:', n_subjects)

for nsubject in range(n_subjects):
    
    sid = datafolders[nsubject]['id']
    print('Current subject:', sid)

    ##### ----- EEG paths
    datafoldereeg = datafolders[nsubject]['eeg']
    if Results_Folder == os.path.join('Results', 'SNR_controls', 'Revision_IN'):
        datafoldereeg = os.path.join(datafoldereeg, 'revision')
    eegfile = os.path.join(datafoldereeg, 'preprocessed_3_ica_eeg.fif')

    ##### ----- OPM paths
    datafolderopm = datafolders[nsubject]['opm']
    opmfile = os.path.join(datafolderopm, 'preprocessed_3_ica.fif')

    try:
        eegdata = mne.io.read_raw(eegfile, preload=True)
        opmdata = mne.io.read_raw(opmfile, preload=True)
    except FileNotFoundError:
        print(f'  Skipping {sid}: preprocessed files not found')
        continue

    ##### ----- EEG: drop VEOG/mastoids, interpolate bads, average reference, pick 10-sensor pool
    drop_list = ['VEOG', 'LM', 'RM']
    to_drop   = [ch for ch in drop_list if ch in eegdata.ch_names]
    if len(to_drop) > 0:
        eegdata.drop_channels(to_drop)
    eegdata.interpolate_bads(reset_bads=True, mode='accurate')
    eegdata.set_eeg_reference(ref_channels='average', projection=False, ch_type='eeg')
    eegchannels = [ch for ch in pool_eeg_10 if ch in eegdata.ch_names]
    eegdata.pick(eegchannels)

    ##### ----- OPM: drop bads, pick 10-position pool
    opmdata     = opmdata.copy().drop_channels(opmdata.info['bads'])
    opmchannels = [ch for ch in pool_opm_10 if ch in opmdata.ch_names]
    opmdata.pick(opmchannels)

    ##### ----- Stimulus events
    eeg_events = get_stim_event_id(eegdata, event_name='STIMON')
    opm_events = get_stim_event_id(opmdata, event_name='STIMON')

    ##### ----- Broadband epochs: same 5 Hz highpass used for the GED input.
    ##### ----- reject_by_annotation=True keeps whatever BAD_* segments were
    ##### ----- already marked during preprocessing (ICA-related exclusions,
    ##### ----- muscle annotations, etc.) -- each device's own full trial set.
    X_eeg = mne.Epochs(eegdata.copy().filter(l_freq=5, h_freq=None),
        event_id=eeg_events, tmin=-0.75, tmax=1,
        baseline=None, proj=False, detrend=1, reject_by_annotation=True, preload=True)
    X_opm = mne.Epochs(opmdata.copy().filter(l_freq=5, h_freq=None),
        event_id=opm_events, tmin=-0.75, tmax=1,
        baseline=None, proj=False, detrend=1, reject_by_annotation=True, preload=True)

    print(f'  EEG trials retained: {len(X_eeg)} | OPM trials retained: {len(X_opm)}')

    ##### ----- Baseline (-0.75 to 0 s) vs. post-stimulus (0 to 0.75 s), matched duration
    for modality, X, datafolder_raw in [('eeg', X_eeg, datafoldereeg), ('opm', X_opm, datafolderopm)]:
        if len(X) == 0:
            print(f'  Skipping {modality} for {sid}: no trials retained')
            continue
        ##### ----- Crop data for pre and post stimulus
        pre_    = X.copy().crop(tmin=-0.75, tmax=0.0)
        post_   = X.copy().crop(tmin=0.0, tmax=0.75)

        ##### ----- Data-defined resolution: no zero-padding
        n_per_seg = pre_.get_data().shape[-1]
        fmin_plot = 5    # matches the 5 Hz highpass already applied to X_eeg/X_opm

        psd_pre  = pre_.compute_psd(method='welch', fmin=fmin_plot, fmax=150,
                                     n_per_seg=n_per_seg, n_fft=n_per_seg)
        psd_post = post_.compute_psd(method='welch', fmin=fmin_plot, fmax=150,
                                      n_per_seg=n_per_seg, n_fft=n_per_seg)

        ##### ----- Average over epochs only -- keep the per-sensor dimension so
        ##### ----- channel averaging (or per-sensor inspection) can happen later,
        ##### ----- at plot time, rather than being baked in here.
        freqs      = psd_pre.freqs
        pre_curve  = psd_pre.get_data().mean(axis=0)    # mean over epochs -> (n_channels, n_freqs)
        post_curve = psd_post.get_data().mean(axis=0)

        ##### ----- Save: one per-sensor PSD (pre, post) per subject per device
        psds_dir = os.path.join(datafolder_raw, 'psds')
        os.makedirs(psds_dir, exist_ok=True)
        np.savez(os.path.join(psds_dir, f'baseline_post_psd_{modality}.npz'),
                 freqs=freqs, pre=pre_curve, post=post_curve, n_trials=len(X),
                 ch_names=np.array(X.info['ch_names']))

    print(f'  Saved PSDs for {sid}')

print('Done.')
