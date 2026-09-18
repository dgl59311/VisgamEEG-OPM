import os
import pandas as pd
from datetime import datetime
from utils_visgam.visgam_paths_local import datafolders, wdir

##### ----- Select subfolder within Results folder (matches the rest of the pipeline)
Results_Folder  = os.path.join('Results', 'SNR_controls', 'Revision_IN')

##### ----- Filenames of the last preprocessing step used for the main analysis
EEG_FILENAME    = 'preprocessed_3_ica_eeg.fif'
OPM_FILENAME    = 'preprocessed_3_ica.fif'

records = []

for d in datafolders:
    subjid      = d['id']

    ### ----- EEG (re-directed to 'revision' subfolder, matching every other script)
    datafoldereeg   = d['eeg']
    if Results_Folder == os.path.join('Results', 'SNR_controls', 'Revision_IN'):
        datafoldereeg   = os.path.join(datafoldereeg, 'revision')
    eegfile         = os.path.join(datafoldereeg, EEG_FILENAME)

    ### ----- OPM (no redirect)
    opmfile         = os.path.join(d['opm'], OPM_FILENAME)

    ### ----- Get creation AND modification dates
    ### ----- (mtime updates when a file is rewritten; ctime does not, on Windows,
    ### ----- so a mismatch between the two flags a file that was overwritten
    ### ----- after its original creation)
    date_eeg_created = date_eeg_modified = None
    date_opm_created = date_opm_modified = None

    if os.path.exists(eegfile):
        date_eeg_created    = datetime.fromtimestamp(os.path.getctime(eegfile))
        date_eeg_modified   = datetime.fromtimestamp(os.path.getmtime(eegfile))
    else:
        print(f'Subject {subjid}: EEG file NOT FOUND at {eegfile}')

    if os.path.exists(opmfile):
        date_opm_created    = datetime.fromtimestamp(os.path.getctime(opmfile))
        date_opm_modified   = datetime.fromtimestamp(os.path.getmtime(opmfile))
    else:
        print(f'Subject {subjid}: OPM file NOT FOUND at {opmfile}')

    rewritten_eeg = (date_eeg_created is not None) and (date_eeg_created != date_eeg_modified)
    rewritten_opm = (date_opm_created is not None) and (date_opm_created != date_opm_modified)

    print(f'Subject {subjid} | EEG modified: {date_eeg_modified}'
          f'{" [REWRITTEN]" if rewritten_eeg else ""}'
          f' | OPM modified: {date_opm_modified}'
          f'{" [REWRITTEN]" if rewritten_opm else ""}')

    records.append(dict(
        subject             = subjid,
        date_eeg_modified   = date_eeg_modified,
        date_eeg_created    = date_eeg_created,
        eeg_rewritten       = rewritten_eeg,
        date_opm_modified   = date_opm_modified,
        date_opm_created    = date_opm_created,
        opm_rewritten       = rewritten_opm,
    ))

##### ----- Save to CSV
df          = pd.DataFrame(records)
outpath     = os.path.join(wdir, Results_Folder, 'preprocessing_file_dates.csv')
df.to_csv(outpath, index=False)
print(f'\nSaved to: {outpath}')