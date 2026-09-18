import os 
import pandas as pd
from pathlib import Path

### ----- Directories
wdir                = r'E:\ongoing\Visgam'          # Working Directory
daten               = r'E:\ongoing\Visgam\Data'     # Data Directory
opmdata             = os.path.join(daten, 'OPMMEG')
eegdata             = os.path.join(daten, 'EEG')

### ----- Local folder to store TFR trials 
local_folder    = r'C:\Users\Dario\Desktop\DarioProjects\visgam_local'
cfg_dir         = os.path.join(wdir, 'Visgam_Code', 'analysis_code', 'cfg_analysis.json')

### ----- OPM subjects
opmsubjects     = os.listdir(opmdata)
### ----- Store subjects directories
opmsubjectsdirs = []
for subj in opmsubjects:
    subj_root = os.path.join(opmdata, subj)
    for root, dirs, files in os.walk(subj_root):
        if 'preprocessed_1.fif' in files:
            opmsubjectsdirs.append(root)
            break  # stop after finding the first match for this subject

### ----- EEG subjects
eegsubjects     = os.listdir(eegdata)
candidate_names = ['VisGam', 'Visgam', 'VisualGamma']
eegrawfilesdir  = []
eegsubjectsdirs = []

for subj in eegsubjects:
    subj_root = os.path.join(eegdata, subj)
    found = False

    ### ----- Special case for subject_hhj24
    if subj == 'subject_hhj24':
        resting_path = os.path.join(subj_root, 'RestingState')
        if os.path.isdir(resting_path):
            eegrawfilesdir.append(resting_path)
            target_dir = os.path.join(resting_path, 'visualgamma_eeg_opm', 'mne')
            eegsubjectsdirs.append(target_dir)
            found = True

    ### ----- Standard folders: VisGam, Visgam, VisualGamma
    if not found:
        for name in candidate_names:
            candidate_path = os.path.join(subj_root, name)
            if os.path.isdir(candidate_path):
                eegrawfilesdir.append(candidate_path)
                target_dir = os.path.join(candidate_path, 'visualgamma_eeg_opm', 'mne')
                eegsubjectsdirs.append(target_dir)
                found = True
                break

    if not found:
        print(f"No valid EEG data folder found for subject {subj}")

### ----- Match subjects and return datafolders
def subject_id(p):
    p = Path(p)
    for part in p.parts:
        if part.startswith("subject_"):
            return part
    raise ValueError(f"Could not find subject_ in path: {p}")

### ----- Build a lookup for OPM by subject id
opm_by_id = {subject_id(p): Path(p) for p in opmsubjectsdirs}

datafolders = []
for eeg_path in eegsubjectsdirs:
    sid = subject_id(eeg_path)
    if sid not in opm_by_id:
        continue  # or raise / log
    datafolders.append({"id": sid, "eeg": Path(eeg_path), "opm": opm_by_id[sid]})

##### Folders to store figures 
local_fig_dir                       = r'C:\Users\Dario\Desktop\DarioProjects\visgam_local\Figures'
figure_dir_gaussian                 = os.path.join(local_fig_dir, 'Gaussian_Fits') 
figure_dir_gaussian_best_sensor     = os.path.join(local_fig_dir, 'Gaussian_Best_Sensor') 
figure_dir_testretest               = os.path.join(local_fig_dir, 'TestRetest')
figure_dir_peakfinder               = os.path.join(local_fig_dir, 'PeakFinder')