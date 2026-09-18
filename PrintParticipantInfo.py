import os 
import pandas as pd

# Directories
wdir        = r'D:\ongoing\Visgam'          # Working Directory
demog_file  = pd.read_csv(os.path.join(wdir, 'DemographicFile.csv'), index_col=0)

# Print Age et Sex
ControlData = demog_file[demog_file['SZ_EEG_INFO'] == 0]
print(ControlData['Age'].mean().round(1))
print(ControlData['Age'].std().round(1))
print(ControlData['Age'].min())
print(ControlData['Age'].max())
print(ControlData['Sex'].value_counts())
