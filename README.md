# Visual Gamma: EEG vs. OPM-MEG

Analysis pipeline for a simultaneous EEG / OPM-MEG comparison of visually-induced gamma-band power, alpha/beta event-related desynchronization (ERD), and inter-trial phase coherence (ITPC), using a moving-grating paradigm.

## Structure

- `analysis_code/` — Python analysis pipeline (spatial filtering, time-frequency analysis, peak detection, control analyses).
- `utils_visgam/` — shared helper functions imported throughout `analysis_code/`.
- `StatisticalAnalysis.ipynb`, `StatisticalAnalysis_ITPC.ipynb` — R notebooks (linear mixed-effects models, effect sizes) for the gamma/alpha and ITPC results respectively.

## Pipeline order

1. **Spatial filters (GED):** `analysis_1_spatial_filters_et_tfr.py` (gamma/alpha), `analysis_2_spatial_filters_itpc.py` (ITPC)
2. **Peak fitting:** `analysis_3_fit_gaussians.py`
3. **Sensor-level TFR / ITPC / topographies:** `analysis_4` through `analysis_9`
4. **Virtual-channel ITPC:** `analysis_10_itpc_virtualchannel.py`
5. **Control analyses (single best sensor):** `analysis_11_best_sensor_tfr.py`, `analysis_12_best_sensor_itpc.py`
6. **Test–retest reliability:** `analysis_13_internal_consistency.py`
7. **Local-maximum peak-finding comparison:** `analysis_14_peak_finder.py`
8. **Supplementary analyses:** `supplementary_*.py`
9. **Statistics:** `StatisticalAnalysis.ipynb`, `StatisticalAnalysis_ITPC.ipynb`

Shared analysis parameters (frequency bands, time windows, fitting criteria) are centralized in `analysis_code/cfg_analysis.json`.

## Dependencies

- Python ≥3.10, MNE-Python 1.9.0, specparam (FOOOF), numpy, pandas, scipy, joblib
- R, with `lme4`, `lmerTest`, `car`, `effectsize`, `emmeans`

## Notes

- Paths are set in `utils_visgam/visgam_paths_local.py` — update these to your local data location before running.
- A few scripts (e.g. `analysis_10`, sensor-config-specific figure scripts) require manually setting the sensor configuration (`10_10` vs. `10_20`) and re-running to generate the alternate-config output.
