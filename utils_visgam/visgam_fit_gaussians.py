# Functions For Peak Fitting Algorithm
import numpy as  np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, peak_widths
from scipy.optimize import curve_fit
from scipy.signal.windows import tukey


### ----- Return candidate peaks
def candidate_peaks(inspec, freqband, minfreq, maxfreq, minwidth_hz, minsep_hz,
                    sig_min, sig_max, default_sig=4.5, verbose=False):

    inspec      = np.asarray(inspec, float)
    freqband    = np.asarray(freqband, float)

    if inspec.ndim != 1 or freqband.ndim != 1 or inspec.size != freqband.size:
        raise ValueError("inspec and freqband must be 1D arrays of the same length.")

    df = float(np.median(np.diff(freqband))) if freqband.size > 1 else 1.0

    peaks, _ = find_peaks(inspec)
    if verbose:
        print('Available peaks:', peaks)

    mu1_0  = float(freqband[int(np.argmax(inspec))])
    sig1_0 = float(np.clip(default_sig, sig_min, sig_max))
    mu2_0  = None
    sig2_0 = None

    cands_freq    = np.array([], dtype=float)
    cands_pow     = np.array([], dtype=float)
    cands_fwhm_hz = np.array([], dtype=float)

    if peaks.size > 0:
        ##### Rel_height 0.5 to match gaussian fit FWHM definition
        widths_bins     = peak_widths(inspec, peaks, rel_height=0.5)[0]
        widths_hz       = widths_bins * df
        ##### Get values for candidate peaks
        cand_freq_all   = freqband[peaks]
        cand_pow_all    = inspec[peaks]
        ##### Ensure validity for candidate peaks
        valid = (
            (widths_hz >= float(minwidth_hz)) &
            (cand_freq_all >= float(minfreq)) &
            (cand_freq_all <= float(maxfreq))
        )

        if np.any(valid):
            cands_freq      = cand_freq_all[valid]
            cands_pow       = cand_pow_all[valid]
            cands_fwhm_hz   = widths_hz[valid]

            order           = np.argsort(cands_pow)[::-1]
            cands_freq      = cands_freq[order]
            cands_pow       = cands_pow[order]
            cands_fwhm_hz   = cands_fwhm_hz[order]

            mu1_0           = float(cands_freq[0])
            fwhm1           = float(cands_fwhm_hz[0])
            sig1_0          = float(np.clip(fwhm1 / 2.35482, sig_min, sig_max))

            if cands_freq.size > 1:
                sep_ok      = np.abs(cands_freq - mu1_0) >= float(minsep_hz)
                sep_ok[0]   = False
                if np.any(sep_ok):
                    mu2_0   = float(cands_freq[sep_ok][0])
                    idx2    = int(np.where(np.isclose(cands_freq, mu2_0))[0][0])
                    fwhm2   = float(cands_fwhm_hz[idx2])
                    sig2_0  = float(np.clip(fwhm2 / 2.35482, sig_min, sig_max))

    return dict(
        mu1_0=mu1_0, sig1_0=sig1_0, mu2_0=mu2_0, sig2_0=sig2_0,
        cands_freq=cands_freq, cands_pow=cands_pow, cands_fwhm_hz=cands_fwhm_hz,
        found_any=bool(cands_freq.size > 0), found_second=bool(mu2_0 is not None)
    )

### ----- Metrics for model selection
def adj_r2(y_true, y_fit, k_params, mask=None):
    if mask is None:
        mask = np.ones_like(y_true, dtype=bool)
    y_t = y_true[mask]
    y_f = y_fit[mask]
    resid = y_t - y_f
    RSS = float(np.sum(resid ** 2))
    y_bar = float(np.mean(y_t))
    TSS = float(np.sum((y_t - y_bar) ** 2) + 1e-20)
    r2 = 1.0 - RSS / TSS
    n = y_t.size
    adj = 1.0 - (1.0 - r2) * (n - 1) / max(1, (n - k_params - 1))
    return float(adj), RSS

def bic_from_RSS(RSS, n, k_params):
    return float(n * np.log(RSS / n + 1e-300) + k_params * np.log(n + 1e-300))

### ----- Peak fitting function
def fit_peaks_tfr(
    tfr,
    fit_range,
    peak_range,
    time_range,
    min_sep,
    sdgaussian_range,
    change_slope,
    plot=False,
    is_fit_negative=False,
    edge_margin=2,
    smooth_sigma=1,
    criterion='adjr2',
    dprime_threshold=0.8,
    downweight_edges=False
):
    ### ----- Get parameters from inputs
    fmin, fmax         = fit_range
    peak_min, peak_max = peak_range
    tmin, tmax         = time_range
    sig_min, sig_max   = sdgaussian_range

    ### ----- Pull data without modifying original object
    data = np.asarray(tfr.data)

    ### ----- Get only frequencies within fit_range
    fmask = (tfr.freqs >= fmin) & (tfr.freqs <= fmax)
    if not np.any(fmask):
        raise ValueError("No frequency bins within [fmin, fmax].")
    f = np.asarray(tfr.freqs)[fmask].astype(float)

    ### ----- Get median frequency for reference (for slope parameterization)
    f_ref = float(np.median(f))

    ### ----- Time indices
    t0, t1 = tfr.time_as_index([tmin, tmax])
    if not (0 <= t0 < t1 <= data.shape[-1]):
        raise ValueError("Invalid time window: no samples within [tmin, tmax].")

    ### ----- Build spectrum: average over channels (if present) and time window
    if data.ndim == 3:  # (n_ch, n_freq, n_time)
        spec = data.mean(axis=0)[fmask, t0:t1].mean(axis=1)
    else:               # (n_freq, n_time)
        spec = data[fmask, t0:t1].mean(axis=1)

    ### ----- Smooth spectrum 
    if smooth_sigma and smooth_sigma > 0:
        freq_step           = f[1] - f[0]
        smooth_sigma_bins   = smooth_sigma / freq_step
        spec                = gaussian_filter1d(spec, sigma=smooth_sigma_bins)

    ### ----- Peak-center bounds with edge margin
    mu_lo = peak_min + edge_margin
    mu_hi = peak_max - edge_margin
    if mu_lo >= mu_hi:
        raise ValueError("Edge margin too large relative to [peak_min, peak_max].")

    ### ----- Define metric mask/Calculate model fit within the peak search range
    metric_mask = (f >= mu_lo) & (f <= mu_hi)
    if not np.any(metric_mask):
        raise ValueError("No bins within effective peak search range after edge_margin.")

    ### ----- Models    
    ##### Baseline model
    def base(x, b0, b1):
        return b0 + b1 * (x - f_ref)
    ##### Gaussian function
    def g(x, A, mu, sig):
        return A * np.exp(-0.5 * ((x - mu) / sig) ** 2)
    ##### Model 1: baseline only
    def m1(x, b0, b1):
        return base(x, b0, b1)
    ##### Model 2: baseline + 1 Gaussian
    def m2(x, b0, b1, A1, mu1, sig1):
        return base(x, b0, b1) + g(x, A1, mu1, sig1)
    ##### Model 3: baseline + 2 Gaussians
    def m3(x, b0, b1, A1, mu1, sig1, A2, mu2, sig2):
        return base(x, b0, b1) + g(x, A1, mu1, sig1) + g(x, A2, mu2, sig2)

    ### ----- Weights for curve_fit/Prioritize data within the peak range
    ##### Downweight frequencies near the edges of the fit range
    if downweight_edges:
        fit_lo, fit_hi      = f[0], f[-1]
        peak_lo, peak_hi    = mu_lo, mu_hi
        if np.isclose(fit_lo, peak_lo) and np.isclose(fit_hi, peak_hi):
            edge_weights        = np.ones(len(f))
        else:
            downweight_width    = (mu_lo - fit_lo) + (fit_hi - mu_hi)
            alpha_tukey         = downweight_width / (fit_hi - fit_lo)
            edge_weights        = tukey(len(f), alpha_tukey)
    else:
        edge_weights        = np.ones(len(f))
    ##### Set weights for model fits
    sc              = spec - spec.min()
    denom           = sc.max() + 1e-12
    w_lin           = sc / denom
    w_total         = w_lin * edge_weights
    sigma_w         = 1.0 / (w_total + 1e-5)

    ##### Baseline bounds
    
    if is_fit_negative:
        ##### For ERD initial baseline starts near the lower end
        b0_min, b0_max  = np.percentile(spec, (1, 95))
        b0_p0           = np.percentile(spec, 2.5)  # Initial
    else:
        b0_min, b0_max  = np.percentile(spec, (1, 95))
        b0_p0           = np.percentile(spec, 2.5)  # Initial

    ##### Slope bounds (assumes relative change units)
    max_total       = 10 * np.log10(1 + float(change_slope))
    b1_hi           = max_total / (fmax - fmin)
    b1_lo           = -b1_hi
    b1_p0           = 0.0 # Initial

    ##### Amplitude bounds
    A_min           = 0.0 # Do not accept peaks below 0

    ##### Set initial parameters for the models
    b_poly          = ([b0_min, b1_lo], [b0_max, b1_hi])
    b_1g            = ([b0_min, b1_lo, A_min,  mu_lo, sig_min],
                        [b0_max, b1_hi, np.inf, mu_hi, sig_max])
    b_2g            = ([b0_min, b1_lo, A_min,  mu_lo, sig_min, A_min,  mu_lo, sig_min],
                        [b0_max, b1_hi, np.inf, mu_hi, sig_max, np.inf, mu_hi, sig_max])

    ##### Get candidate peaks for initialization of Gaussian fits
    band_mask   = (f >= peak_min) & (f <= peak_max)
    f_band      = f[band_mask]
    s_band      = spec[band_mask]
    if f_band.size == 0:
        raise ValueError("No bins within peak_range inside fit_range.")

    minwidth_hz = 2.35482 * sig_min
    specpeaks = candidate_peaks(
        s_band, f_band,
        minfreq=mu_lo, maxfreq=mu_hi,
        minwidth_hz=minwidth_hz, minsep_hz=min_sep,
        sig_min=sig_min, sig_max=sig_max
    )
    ##### Set initial parameters for Gaussian fits
    mu1_0, sig1_0 = specpeaks['mu1_0'], specpeaks['sig1_0']
    mu2_0, sig2_0 = specpeaks['mu2_0'], specpeaks['sig2_0']

    if is_fit_negative:
        ##### Initial amplitude based on dynamic range - alpha/beta
        A1_0 = float(max(1e-6, np.percentile(s_band, 95) - np.percentile(s_band, 5)))
    else:
        ##### Initial amplitude based on distance from median - gamma baseline closer to zero
        A1_0 = float(max(1e-6, np.percentile(s_band, 95) - np.median(s_band)))

    ##### For 2nd peak, assume it is smaller than the 1st one for initialization
    A2_0    = float(max(1e-6, 0.75 * A1_0))

    ##### ---- Fit models
    results     = {}
    edge_tol    = 1e-6

    ##### Fit model 1: only polynomial
    try:
        popt1, _    = curve_fit(m1, f, spec, p0=[b0_p0, b1_p0],
                             bounds=b_poly, sigma=sigma_w,
                             absolute_sigma=False, maxfev=8000)
        y1              = m1(f, *popt1)
        ##### ----- Calculate model fits considering 2 free parameters
        r2_1, RSS1      = adj_r2(spec, y1, 2, mask=metric_mask)
        bic1            = bic_from_RSS(RSS1, int(metric_mask.sum()), 2)
        results['poly'] = dict(params=popt1, yfit=y1, adjr2=r2_1, bic=bic1, k=2, valid=True)
    except Exception:
        results['poly'] = dict(params=None, yfit=None, adjr2=-np.inf, bic=np.inf, k=2, valid=False)

    ##### Fit model 2: polynomial + 1 Gaussian
    try:
        popt2, _            = curve_fit(m2, f, spec,
                                p0=[b0_p0, b1_p0, A1_0, mu1_0, sig1_0],
                                bounds=b_1g, sigma=sigma_w,
                                absolute_sigma=False, maxfev=30000)
        y2                  = m2(f, *popt2)
        b0, b1, A1, mu1, s1 = popt2
        ##### ----- Check if peaks are valid
        at_edge_1g          = (np.isclose(mu1, mu_lo, atol=edge_tol) or np.isclose(mu1, mu_hi, atol=edge_tol))
        baseline_at_mu1     = b0 + b1 * (mu1 - f_ref)
        peak_model_db       = baseline_at_mu1 + A1
        print(peak_model_db)
        ##### ----- Calculate model fits
        r2_2, RSS2          = adj_r2(spec, y2, 5, mask=metric_mask)
        bic2                = bic_from_RSS(RSS2, int(metric_mask.sum()), 5)
        results['poly+g1']  = dict(params=popt2, yfit=y2, adjr2=r2_2, bic=bic2, k=5,
                                  valid=(peak_model_db > 0) and (not at_edge_1g))
    except Exception:
        results['poly+g1'] = dict(params=None, yfit=None, adjr2=-np.inf, bic=np.inf, k=5, valid=False)

    ##### Fit model 3: polynomial + 2 Gaussians
    try:
        ##### ----- Ensure we have valid initial parameters for 2-G fit
        if (mu2_0 is None) or (sig2_0 is None):
            raise RuntimeError("No 2nd peak init; skipping 2-G fit.")

        popt3, _ = curve_fit(m3, f, spec,
                            p0=[b0_p0, b1_p0, A1_0, mu1_0, sig1_0, A2_0, mu2_0, sig2_0],
                            bounds=b_2g, sigma=sigma_w,
                            absolute_sigma=False, maxfev=60000)
        y3      = m3(f, *popt3)
        b0, b1, A1, mu1, s1, A2, mu2, s2 = popt3

        # Calculate modeled dB at each peak (including cross-term)
        baseline1   = b0 + b1 * (mu1 - f_ref)
        baseline2   = b0 + b1 * (mu2 - f_ref)
        cross1      = A2 * np.exp(-0.5 * ((mu1 - mu2) / s2) ** 2)
        cross2      = A1 * np.exp(-0.5 * ((mu2 - mu1) / s1) ** 2)
        peak1_db    = baseline1 + A1 + cross1
        peak2_db    = baseline2 + A2 + cross2

        too_close_abs = (abs(mu2 - mu1) < min_sep)
        dprime = abs(mu2 - mu1) / (np.sqrt(0.5 * (s1**2 + s2**2)) + 1e-12)
        dprime_fail = (dprime < dprime_threshold) if (dprime_threshold and dprime_threshold > 0) else False

        at_edge_2g = (
            np.isclose(mu1, mu_lo, atol=edge_tol) or np.isclose(mu1, mu_hi, atol=edge_tol) or
            np.isclose(mu2, mu_lo, atol=edge_tol) or np.isclose(mu2, mu_hi, atol=edge_tol)
        )

        valid_2g = (
            (peak1_db > 0) and (peak2_db > 0) and
            (not too_close_abs) and (not dprime_fail) and (not at_edge_2g)
        )

        r2_3, RSS3 = adj_r2(spec, y3, 8, mask=metric_mask)
        bic3 = bic_from_RSS(RSS3, int(metric_mask.sum()), 8)

        results['poly+g2'] = dict(params=popt3, yfit=y3, adjr2=r2_3, bic=bic3, k=8,
                                  valid=bool(valid_2g), dprime=float(dprime))
    except Exception:
        results['poly+g2'] = dict(params=None, yfit=None, adjr2=-np.inf, bic=np.inf, k=8, valid=False)


    ##### Select best model
    candidates  = {k: v for k, v in results.items() if v.get('valid', False)}
    if not candidates:
        raise RuntimeError("All model fits invalid.")

    crit = criterion.lower()
    if crit == 'bic':
        best_name   = min(candidates, key=lambda k: candidates[k]['bic'])
        best_metric = candidates[best_name]['bic']
        r2_metric   = candidates[best_name]['adjr2']
    elif crit in ('adjr2', 'r2', 'adj_r2'):
        best_name   = max(candidates, key=lambda k: candidates[k]['adjr2'])
        best_metric = candidates[best_name]['adjr2']
        r2_metric   = candidates[best_name]['adjr2']
    else:
        raise ValueError("criterion must be 'bic' or 'adjr2'.")

    ##### Get best model
    best = results[best_name]
    popt = best['params']
    yfit = best['yfit']

    ##### ----- Extract parameters and peak frequencies from best model
    params_out = {}
    peak_freqs = []

    if best_name == 'poly':
        s_band_full = spec[(f >= mu_lo) & (f <= mu_hi)]
        f_band_full = f[(f >= mu_lo) & (f <= mu_hi)]
        peak_freqs = [float(f_band_full[int(np.argmax(s_band_full))])]
        b0, b1 = popt
        params_out = {'b0': float(b0), 'b1': float(b1), 'f_ref': f_ref}

    elif best_name == 'poly+g1':
        b0, b1, A1, mu1, s1 = popt
        params_out = {'b0': float(b0), 'b1': float(b1), 'f_ref': f_ref,
                      'A1': float(A1), 'mu1': float(mu1), 'sig1': float(s1)}
        peak_freqs = [float(mu1)]

    else:  # poly+g2
        b0, b1, A1, mu1, s1, A2, mu2, s2 = popt
        params_out = {'b0': float(b0), 'b1': float(b1), 'f_ref': f_ref,
                      'A1': float(A1), 'mu1': float(mu1), 'sig1': float(s1),
                      'A2': float(A2), 'mu2': float(mu2), 'sig2': float(s2)}
        peak_freqs = sorted([float(mu1), float(mu2)])

    fig = None
    if plot:
        fig, ax = plt.subplots(figsize=(6, 3))
        lbl = 'Spectrum (smoothed)' if (smooth_sigma and smooth_sigma > 0) else 'Spectrum'
        ax.plot(f, spec, lw=1.5, label=lbl)
        if yfit is not None:
            ax.plot(f, yfit, lw=1.5, ls='--', label=f"Fit ({best_name}, {criterion.upper()}={best_metric:.3f})")
        for pf in peak_freqs:
            ax.axvline(pf, ls=':', lw=1)
        ax.set_xlim(fmin, fmax)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power (a.u.)')  # don't claim dB unless guaranteed
        plt.legend(frameon=False)
        plt.tight_layout()
        plt.show(block=False)


    ##### Return fit metrics for all models (with NaN for invalid fits, and dprime for poly+g2 if available)
    metrics_all = {
        k: {m: (float(v[m]) if np.isfinite(v[m]) else v[m]) for m in ('adjr2', 'bic') if m in v}
        for k, v in results.items()
    }
    if 'poly+g2' in results and 'dprime' in results['poly+g2']:
        metrics_all['poly+g2']['dprime'] = float(results['poly+g2']['dprime'])

    return {
        'model': best_name,
        'peak_freqs': peak_freqs,
        'params': params_out,
        'metric': float(best_metric),
        'metric_r2': float(r2_metric),
        'criterion': criterion.lower(),
        'metrics_all': metrics_all,
        'freqs': f,
        'spec': spec,
        'fit': yfit
    }, fig 

# Get Peak Results and Extract Peak for Comparison Across Modalities
def extract_peak_info(fit_result):
    """
    Returns:
      - peaks: list of dicts with
          freq_fit, amp_fit, baseline_at_mu, model_at_mu, freq_data, data_at_mu
      - chosen peak summary:
          peak_freq_fit, peak_freq_data, peak_amp_fit, peak_model_db, peak_data_db
      - fit metrics passthrough:
          fit_model, fit_criterion, fit_metric_best, fit_adj_r2_best, fit_metrics_all

    Notes:
      - fit_adj_r2_best is the *adjusted R^2* of the selected model, computed in the gamma band
        (because your fitter computed metrics using metric_mask).
    """
    model  = fit_result['model']
    params = fit_result['params']
    f      = np.asarray(fit_result['freqs'], float)
    spec   = np.asarray(fit_result['spec'],  float)

    if 'b0' in params and 'b1' in params:
        b0 = float(params['b0']); b1 = float(params['b1'])
        f_ref = float(params.get('f_ref', 0.0))
        baseline = lambda x: b0 + b1 * (x - f_ref)
    else:
        raise ValueError("No baseline params found in fit_result['params'] (expected b0/b1[/f_ref] or a0/a1).")


    def nearest_data_value(mu):
        idx = int(np.argmin(np.abs(f - mu)))
        return float(f[idx]), float(spec[idx])

    def g(x, A, mu, sig):
        return A * np.exp(-0.5 * ((x - mu) / sig) ** 2)

    out = {
        'model': model,
        'peaks': [],
        # ---- fit metrics passthrough
        'fit_model': fit_result.get('model', None),
        'fit_criterion': fit_result.get('criterion', None),
        'fit_metric_best': float(fit_result['metric']) if 'metric' in fit_result else None,
        'fit_adj_r2_best': float(fit_result['metric_r2']) if 'metric_r2' in fit_result else None,
        'fit_metrics_all': fit_result.get('metrics_all', None),
    }

    if model == 'poly':
        mu = float(fit_result['peak_freqs'][0])
        f_data, data_at_mu = nearest_data_value(mu)
        base_at_mu = float(baseline(mu))
        out['peaks'].append({
            'freq_fit': mu,
            'amp_fit': None,
            'baseline_at_mu': base_at_mu,
            'model_at_mu': base_at_mu,
            'freq_data': f_data,
            'data_at_mu': data_at_mu,
        })

    elif model == 'poly+g1':
        A1              = float(params['A1'])
        mu1             = float(params['mu1'])
        s1              = float(params['sig1'])

        base_at_mu      = float(baseline(mu1))
        model_at_mu     = float(base_at_mu + A1)

        f_data, data_at_mu = nearest_data_value(mu1)

        out['peaks'].append({
            'freq_fit': mu1,
            'amp_fit': A1,
            'baseline_at_mu': base_at_mu,
            'model_at_mu': model_at_mu,
            'freq_data': f_data,
            'data_at_mu': data_at_mu,
        })

    elif model == 'poly+g2':
        A1      = float(params['A1']);  mu1 = float(params['mu1']);  s1 = float(params['sig1'])
        A2      = float(params['A2']);  mu2 = float(params['mu2']);  s2 = float(params['sig2'])

        base1   = float(baseline(mu1))
        base2   = float(baseline(mu2))

        model1  = float(base1 + A1 + g(mu1, A2, mu2, s2))
        model2  = float(base2 + A2 + g(mu2, A1, mu1, s1))

        f1_data, y1_data = nearest_data_value(mu1)
        f2_data, y2_data = nearest_data_value(mu2)

        out['peaks'].append({
            'freq_fit': mu1,
            'amp_fit': A1,
            'baseline_at_mu': base1,
            'model_at_mu': model1,
            'freq_data': f1_data,
            'data_at_mu': y1_data,
        })
        out['peaks'].append({
            'freq_fit': mu2,
            'amp_fit': A2,
            'baseline_at_mu': base2,
            'model_at_mu': model2,
            'freq_data': f2_data,
            'data_at_mu': y2_data,
        })

    else:
        raise ValueError(f"Unknown model: {model}")

    return out

### ----- Function to obtain peaks
def peak_info(peaks_out, target_freq=None):

    peaks_result = extract_peak_info(peaks_out)
    peak_list = peaks_result['peaks']
    
    # Select by target frequency if specified and multiple peaks
    if target_freq is not None and len(peak_list) > 1:
        selected = min(peak_list, key=lambda d: abs(d['freq_fit'] - float(target_freq)))
    else:
        # Select strongest (highest power)
        selected = max(peak_list, key=lambda d: d['model_at_mu'])
    
    return {
        'peak_freq_fit': selected['freq_fit'],
        'peak_freq_data': selected['freq_data'],
        'amp_fit': selected['amp_fit'],
        'peak_model_db': selected['model_at_mu'],
        'peak_data_db':selected['data_at_mu'],
        'peaks': peak_list
    }

##### ----- Return peaks for EEG and OPM
def get_matched_peaks_status_closest_peak(eeg_info, opm_info, target_freq=10.0, tolerance=5.0, return_status=True):
    """
    Like get_matched_peaks_status, but if no pair is found within tolerance,
    selects the pair of peaks (one from EEG, one from OPM) with the smallest frequency difference.
    This improves comparability across modalities.
    """
    eeg_peaks = eeg_info['peaks']
    opm_peaks = opm_info['peaks']


    # 1. Find all pairs where the cross-modality frequency difference is within tolerance
    modality_pairs = []
    for ep in eeg_peaks:
        for op in opm_peaks:
            pair_dist = abs(ep['freq_fit'] - op['freq_fit'])
            if pair_dist <= tolerance:
                avg_to_target = abs((ep['freq_fit'] + op['freq_fit']) / 2.0 - target_freq)
                modality_pairs.append({
                    'eeg': ep,
                    'opm': op,
                    'pair_dist': pair_dist,
                    'avg_to_target': avg_to_target
                })

    if modality_pairs:
        # If multiple pairs, select the one whose average frequency is closest to target_freq
        best_pair = min(modality_pairs, key=lambda x: x['avg_to_target'])
        sel_eeg = best_pair['eeg']
        sel_opm = best_pair['opm']
        has_consensus = True
    else:
        # 2. Fallback: select the pair with the smallest frequency difference across modalities
        min_dist = float('inf')
        best_pair = None
        for ep in eeg_peaks:
            for op in opm_peaks:
                dist = abs(ep['freq_fit'] - op['freq_fit'])
                if dist < min_dist:
                    min_dist = dist
                    avg_to_target = abs((ep['freq_fit'] + op['freq_fit']) / 2.0 - target_freq)
                    best_pair = {'eeg': ep, 'opm': op, 'dist': dist, 'avg_to_target': avg_to_target}
        sel_eeg = best_pair['eeg']
        sel_opm = best_pair['opm']
        has_consensus = False

    sel_eeg['peak_model_db'] = sel_eeg['model_at_mu']
    sel_opm['peak_model_db'] = sel_opm['model_at_mu']
    sel_eeg['peak_freq_fit'] = sel_eeg['freq_fit']
    sel_opm['peak_freq_fit'] = sel_opm['freq_fit']

    if return_status:
        return sel_eeg, sel_opm, has_consensus
    return sel_eeg, sel_opm



##### ----- Return peaks for EEG and OPM
def match_all_peaks(eeg_info, opm_info, tolerance=5.0):
    """
    Return all EEG–OPM peak pairs whose fitted frequencies differ by <= tolerance.

    Returns
    -------
    matched_pairs : list of dict
        Each entry contains:
        - 'eeg': EEG peak dict
        - 'opm': OPM peak dict
        - 'pair_dist': absolute frequency difference
    """

    eeg_peaks = eeg_info['peaks']
    opm_peaks = opm_info['peaks']

    matched_pairs = []

    for ep in eeg_peaks:
        for op in opm_peaks:
            pair_dist = abs(ep['freq_fit'] - op['freq_fit'])

            if pair_dist <= tolerance:
                ep_out = ep.copy()
                op_out = op.copy()

                ep_out['peak_model_db'] = ep_out['model_at_mu']
                op_out['peak_model_db'] = op_out['model_at_mu']

                ep_out['peak_freq_fit'] = ep_out['freq_fit']
                op_out['peak_freq_fit'] = op_out['freq_fit']

                matched_pairs.append({
                    'eeg': ep_out,
                    'opm': op_out,
                    'pair_dist': pair_dist
                })

    return matched_pairs


### ----- Peak fitting function
def peak_finder_tfr(
    tfr,
    fit_range,
    peak_range,
    time_range,
    min_sep,
    sdgaussian_range,
    plot=False,
    is_fit_negative=False,
    edge_margin=2,
    smooth_sigma=1,
):
    ### ----- Get parameters from inputs
    fmin, fmax         = fit_range
    peak_min, peak_max = peak_range
    tmin, tmax         = time_range
    sig_min, sig_max   = sdgaussian_range

    ### ----- Pull data without modifying original object
    data = np.asarray(tfr.data)

    ### ----- Get only frequencies within fit_range
    fmask = (tfr.freqs >= fmin) & (tfr.freqs <= fmax)
    if not np.any(fmask):
        raise ValueError("No frequency bins within [fmin, fmax].")
    f = np.asarray(tfr.freqs)[fmask].astype(float)

    ### ----- Get median frequency for reference (for slope parameterization)
    f_ref = float(np.median(f))

    ### ----- Time indices
    t0, t1 = tfr.time_as_index([tmin, tmax])
    if not (0 <= t0 < t1 <= data.shape[-1]):
        raise ValueError("Invalid time window: no samples within [tmin, tmax].")

    ### ----- Build spectrum: average over channels (if present) and time window
    if data.ndim == 3:  # (n_ch, n_freq, n_time)
        spec = data.mean(axis=0)[fmask, t0:t1].mean(axis=1)
    else:               # (n_freq, n_time)
        spec = data[fmask, t0:t1].mean(axis=1)

    ### ----- Smooth spectrum 
    if smooth_sigma and smooth_sigma > 0:
        freq_step           = f[1] - f[0]
        smooth_sigma_bins   = smooth_sigma / freq_step
        spec                = gaussian_filter1d(spec, sigma=smooth_sigma_bins)

    ### ----- Peak-center bounds with edge margin
    mu_lo = peak_min + edge_margin
    mu_hi = peak_max - edge_margin
    if mu_lo >= mu_hi:
        raise ValueError("Edge margin too large relative to [peak_min, peak_max].")

    ##### Get candidate peaks for initialization of Gaussian fits
    band_mask   = (f >= peak_min) & (f <= peak_max)
    f_band      = f[band_mask]
    s_band      = spec[band_mask]
    if f_band.size == 0:
        raise ValueError("No bins within peak_range inside fit_range.")

    minwidth_hz = 2.35482 * sig_min
    specpeaks = candidate_peaks(
        s_band, f_band,
        minfreq=mu_lo, maxfreq=mu_hi,
        minwidth_hz=minwidth_hz, minsep_hz=min_sep,
        sig_min=sig_min, sig_max=sig_max
    )
    print(specpeaks)

    ##### ----- If no valid local maximum was found (e.g. no clean peak shape
    ##### ----- passing the width/separation/frequency-range checks), fall back
    ##### ----- to the global maximum of the search band (candidate_peaks already
    ##### ----- computes this as mu1_0, used elsewhere only as a Gaussian-fit
    ##### ----- initialization) and report it as the single peak instead of none.
    if not specpeaks['found_any']:
        fallback_freq   = specpeaks['mu1_0']
        idx_fallback    = int(np.argmin(np.abs(f_band - fallback_freq)))
        fallback_pow    = float(s_band[idx_fallback])
        specpeaks       = dict(specpeaks)
        specpeaks['cands_freq'] = np.array([fallback_freq])
        specpeaks['cands_pow']  = np.array([fallback_pow])
        print(f"No valid local maximum found; falling back to global max at "
              f"{fallback_freq:.2f} Hz.")

    ##### ---- Fit peaks
    results     = {}
    fig = None
    if plot:
        fig, ax = plt.subplots(figsize=(6, 3))
        lbl = 'Spectrum (smoothed)' if (smooth_sigma and smooth_sigma > 0) else 'Spectrum'
        ax.plot(f, spec, lw=1.5, label=lbl)
        if len(specpeaks['cands_freq']) > 2:
            print("Warning: More than 2 candidate peaks found; only plotting the first 2.")
            specpeaks['cands_freq'] = specpeaks['cands_freq'][:2]
            specpeaks['cands_pow']  = specpeaks['cands_pow'][:2]
        for pf in specpeaks['cands_freq']:
            ax.axvline(pf, ls=':', lw=2, label=f"Candidate peak: {pf:.2f} Hz")
        ax.set_xlim(fmin, fmax)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power (a.u.)')  # don't claim dB unless guaranteed
        plt.legend(frameon=False)
        plt.tight_layout()
        plt.show(block=False)


    return {
        'model': None,
        'params': None,
        'peak_freqs': specpeaks['cands_freq'],
        'peak_amps': specpeaks['cands_pow'],
        'freqs': f,
        'spec': spec,
    }, fig 


def extract_peak_info_peakfinder(fit_result, min_sep=None):
    """
    Returns up to 2 peaks from fit_result. peak_finder_tfr() returns
    peak_freqs/peak_amps sorted by power (descending), with NO separation
    filtering applied -- the top 2 by power could be arbitrarily close in
    frequency. This function enforces min_sep on the pair it reports: the
    strongest peak is always kept, and the second peak is the strongest
    REMAINING candidate that is at least min_sep Hz away from it -- the
    same logic candidate_peaks() already uses to compute mu2_0 (which
    peak_finder_tfr computes but never returns), and the same criterion
    fit_peaks_tfr()'s Gaussian fit enforces on its own two-peak model
    (too_close_abs). If no candidate satisfies min_sep, only the single
    strongest peak is returned, mirroring the Gaussian fit's fallback to a
    one-peak model when no valid second peak exists.

    min_sep=None keeps the old, unconstrained "top 2 by power" behavior --
    kept only for backward compatibility; pass min_sep explicitly (e.g.
    cf_peak['min_sep']) to get separation-enforced peaks.
    """
    out = {'peaks': []}
    allpeaks    = list(fit_result['peak_freqs'])
    allamps     = list(fit_result['peak_amps'])

    if len(allpeaks) == 0 or len(allpeaks) != len(allamps):
        print("Error: No Peaks or inconsistent lengths of peak_freqs and peak_amps in fit_result.")
        return out

    ##### ----- Strongest peak is always kept
    selected_freqs = [allpeaks[0]]
    selected_amps  = [allamps[0]]

    if min_sep is None:
        ##### ----- Old, unconstrained behavior: just take the next peak by
        ##### ----- power, regardless of frequency separation
        if len(allpeaks) > 1:
            selected_freqs.append(allpeaks[1])
            selected_amps.append(allamps[1])
    else:
        ##### ----- Walk the remaining candidates in power order; keep the
        ##### ----- first one that is >= min_sep away from the strongest peak
        for freq, amp in zip(allpeaks[1:], allamps[1:]):
            if abs(freq - selected_freqs[0]) >= min_sep:
                selected_freqs.append(freq)
                selected_amps.append(amp)
                break
        if len(selected_freqs) == 1 and len(allpeaks) > 1:
            print(f"Note: no candidate peak >= {min_sep} Hz from the strongest peak "
                  f"({selected_freqs[0]:.2f} Hz); reporting a single peak.")

    for freq, amp in zip(selected_freqs, selected_amps):
        out['peaks'].append({
            'freq_fit': freq,
            'model_at_mu': amp,
        })

    return out