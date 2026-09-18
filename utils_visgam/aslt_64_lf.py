# aslt with reduced SD
import numpy as np
try:
    import cupy as cp
    from cusignal.convolution.convolve import fftconvolve
except Exception:
    cp = np
    from scipy.signal import fftconvolve
    np.asnumpy = np.asarray

def cxmorlet(fc, n_cycles, sfreq):
    sd = (n_cycles / 2) * (1.0 / fc) / 2.5
    wl = int(2 * np.floor(np.fix(3 * sd * sfreq) / 2) + 1) # Reduced from 6 to 3 to enable shorter windows for lower frequencies
    w = np.zeros(wl, dtype=np.complex128)
    off = int(np.fix(wl / 2))
    gi = 0.0
    for i in range(wl):
        t = (i - off) / sfreq
        g = (1.0 / (sd * np.sqrt(2 * np.pi))) * np.exp(-(t**2) / (2 * sd**2))
        w[i] = g * np.exp(2j * np.pi * fc * t)
        gi += g
    w /= gi
    return w

def aslt(data, sfreq, foi, n_cycles, order=None, mult=False):
    data = np.atleast_2d(np.asarray(data, dtype=np.float64))
    n_epochs, n_times = data.shape
    foi = np.asarray(foi, dtype=float)
    n_freqs = len(foi)
    order_ls = (np.fix(np.linspace(order[0], order[1], n_freqs)).astype(int)
                if order is not None else np.ones(n_freqs, dtype=int))

    padding = 0
    wavelets = {}
    for i_f in range(n_freqs):
        for i_o in range(order_ls[i_f]):
            n_cyc = (n_cycles * (i_o + 1)) if mult else (n_cycles + i_o)
            w = cxmorlet(foi[i_f], n_cyc, sfreq)
            padding = max(padding, int(np.fix(len(w) / 2)))
            wavelets[(i_f, i_o)] = w

    buffer = cp.zeros((n_epochs, int(n_times + 2 * padding)), dtype=cp.float64)
    bufbegin = int(padding); bufend = int(padding + n_times)
    buffer[:, bufbegin:bufend] = cp.asarray(data, dtype=cp.float64)

    wtresult = cp.zeros((n_epochs, n_freqs, n_times), dtype=cp.float64)

    for i_f in range(n_freqs):
        temp = cp.ones((n_epochs, n_times), dtype=cp.float64)
        for i_o in range(order_ls[i_f]):
            sw = cp.asarray(wavelets[(i_f, i_o)], dtype=cp.complex128).reshape(1, -1)
            conv = fftconvolve(buffer, sw, mode='same', axes=1)
            mag = 2.0 * cp.abs(conv[:, bufbegin:bufend])
            temp *= mag
        root = 1.0 / float(order_ls[i_f])
        amp = temp ** root
        wtresult[:, i_f, :] = amp * amp  # power

    return cp.asnumpy(wtresult)
