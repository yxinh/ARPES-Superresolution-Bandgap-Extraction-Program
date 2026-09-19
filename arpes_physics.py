"""Physical models and numerical kernels for ARPES gap extraction.

Notation follows the companion RSI manuscript:

    xi_k          normal-state dispersion measured from mu
    delta         superconducting gap
    gamma         Dynes scattering rate
    delta_app     apparent gap before the chemical-potential correction
    delta_corr    gap after subtracting xi_mu in quadrature
    delta_best    inverse-variance (MMWA) combination of per-EDC gaps
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from scipy.special import expit

# Boltzmann constant (eV/K). FWHM_TO_SIGMA converts a Gaussian FWHM to sigma.
KB = 8.617333262145e-5
FWHM_TO_SIGMA = 2.3548

# Kept as aliases so existing GUI code can import the historical names.
KB_CONSTANT = KB


def energy_sigma_from_fwhm(fwhm_eV):
    """Instrumental energy resolution: Gaussian sigma from FWHM (eV)."""
    return float(fwhm_eV) / FWHM_TO_SIGMA


def fermi_dirac(energy, temperature, k_B=KB):
    """Occupied Fermi--Dirac factor f(E, T) = 1 / (1 + exp(E / k_B T))."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    return expit(-np.asarray(energy, dtype=float) / (k_B * temperature))


def intensity_sigma(raw_intensity, alpha, bg_variance):
    """Poisson-plus-background standard deviation used in weighted fits."""
    raw = np.asarray(raw_intensity, dtype=float)
    return np.sqrt(np.abs(raw) * alpha ** 2 + bg_variance + 1e-12)


def load_arpes_dat(path):
    """Load a tab-separated ARPES map.

    File layout (SSRL-style ``.dat``):
        first row  -- energy axis (eV)
        later rows -- unused index, momentum, then intensity along energy

    Returns
    -------
    intensity : ndarray, shape (n_energy, n_momentum)
    momentum : ndarray, shape (n_momentum,)
    energy : ndarray, shape (n_energy,)
    """
    with open(path, "r") as fh:
        raw_lines = [line.rstrip("\n") for line in fh if line.strip() != ""]

    first_line = raw_lines[0].split("\t")
    energy = np.array([float(x) for x in first_line if x.strip() != ""])
    k_list, intensity_rows = [], []
    for line in raw_lines[1:]:
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        try:
            k_list.append(float(parts[1]))
        except ValueError:
            continue
        intensity_rows.append(
            [float(x) if x.strip() != "" else 0.0 for x in parts[2:]]
        )

    momentum = np.array(k_list, dtype=float)
    intensity = np.array(intensity_rows, dtype=float).T
    if momentum[0] > momentum[-1]:
        momentum, intensity = momentum[::-1], intensity[:, ::-1]
    if energy[0] > energy[-1]:
        energy, intensity = energy[::-1], intensity[::-1, :]
    return intensity, momentum, energy


def shirley_background_1d(energy, intensity, max_iter, tol):
    """Iterative Shirley background on a single EDC.

    Intensity is first shifted so its minimum is zero (as in the GUI
    implementation). Returns ``(background, converged, last_diff)`` on the
    original intensity scale.
    """
    energy = np.asarray(energy, dtype=float)
    y = np.asarray(intensity, dtype=float)
    y_min = np.min(y)
    y_proc = y - y_min
    bg = np.zeros_like(y_proc)
    converged = False
    last_diff = 0.0
    for _ in range(int(max_iter)):
        y_eff = np.maximum(y_proc - bg, 0.0)
        integral = np.zeros_like(y_eff)
        for i in range(len(y_eff) - 2, -1, -1):
            integral[i] = integral[i + 1] + 0.5 * (y_eff[i + 1] + y_eff[i]) * (
                energy[i + 1] - energy[i]
            )
        if integral[0] == 0:
            break
        new_bg = y_proc[-1] + ((y_proc[0] - y_proc[-1]) / integral[0]) * integral
        last_diff = float(np.max(np.abs(new_bg - bg)))
        if last_diff < tol:
            bg = new_bg
            converged = True
            break
        bg = new_bg
    return bg + y_min, converged, last_diff


def shirley_background_2d(energy, intensity, max_iter, tol, smooth_k_pts=0.0):
    """Shirley background on a (energy, momentum) map, then optional k-smoothing.

    Returns
    -------
    background : ndarray
    all_converged : bool
    max_err : float
        Largest unconverged residual; 0 if every EDC converged.
    max_err_k_index : int
        Momentum index of ``max_err``, or -1 if every EDC converged.
    """
    energy = np.asarray(energy, dtype=float)
    intensity = np.asarray(intensity, dtype=float)
    background = np.zeros_like(intensity)
    all_converged = True
    max_err = 0.0
    max_err_k_index = -1
    for j in range(intensity.shape[1]):
        bg, converged, last_diff = shirley_background_1d(
            energy, intensity[:, j], max_iter, tol
        )
        if not converged:
            all_converged = False
            if last_diff > max_err:
                max_err, max_err_k_index = last_diff, j
        background[:, j] = bg

    if smooth_k_pts > 0:
        pad_w = int(np.ceil(4 * smooth_k_pts))
        if pad_w > 0:
            padded = np.pad(background, pad_width=((0, 0), (pad_w, pad_w)), mode="edge")
            smoothed = gaussian_filter1d(padded, sigma=smooth_k_pts, axis=1)
            background = smoothed[:, pad_w:-pad_w]
        else:
            background = gaussian_filter1d(background, sigma=smooth_k_pts, axis=1)
    return background, all_converged, max_err, max_err_k_index


def poisson_scale(roi, smooth_sigma):
    """Estimate the Poisson scale alpha from a weakly smoothed ROI.

    Returns ``(alpha, roi_lowpass, residual)``.
    """
    roi = np.asarray(roi, dtype=float)
    pad_w = int(np.ceil(4 * smooth_sigma))
    if pad_w > 0:
        padded = np.pad(roi, pad_width=pad_w, mode="edge")
        smoothed = gaussian_filter(padded, sigma=smooth_sigma)
        roi_lp = smoothed[pad_w:-pad_w, pad_w:-pad_w]
    else:
        roi_lp = gaussian_filter(roi, sigma=smooth_sigma)
    residual = roi - roi_lp
    positive = roi_lp[roi_lp > 0]
    mean_signal = np.mean(positive) if positive.size > 0 else 1e-12
    alpha = np.sqrt(np.var(residual) / abs(mean_signal) + 1e-12)
    return float(alpha), roi_lp, residual


def dynes_photocurrent(
    energy,
    delta,
    gamma,
    scale,
    xi_k,
    temperature,
    energy_res_sigma,
    k_B=KB,
):
    """Resolution-convolved Dynes photocurrent at one momentum.

    The overall ``scale`` absorbs the conventional 1/pi prefactor of A(k, omega).
    Convolution uses a Gaussian of width ``energy_res_sigma`` (eV).
    """
    energy = np.asarray(energy, dtype=float)
    dE = (energy[-1] - energy[0]) / max(len(energy) - 1, 1)
    abs_dE = abs(dE)
    sigma_pixels = energy_res_sigma / abs_dE if abs_dE > 0 else 1.0

    pad_e = 4.0 * energy_res_sigma
    pad_n = int(np.ceil(pad_e / abs_dE)) if abs_dE > 0 else 0
    if pad_n > 0:
        e_left = np.array([energy[0] - (pad_n - i) * dE for i in range(pad_n)])
        e_right = np.array([energy[-1] + (i + 1) * dE for i in range(pad_n)])
        e_ext = np.concatenate((e_left, energy, e_right))
    else:
        e_ext = energy

    e_k = np.sqrt(xi_k ** 2 + delta ** 2)
    u2 = 0.5 * (1.0 + xi_k / e_k)
    v2 = 1.0 - u2
    spectral = scale * gamma * (
        u2 / ((e_ext - e_k) ** 2 + gamma ** 2)
        + v2 / ((e_ext + e_k) ** 2 + gamma ** 2)
    )
    intensity = spectral * fermi_dirac(e_ext, temperature, k_B=k_B)
    blurred = gaussian_filter1d(intensity, sigma=sigma_pixels)
    if pad_n > 0:
        return blurred[pad_n:-pad_n]
    return blurred


def _inverse_variance_mean(values, errors):
    weights = 1.0 / (np.asarray(errors, dtype=float) ** 2 + 1e-12)
    mean = float(np.sum(weights * values) / np.sum(weights))
    sigma = float(np.sqrt(1.0 / np.sum(weights)))
    return mean, sigma, weights


def _grow_consistent_window(k_vals, delta_vals, err_vals, valid_k, center, n_sigma):
    """Expand a contiguous window about ``center`` while Delta stays in-bounds."""
    lower = delta_vals[center] - n_sigma * err_vals[center]
    upper = delta_vals[center] + n_sigma * err_vals[center]
    left = center
    while left > 0:
        nxt = left - 1
        if not valid_k[nxt]:
            break
        if not (lower <= delta_vals[nxt] <= upper):
            break
        left = nxt
    right = center
    while right < len(k_vals) - 1:
        nxt = right + 1
        if not valid_k[nxt]:
            break
        if not (lower <= delta_vals[nxt] <= upper):
            break
        right = nxt
    return slice(left, right + 1)


def _window_combination(k_vals, delta_vals, err_vals, valid_k, sel, gamma_vals, gamma_errs):
    sel_delta = delta_vals[sel]
    sel_err = err_vals[sel]
    sel_k = k_vals[sel]
    mask = valid_k[sel] & np.isfinite(sel_err) & (sel_err > 0)
    if not np.any(mask):
        return None
    sel_delta, sel_err, sel_k = sel_delta[mask], sel_err[mask], sel_k[mask]
    delta_best, error_best, _ = _inverse_variance_mean(sel_delta, sel_err)
    dof = len(sel_delta) - 1
    chi2_nu = (
        float(np.sum(((sel_delta - delta_best) / sel_err) ** 2) / dof) if dof > 0 else 0.0
    )

    gamma_best, gamma_err = np.nan, np.nan
    if gamma_vals is not None and gamma_errs is not None:
        sel_g = np.asarray(gamma_vals)[sel][mask]
        sel_ge = np.asarray(gamma_errs)[sel][mask]
        g_ok = np.isfinite(sel_ge) & (sel_ge > 0)
        if np.any(g_ok):
            gamma_best, gamma_err, _ = _inverse_variance_mean(sel_g[g_ok], sel_ge[g_ok])

    return {
        "sel_k": sel_k,
        "sel_delta": sel_delta,
        "sel_err": sel_err,
        "delta_best": delta_best,
        "error_best": error_best,
        "gamma_best": gamma_best,
        "gamma_err": gamma_err,
        "chi2_nu": chi2_nu,
    }


def mmwa_combine(
    k_vals,
    delta_vals,
    err_vals,
    valid_k,
    k_f=None,
    n_sigma=2.0,
    err_cap_mult=3.0,
    gamma_vals=None,
    gamma_errs=None,
):
    """Multi-momentum weighted-average (MMWA) combination of per-EDC gaps.

    For each valid center whose uncertainty is within ``err_cap_mult`` of the
    smallest error on the cut, grow a window in which every Delta lies inside
    ``n_sigma`` of the center. Keep the window with the smallest combined
    inverse-variance uncertainty.

    Returns a dict with keys expected by the Step-2 GUI / Step-3 loader, or
    ``None`` if no valid window exists.
    """
    k_vals = np.asarray(k_vals, dtype=float)
    delta_vals = np.asarray(delta_vals, dtype=float)
    err_vals = np.asarray(err_vals, dtype=float)
    valid_k = np.asarray(valid_k, dtype=bool)
    if gamma_vals is None or np.asarray(gamma_vals).size != len(k_vals):
        gamma_vals = np.full(len(k_vals), np.nan)
    if gamma_errs is None or np.asarray(gamma_errs).size != len(k_vals):
        gamma_errs = np.full(len(k_vals), np.nan)
    gamma_vals = np.asarray(gamma_vals, dtype=float)
    gamma_errs = np.asarray(gamma_errs, dtype=float)

    if k_f is None:
        k_f = 0.5 * (np.min(k_vals) + np.max(k_vals))
    kf_idx = int(np.argmin(np.abs(k_vals - k_f)))
    if 0 <= kf_idx < len(err_vals) and valid_k[kf_idx]:
        kf_err = err_vals[kf_idx]
        kf_delta = delta_vals[kf_idx]
    else:
        kf_err, kf_delta = np.nan, np.nan

    finite_pos = err_vals[valid_k & np.isfinite(err_vals) & (err_vals > 0)]
    err_min = float(np.min(finite_pos)) if finite_pos.size > 0 else np.nan
    if err_cap_mult > 0 and np.isfinite(err_min) and err_min > 0:
        max_err = err_min * err_cap_mult
    else:
        max_err = np.inf

    best_result = None
    best_error = np.inf
    for i in range(len(k_vals)):
        if not valid_k[i]:
            continue
        err_i = err_vals[i]
        if not np.isfinite(err_i) or err_i <= 0 or err_i > max_err:
            continue
        sel = _grow_consistent_window(k_vals, delta_vals, err_vals, valid_k, i, n_sigma)
        combo = _window_combination(
            k_vals, delta_vals, err_vals, valid_k, sel, gamma_vals, gamma_errs
        )
        if combo is None:
            continue
        if combo["error_best"] < best_error:
            best_error = combo["error_best"]
            best_result = {
                "kF": k_f,
                "mid_idx": i,
                "delta_mid": delta_vals[i],
                "err_mid": err_i,
                **combo,
            }

    if best_result is None:
        valid_idx = np.where(valid_k)[0]
        if valid_idx.size == 0:
            return None
        mid_idx = int(valid_idx[np.argmin(np.abs(k_vals[valid_idx] - k_f))])
        err_mid = err_vals[mid_idx]
        if not np.isfinite(err_mid) or err_mid <= 0:
            return None
        sel = _grow_consistent_window(
            k_vals, delta_vals, err_vals, valid_k, mid_idx, n_sigma
        )
        combo = _window_combination(
            k_vals, delta_vals, err_vals, valid_k, sel, gamma_vals, gamma_errs
        )
        if combo is None:
            return None
        best_result = {
            "kF": k_f,
            "mid_idx": mid_idx,
            "delta_mid": delta_vals[mid_idx],
            "err_mid": err_mid,
            **combo,
        }

    best_result["delta_kf"] = kf_delta
    best_result["err_kf"] = kf_err
    return best_result


def weighted_linear_fit(x, y, sigma):
    """Inverse-variance linear fit y = a + b x. Returns (a, b) or None."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    good = np.isfinite(x) & np.isfinite(y) & np.isfinite(sigma) & (sigma > 0)
    if int(np.count_nonzero(good)) < 2:
        return None
    slope, intercept = np.polyfit(x[good], y[good], 1, w=1.0 / sigma[good])
    return float(intercept), float(slope)


def linear_drift(temperature, intercept, slope):
    """xi_mu(T) = a + b T."""
    return intercept + slope * np.asarray(temperature, dtype=float)


def apply_gap_correction(delta, xi_mu):
    """delta_corr = sqrt(max(delta^2 - xi_mu^2, 0))."""
    delta = np.asarray(delta, dtype=float)
    xi_mu = np.asarray(xi_mu, dtype=float)
    return np.sqrt(np.clip(delta ** 2 - xi_mu ** 2, 0.0, None))


def bcs_gap(temperature, delta_0, t_bcs):
    """Approximate BCS interpolation Delta(T) = Delta_0 tanh(1.74 sqrt(T_BCS/T - 1))."""
    temperature = np.asarray(temperature, dtype=float)
    out = np.zeros_like(temperature, dtype=float)
    ok = np.isfinite(temperature) & (temperature > 0) & (temperature < t_bcs)
    out[ok] = delta_0 * np.tanh(
        1.74 * np.sqrt(np.maximum(t_bcs / temperature[ok] - 1.0, 0.0))
    )
    return out
