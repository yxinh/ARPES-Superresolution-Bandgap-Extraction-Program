"""Step 3: temperature dependence, chemical-potential correction, and BCS interpolation."""

import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import numpy as np
from scipy.optimize import curve_fit
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

from arpes_physics import (
    linear_drift,
    weighted_linear_fit,
    apply_gap_correction,
    bcs_gap,
)
from prl_plot_style import (
    SERIES, COLORS, PRL_MARKERSIZE_LARGE, PRL_DPI, PRL_LINEWIDTH,
    OVERFLOW_BAR_EDGE,
    step3_panel_figsize, step3_apply_panel_layout, step3_finalize_axes, step3_legend_kwargs, step3_save_panel,
    set_axis_labels,
    errorbar_kwargs, plot_curve_kwargs, reference_line_kwargs,
    shade_gapless, temperature_bar_spacing, plot_offset_uncertainty_bars,
)

# Slightly larger markers on sparse T-series plots (≥ ~1 mm at single-column print)
_STEP3_MS = PRL_MARKERSIZE_LARGE

# Panel order / labels for 2×2 composite figures (+ standalone uncertainty panel)
_STEP3_PANELS = (
    ("RSS Comparison vs T", "(a)", "RSS"),
    ("P-value vs T", "(b)", r"$\log_{10}(p)$"),
    ("SC Gap (Delta) vs T", "(c)", r"$\Delta$"),
    ("Gamma vs T", "(d)", r"$\Gamma$"),
)
_STEP3_PANEL_MODES = [p[0] for p in _STEP3_PANELS]
_STEP3_UNCERTAINTY_PANELS = (
    ("Delta Uncertainty vs T", "(e)", r"$|\sigma_\Delta|$"),
    ("Gamma Uncertainty vs T", "(f)", r"$|\sigma_\Gamma|$"),
)
_STEP3_UNCERTAINTY_MODES = [p[0] for p in _STEP3_UNCERTAINTY_PANELS]

class Step3TemperatureDependence(ttk.Frame):
    def __init__(self, parent, controller=None, **kwargs):
        super().__init__(parent, **kwargs)
        self.controller = controller 
        
        self.temp_data = [] 
        self.extracted_physics = [] 
        self.Tc_estimate = None

        self.show_mode = tk.StringVar(value="RSS Comparison vs T")

        # Chemical-potential μ(T) correction for the Δ-vs-T panel (weighted only)
        self.var_mu_correct = tk.BooleanVar(value=False)
        self.var_show_uncertainty = tk.BooleanVar(value=True)
        self.var_bcs_fit = tk.BooleanVar(value=True)

        self._build_ui()

    def _apply_gapless_shading(self, ax, T_max):
        """Mark gapless (normal-state) region on the lowest drawing layer.

        Always call this on the back-most axis (``ax2`` when a twin exists)
        *before* plotting bars/curves, so the opaque band cannot cover data.
        """
        if self.Tc_estimate is not None and self.Tc_estimate <= T_max:
            artist = shade_gapless(ax, self.Tc_estimate, T_max + 50)
            if artist is not None:
                try:
                    artist.set_zorder(-100)
                except Exception:
                    pass
            ax.set_axisbelow(True)

    @staticmethod
    def _finite_sorted_series(T, y, err, mask):
        """Return (T, y, err) at finite points in mask, sorted by T (avoids NaN gaps in lines).

        ``err`` may be a 1-D array or ``(err_lo, err_hi)``; the same index selection
        is applied in either case.
        """
        idx = np.where(mask)[0]
        if idx.size == 0:
            return None, None, None
        idx = idx[np.argsort(T[idx])]
        if isinstance(err, tuple):
            return T[idx], y[idx], (err[0][idx], err[1][idx])
        return T[idx], y[idx], err[idx]

    @staticmethod
    def _parse_limit(entry, default=None):
        try:
            val = entry.get().strip()
            if val.lower() == 'auto' or val == '':
                return default
            return float(val)
        except Exception:
            return default

    # =============================================================================
    # --- Chemical-potential μ(T) correction (linear high-T extrapolation) ---
    # =============================================================================
    @staticmethod
    def _linear_drift(T, a, b):
        """xi_mu(T) = a + b T."""
        return linear_drift(T, a, b)

    @staticmethod
    def _weighted_linear_fit(T, y, sigma):
        """Weighted least-squares fit y = a + b T. Returns (a, b) or None."""
        return weighted_linear_fit(T, y, sigma)

    @staticmethod
    def _apply_gap_correction(delta, ksi):
        """delta_corr = sqrt(max(delta^2 - xi_mu^2, 0))."""
        return apply_gap_correction(delta, ksi)

    def _fit_mu_drift(self, T, delta_eV, err_eV):
        """Fit ξ(T) = a + b·T on high-T points of one Δ series. Returns (a, b)."""
        T = np.asarray(T, dtype=float)
        delta = np.abs(np.asarray(delta_eV, dtype=float))
        err = np.abs(np.asarray(err_eV, dtype=float))

        t_cut = self._parse_limit(self.ent_corr_tmin)
        if t_cut is None:
            t_cut = self.Tc_estimate
        if t_cut is None:
            raise ValueError("specify the linear-fit temperature (K), or estimate Tc first.")

        mask = np.isfinite(T) & np.isfinite(delta) & (T >= t_cut)
        if int(np.count_nonzero(mask)) < 2:
            raise RuntimeError("need at least two points at or above the fit temperature.")

        Th, dh, eh = T[mask], delta[mask], err[mask]
        eh = np.where(np.isfinite(eh) & (eh > 0), eh, np.nan)
        med = np.nanmedian(eh)
        if not np.isfinite(med) or med <= 0:
            med = 1.0
        sigma = np.where(np.isfinite(eh) & (eh > 0), eh, med)

        fit = self._weighted_linear_fit(Th, dh, sigma)
        if fit is None:
            raise RuntimeError("linear fit failed; check the fit temperature and data.")
        return fit

    def _apply_mu_correction_with_fit(self, T, delta_eV, err_eV, a, b):
        """Correct one Δ series with a fixed ξ(T) = a + b·T (inputs/outputs in eV).

        Error bars from endpoint correction with the same ξ:
            Δ_corr(+σ) = sqrt((Δ+σ)² − ξ²)
            Δ_corr(−σ) = sqrt(max(Δ−σ, 0)² − ξ²)
        Returns (corr, err_lo, err_hi). Failed / missing inputs stay NaN so the
        uncertainty panel can flag them as overflow.
        """
        T = np.asarray(T, dtype=float)
        delta = np.abs(np.asarray(delta_eV, dtype=float))
        err = np.abs(np.asarray(err_eV, dtype=float))

        ksi = self._linear_drift(T, a, b)
        valid = np.isfinite(delta) & np.isfinite(err)
        err_use = np.where(valid, err, 0.0)

        corr = np.full_like(delta, np.nan, dtype=float)
        err_lo = np.full_like(delta, np.nan, dtype=float)
        err_hi = np.full_like(delta, np.nan, dtype=float)
        if not np.any(valid):
            return corr, err_lo, err_hi

        corr_v = self._apply_gap_correction(delta[valid], ksi[valid] if np.ndim(ksi) else ksi)
        corr_hi = self._apply_gap_correction(delta[valid] + err_use[valid], ksi[valid] if np.ndim(ksi) else ksi)
        corr_lo = self._apply_gap_correction(
            np.maximum(delta[valid] - err_use[valid], 0.0),
            ksi[valid] if np.ndim(ksi) else ksi,
        )
        corr[valid] = corr_v
        err_lo[valid] = np.maximum(corr_v - corr_lo, 0.0)
        err_hi[valid] = np.maximum(corr_hi - corr_v, 0.0)
        # Correction undefined (e.g. |ξ| ≥ |Δ| with non-finite result): keep NaN
        bad = valid & ~np.isfinite(corr)
        err_lo[bad] = np.nan
        err_hi[bad] = np.nan
        return corr, err_lo, err_hi

    @staticmethod
    def _bcs_gap(T, delta0, Tc):
        """Approximate BCS interpolation used in the Delta(T) panel."""
        return bcs_gap(T, delta0, Tc)

    def _bcs_fit_window(self, T):
        """User BCS-fit interval (K). ``auto`` min = data min; ``auto`` max = 0.70 Tc.

        Near-Tc points are left out of the fit so the extrapolation can be
        compared with the two extraction methods where they differ most.
        """
        T = np.asarray(T, dtype=float)
        finite = T[np.isfinite(T)]
        if finite.size == 0:
            return 0.0, 1.0
        t_data_min = float(np.min(finite))
        t_data_max = float(np.max(finite))
        user_min = self._parse_limit(self.ent_bcs_tmin)
        user_max = self._parse_limit(self.ent_bcs_tmax)
        tmin = t_data_min if user_min is None else float(user_min)
        if user_max is None:
            if self.Tc_estimate is not None and np.isfinite(self.Tc_estimate) and self.Tc_estimate > 0:
                tmax = 0.70 * float(self.Tc_estimate)
            else:
                tmax = 0.70 * t_data_max
        else:
            tmax = float(user_max)
        if tmax < tmin:
            tmin, tmax = tmax, tmin
        return tmin, tmax

    def _fit_bcs_gap(self, T, y, err=None, t_lo=None, t_hi=None):
        """Fit BCS Δ(T) only on ``[t_lo, t_hi]``. Returns ``(delta0, Tc)`` or None."""
        T = np.asarray(T, dtype=float)
        y = np.asarray(y, dtype=float)
        T_all_max = float(np.nanmax(T)) if np.any(np.isfinite(T)) else 1.0
        good = np.isfinite(T) & np.isfinite(y) & (y >= 0)
        if t_lo is not None:
            good &= T >= t_lo
        if t_hi is not None:
            good &= T <= t_hi
        if err is not None:
            e = self._symmetric_err_width(err)
            if e is not None:
                e = np.asarray(e, dtype=float)
                good &= np.isfinite(e) & (e > 0)
        if int(np.count_nonzero(good)) < 3:
            return None

        Tg, yg = T[good], y[good]
        sigma = None
        if err is not None:
            e = self._symmetric_err_width(err)
            if e is not None:
                sigma = np.asarray(e, dtype=float)[good]

        d0_guess = float(np.nanmax(yg)) if np.any(yg > 0) else 1.0
        if d0_guess <= 0:
            d0_guess = 1.0
        tc_guess = float(self.Tc_estimate) if self.Tc_estimate is not None else T_all_max
        if not np.isfinite(tc_guess) or tc_guess <= 0:
            tc_guess = max(T_all_max, float(np.nanmax(Tg)))
        tc_lo = max(float(np.nanmin(Tg)) * 0.5, 1e-3)
        tc_hi = max(T_all_max * 1.5, tc_guess * 1.5, float(np.nanmax(Tg)) * 1.5, tc_lo + 1.0)

        try:
            popt, _ = curve_fit(
                self._bcs_gap, Tg, yg,
                p0=[d0_guess, tc_guess],
                sigma=sigma, absolute_sigma=True if sigma is not None else False,
                bounds=([0.0, tc_lo], [max(d0_guess * 20.0, 1.0), tc_hi]),
                maxfev=50000,
            )
            return float(popt[0]), float(popt[1])
        except Exception:
            return None

    def _mark_bcs_fit_window(self, ax, T_arr, t_lo=None, t_hi=None):
        """Dotted guides at the BCS-fit interval (drawn once per panel)."""
        if not self.var_bcs_fit.get():
            return
        if t_lo is None or t_hi is None:
            t_lo, t_hi = self._bcs_fit_window(T_arr)
        for t in (t_lo, t_hi):
            ax.axvline(
                t, color=COLORS["gray"], linestyle=":",
                linewidth=max(0.7, PRL_LINEWIDTH * 0.55), zorder=1,
            )

    def _plot_bcs_fit_curve(
        self, ax, T_arr, y, err, color, label_prefix, amp_symbol=r'\Delta_0',
        t_lo=None, t_hi=None, T_plot_max=None,
    ):
        """BCS fit on the user T-window; solid in-window, dashed extrapolation."""
        if not self.var_bcs_fit.get():
            return
        if t_lo is None or t_hi is None:
            t_lo, t_hi = self._bcs_fit_window(T_arr)
        fit = self._fit_bcs_gap(T_arr, y, err, t_lo=t_lo, t_hi=t_hi)
        if fit is None:
            return
        d0, tc = fit
        T_arr = np.asarray(T_arr, dtype=float)
        T_min = float(np.nanmin(T_arr))
        T_max = float(np.nanmax(T_arr))
        T_end = max(T_max, float(tc), t_hi)
        if T_plot_max is not None and np.isfinite(T_plot_max):
            T_end = max(T_end, float(T_plot_max))
        T_dense = np.linspace(max(min(T_min, t_lo), 1e-6), T_end, 600)
        y_dense = self._bcs_gap(T_dense, d0, tc)
        t_solid_hi = min(t_hi, float(tc)) if np.isfinite(tc) else t_hi
        bcs_label = (
            fr'{label_prefix} BCS (${amp_symbol}={d0:.2f}$ meV, $T_c={tc:.1f}$ K)'
        )
        used_label = False
        m_in = (T_dense >= t_lo) & (T_dense <= t_solid_hi)
        if int(np.count_nonzero(m_in)) >= 2:
            ax.plot(
                T_dense[m_in], y_dense[m_in],
                color=color, linestyle="-", linewidth=PRL_LINEWIDTH,
                marker="", zorder=7, label=bcs_label,
            )
            used_label = True
        for mseg in (T_dense < t_lo, T_dense > t_solid_hi):
            if int(np.count_nonzero(mseg)) < 2:
                continue
            ax.plot(
                T_dense[mseg], y_dense[mseg],
                color=color, linestyle="--", linewidth=PRL_LINEWIDTH,
                marker="", zorder=7,
                label=bcs_label if not used_label else "_nolegend_",
            )
            used_label = True

    def _uncertainty_ymax(self, entry, values_list):
        """Y-axis top for uncertainty histograms: user value, else data-driven fallback."""
        ymax = self._parse_limit(entry)
        if ymax is not None and ymax > 0:
            return float(ymax)
        vals = []
        for v in values_list:
            if v is None:
                continue
            a = np.asarray(v, dtype=float)
            vals.append(a[np.isfinite(a)])
        if not vals:
            return 1.0
        m = float(np.nanmax(np.concatenate(vals))) if any(len(v) for v in vals) else 1.0
        return max(m * 1.15, 1e-6)

    def _plot_capped_uncertainty_bars(
        self, ax, T_all, values, color, label, side, group_width, y_max,
        overflow_label="Failed / over limit",
    ):
        """Plot uncertainty bars capped at ``y_max``.

        Over-limit or missing / failed-fit values are drawn as light-gray
        bars at full axis height (same bar geometry; not red, not background).
        """
        T_all = np.asarray(T_all, dtype=float)
        values = np.asarray(values, dtype=float)
        if T_all.size == 0:
            return

        finite = np.isfinite(values)
        in_range = finite & (values <= y_max) & (values >= 0)
        overflow = (~finite) | (finite & ((values > y_max) | (values < 0)))

        if np.any(in_range):
            plot_offset_uncertainty_bars(
                ax, T_all[in_range], values[in_range], color,
                label=label, side=side, group_width=group_width,
            )
        if np.any(overflow):
            plot_offset_uncertainty_bars(
                ax, T_all[overflow],
                np.full(int(np.count_nonzero(overflow)), y_max),
                OVERFLOW_BAR_EDGE,
                label=overflow_label, side=side, group_width=group_width,
                style="frame",
            )

    @staticmethod
    def _symmetric_err_width(err):
        """Convert symmetric σ or (err_lo, err_hi) into a single |σ|-like width for bar plots."""
        if err is None:
            return None
        if isinstance(err, tuple):
            err_lo, err_hi = np.asarray(err[0], dtype=float), np.asarray(err[1], dtype=float)
            return 0.5 * (err_lo + err_hi)
        return np.asarray(err, dtype=float)

    @staticmethod
    def _slice_err(err, mask):
        """Index a symmetric σ array or ``(err_lo, err_hi)`` with the same mask."""
        if err is None:
            return None
        if isinstance(err, tuple):
            return (np.asarray(err[0])[mask], np.asarray(err[1])[mask])
        return np.asarray(err)[mask]

    def _plot_value_series(
        self, ax, T, y, err, series_key, label, marker, linestyle='-',
        markerfacecolor=None,
    ):
        """Plot a T-series with optional error bars controlled by ``var_show_uncertainty``.

        ``err`` may be a 1-D array (symmetric) or ``(err_lo, err_hi)`` (asymmetric).
        Points with non-finite σ are drawn as markers only (no bar).
        ``markerfacecolor='white'`` marks points held out of the BCS fit.
        """
        if T is None:
            return
        kw = plot_curve_kwargs(series_key, marker=marker, linestyle=linestyle, markersize=_STEP3_MS)
        ekw = errorbar_kwargs(series_key, marker=marker, linestyle=linestyle, markersize=_STEP3_MS)
        kw["zorder"] = 5
        ekw["zorder"] = 5
        if markerfacecolor is not None:
            kw["markerfacecolor"] = markerfacecolor
            kw["markeredgecolor"] = series_key
            ekw["markerfacecolor"] = markerfacecolor
            ekw["markeredgecolor"] = series_key
        if self.var_show_uncertainty.get() and err is not None:
            if isinstance(err, tuple):
                elo = np.asarray(err[0], dtype=float)
                ehi = np.asarray(err[1], dtype=float)
                ok = np.isfinite(elo) & np.isfinite(ehi)
                if np.any(ok):
                    ax.errorbar(
                        np.asarray(T)[ok], np.asarray(y)[ok],
                        yerr=np.vstack([elo[ok], ehi[ok]]),
                        **ekw, label=label,
                    )
                bad = ~ok
                if np.any(bad):
                    ax.plot(np.asarray(T)[bad], np.asarray(y)[bad], **kw,
                            label=None if np.any(ok) else label)
            else:
                e = np.asarray(err, dtype=float)
                ok = np.isfinite(e)
                if np.any(ok):
                    ax.errorbar(
                        np.asarray(T)[ok], np.asarray(y)[ok], yerr=e[ok],
                        **ekw, label=label,
                    )
                bad = ~ok
                if np.any(bad):
                    ax.plot(np.asarray(T)[bad], np.asarray(y)[bad], **kw,
                            label=None if np.any(ok) else label)
        else:
            ax.plot(T, y, **kw, label=label)

    def _plot_delta_series_with_bcs_window(
        self, ax, T, y, err, series_key, label, marker, t_lo, t_hi,
    ):
        """Filled markers inside the BCS window; open markers held out (near Tc)."""
        if T is None:
            return
        T = np.asarray(T, dtype=float)
        y = np.asarray(y, dtype=float)
        if (not self.var_bcs_fit.get()) or T.size == 0:
            self._plot_value_series(
                ax, T, y, err, series_key, label, marker=marker, linestyle='none',
            )
            return
        in_w = (T >= t_lo) & (T <= t_hi)
        if np.any(in_w):
            self._plot_value_series(
                ax, T[in_w], y[in_w], self._slice_err(err, in_w),
                series_key, label, marker=marker, linestyle='none',
            )
        if np.any(~in_w):
            self._plot_value_series(
                ax, T[~in_w], y[~in_w], self._slice_err(err, ~in_w),
                series_key, label if not np.any(in_w) else "_nolegend_",
                marker=marker, linestyle='none',
                markerfacecolor="white",
            )

    def _prepare_delta_pair_correction(self, T_arr, sp_d, sp_e, w_d, w_e):
        """Fit ξ(T) on weighted Δ only; apply the same ξ to both SP and weighted.

        Returns
        -------
        sp_d, sp_e, w_d, w_e, fit_params
            Corrected series when correction is on (``err`` as ``(err_lo, err_hi)``);
            raw series when off. ``fit_params`` is the weighted ``(a, b)`` for the
            preview line when correction is off (else None).
        """
        fit_params = None
        try:
            a, b = self._fit_mu_drift(T_arr, w_d, w_e)
        except Exception as e:
            if self.var_mu_correct.get():
                messagebox.showwarning("μ(T) Correction", f"Correction skipped: {e}")
            return sp_d, sp_e, w_d, w_e, None

        if self.var_mu_correct.get():
            sp_c, sp_lo, sp_hi = self._apply_mu_correction_with_fit(T_arr, sp_d, sp_e, a, b)
            w_c, w_lo, w_hi = self._apply_mu_correction_with_fit(T_arr, w_d, w_e, a, b)
            return sp_c, (sp_lo, sp_hi), w_c, (w_lo, w_hi), None

        # Correction off: keep raw data, preview the weighted ξ(T) fit only
        return sp_d, sp_e, w_d, w_e, (a, b)

    # =============================================================================
    # --- UI Building ---
    # =============================================================================
    def _build_ui(self):
        # Scrollable container for the left control panel
        left_container = ttk.Frame(self, width=320)
        left_container.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        left_container.pack_propagate(False)

        self.left_canvas = tk.Canvas(left_container, width=320, highlightthickness=0, borderwidth=0)
        left_scroll = ttk.Scrollbar(left_container, orient=tk.VERTICAL, command=self.left_canvas.yview)
        self.left_canvas.configure(yscrollcommand=left_scroll.set)
        left_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.left_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Interior frame that actually holds the controls
        self.left_panel = ttk.Frame(self.left_canvas)
        self._left_window = self.left_canvas.create_window((0, 0), window=self.left_panel, anchor="nw")

        # Keep the scrollregion and interior width in sync with the content/canvas
        self.left_panel.bind(
            "<Configure>",
            lambda e: self.left_canvas.configure(scrollregion=self.left_canvas.bbox("all")),
        )
        self.left_canvas.bind(
            "<Configure>",
            lambda e: self.left_canvas.itemconfigure(self._left_window, width=e.width),
        )

        # Mouse-wheel scrolling (only while the pointer is over the left panel)
        self.left_canvas.bind("<Enter>", self._bind_left_mousewheel)
        self.left_canvas.bind("<Leave>", self._unbind_left_mousewheel)

        self.right_panel = ttk.Frame(self)
        self.right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self._build_left_panel()
        self._build_plot_area()

    def _bind_left_mousewheel(self, _event=None):
        # Windows / macOS
        self.left_canvas.bind_all("<MouseWheel>", self._on_left_mousewheel)
        # Linux (X11) uses Button-4 / Button-5
        self.left_canvas.bind_all("<Button-4>", self._on_left_mousewheel)
        self.left_canvas.bind_all("<Button-5>", self._on_left_mousewheel)

    def _unbind_left_mousewheel(self, _event=None):
        self.left_canvas.unbind_all("<MouseWheel>")
        self.left_canvas.unbind_all("<Button-4>")
        self.left_canvas.unbind_all("<Button-5>")

    def _on_left_mousewheel(self, event):
        if getattr(event, "num", None) == 4:
            delta = -1
        elif getattr(event, "num", None) == 5:
            delta = 1
        else:
            # macOS gives small deltas; Windows gives multiples of 120
            delta = -1 if event.delta > 0 else 1
        self.left_canvas.yview_scroll(delta, "units")

    def _build_left_panel(self):
        # 1. Data Loading
        frame_load = ttk.LabelFrame(self.left_panel, text="1. Load Data", padding=5)
        frame_load.pack(fill=tk.X, pady=5)
        ttk.Button(frame_load, text="Load from Directory (Files)", command=self.load_from_folder).pack(fill=tk.X, pady=2)
        ttk.Button(frame_load, text="Load from Step 2 (Memory)", command=self.load_from_step2).pack(fill=tk.X, pady=2)
        self.lbl_status = ttk.Label(frame_load, text="Status: No data loaded.", foreground="blue")
        self.lbl_status.pack(fill=tk.X, pady=2)

        # 2. Analysis Parameters
        frame_param = ttk.LabelFrame(self.left_panel, text="2. Analysis Parameters", padding=5)
        frame_param.pack(fill=tk.X, pady=5)
        
        f1 = ttk.Frame(frame_param); f1.pack(fill=tk.X, pady=2)
        ttk.Label(f1, text="P-value Threshold:").pack(side=tk.LEFT)
        self.ent_p_thresh = ttk.Entry(f1, width=8); self.ent_p_thresh.insert(0, "0.05"); self.ent_p_thresh.pack(side=tk.RIGHT)
        
        f2 = ttk.Frame(frame_param); f2.pack(fill=tk.X, pady=2)
        ttk.Label(f2, text="Target k (or 'kF'):").pack(side=tk.LEFT)
        self.ent_k_ref = ttk.Entry(f2, width=8); self.ent_k_ref.insert(0, "kF"); self.ent_k_ref.pack(side=tk.RIGHT)

        # Note: Err Mult for Weighting removed — weighted results must be provided by Step 2 export or memory
        
        ttk.Button(frame_param, text="Recalculate Physics", command=self._calculate_physics).pack(fill=tk.X, pady=5)

        # 2b. Chemical-potential μ(T) correction (applied to the Δ-vs-T panel)
        frame_corr = ttk.LabelFrame(self.left_panel, text="2b. μ(T) Correction (Δ panel)", padding=5)
        frame_corr.pack(fill=tk.X, pady=5)
        ttk.Checkbutton(
            frame_corr, text="Enable μ(T) correction", variable=self.var_mu_correct,
            command=self._update_plot,
        ).pack(anchor=tk.W, pady=2)

        fc1 = ttk.Frame(frame_corr); fc1.pack(fill=tk.X, pady=2)
        ttk.Label(fc1, text="Fit T ≥ (K):").pack(side=tk.LEFT)
        self.ent_corr_tmin = ttk.Entry(fc1, width=8)
        self.ent_corr_tmin.insert(0, "auto")
        self.ent_corr_tmin.pack(side=tk.RIGHT)

        ttk.Button(frame_corr, text="Recalculate", command=self._update_plot).pack(fill=tk.X, pady=5)

        # 3. Visualization Mode
        frame_vis = ttk.LabelFrame(self.left_panel, text="3. Visualization Mode", padding=5)
        frame_vis.pack(fill=tk.X, pady=5)
        for m in _STEP3_PANEL_MODES:
            ttk.Radiobutton(frame_vis, text=m, variable=self.show_mode, value=m, command=self._update_plot).pack(anchor=tk.W, pady=2)
        for m in _STEP3_UNCERTAINTY_MODES:
            ttk.Radiobutton(frame_vis, text=m, variable=self.show_mode, value=m, command=self._update_plot).pack(anchor=tk.W, pady=2)
        ttk.Checkbutton(
            frame_vis, text="Show error bars (Δ, Γ)", variable=self.var_show_uncertainty,
            command=self._update_plot,
        ).pack(anchor=tk.W, pady=2)
        ttk.Checkbutton(
            frame_vis, text="Show BCS fit (Δ, Γ)", variable=self.var_bcs_fit,
            command=self._update_plot,
        ).pack(anchor=tk.W, pady=2)
        fbcs = ttk.Frame(frame_vis)
        fbcs.pack(fill=tk.X, pady=2)
        ttk.Label(fbcs, text="BCS fit T (K):").pack(side=tk.LEFT)
        self.ent_bcs_tmin = ttk.Entry(fbcs, width=6)
        self.ent_bcs_tmin.insert(0, "auto")
        self.ent_bcs_tmin.pack(side=tk.LEFT, padx=1)
        ttk.Label(fbcs, text="to").pack(side=tk.LEFT)
        self.ent_bcs_tmax = ttk.Entry(fbcs, width=6)
        self.ent_bcs_tmax.insert(0, "auto")
        self.ent_bcs_tmax.pack(side=tk.LEFT, padx=1)
        for _w in (self.ent_bcs_tmin, self.ent_bcs_tmax):
            _w.bind("<Return>", lambda _e: self._update_plot())

        frame_export = ttk.LabelFrame(self.left_panel, text="5. Export Panels", padding=5)
        frame_export.pack(fill=tk.X, pady=5)
        ttk.Button(
            frame_export, text="Export All Panels (PDF)",
            command=lambda: self._export_all_panels("pdf"),
        ).pack(fill=tk.X, pady=2)
        ttk.Button(
            frame_export, text="Export All Panels (SVG)",
            command=lambda: self._export_all_panels("svg"),
        ).pack(fill=tk.X, pady=2)
        ttk.Button(
            frame_export, text="Export 2×2 Composite (PDF)",
            command=lambda: self._export_composite("pdf"),
        ).pack(fill=tk.X, pady=2)

        # 4. Display Range Override
        frame_lim = ttk.LabelFrame(self.left_panel, text="4. Display Ranges", padding=5)
        frame_lim.pack(fill=tk.X, pady=5)
        
        # Temp limits
        f_t = ttk.Frame(frame_lim); f_t.pack(fill=tk.X, pady=2)
        ttk.Label(f_t, text="Temp (K):", width=12).pack(side=tk.LEFT)
        self.ent_t_min = ttk.Entry(f_t, width=6); self.ent_t_min.insert(0, "auto"); self.ent_t_min.pack(side=tk.LEFT, padx=1)
        ttk.Label(f_t, text="to").pack(side=tk.LEFT); 
        self.ent_t_max = ttk.Entry(f_t, width=6); self.ent_t_max.insert(0, "auto"); self.ent_t_max.pack(side=tk.LEFT, padx=1)

        # Delta limits
        f_d = ttk.Frame(frame_lim); f_d.pack(fill=tk.X, pady=2)
        ttk.Label(f_d, text="Delta (meV):", width=12).pack(side=tk.LEFT)
        self.ent_d_min = ttk.Entry(f_d, width=6); self.ent_d_min.insert(0, "auto"); self.ent_d_min.pack(side=tk.LEFT, padx=1)
        ttk.Label(f_d, text="to").pack(side=tk.LEFT); 
        self.ent_d_max = ttk.Entry(f_d, width=6); self.ent_d_max.insert(0, "auto"); self.ent_d_max.pack(side=tk.LEFT, padx=1)

        # |σ_Δ| display: always from 0 to user max (over/missing → light-gray frames)
        f_sd = ttk.Frame(frame_lim); f_sd.pack(fill=tk.X, pady=2)
        ttk.Label(f_sd, text="|σ_Δ| max (meV):", width=14).pack(side=tk.LEFT)
        self.ent_sig_d_max = ttk.Entry(f_sd, width=8)
        self.ent_sig_d_max.insert(0, "2")
        self.ent_sig_d_max.pack(side=tk.RIGHT)

        # Gamma limits (left axis)
        f_g = ttk.Frame(frame_lim); f_g.pack(fill=tk.X, pady=2)
        ttk.Label(f_g, text="Gamma (meV):", width=12).pack(side=tk.LEFT)
        self.ent_g_min = ttk.Entry(f_g, width=6); self.ent_g_min.insert(0, "auto"); self.ent_g_min.pack(side=tk.LEFT, padx=1)
        ttk.Label(f_g, text="to").pack(side=tk.LEFT); 
        self.ent_g_max = ttk.Entry(f_g, width=6); self.ent_g_max.insert(0, "auto"); self.ent_g_max.pack(side=tk.LEFT, padx=1)

        # |σ_Γ| display: always from 0 to user max
        f_sg = ttk.Frame(frame_lim); f_sg.pack(fill=tk.X, pady=2)
        ttk.Label(f_sg, text="|σ_Γ| max (meV):", width=14).pack(side=tk.LEFT)
        self.ent_sig_g_max = ttk.Entry(f_sg, width=8)
        self.ent_sig_g_max.insert(0, "2")
        self.ent_sig_g_max.pack(side=tk.RIGHT)

        ttk.Button(frame_lim, text="Apply Limits", command=self._update_plot).pack(fill=tk.X, pady=5)

    def _build_plot_area(self):
        self.fig = plt.Figure(figsize=step3_panel_figsize(for_gui=True), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.right_panel)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.right_panel)
        self.toolbar.update()

    # =============================================================================
    # --- Data Loading (Updated with kF Header Parsing) ---
    # =============================================================================
    def load_from_folder(self):
        folder_path = filedialog.askdirectory(title="Select Step 2 Output Directory")
        if not folder_path: return
        
        try:
            self.temp_data = []
            files = [f for f in os.listdir(folder_path) if f.startswith('fit_results_') and f.endswith('.txt')]
            if not files: return messagebox.showerror("Error", "No 'fit_results_*.txt' files found.")
                
            for fname in files:
                fpath = os.path.join(folder_path, fname)
                
                T_val, kF_val = None, None
                weighted_delta = weighted_err = weighted_gamma = weighted_gamma_err = None
                weighted_sel_k = None
                
                # Parse the 4-line header for T, kF, and the MMWA summary
                with open(fpath, 'r') as f:
                    lines = [f.readline() for _ in range(4)]
                    # Line 2: Temperature: 14.0 K, kF: ..., WeightedDelta: ..., WeightedErr: ...
                    if len(lines) >= 2 and 'Temperature' in lines[1]:
                        parts = [p.strip() for p in lines[1].split(',')]
                        for p in parts:
                            if ':' not in p:
                                continue
                            key, val = [x.strip() for x in p.split(':', 1)]
                            if key == 'Temperature':
                                try:
                                    T_val = float(val.replace('K', '').strip())
                                except Exception:
                                    pass
                            elif key == 'kF':
                                try:
                                    kF_val = float(val)
                                except Exception:
                                    pass
                            elif key == 'WeightedDelta':
                                try:
                                    weighted_delta = float(val)
                                except Exception:
                                    weighted_delta = None
                            elif key == 'WeightedErr':
                                try:
                                    weighted_err = float(val)
                                except Exception:
                                    weighted_err = None
                            elif key == 'WeightedGamma':
                                try:
                                    weighted_gamma = float(val)
                                except Exception:
                                    weighted_gamma = None
                            elif key == 'WeightedGammaErr':
                                try:
                                    weighted_gamma_err = float(val)
                                except Exception:
                                    weighted_gamma_err = None
                            elif key == 'WeightedSelK':
                                try:
                                    weighted_sel_k = np.array([float(x) for x in val.split(';') if x.strip()])
                                except Exception:
                                    weighted_sel_k = None
                    else:
                        weighted_delta = None
                        weighted_err = None
                        weighted_gamma = None
                        weighted_gamma_err = None
                        weighted_sel_k = None
                
                if T_val is None:
                    T_val = float(fname.replace('fit_results_', '').replace('K.txt', ''))

                # Column matrix after the 4-line header
                data = np.loadtxt(fpath, skiprows=4)
                if data.shape[1] < 8:
                    continue

                nrows = data.shape[0]
                if data.shape[1] >= 9:
                    delta_valid_arr = data[:, 8].astype(float) >= 0.5
                else:
                    delta_valid_arr = np.ones(nrows, dtype=bool)

                # Fix very small or zero p-values that may have been written as 0.0
                p_col = data[:, 7].astype(float)
                p_col = np.where(p_col <= 0.0, np.finfo(float).tiny, p_col)

                # Build weighted_res only if all four weighted values are present
                weighted_res_obj = None
                if ('weighted_delta' in locals() and weighted_delta is not None and
                    'weighted_err' in locals() and weighted_err is not None and
                    'weighted_gamma' in locals() and weighted_gamma is not None and
                    'weighted_gamma_err' in locals() and weighted_gamma_err is not None):
                    weighted_res_obj = {
                        'delta_best': weighted_delta,
                        'error_best': weighted_err,
                        'gamma_best': weighted_gamma,
                        'gamma_err': weighted_gamma_err
                    }
                    if weighted_sel_k is not None and len(np.asarray(weighted_sel_k, dtype=float).ravel()) > 0:
                        weighted_res_obj['sel_k'] = np.asarray(weighted_sel_k, dtype=float).ravel()

                self.temp_data.append({
                    'T': T_val, 'kF': kF_val,
                    'k_vals': data[:, 0], 'delta_vals': data[:, 1], 'err_vals': data[:, 2],
                    'gamma_vals': data[:, 3], 'gamma_err_vals': data[:, 4],
                    'RSS_gap': data[:, 5], 'RSS_met': data[:, 6], 'p_vals': p_col,
                    'delta_valid': delta_valid_arr,
                    'weighted_res': weighted_res_obj
                })
            
            # Verify every file contains weighted results; otherwise prompt format error
            missing = [d for d in self.temp_data if d.get('weighted_res') is None]
            if missing:
                self.temp_data = []
                return messagebox.showerror("Format Error", "One or more files are missing weighted results. Please export from Step 2 with WeightedDelta/WeightedErr or use 'Load from Step 2 (Memory)'.")

            self.temp_data.sort(key=lambda x: x['T'])
            self.lbl_status.config(text=f"Loaded {len(self.temp_data)} files.", foreground="green")
            self._calculate_physics()
        except Exception as e:
            messagebox.showerror("Load Error", str(e))

    def load_from_step2(self):
        if not self.controller or not hasattr(self.controller, 'step2_module'): return
        step2 = self.controller.step2_module
        if not hasattr(step2, 'saved_results') or not step2.saved_results: return
            
        try:
            self.temp_data = []
            for key, res in step2.saved_results.items():
                stats = res.get('final_stats', {})
                # Handle p_vals from memory: replace non-positive with tiny positive float
                p_vals_arr = np.array(stats.get('p_vals', []), dtype=float)
                if p_vals_arr.size > 0:
                    p_vals_arr = np.where(p_vals_arr <= 0.0, np.finfo(float).tiny, p_vals_arr)

                k_vals_mem = np.array(res.get('k_points', []))
                dv = stats.get('delta_point_valid')
                if dv is None:
                    delta_valid_mem = np.ones(len(k_vals_mem), dtype=bool)
                else:
                    delta_valid_mem = np.asarray(dv, dtype=bool).ravel()
                    if delta_valid_mem.size != len(k_vals_mem):
                        delta_valid_mem = np.ones(len(k_vals_mem), dtype=bool)

                self.temp_data.append({
                    'T': res.get('Temperature', 0),
                    'kF': res.get('kF', None),
                    'k_vals': k_vals_mem,
                    'delta_vals': np.array(stats.get('delta_fit', [])),
                    'err_vals': np.array(stats.get('delta_err', [])),
                    'gamma_vals': np.array(stats.get('gamma_fit', [])),
                    'gamma_err_vals': np.array(stats.get('gamma_err', [])),
                    'RSS_gap': np.array(stats.get('RSS_gap', [])),
                    'RSS_met': np.array(stats.get('RSS_met', [])),
                    'p_vals': p_vals_arr,
                    'delta_valid': delta_valid_mem,
                    'weighted_res': res.get('weighted_res', None)
                })
            # If any memory entries lack weighted_res or required keys, prompt Format Error
            def has_required_weighted(w):
                return (w is not None and isinstance(w, dict) and 'delta_best' in w and 'error_best' in w and 'gamma_best' in w and 'gamma_err' in w)

            missing_mem = [d for d in self.temp_data if not has_required_weighted(d.get('weighted_res'))]
            if missing_mem:
                self.temp_data = []
                return messagebox.showerror("Format Error", "Saved results from Step 2 are missing complete weighted results (delta and gamma). Please ensure Step 2 saving includes weighted results.")
            self.temp_data.sort(key=lambda x: x['T'])
            self.lbl_status.config(text=f"Loaded {len(self.temp_data)} sets (Memory).", foreground="green")
            self._calculate_physics()
        except Exception as e:
            messagebox.showerror("Load Error", str(e))

    # =============================================================================
    # --- Physics Calculation (Single Point & Dynamic Weighted) ---
    # =============================================================================
    @staticmethod
    def _rss_mean_on_weighted_k(k_vals, rss_gap, rss_met, weighted):
        """Mean RSS on momentum points used for inverse-variance Δ (Step 2 ``sel_k``)."""
        k_vals = np.asarray(k_vals, dtype=float)
        rss_gap = np.asarray(rss_gap, dtype=float)
        rss_met = np.asarray(rss_met, dtype=float)
        if weighted is None or k_vals.size == 0:
            return float(np.mean(rss_gap)), float(np.mean(rss_met))
        sel_k = weighted.get('sel_k')
        if sel_k is None:
            return float(np.mean(rss_gap)), float(np.mean(rss_met))
        sel_k = np.asarray(sel_k, dtype=float).ravel()
        if sel_k.size == 0:
            return float(np.mean(rss_gap)), float(np.mean(rss_met))
        idxs = sorted({int(np.argmin(np.abs(k_vals - sk))) for sk in sel_k})
        if not idxs:
            return float(np.mean(rss_gap)), float(np.mean(rss_met))
        ix = np.array(idxs, dtype=int)
        return float(np.mean(rss_gap[ix])), float(np.mean(rss_met[ix]))

    def _calculate_physics(self):
        if not self.temp_data: return
        
        try:
            p_thresh = float(self.ent_p_thresh.get())
            k_ref_str = self.ent_k_ref.get()
        except ValueError:
            return messagebox.showerror("Error", "Invalid analysis parameters.")
            
        self.extracted_physics = []
        self.Tc_estimate = None
        gap_closed = False
        
        for item in self.temp_data:
            T = item['T']
            kF_val = item['kF']
            k_vals, p_vals = item['k_vals'], item['p_vals']
            delta_valid = item.get('delta_valid')
            if delta_valid is None:
                delta_valid = np.ones(len(k_vals), dtype=bool)
            else:
                delta_valid = np.asarray(delta_valid, dtype=bool).ravel()
                if delta_valid.size != len(k_vals):
                    delta_valid = np.ones(len(k_vals), dtype=bool)
            
            # Determine Target k 
            if k_ref_str.lower() == 'kf':
                if kF_val is not None:
                    k_target = kF_val
                else:
                    k_target = k_vals[len(k_vals) // 2] 
            else:
                k_target = float(k_ref_str)
                
            target_idx = np.argmin(np.abs(k_vals - k_target))
            
            # P-value at target k: always from full fit (same with/without delta_valid in files)
            sp_p_val = p_vals[target_idx]

            # Single-point Δ / Γ: omit invalid points for those curves only
            if delta_valid[target_idx]:
                sp_delta = item['delta_vals'][target_idx]
                sp_err = item['err_vals'][target_idx]
                sp_gamma = item['gamma_vals'][target_idx]
                sp_g_err = item['gamma_err_vals'][target_idx]
            else:
                sp_delta = sp_err = sp_gamma = sp_g_err = np.nan
            
            # Estimate Tc (from p-value at target k, independent of Δ validity flag)
            if np.isfinite(sp_p_val) and sp_p_val > p_thresh and not gap_closed:
                self.Tc_estimate = T
                gap_closed = True
                
            # 2. Weighted Method: use Step 2's precomputed weighted result
            weighted = item.get('weighted_res', None)
            if weighted is None or weighted.get('delta_best', None) is None or weighted.get('gamma_best', None) is None:
                return messagebox.showerror("Format Error", "Missing weighted delta/gamma results for T={}. Please export from Step 2 with WeightedDelta/WeightedErr/WeightedGamma/WeightedGammaErr or load from Step 2 memory.".format(T))

            w_delta = weighted.get('delta_best')
            w_err = weighted.get('error_best')
            w_gamma = weighted.get('gamma_best')
            w_g_err = weighted.get('gamma_err')

            rss_gap_mean, rss_met_mean = self._rss_mean_on_weighted_k(
                k_vals, item['RSS_gap'], item['RSS_met'], weighted,
            )

            self.extracted_physics.append({
                'T': T, 'k_target': k_target,
                'sp_p_val': sp_p_val,
                'rss_gap_mean': rss_gap_mean,
                'rss_met_mean': rss_met_mean,
                'sp_delta': sp_delta, 'sp_err': sp_err,
                'w_delta': w_delta, 'w_err': w_err,
                'sp_gamma': sp_gamma, 'sp_g_err': sp_g_err,
                'w_gamma': w_gamma, 'w_g_err': w_g_err,
                'weighted_res': weighted
            })
            
        self._update_plot()

    # =============================================================================
    # --- Plotting Logic ---
    # =============================================================================
    @staticmethod
    def _panel_title(mode):
        for m, tag, short in list(_STEP3_PANELS) + list(_STEP3_UNCERTAINTY_PANELS):
            if m == mode:
                return f"{tag} {short}"
        return mode

    def _draw_uncertainty_only_panel(self, ax, mode, T_arr, T_max, title):
        """Standalone |σ_Δ| or |σ_Γ| bar panel (y from 0 to user max; over/missing as light-gray bars)."""
        self._apply_gapless_shading(ax, T_max)
        group_w = temperature_bar_spacing(T_arr)
        T_arr = np.asarray(T_arr, dtype=float)

        if mode == "Delta Uncertainty vs T":
            sp_d = np.array([p['sp_delta'] for p in self.extracted_physics], dtype=float)
            sp_e = np.abs(np.array([p['sp_err'] for p in self.extracted_physics], dtype=float))
            w_d = np.array([p['w_delta'] for p in self.extracted_physics], dtype=float)
            w_e = np.abs(np.array([p['w_err'] for p in self.extracted_physics], dtype=float))
            _, sp_err_c, _, w_err_c, _ = self._prepare_delta_pair_correction(
                T_arr, sp_d, sp_e, w_d, w_e,
            )
            sp_w = self._symmetric_err_width(sp_err_c) * 1000
            w_w = self._symmetric_err_width(w_err_c) * 1000
            # Align to full T grid (missing → NaN → light-gray bars)
            sp_full = np.full(T_arr.shape, np.nan)
            w_full = np.full(T_arr.shape, np.nan)
            if sp_w is not None:
                sp_full[:] = sp_w
            if w_w is not None:
                w_full[:] = w_w
            y_max = self._uncertainty_ymax(self.ent_sig_d_max, [sp_full, w_full])
            self._plot_capped_uncertainty_bars(
                ax, T_arr, sp_full, SERIES["single_point"],
                r'Single-point $|\sigma_\Delta|$', "left", group_w, y_max,
            )
            self._plot_capped_uncertainty_bars(
                ax, T_arr, w_full, SERIES["weighted"],
                r'Weighted $|\sigma_\Delta|$', "right", group_w, y_max,
            )
            set_axis_labels(ax, xlabel='Temperature $T$ (K)', ylabel=r'$|\sigma_\Delta|$ (meV)', title=title)
            ax.set_ylim(0.0, y_max)
        else:
            sp_ge = np.abs(np.array([p['sp_g_err'] for p in self.extracted_physics], dtype=float)) * 1000
            w_ge = np.abs(np.array([p['w_g_err'] for p in self.extracted_physics], dtype=float)) * 1000
            y_max = self._uncertainty_ymax(self.ent_sig_g_max, [sp_ge, w_ge])
            self._plot_capped_uncertainty_bars(
                ax, T_arr, sp_ge, SERIES["single_point"],
                r'Single-point $|\sigma_\Gamma|$', "left", group_w, y_max,
            )
            self._plot_capped_uncertainty_bars(
                ax, T_arr, w_ge, SERIES["gamma_weighted"],
                r'Weighted $|\sigma_\Gamma|$', "right", group_w, y_max,
            )
            set_axis_labels(ax, xlabel='Temperature $T$ (K)', ylabel=r'$|\sigma_\Gamma|$ (meV)', title=title)
            ax.set_ylim(0.0, y_max)

        # Deduplicate overflow legend
        handles, labels = ax.get_legend_handles_labels()
        seen, H, L = set(), [], []
        for h, lab in zip(handles, labels):
            if lab in seen:
                continue
            seen.add(lab)
            H.append(h)
            L.append(lab)
        ax.legend(H, L, **step3_legend_kwargs())
        return None

    def _apply_temperature_xlim(self, ax, T_arr, T_max):
        tmin = self._parse_limit(self.ent_t_min)
        tmax = self._parse_limit(self.ent_t_max)
        if tmin is not None or tmax is not None:
            ax.set_xlim(left=tmin, right=tmax)
        else:
            pad = max(0.08 * (T_max - np.min(T_arr)), 1.5)
            ax.set_xlim(left=np.min(T_arr) - pad, right=T_max + pad)

    def _draw_panel(self, ax, mode, T_arr, T_max):
        """Draw one Step 3 panel. Returns twin axis when used, else None."""
        ax2 = None
        title = self._panel_title(mode)

        if mode == "RSS Comparison vs T":
            rss_g = [p['rss_gap_mean'] for p in self.extracted_physics]
            rss_m = [p['rss_met_mean'] for p in self.extracted_physics]

            ax.plot(T_arr, rss_g, **plot_curve_kwargs(
                SERIES["gap_model"], marker='o', linestyle='-', markersize=_STEP3_MS),
                label='Gap model')
            ax.plot(T_arr, rss_m, **plot_curve_kwargs(
                SERIES["metal_model"], marker='s', linestyle='--', markersize=_STEP3_MS),
                label='Zero-gap model')
            set_axis_labels(ax, xlabel='Temperature $T$ (K)', ylabel='RSS', title=title)
            self._apply_gapless_shading(ax, T_max)
            ax.legend(**step3_legend_kwargs())

        elif mode == "P-value vs T":
            p_vals = np.array([p['sp_p_val'] for p in self.extracted_physics], dtype=float)
            log_p = np.log10(np.clip(p_vals, np.finfo(float).tiny, 1.0))

            ax.plot(T_arr, log_p, **plot_curve_kwargs(
                SERIES["pvalue"], marker='D', linestyle='-', markersize=_STEP3_MS),
                label=r'$\log_{10}(p)$')
            try:
                thresh = float(self.ent_p_thresh.get())
                ax.axhline(np.log10(thresh), label=f'Threshold ($p={thresh}$)', **reference_line_kwargs())
            except Exception:
                pass
            set_axis_labels(ax, xlabel='Temperature $T$ (K)', ylabel=r'$\log_{10}(p)$', title=title)
            self._apply_gapless_shading(ax, T_max)
            ax.legend(**step3_legend_kwargs())

        elif mode == "SC Gap (Delta) vs T":
            sp_d = np.array([p['sp_delta'] for p in self.extracted_physics], dtype=float)
            sp_e = np.abs(np.array([p['sp_err'] for p in self.extracted_physics], dtype=float))
            w_d = np.array([p['w_delta'] for p in self.extracted_physics], dtype=float)
            w_e = np.abs(np.array([p['w_err'] for p in self.extracted_physics], dtype=float))

            # ξ(T) fit on weighted only; same ξ corrects both SP and weighted
            sp_d, sp_e, w_d, w_e, w_fit = self._prepare_delta_pair_correction(
                T_arr, sp_d, sp_e, w_d, w_e,
            )

            sp_d = sp_d * 1000
            w_d = w_d * 1000
            if isinstance(sp_e, tuple):
                sp_e = (sp_e[0] * 1000, sp_e[1] * 1000)
            else:
                sp_e = sp_e * 1000
            if isinstance(w_e, tuple):
                w_e = (w_e[0] * 1000, w_e[1] * 1000)
            else:
                w_e = w_e * 1000

            m_sp = np.isfinite(sp_d)
            m_w = np.isfinite(w_d)
            T_sp, sp_ds, sp_es = self._finite_sorted_series(T_arr, sp_d, sp_e, m_sp)
            T_w, w_ds, w_es = self._finite_sorted_series(T_arr, w_d, w_e, m_w)

            t_lo, t_hi = self._bcs_fit_window(T_arr)

            # Scatter only. Filled markers = BCS-fit window; open = held out (near Tc).
            self._plot_delta_series_with_bcs_window(
                ax, T_sp, sp_ds, sp_es, SERIES["single_point"],
                r'Single-point $\Delta$', 'o', t_lo, t_hi,
            )
            self._plot_delta_series_with_bcs_window(
                ax, T_w, w_ds, w_es, SERIES["weighted"],
                r'Weighted $\Delta$', 's', t_lo, t_hi,
            )

            # BCS fits on the user window only; dashed outside (extrapolation toward Tc)
            if T_sp is not None:
                self._plot_bcs_fit_curve(
                    ax, T_sp, sp_ds, sp_es, SERIES["single_point"], r'Single-point',
                    t_lo=t_lo, t_hi=t_hi, T_plot_max=T_max,
                )
            if T_w is not None:
                self._plot_bcs_fit_curve(
                    ax, T_w, w_ds, w_es, SERIES["weighted"], r'Weighted',
                    t_lo=t_lo, t_hi=t_hi, T_plot_max=T_max,
                )

            # Preview the weighted ξ(T) fit when correction is off
            if w_fit is not None:
                T_dense = np.linspace(np.min(T_arr), np.max(T_arr), 400)
                a, b = w_fit
                ax.plot(T_dense, self._linear_drift(T_dense, a, b) * 1000,
                        color=SERIES["weighted"], linestyle=':', linewidth=PRL_LINEWIDTH,
                        marker='', zorder=2,
                        label=fr'$\xi(T)$ linear fit, $b={b*1e3:.2f}$ meV/K')

            set_axis_labels(ax, xlabel='Temperature $T$ (K)', ylabel=r'$\Delta$ (meV)', title=title)
            self._apply_gapless_shading(ax, T_max)
            self._mark_bcs_fit_window(ax, T_arr, t_lo=t_lo, t_hi=t_hi)
            if ax.get_legend_handles_labels()[0]:
                ax.legend(**step3_legend_kwargs(loc="upper right"))
            ax.set_ylim(bottom=self._parse_limit(self.ent_d_min), top=self._parse_limit(self.ent_d_max))

        elif mode == "Gamma vs T":
            sp_g = np.array([p['sp_gamma'] for p in self.extracted_physics], dtype=float) * 1000
            sp_ge = np.abs(np.array([p['sp_g_err'] for p in self.extracted_physics], dtype=float)) * 1000
            w_g = np.array([p['w_gamma'] for p in self.extracted_physics], dtype=float) * 1000
            w_ge = np.abs(np.array([p['w_g_err'] for p in self.extracted_physics], dtype=float)) * 1000

            m_sp = np.isfinite(sp_g)
            m_w = np.isfinite(w_g)
            T_sp, sp_gs, sp_ges = self._finite_sorted_series(T_arr, sp_g, sp_ge, m_sp)
            T_w, w_gs, w_ges = self._finite_sorted_series(T_arr, w_g, w_ge, m_w)

            t_lo_g, t_hi_g = self._bcs_fit_window(T_arr)
            self._plot_delta_series_with_bcs_window(
                ax, T_sp, sp_gs, sp_ges, SERIES["single_point"],
                r'Single-point $\Gamma$', 'o', t_lo_g, t_hi_g,
            )
            self._plot_delta_series_with_bcs_window(
                ax, T_w, w_gs, w_ges, SERIES["gamma_weighted"],
                r'Weighted $\Gamma$', 's', t_lo_g, t_hi_g,
            )

            # BCS-form fits for SP / weighted Γ (same phenomenological form)
            if T_sp is not None:
                self._plot_bcs_fit_curve(
                    ax, T_sp, sp_gs, sp_ges, SERIES["single_point"], r'Single-point',
                    amp_symbol=r'\Gamma_0', t_lo=t_lo_g, t_hi=t_hi_g, T_plot_max=T_max,
                )
            if T_w is not None:
                self._plot_bcs_fit_curve(
                    ax, T_w, w_gs, w_ges, SERIES["gamma_weighted"], r'Weighted',
                    amp_symbol=r'\Gamma_0', t_lo=t_lo_g, t_hi=t_hi_g, T_plot_max=T_max,
                )

            set_axis_labels(ax, xlabel='Temperature $T$ (K)', ylabel=r'$\Gamma$ (meV)', title=title)
            self._apply_gapless_shading(ax, T_max)
            self._mark_bcs_fit_window(ax, T_arr, t_lo=t_lo_g, t_hi=t_hi_g)
            if ax.get_legend_handles_labels()[0]:
                ax.legend(**step3_legend_kwargs())
            ax.set_ylim(bottom=self._parse_limit(self.ent_g_min), top=self._parse_limit(self.ent_g_max))

        elif mode in _STEP3_UNCERTAINTY_MODES:
            ax2 = self._draw_uncertainty_only_panel(ax, mode, T_arr, T_max, title)

        return ax2

    def _render_panel_figure(self, mode):
        """Build one publication-sized panel figure."""
        T_arr = np.array([p['T'] for p in self.extracted_physics])
        T_max = np.max(T_arr)
        fig = plt.Figure(figsize=step3_panel_figsize(for_gui=False), dpi=PRL_DPI)
        ax = fig.add_subplot(111)
        ax2 = self._draw_panel(ax, mode, T_arr, T_max)
        self._apply_temperature_xlim(ax, T_arr, T_max)
        step3_apply_panel_layout(fig, ax, ax2=ax2)
        return fig

    def _export_all_panels(self, fmt):
        if not self.extracted_physics:
            return messagebox.showwarning("Export", "No data loaded.")
        folder = filedialog.askdirectory(title="Select export folder")
        if not folder:
            return
        try:
            export_panels = list(_STEP3_PANELS) + list(_STEP3_UNCERTAINTY_PANELS)
            for mode, tag, _short in export_panels:
                stem = tag.strip("()").lower()
                fig = self._render_panel_figure(mode)
                path = os.path.join(folder, f"step3_panel_{stem}.{fmt}")
                step3_save_panel(fig, path)
                plt.close(fig)
            messagebox.showinfo("Export", f"Saved {len(export_panels)} panels to:\n{folder}")
        except Exception as e:
            messagebox.showerror("Export Error", str(e))

    def _export_composite(self, fmt):
        if not self.extracted_physics:
            return messagebox.showwarning("Export", "No data loaded.")
        folder = filedialog.askdirectory(title="Select export folder")
        if not folder:
            return
        try:
            from prl_plot_style import PRL_DOUBLE_COL, STEP3_PANEL_ASPECT
            T_arr = np.array([p['T'] for p in self.extracted_physics])
            T_max = np.max(T_arr)
            panel_h = PRL_DOUBLE_COL / 2 * STEP3_PANEL_ASPECT
            fig = plt.Figure(figsize=(PRL_DOUBLE_COL, 2 * panel_h), dpi=PRL_DPI)
            axes = fig.subplots(2, 2)
            for ax, (mode, _tag, _short) in zip(axes.ravel(), _STEP3_PANELS):
                ax2 = self._draw_panel(ax, mode, T_arr, T_max)
                self._apply_temperature_xlim(ax, T_arr, T_max)
                step3_finalize_axes(ax, ax2=ax2)
            fig.subplots_adjust(left=0.10, right=0.98, bottom=0.08, top=0.96, wspace=0.42, hspace=0.38)
            path = os.path.join(folder, f"step3_composite_2x2.{fmt}")
            step3_save_panel(fig, path)
            plt.close(fig)
            messagebox.showinfo("Export", f"Saved composite figure:\n{path}")
        except Exception as e:
            messagebox.showerror("Export Error", str(e))

    def _update_plot(self):
        if not self.extracted_physics:
            return
        try:
            self.fig.clf()

            mode = self.show_mode.get()
            T_arr = np.array([p['T'] for p in self.extracted_physics])
            T_max = np.max(T_arr)

            ax = self.fig.add_subplot(111)
            ax2 = self._draw_panel(ax, mode, T_arr, T_max)
            self._apply_temperature_xlim(ax, T_arr, T_max)
            step3_apply_panel_layout(self.fig, ax, ax2=ax2)
            self.canvas.draw()
        except Exception as e:
            messagebox.showerror("Plot Error", str(e))


Step3_TemperatureDependence = Step3TemperatureDependence


if __name__ == "__main__":
    root = tk.Tk()
    root.title("ARPES Tool - Step 3")
    root.geometry("1000x700")
    app = Step3TemperatureDependence(root)
    app.pack(fill=tk.BOTH, expand=True)
    root.mainloop()