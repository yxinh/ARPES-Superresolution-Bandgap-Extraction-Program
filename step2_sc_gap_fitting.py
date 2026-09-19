"""Step 2: per-EDC Dynes fits, F-test, and multi-momentum weighted-average (MMWA)."""

import os
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import numpy as np
import threading
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.optimize import curve_fit, root_scalar
from scipy.integrate import cumulative_trapezoid
import scipy.stats as stats

from arpes_physics import (
    KB as KB_CONSTANT,
    FWHM_TO_SIGMA,
    load_arpes_dat,
    shirley_background_2d,
    poisson_scale,
    intensity_sigma,
    dynes_photocurrent,
    mmwa_combine,
)
from gui_common import bind_mousewheel, mousewheel_delta
from prl_plot_style import (
    apply_style, apply_twinx_style, COLORS, SERIES, PRL_LINEWIDTH, PRL_LINEWIDTH_THICK,
    PRL_MARKERSIZE, PRL_LABEL_SIZE, PRL_TITLE_SIZE,
    gui_figsize, legend_kwargs, set_axis_labels, style_colorbar,
    imshow_intensity, imshow_diverging, add_intensity_colorbar,
    plot_fit_line_kwargs, plot_data_points_kwargs,
    errorbar_kwargs, plot_curve_kwargs, shade_region,
    shade_significant_pvalue, shade_insignificant_pvalue,
    get_intensity_cmap, get_diverging_cmap,
)

class Step2GapFitting(ttk.Frame):
    def __init__(self, parent, controller=None, **kwargs):
        super().__init__(parent, **kwargs)
        self.controller = controller 
        
        # --- Internal Data Storage ---
        self.file_path = None
        self.I_raw, self.k_raw, self.e_raw = None, None, None
        self.I_raw_roi, self.I_shirley_bg, self.I_proc = None, None, None
        self.k_proc, self.e_proc = None, None 
        
        # Fitting ROI storage
        self.fit_k_vals, self.fit_e_vals = None, None
        self.I_fit_raw, self.I_fit_bg, self.I_fit_proc = None, None, None
        self.I_recon_gap, self.I_recon_gap_plus_bg = None, None
        self.I_diff = None  
        
        self._temp_I_raw_roi = None
        self._temp_I_bg_total = None
        
        self.T = 4.2  
        self.energy_res_sigma = 0.008 / FWHM_TO_SIGMA 
        self.bg_noise_val = 0.0  
        self.bg_noise_data = None
        self.alpha_est = 0.0     
        self.noise_data = None   
        self.use_shirley_var = tk.BooleanVar(value=True)
        
        # SC Fitting Results Storage
        self.kF_actual = None
        self.fit_k_points = []
        self.fit_results_gap = []
        self.fit_results_metal = []
        self.final_stats = {} 
        
        # Datastore for Step 3
        self.saved_results = {}
        
        self.show_mode = tk.StringVar(value="Full Raw Spectrum") 
        
        self._build_ui()

    # =============================================================================
    # --- UI Construction ---
    # =============================================================================
    def _build_ui(self):
        self.control_canvas = tk.Canvas(self, width=370) 
        self.control_scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.control_canvas.yview)
        self.control_frame = ttk.Frame(self.control_canvas)
        
        self.control_window = self.control_canvas.create_window((0, 0), window=self.control_frame, anchor="nw")
        self.control_frame.bind("<Configure>", lambda e: self.control_canvas.configure(scrollregion=self.control_canvas.bbox("all")))
        self.control_canvas.bind("<Configure>", lambda e: self.control_canvas.itemconfig(self.control_window, width=e.width))
        
        bind_mousewheel(self.control_canvas, self._on_mousewheel)

        self.control_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=False)
        self.control_scrollbar.pack(side=tk.LEFT, fill=tk.Y)
        
        self.plot_frame = ttk.Frame(self, padding=10)
        self.plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        self.plot_top_frame = ttk.Frame(self.plot_frame)
        self.plot_top_frame.pack(side=tk.TOP, fill=tk.X, pady=(0, 5))
        ttk.Label(self.plot_top_frame, text="Display View: ", font=('Arial', 10, 'bold')).pack(side=tk.LEFT, padx=(0, 5))
        
        self.cb_view = ttk.Combobox(self.plot_top_frame, textvariable=self.show_mode, state="readonly", width=40)
        self.cb_view['values'] = ["Full Raw Spectrum"]
        self.cb_view.current(0)
        self.cb_view.pack(side=tk.LEFT, padx=5)
        self.cb_view.bind("<<ComboboxSelected>>", self._on_display_mode_change)
        # Ensure radio/check labels look enabled across themes
        try:
            style = ttk.Style()
            style.configure('TRadiobutton', foreground='black')
            style.configure('TCheckbutton', foreground='black')
        except Exception:
            pass
        
        self.fig = plt.figure(figsize=gui_figsize())
        self.ax = self.fig.add_subplot(111)
        self.divider = make_axes_locatable(self.ax)
        self.cax = self.divider.append_axes("right", size="5%", pad=0.05)
        self.fig.tight_layout()
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.draw()
        
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.plot_frame)
        self.toolbar.update()
        self.toolbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        self._build_constants_panel()
        self._build_step1_controls()
        self._build_step2_preprocessing() 
        self._build_step3_kf_search()
        self._build_step4_fitting()
        self._build_step5_saving()

    def _on_mousewheel(self, event):
        self.control_canvas.yview_scroll(mousewheel_delta(event), "units")

    def _build_constants_panel(self):
        frame = tk.Frame(self.control_frame, bg='#e0e0e0', padx=2, pady=2, relief=tk.GROOVE, borderwidth=2)
        frame.pack(fill=tk.X, pady=(0, 5), padx=2)
        tk.Label(frame, text="Physical Constants & Params", bg='#e0e0e0', font=('Arial', 10, 'bold')).pack(anchor=tk.W)
        tk.Label(frame, text=f"kB = {KB_CONSTANT} eV/K", bg='#e0e0e0').pack(anchor=tk.W)
        
        res_frame = tk.Frame(frame, bg='#e0e0e0'); res_frame.pack(fill=tk.X, pady=2)
        tk.Label(res_frame, text="Energy Res (FWHM, eV):", bg='#e0e0e0').pack(side=tk.LEFT)
        self.ent_res = ttk.Entry(res_frame, width=6); self.ent_res.insert(0, "0.008"); self.ent_res.pack(side=tk.LEFT, padx=2)

    def _build_step1_controls(self):
        frame = ttk.LabelFrame(self.control_frame, text="1. Load Low-T SC Band Data", padding=2)
        frame.pack(fill=tk.X, pady=2, padx=2)
        ttk.Button(frame, text="Select .dat File", command=self.load_file).pack(fill=tk.X)
        self.lbl_file = ttk.Label(frame, text="No file selected", foreground="gray"); self.lbl_file.pack(fill=tk.X)
        
        temp_frame = ttk.Frame(frame); temp_frame.pack(fill=tk.X, pady=2)
        ttk.Label(temp_frame, text="Temperature T (K):").pack(side=tk.LEFT)
        self.ent_temp = ttk.Entry(temp_frame, width=6); self.ent_temp.insert(0, "14"); self.ent_temp.pack(side=tk.LEFT, padx=2)
        self.btn_auto_calc = ttk.Button(frame, text="Auto Calculate", command=self.auto_calculate)
        self.btn_auto_calc.pack(fill=tk.X, pady=5)
        self.btn_plot_raw = ttk.Button(frame, text="Load & Plot Spectrum", command=self.plot_raw_data, state=tk.DISABLED)
        self.btn_plot_raw.pack(fill=tk.X, pady=2)

    def _build_step2_preprocessing(self):
        self.frame_preproc = ttk.LabelFrame(self.control_frame, text="2. Preprocessing (Noise & Wide BG)", padding=2)
        self.frame_preproc.pack(fill=tk.X, pady=2, padx=2)
        
        bg_noise_frame = ttk.LabelFrame(self.frame_preproc, text="Constant Background Variance", padding=2)
        bg_noise_frame.pack(fill=tk.X, pady=2)
        bg_input_f = ttk.Frame(bg_noise_frame); bg_input_f.pack(fill=tk.X, pady=2)
        ttk.Label(bg_input_f, text="Bg Min Energy (eV):").pack(side=tk.LEFT)
        self.ent_bg_min_e = ttk.Entry(bg_input_f, width=6); self.ent_bg_min_e.pack(side=tk.LEFT, padx=5); self.ent_bg_min_e.insert(0, "0.02")
        
        bg_btn_f = ttk.Frame(bg_noise_frame); bg_btn_f.pack(fill=tk.X, pady=2)
        self.btn_bg_noise = ttk.Button(bg_btn_f, text="Estimate Bg Var.", command=self.estimate_bg_noise, state=tk.DISABLED)
        self.btn_bg_noise.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=1)
        self.btn_insp_bg = ttk.Button(bg_btn_f, text="Inspect", command=self.inspect_bg_noise, state=tk.DISABLED)
        self.btn_insp_bg.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=1)
        
        bg_res_f = ttk.Frame(bg_noise_frame); bg_res_f.pack(fill=tk.X, pady=2)
        ttk.Label(bg_res_f, text="Background Variance:").pack(side=tk.LEFT)
        self.var_bg_noise_disp = tk.StringVar(value="N/A")
        tk.Entry(bg_res_f, textvariable=self.var_bg_noise_disp, state='readonly', readonlybackground='#d3d3d3', width=10).pack(side=tk.LEFT, padx=2)
        
        noise_frame = ttk.LabelFrame(self.frame_preproc, text="Poisson Noise Estimation", padding=2)
        noise_frame.pack(fill=tk.X, pady=2)
        k_roi_frame = ttk.Frame(noise_frame); k_roi_frame.pack(fill=tk.X, pady=2)
        ttk.Label(k_roi_frame, text="Momentum ROI (Å⁻¹):").pack(side=tk.LEFT)
        self.ent_k_left = ttk.Entry(k_roi_frame, width=5); self.ent_k_left.pack(side=tk.LEFT, padx=1); self.ent_k_left.insert(0, "-0.2")
        ttk.Label(k_roi_frame, text="to").pack(side=tk.LEFT)
        self.ent_k_right = ttk.Entry(k_roi_frame, width=5); self.ent_k_right.pack(side=tk.LEFT, padx=1); self.ent_k_right.insert(0, "0.0")
        
        e_roi_frame = ttk.Frame(noise_frame); e_roi_frame.pack(fill=tk.X, pady=2)
        ttk.Label(e_roi_frame, text="Energy ROI (eV):").pack(side=tk.LEFT)
        self.ent_e_left = ttk.Entry(e_roi_frame, width=5); self.ent_e_left.pack(side=tk.LEFT, padx=1); self.ent_e_left.insert(0, "-0.15")
        ttk.Label(e_roi_frame, text="to").pack(side=tk.LEFT)
        self.ent_e_right = ttk.Entry(e_roi_frame, width=5); self.ent_e_right.pack(side=tk.LEFT, padx=1); self.ent_e_right.insert(0, "-0.10")
        
        sigma_frame = ttk.Frame(noise_frame); sigma_frame.pack(fill=tk.X, pady=2)
        ttk.Label(sigma_frame, text="Gaussian Sigma:").pack(side=tk.LEFT)
        self.ent_noise_sigma = ttk.Entry(sigma_frame, width=4); self.ent_noise_sigma.pack(side=tk.LEFT, padx=2); self.ent_noise_sigma.insert(0, "4.0")
        
        btn_n_frame = ttk.Frame(noise_frame); btn_n_frame.pack(fill=tk.X, pady=2)
        self.btn_noise = ttk.Button(btn_n_frame, text="Est. Alpha", command=self.estimate_poisson_level, state=tk.DISABLED)
        self.btn_noise.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=1)
        self.btn_insp_noise = ttk.Button(btn_n_frame, text="Inspect", command=self.inspect_noise, state=tk.DISABLED)
        self.btn_insp_noise.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=1)
        
        alpha_frame = ttk.Frame(noise_frame); alpha_frame.pack(fill=tk.X)
        ttk.Label(alpha_frame, text="Poisson Alpha:").pack(side=tk.LEFT)
        self.var_alpha = tk.StringVar(value="N/A")
        tk.Entry(alpha_frame, textvariable=self.var_alpha, state='readonly', readonlybackground='#d3d3d3', width=8).pack(side=tk.LEFT, padx=2)
        
        shirley_frame = ttk.LabelFrame(self.frame_preproc, text="Shirley BG Removal (Wide ROI Crop)", padding=2)
        shirley_frame.pack(fill=tk.X, pady=2)
        # Optional checkbox to enable/disable Shirley background removal
        chk_frame = ttk.Frame(shirley_frame)
        chk_frame.pack(fill=tk.X, pady=(0, 2))
        self.chk_use_shirley = ttk.Checkbutton(chk_frame, text="Enable Shirley BG removal", variable=self.use_shirley_var)
        self.chk_use_shirley.pack(side=tk.LEFT)
        s_k_frame = ttk.Frame(shirley_frame); s_k_frame.pack(fill=tk.X, pady=2)
        ttk.Label(s_k_frame, text="Crop Mom. (Å⁻¹):").pack(side=tk.LEFT)
        self.ent_s_k_left = ttk.Entry(s_k_frame, width=5); self.ent_s_k_left.pack(side=tk.LEFT, padx=1); self.ent_s_k_left.insert(0, "-0.15")
        ttk.Label(s_k_frame, text="to").pack(side=tk.LEFT)
        self.ent_s_k_right = ttk.Entry(s_k_frame, width=5); self.ent_s_k_right.pack(side=tk.LEFT, padx=1); self.ent_s_k_right.insert(0, "0.15")
        
        s_e_frame = ttk.Frame(shirley_frame); s_e_frame.pack(fill=tk.X, pady=2)
        ttk.Label(s_e_frame, text="Crop Energy (eV):").pack(side=tk.LEFT)
        self.ent_s_e_left = ttk.Entry(s_e_frame, width=5); self.ent_s_e_left.pack(side=tk.LEFT, padx=1); self.ent_s_e_left.insert(0, "-0.10")
        ttk.Label(s_e_frame, text="to").pack(side=tk.LEFT)
        self.ent_s_e_right = ttk.Entry(s_e_frame, width=5); self.ent_s_e_right.pack(side=tk.LEFT, padx=1); self.ent_s_e_right.insert(0, "0.03")

        s_param_frame = ttk.Frame(shirley_frame); s_param_frame.pack(fill=tk.X, pady=2)
        ttk.Label(s_param_frame, text="Max Iterations:").pack(side=tk.LEFT)
        self.ent_shirley_iter = ttk.Entry(s_param_frame, width=3); self.ent_shirley_iter.pack(side=tk.LEFT, padx=1); self.ent_shirley_iter.insert(0, "100")
        ttk.Label(s_param_frame, text="Tol:").pack(side=tk.LEFT, padx=(2,0))
        self.ent_shirley_tol = ttk.Entry(s_param_frame, width=5); self.ent_shirley_tol.pack(side=tk.LEFT, padx=1); self.ent_shirley_tol.insert(0, "1e-9")
        
        ttk.Label(s_param_frame, text="Smooth k (pts):").pack(side=tk.LEFT, padx=(2,0))
        self.ent_shirley_smooth = ttk.Entry(s_param_frame, width=4)
        self.ent_shirley_smooth.pack(side=tk.LEFT, padx=1)
        self.ent_shirley_smooth.insert(0, "4")
        
        insp_frame = ttk.Frame(shirley_frame); insp_frame.pack(fill=tk.X, pady=2)
        ttk.Label(insp_frame, text="Start k (Å⁻¹):").pack(side=tk.LEFT)
        self.ent_insp_k = ttk.Entry(insp_frame, width=5); self.ent_insp_k.pack(side=tk.LEFT, padx=1); self.ent_insp_k.insert(0, "0.0")
        ttk.Label(insp_frame, text="Step:").pack(side=tk.LEFT, padx=(2,0))
        self.ent_insp_step = ttk.Entry(insp_frame, width=3); self.ent_insp_step.pack(side=tk.LEFT, padx=1); self.ent_insp_step.insert(0, "1")
        
        self.btn_insp_shirley = ttk.Button(shirley_frame, text="Inspect Shirley Iterations", command=self.open_shirley_inspector, state=tk.DISABLED)
        self.btn_insp_shirley.pack(fill=tk.X, pady=2)
        self.btn_shirley = ttk.Button(shirley_frame, text="Crop ROI & Remove Background", command=self.run_shirley_bg, state=tk.DISABLED)
        self.btn_shirley.pack(fill=tk.X, pady=2)
        
        err_frame = ttk.Frame(shirley_frame); err_frame.pack(fill=tk.X, pady=2)
        ttk.Label(err_frame, text="Max Err:").pack(side=tk.LEFT)
        self.var_shirley_err = tk.StringVar(value="N/A")
        tk.Entry(err_frame, textvariable=self.var_shirley_err, state='readonly', readonlybackground='#d3d3d3', width=8).pack(side=tk.LEFT, padx=1)
        ttk.Label(err_frame, text="at k:").pack(side=tk.LEFT, padx=(1,0))
        self.var_shirley_err_k = tk.StringVar(value="N/A")
        tk.Entry(err_frame, textvariable=self.var_shirley_err_k, state='readonly', readonlybackground='#d3d3d3', width=6).pack(side=tk.LEFT, padx=1)

    def _build_step3_kf_search(self):
        self.frame_kf = ttk.LabelFrame(self.control_frame, text="3. Fermi Momentum (kF) Search", padding=2)
        self.frame_kf.pack(fill=tk.X, pady=2, padx=2)
        
        bracket_f = ttk.Frame(self.frame_kf); bracket_f.pack(fill=tk.X, pady=2)
        ttk.Label(bracket_f, text="Search Bracket:").pack(side=tk.LEFT)
        self.ent_kf_min = ttk.Entry(bracket_f, width=5); self.ent_kf_min.pack(side=tk.LEFT, padx=1); self.ent_kf_min.insert(0, "0.0")
        ttk.Label(bracket_f, text="to").pack(side=tk.LEFT)
        self.ent_kf_max = ttk.Entry(bracket_f, width=5); self.ent_kf_max.pack(side=tk.LEFT, padx=1); self.ent_kf_max.insert(0, "0.1")
        
        btn_f = ttk.Frame(self.frame_kf); btn_f.pack(fill=tk.X, pady=2)
        self.btn_search_kf = ttk.Button(btn_f, text="Find kF", command=self.search_kf, state=tk.DISABLED)
        self.btn_search_kf.pack(side=tk.LEFT, padx=2)
        
        ttk.Label(btn_f, text="kF =").pack(side=tk.LEFT, padx=2)
        self.var_kf_result = tk.StringVar(value="N/A")
        tk.Entry(btn_f, textvariable=self.var_kf_result, state='readonly', readonlybackground='#d3d3d3', width=8).pack(side=tk.LEFT, padx=2)
        
        self.lbl_kf_status = ttk.Label(self.frame_kf, text="Waiting for Background Removal...", foreground="gray")
        self.lbl_kf_status.pack(fill=tk.X, pady=2)

    def _build_step4_fitting(self):
        self.frame_fit = ttk.LabelFrame(self.control_frame, text="4. Superconducting Gap Fitting", padding=2)
        self.frame_fit.pack(fill=tk.X, pady=2, padx=2)
        
        roi_f_k = ttk.Frame(self.frame_fit); roi_f_k.pack(fill=tk.X, pady=2)
        ttk.Label(roi_f_k, text="Fit Mom. (Å⁻¹):").pack(side=tk.LEFT)
        self.ent_fit_k_min = ttk.Entry(roi_f_k, width=5); self.ent_fit_k_min.pack(side=tk.LEFT, padx=1); self.ent_fit_k_min.insert(0, "0.0")
        ttk.Label(roi_f_k, text="to").pack(side=tk.LEFT)
        self.ent_fit_k_max = ttk.Entry(roi_f_k, width=5); self.ent_fit_k_max.pack(side=tk.LEFT, padx=1); self.ent_fit_k_max.insert(0, "0.1")
        
        roi_f_e = ttk.Frame(self.frame_fit); roi_f_e.pack(fill=tk.X, pady=2)
        ttk.Label(roi_f_e, text="Fit Energy (eV):").pack(side=tk.LEFT)
        self.ent_fit_e_min = ttk.Entry(roi_f_e, width=5); self.ent_fit_e_min.pack(side=tk.LEFT, padx=1); self.ent_fit_e_min.insert(0, "-0.02")
        ttk.Label(roi_f_e, text="to").pack(side=tk.LEFT)
        self.ent_fit_e_max = ttk.Entry(roi_f_e, width=5); self.ent_fit_e_max.pack(side=tk.LEFT, padx=1); self.ent_fit_e_max.insert(0, "0.01")

        self.f_guesses = ttk.LabelFrame(self.frame_fit, text="Initial Guesses (Gap & Metal)")
        self.f_guesses.pack(fill=tk.X, pady=2)
        
        ttk.Label(self.f_guesses, text="Delta (eV):").grid(row=0, column=0, padx=2, pady=2, sticky=tk.E)
        self.ent_guess_delta = ttk.Entry(self.f_guesses, width=6); self.ent_guess_delta.insert(0, "10e-3"); self.ent_guess_delta.grid(row=0, column=1)
        
        ttk.Label(self.f_guesses, text="Gamma (eV):").grid(row=0, column=2, padx=2, pady=2, sticky=tk.E)
        self.ent_guess_gamma = ttk.Entry(self.f_guesses, width=6); self.ent_guess_gamma.insert(0, "2e-3"); self.ent_guess_gamma.grid(row=0, column=3)
        
        ttk.Label(self.f_guesses, text="Amplitude:").grid(row=1, column=0, padx=2, pady=2, sticky=tk.E)
        self.ent_guess_scale = ttk.Entry(self.f_guesses, width=6); self.ent_guess_scale.insert(0, "1e-3"); self.ent_guess_scale.grid(row=1, column=1)
        
        self.btn_fit = ttk.Button(self.frame_fit, text="Run Gap & Metal Fitting (F-Test)", command=self.run_gap_fitting, state=tk.DISABLED)
        self.btn_fit.pack(fill=tk.X, pady=5)
        
        self.btn_inspect_fits = ttk.Button(self.frame_fit, text="Inspect Fits", command=self.open_fit_inspector, state=tk.DISABLED)
        self.btn_inspect_fits.pack(fill=tk.X, pady=2)

        ext_f = ttk.LabelFrame(self.frame_fit, text="Delta Extraction & Averaging", padding=2)
        ext_f.pack(fill=tk.X, pady=2)
        
        mult_f = ttk.Frame(ext_f); mult_f.pack(fill=tk.X, pady=1)
        ttk.Label(mult_f, text="Tolerance (N*sigma):").pack(side=tk.LEFT)
        self.ent_err_mult = ttk.Entry(mult_f, width=5)
        self.ent_err_mult.pack(side=tk.LEFT, padx=5)
        self.ent_err_mult.insert(0, "1.5")

        # Upper bound on fitted Δ uncertainty = (min err over all fitted k) × multiplier
        cap_f = ttk.Frame(ext_f); cap_f.pack(fill=tk.X, pady=1)
        ttk.Label(cap_f, text="Δ err cap (× min err @ valid k):").pack(side=tk.LEFT)
        self.ent_delta_err_cap_mult = ttk.Entry(cap_f, width=5)
        self.ent_delta_err_cap_mult.pack(side=tk.LEFT, padx=5)
        self.ent_delta_err_cap_mult.insert(0, "3.0")

        # Bind updates to both entries so plot refreshes when values change
        for ent in (self.ent_err_mult, self.ent_delta_err_cap_mult):
            ent.bind("<Return>", lambda e: self._on_display_mode_change())
            ent.bind("<FocusOut>", lambda e: self._on_display_mode_change())
        
        res1_f = ttk.Frame(ext_f); res1_f.pack(fill=tk.X, pady=1)
        ttk.Label(res1_f, text="Delta at kF:").pack(side=tk.LEFT)
        self.var_delta_kf = tk.StringVar(value="N/A")
        tk.Entry(res1_f, textvariable=self.var_delta_kf, state='readonly', readonlybackground='#d3d3d3', width=18).pack(side=tk.RIGHT, padx=2)
        
        res2_f = ttk.Frame(ext_f); res2_f.pack(fill=tk.X, pady=1)
        ttk.Label(res2_f, text="Weighted Delta:").pack(side=tk.LEFT)
        self.var_delta_best = tk.StringVar(value="N/A")
        tk.Entry(res2_f, textvariable=self.var_delta_best, state='readonly', readonlybackground='#d3d3d3', width=18).pack(side=tk.RIGHT, padx=2)

        lim_f = ttk.LabelFrame(self.frame_fit, text="Plot Limits & Adjustments", padding=2)
        lim_f.pack(fill=tk.X, pady=2)
        
        lim_x = ttk.Frame(lim_f); lim_x.pack(fill=tk.X, pady=1)
        ttk.Label(lim_x, text="k X-axis: ").pack(side=tk.LEFT)
        self.ent_lim_k_min = ttk.Entry(lim_x, width=5); self.ent_lim_k_min.pack(side=tk.LEFT, padx=1)
        ttk.Label(lim_x, text="to").pack(side=tk.LEFT)
        self.ent_lim_k_max = ttk.Entry(lim_x, width=5); self.ent_lim_k_max.pack(side=tk.LEFT, padx=1)
        
        lim_y_d = ttk.Frame(lim_f); lim_y_d.pack(fill=tk.X, pady=1)
        ttk.Label(lim_y_d, text="Δ Y-axis (meV):").pack(side=tk.LEFT)
        self.ent_lim_d_min = ttk.Entry(lim_y_d, width=5); self.ent_lim_d_min.pack(side=tk.LEFT, padx=1)
        ttk.Label(lim_y_d, text="to").pack(side=tk.LEFT)
        self.ent_lim_d_max = ttk.Entry(lim_y_d, width=5); self.ent_lim_d_max.pack(side=tk.LEFT, padx=1)

        lim_y_g = ttk.Frame(lim_f); lim_y_g.pack(fill=tk.X, pady=1)
        ttk.Label(lim_y_g, text="Γ Y-axis (meV):").pack(side=tk.LEFT)
        self.ent_lim_g_min = ttk.Entry(lim_y_g, width=5); self.ent_lim_g_min.pack(side=tk.LEFT, padx=1)
        ttk.Label(lim_y_g, text="to").pack(side=tk.LEFT)
        self.ent_lim_g_max = ttk.Entry(lim_y_g, width=5); self.ent_lim_g_max.pack(side=tk.LEFT, padx=1)
        
        lim_p = ttk.Frame(lim_f); lim_p.pack(fill=tk.X, pady=1)
        ttk.Label(lim_p, text="P-val Min:").pack(side=tk.LEFT)
        self.ent_lim_p_min = ttk.Entry(lim_p, width=6); self.ent_lim_p_min.insert(0, "1e-20"); self.ent_lim_p_min.pack(side=tk.LEFT, padx=1)
        ttk.Label(lim_p, text="Thresh:").pack(side=tk.LEFT)
        self.ent_p_thresh = ttk.Entry(lim_p, width=5); self.ent_p_thresh.insert(0, "1e-10"); self.ent_p_thresh.pack(side=tk.LEFT, padx=1)
        
        for widget in [self.ent_lim_k_min, self.ent_lim_k_max, self.ent_lim_d_min, self.ent_lim_d_max, 
                       self.ent_lim_g_min, self.ent_lim_g_max, self.ent_lim_p_min, self.ent_p_thresh]:
            widget.bind("<Return>", lambda e: self._on_display_mode_change())
            widget.bind("<FocusOut>", lambda e: self._on_display_mode_change())

        self.lbl_fit_status = ttk.Label(self.frame_fit, text="Awaiting kF Search...", foreground="gray")
        self.lbl_fit_status.pack(fill=tk.X, pady=5)

    def _build_step5_saving(self):
        self.frame_save = ttk.LabelFrame(self.control_frame, text="5. Save Results for Step 3", padding=2)
        self.frame_save.pack(fill=tk.X, pady=2, padx=2)
        
        list_frame = ttk.Frame(self.frame_save)
        list_frame.pack(side=tk.TOP, fill=tk.X, padx=2, pady=2)
        
        self.listbox_saved = tk.Listbox(list_frame, height=5)
        scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=self.listbox_saved.yview)
        self.listbox_saved.config(yscrollcommand=scrollbar.set)
        
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.listbox_saved.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        btn_f = ttk.Frame(self.frame_save)
        btn_f.pack(fill=tk.X, pady=2)
        
        self.btn_save_res = ttk.Button(btn_f, text="Save Result", command=self.save_current_result, state=tk.DISABLED)
        self.btn_save_res.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=1)
        
        self.btn_clear_res = ttk.Button(btn_f, text="Clear Selected", command=self.clear_selected_result)
        self.btn_clear_res.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=1)
        
        # Add an export button to save all fitted results to directory for Step 3
        self.btn_export = ttk.Button(self.frame_save, text="Export All Results", command=self.export_all_results)
        self.btn_export.pack(fill=tk.X, pady=2)  # Note: Adjust 'self' to the specific frame you want to place it in, e.g., your control frame
        
        self.btn_next_step = ttk.Button(self.frame_save, text="Proceed to Step 3 >>", command=self.go_to_step_3, state=tk.DISABLED)
        self.btn_next_step.pack(fill=tk.X, pady=5)

    # =============================================================================
    # --- File Loading & Preprocessing Logic ---
    # =============================================================================
    def _on_display_mode_change(self, event=None):
        self._update_plot(preserve_limits=False)

    def load_file(self):
        filepath = filedialog.askopenfilename(filetypes=[("DAT files", "*.dat"), ("All files", "*.*")])
        if filepath:
            self.file_path = filepath
            self.lbl_file.config(text=filepath.split("/")[-1])
            self.btn_plot_raw.config(state=tk.NORMAL)
            
    def auto_calculate(self):
        self.plot_raw_data() 
        if getattr(self, 'I_raw', None) is None:
            return 
            
        self.btn_save_res.config(state=tk.DISABLED) 
        
        # Add all steps to the queue, including the final save
        self.auto_step_queue = [
            self.estimate_bg_noise,
            self.estimate_poisson_level,
            self.run_shirley_bg,
            self.search_kf,
            self.run_gap_fitting,
            self.save_current_result
        ]
        
        self.after(200, self._execute_next_auto_step)

    def _execute_next_auto_step(self):
        if not self.auto_step_queue:
            return
        func = self.auto_step_queue[0]
        # Pause queue execution if it's the save step and fitting is still running
        if func == self.save_current_result and str(self.btn_save_res['state']) != tk.NORMAL:
            self.after(500, self._execute_next_auto_step)
            return
            
        try:
            self.auto_step_queue.pop(0)()
            self.after(300, self._execute_next_auto_step)
            
        except Exception as e:
            messagebox.showerror("Auto Calculate Error", f"Process failed at {func.__name__}:\n{str(e)}")
            
    def plot_raw_data(self):
        try:
            self.T = float(self.ent_temp.get())
            raw_res = float(self.ent_res.get())
            self.energy_res_sigma = raw_res / FWHM_TO_SIGMA
            self.I_raw, self.k_raw, self.e_raw = load_arpes_dat(self.file_path)
            self.I_raw_roi, self.I_shirley_bg, self.I_proc = None, None, None
            
            self.cb_view['values'] = ["Full Raw Spectrum"]
            self.show_mode.set("Full Raw Spectrum")
            self._update_plot()
            
            self.btn_bg_noise.config(state=tk.NORMAL)
            self.btn_noise.config(state=tk.NORMAL)
            self.btn_shirley.config(state=tk.NORMAL)
            self.btn_insp_shirley.config(state=tk.NORMAL)
            
            self.btn_search_kf.config(state=tk.DISABLED)
            self.btn_fit.config(state=tk.DISABLED)
            self.btn_inspect_fits.config(state=tk.DISABLED)
            self.btn_save_res.config(state=tk.DISABLED)
            self.btn_next_step.config(state=tk.DISABLED)
            self.lbl_kf_status.config(text="Awaiting Background Removal...", foreground="gray")
            self.lbl_fit_status.config(text="Awaiting kF Search...", foreground="gray")
        except Exception as e:
            messagebox.showerror("Read Error", str(e))

    def estimate_bg_noise(self):
        try:
            min_e = float(self.ent_bg_min_e.get())
            bg_mask = self.e_raw > min_e
            if not np.any(bg_mask): return messagebox.showwarning("Warning", "No data points found above Bg Min Energy.")
            roi_bg = self.I_raw[bg_mask, :]
            self.bg_noise_val = np.var(roi_bg)
            self.var_bg_noise_disp.set(f"{self.bg_noise_val:.2e}")
            self.bg_noise_data = (self.k_raw, self.e_raw[bg_mask], roi_bg)
            self.btn_insp_bg.config(state=tk.NORMAL)
        except Exception as e: messagebox.showerror("Error", str(e))

    def inspect_bg_noise(self):
        if self.bg_noise_data is None: return
        k_vals, e_vals_bg, roi = self.bg_noise_data
        top = tk.Toplevel(self.winfo_toplevel()); top.title("Background Noise Region Inspector")
        fig, ax = plt.subplots(figsize=gui_figsize())
        
        canvas = FigureCanvasTkAgg(fig, master=top)
        canvas.draw()
        toolbar = NavigationToolbar2Tk(canvas, top)
        toolbar.update()
        toolbar.pack(side=tk.BOTTOM, fill=tk.X)
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        im = imshow_intensity(ax, roi, [k_vals[0], k_vals[-1], e_vals_bg[0]*1000, e_vals_bg[-1]*1000])
        cbar = fig.colorbar(im, ax=ax)
        style_colorbar(cbar, label='Intensity (a.u.)')
        set_axis_labels(
            ax,
            xlabel=fr'Momentum $k$ ($\mathrm{{\AA}}^{{-1}}$)',
            ylabel='Energy $E$ (meV)',
            title='Background Estimation Region',
        )
        apply_style(ax)
        fig.tight_layout(); canvas.draw()

    def estimate_poisson_level(self):
        try:
            k_l, k_r = float(self.ent_k_left.get()), float(self.ent_k_right.get())
            e_l, e_r = float(self.ent_e_left.get()), float(self.ent_e_right.get())
            smooth_sigma = float(self.ent_noise_sigma.get()) 
            k_mask = (self.k_raw >= k_l) & (self.k_raw <= k_r)
            e_mask = (self.e_raw >= e_l) & (self.e_raw <= e_r)
            if not np.any(k_mask) or not np.any(e_mask): return messagebox.showwarning("Warning", "Selected ROI is empty!")
            roi = self.I_raw[np.ix_(e_mask, k_mask)]
            self.alpha_est, roi_lp, residual = poisson_scale(roi, smooth_sigma)
            self.var_alpha.set(f"{self.alpha_est:.5f}")
            self.noise_data = (roi, roi_lp, residual, k_mask, e_mask)
            self.btn_insp_noise.config(state=tk.NORMAL) 
        except Exception as e: messagebox.showerror("Noise Est. Error", str(e))

    def inspect_noise(self):
        if self.noise_data is not None: 
            k_mask, e_mask = self.noise_data[3], self.noise_data[4]
            k_roi, e_roi = self.k_raw[k_mask], self.e_raw[e_mask]
            top = tk.Toplevel(self.winfo_toplevel())
            top.title("Noise Estimation Details")
            top.geometry("1200x450")
            
            fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharex=True, sharey=True)
            
            canvas = FigureCanvasTkAgg(fig, master=top)
            canvas.draw()
            toolbar = NavigationToolbar2Tk(canvas, top)
            toolbar.update()
            toolbar.pack(side=tk.BOTTOM, fill=tk.X)
            canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
            
            ext = [k_roi[0], k_roi[-1], e_roi[0]*1000, e_roi[-1]*1000]
            im0 = imshow_intensity(axes[0], self.noise_data[0], ext)
            axes[0].set_title("Original ROI", fontsize=PRL_TITLE_SIZE)
            style_colorbar(fig.colorbar(im0, ax=axes[0]))
            im1 = imshow_intensity(axes[1], self.noise_data[1], ext)
            axes[1].set_title("Smoothed (Signal)", fontsize=PRL_TITLE_SIZE)
            style_colorbar(fig.colorbar(im1, ax=axes[1]))
            std_res = np.std(self.noise_data[2])
            im2 = imshow_diverging(axes[2], self.noise_data[2], ext, std_res * 3)
            axes[2].set_title(f"Residual (Noise)\n$\\alpha_{{est}}$ = {self.alpha_est:.4f}", fontsize=PRL_TITLE_SIZE)
            style_colorbar(fig.colorbar(im2, ax=axes[2]))
            for a in axes:
                apply_style(a)
            
            fig.tight_layout(pad=2.0, w_pad=3.0)
            canvas.draw()

    def open_shirley_inspector(self):
        if self.I_raw is None: return
        try:
            start_k = float(self.ent_insp_k.get())
            plot_step, max_iter = int(self.ent_insp_step.get()), int(self.ent_shirley_iter.get())
            e_left, e_right, tol = float(self.ent_s_e_left.get()), float(self.ent_s_e_right.get()), float(self.ent_shirley_tol.get())
        except ValueError: return messagebox.showerror("Input Error", "Invalid parameters!")
        
        e_mask = (self.e_raw >= e_left) & (self.e_raw <= e_right)
        if not np.any(e_mask): return messagebox.showwarning("Warning", "Energy crop window out of bounds.")
        
        top = tk.Toplevel(self.winfo_toplevel())
        top.title("Shirley Iteration Inspector")
        top.geometry("1100x700")  # Start large
        
        current_k_idx = [np.argmin(np.abs(self.k_raw - start_k))]
        
        global_E_min, global_E_max = np.min(self.e_raw[e_mask]), np.max(self.e_raw[e_mask])
        global_I_min, global_I_max = np.min(self.I_raw[e_mask, :]), np.max(self.I_raw[e_mask, :])
        pad_I = (global_I_max - global_I_min) * 0.05
        
        ctrl_frame = ttk.Frame(top, padding=5)
        ctrl_frame.pack(side=tk.TOP, fill=tk.X)
        ttk.Label(ctrl_frame, text="Jump to k (Å⁻¹):").pack(side=tk.LEFT)
        ent_goto = ttk.Entry(ctrl_frame, width=8); ent_goto.pack(side=tk.LEFT, padx=2)
        
        nav_frame = ttk.Frame(top, padding=5)
        nav_frame.pack(side=tk.BOTTOM, fill=tk.X)
        btn_prev = ttk.Button(nav_frame, text="<< Prev k-slice", command=lambda: update_plot(step=-1))
        btn_prev.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=5)
        btn_next = ttk.Button(nav_frame, text="Next k-slice >>", command=lambda: update_plot(step=1))
        btn_next.pack(side=tk.RIGHT, expand=True, fill=tk.X, padx=5)

        plot_frame = ttk.Frame(top)
        plot_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        fig, ax = plt.subplots(figsize=gui_figsize())
        fig.subplots_adjust(left=0.1, right=0.75, bottom=0.12, top=0.92) 
        
        canvas = FigureCanvasTkAgg(fig, master=plot_frame)
        canvas.draw()
        
        toolbar = NavigationToolbar2Tk(canvas, plot_frame)
        toolbar.update()
        toolbar.pack(side=tk.BOTTOM, fill=tk.X)
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        def goto_val():
            try:
                target = float(ent_goto.get())
                current_k_idx[0] = np.argmin(np.abs(self.k_raw - target)); update_plot()
            except: pass
        ttk.Button(ctrl_frame, text="Go", command=goto_val).pack(side=tk.LEFT)

        def update_plot(step=0):
            current_k_idx[0] = max(0, min(len(self.k_raw) - 1, current_k_idx[0] + step))
            idx = current_k_idx[0]; actual_k = self.k_raw[idx]
            
            E_sorted, I_sorted = self.e_raw[e_mask], self.I_raw[e_mask, idx]
            sort_idx = np.argsort(E_sorted); E_sorted, I_sorted = E_sorted[sort_idx], I_sorted[sort_idx]
            E_plot = E_sorted * 1000
            N, I_left, I_right = len(E_sorted), I_sorted[0], I_sorted[-1]
            
            ax.clear(); B = np.linspace(I_left, I_right, N)
            ax.plot(E_plot, I_sorted, label='Original $I(E)$', **plot_fit_line_kwargs(SERIES["reference"]))
            
            for n in range(max_iter):
                B_old = np.copy(B); Y = np.maximum(I_sorted - B_old, 0)
                cum_int = np.zeros(N); cum_int[1:] = cumulative_trapezoid(Y, E_sorted)
                if cum_int[-1] == 0: break
                B = I_right + (I_left - I_right) * ((cum_int[-1] - cum_int) / cum_int[-1])
                if (n + 1) % plot_step == 0:
                    ax.plot(E_plot, B, label=f'Iter {n+1}', color=COLORS['gray'], linestyle='--', linewidth=PRL_LINEWIDTH, alpha=0.7)
                if np.max(np.abs(B - B_old)) < tol: break
                
            ax.plot(E_plot, B, label='Final Shirley BG', **plot_fit_line_kwargs(SERIES["background"]))
            ax.plot(E_plot, np.maximum(I_sorted - B, 1e-4), **plot_fit_line_kwargs(SERIES["gap_model"]),
                     label='Subtracted Signal')
            
            apply_style(ax)
            set_axis_labels(
                ax,
                xlabel='Energy (meV)',
                ylabel='Intensity (a.u.)',
                title=fr"Shirley BG Tuning | $k = {actual_k:.4f}$ $\mathrm{{\AA}}^{{-1}}$",
            )
            
            pad_I = (np.max(I_sorted) - np.min(I_sorted)) * 0.1
            ax.set_xlim(np.min(E_plot), np.max(E_plot))
            ax.set_ylim(0, np.max(I_sorted) + pad_I)
            
            ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left", **legend_kwargs())
            
            canvas.draw()

        update_plot()

    def run_shirley_bg(self):
        try:
            k_l, k_r = float(self.ent_s_k_left.get()), float(self.ent_s_k_right.get())
            e_l, e_r = float(self.ent_s_e_left.get()), float(self.ent_s_e_right.get())
            max_iter, tol = int(self.ent_shirley_iter.get()), float(self.ent_shirley_tol.get())
            smooth_k_pts = float(self.ent_shirley_smooth.get())
        except ValueError: return messagebox.showerror("Input Error", "Invalid parameters for Shirley!")
        
        k_mask = (self.k_raw >= k_l) & (self.k_raw <= k_r)
        e_mask = (self.e_raw >= e_l) & (self.e_raw <= e_r)
        if not np.any(k_mask) or not np.any(e_mask): return messagebox.showwarning("Warning", "Crop window out of bounds.")
        
        self.k_proc, self.e_proc = self.k_raw[k_mask], self.e_raw[e_mask]
        I_crop = self.I_raw[np.ix_(e_mask, k_mask)].copy()
        # If user disabled Shirley background removal, skip heavy computation
        if not getattr(self, 'use_shirley_var', tk.BooleanVar(value=True)).get():
            self._temp_I_raw_roi = I_crop
            self._temp_I_bg_total = np.zeros_like(I_crop)
            self.k_proc, self.e_proc = self.k_raw[k_mask], self.e_raw[e_mask]
            self.var_shirley_err.set("Skipped")
            self.var_shirley_err_k.set("-")
            self._shirley_done(True, 0.0, None)
            return
        
        self.btn_shirley.config(state=tk.DISABLED, text="Processing...")
        threading.Thread(target=self._shirley_thread, args=(I_crop, max_iter, tol, smooth_k_pts), daemon=True).start()

    def _shirley_thread(self, I_crop, max_iter, tol, smooth_k_pts):
        try:
            I_bg_total, all_converged, max_err_val, max_err_k_idx = shirley_background_2d(
                self.e_proc, I_crop, max_iter, tol, smooth_k_pts
            )
            self._temp_I_raw_roi = I_crop
            self._temp_I_bg_total = I_bg_total
            err_k_val = self.k_proc[max_err_k_idx] if max_err_k_idx != -1 else None
            self.after(0, lambda: self._shirley_done(all_converged, max_err_val, err_k_val))
        except Exception as err:
            self.after(0, lambda: messagebox.showerror("Error", str(err)))
            self.after(0, lambda: self.btn_shirley.config(state=tk.NORMAL, text="Crop ROI & Remove Background"))

    def _shirley_done(self, all_converged, max_err, err_k):
        self.btn_shirley.config(state=tk.NORMAL, text="Crop ROI & Remove Background")
        self.I_raw_roi = self._temp_I_raw_roi
        self.I_shirley_bg = self._temp_I_bg_total
        self.I_proc = np.maximum(self.I_raw_roi - self.I_shirley_bg, 0) 
        
        if all_converged:
            self.var_shirley_err.set("Converged")
            self.var_shirley_err_k.set("-")
        else:
            self.var_shirley_err.set(f"{max_err:.2e}")
            self.var_shirley_err_k.set(f"{err_k:.4f}")
            
        self.cb_view['values'] = [
            "Full Raw Spectrum", 
            "Shirley ROI: Raw", 
            "Shirley ROI: Background", 
            "Shirley ROI: Processed"
        ]
        self.show_mode.set("Shirley ROI: Processed") 
        self._update_plot()
        
        self.btn_search_kf.config(state=tk.NORMAL)
        self.lbl_kf_status.config(text="Ready to search kF.", foreground="blue")

    # =============================================================================
    # --- SC Gap Fitting Core Logic ---
    # =============================================================================
    def search_kf(self):
        try:
            kf_min = float(self.ent_kf_min.get())
            kf_max = float(self.ent_kf_max.get())
            
            if not self.controller or not hasattr(self.controller, 'step1_module') or self.controller.step1_module.spline_func is None:
                self.lbl_kf_status.config(text="Error: Valid Spline function not found from Step 1!", foreground="red")
                self.var_kf_result.set("Error")
                self.btn_fit.config(state=tk.DISABLED)
                return
                
            spline_func = self.controller.step1_module.spline_func
            sol = root_scalar(spline_func, bracket=[kf_min, kf_max], method='bisect')
            self.kF_actual = sol.root
            self.var_kf_result.set(f"{self.kF_actual:.6f}")
            self.lbl_kf_status.config(text="kF found successfully.", foreground="green")
            
            self.btn_fit.config(state=tk.NORMAL)
            self.lbl_fit_status.config(text="Ready to run Gap Fitting.", foreground="blue")
                
        except Exception as e:
            self.lbl_kf_status.config(text=f"Search Failed: {str(e)}", foreground="red")
            self.var_kf_result.set("Error")
            self.btn_fit.config(state=tk.DISABLED)

    def calc_spectrum(self, e, delta, gamma, scale, edc_k, spline_func):
        """Forward model I(k, omega) at one momentum (kept as the fit callback)."""
        return dynes_photocurrent(
            e, delta, gamma, scale,
            xi_k=float(spline_func(edc_k)),
            temperature=self.T,
            energy_res_sigma=self.energy_res_sigma,
        )

    def run_gap_fitting(self):
        self.lbl_fit_status.config(text="Fitting Models in ROI... Please wait", foreground="orange")
        self.btn_fit.config(state=tk.DISABLED)
        threading.Thread(target=self._fitting_thread, args=(self.controller.step1_module.spline_func,), daemon=True).start()

    def _fitting_thread(self, spline_func):
        try:
            k_min, k_max = float(self.ent_fit_k_min.get()), float(self.ent_fit_k_max.get())
            e_min, e_max = float(self.ent_fit_e_min.get()), float(self.ent_fit_e_max.get())
            
            k_mask = (self.k_proc >= k_min) & (self.k_proc <= k_max)
            e_mask = (self.e_proc >= e_min) & (self.e_proc <= e_max)
            
            if not np.any(k_mask) or not np.any(e_mask):
                raise ValueError("Fitting ROI out of bounds or empty!")
            
            self.fit_k_vals = self.k_proc[k_mask]
            self.fit_e_vals = self.e_proc[e_mask]
            self.I_fit_raw = self.I_raw_roi[np.ix_(e_mask, k_mask)]
            self.I_fit_bg = self.I_shirley_bg[np.ix_(e_mask, k_mask)]
            self.I_fit_proc = self.I_proc[np.ix_(e_mask, k_mask)]
            
            N_k = len(self.fit_k_vals)
            
            d_str = self.ent_guess_delta.get().strip()
            g_str = self.ent_guess_gamma.get().strip()
            s_str = self.ent_guess_scale.get().strip()
            
            d_g = 10e-3 if d_str.lower() == 'auto' else float(d_str)
            g_g = 1e-3 if g_str.lower() == 'auto' else float(g_str)
            s_g = 0.1 if s_str.lower() == 'auto' else float(s_str)
            
            last_1 = [d_g, g_g, s_g]
            last_2 = [g_g, s_g] 
            
            delta_fit, gamma_fit = np.zeros(N_k), np.zeros(N_k)
            delta_err, gamma_err = np.zeros(N_k), np.zeros(N_k)
            RSS_gap, RSS_met = np.zeros(N_k), np.zeros(N_k)
            gap_fit_ok = np.zeros(N_k, dtype=bool)
            delta_point_valid = np.zeros(N_k, dtype=bool)
            
            self.fit_k_points = []
            self.fit_results_gap = []
            self.fit_results_metal = []
            
            self.I_recon_gap = np.zeros_like(self.I_fit_proc)
            
            flag_1, flag_2 = True, True
            
            for a in range(N_k):
                edc_k = self.fit_k_vals[a]
                self.fit_k_points.append(edc_k)
                
                I_ori = self.I_fit_raw[:, a]
                I_edc = self.I_fit_proc[:, a]
                
                sigma_arr = intensity_sigma(I_ori, self.alpha_est, self.bg_noise_val)

                try:
                    popt_1, pcov_1 = curve_fit(
                        lambda e, d, g, s: self.calc_spectrum(e, d, g, s, edc_k, spline_func),
                        self.fit_e_vals, I_edc, p0=last_1,
                        bounds=([0, 0, 0], [max(abs(self.fit_e_vals)), np.inf, np.inf]),
                        maxfev=5000, ftol=1e-9, xtol=1e-9, gtol=1e-9, method='trf',
                        sigma=sigma_arr, absolute_sigma=True
                    )
                    flag_1 = False
                    gap_fit_ok[a] = True
                except RuntimeError:
                    popt_1, pcov_1 = [0, 0, 0], np.full((3, 3), np.inf)
                    flag_1 = True
                    gap_fit_ok[a] = False
                
                if not flag_1: last_1 = popt_1.copy()
                delta_fit[a], gamma_fit[a] = popt_1[0], popt_1[1]
                ci95_1 = 1.96 * np.sqrt(np.diag(pcov_1))
                delta_err[a], gamma_err[a] = ci95_1[0], ci95_1[1]
                # Invalid when gap fit failed or Δ is smaller than its own fit uncertainty
                delta_point_valid[a] = bool(
                    gap_fit_ok[a]
                    and np.isfinite(delta_fit[a])
                    and np.isfinite(delta_err[a]) and (delta_err[a] > 0)
                    and (delta_fit[a] >= delta_err[a])
                )
                I_fit_gap = self.calc_spectrum(self.fit_e_vals, *popt_1, edc_k, spline_func)
                RSS_gap[a] = np.sum(((I_edc - I_fit_gap) / sigma_arr)**2)
                
                self.I_recon_gap[:, a] = I_fit_gap
                
                try:
                    popt_2, pcov_2 = curve_fit(
                        lambda e, g, s: self.calc_spectrum(e, 0, g, s, edc_k, spline_func),
                        self.fit_e_vals, I_edc, p0=last_2,
                        bounds=([0, 0], [np.inf, np.inf]),
                        maxfev=5000, ftol=1e-9, xtol=1e-9, gtol=1e-9, method='trf',
                        sigma=sigma_arr, absolute_sigma=True
                    )
                    flag_2 = False
                except RuntimeError:
                    popt_2, pcov_2 = [0, 0], np.full((2, 2), np.inf)
                    flag_2 = True
                
                if not flag_2: last_2 = popt_2.copy()
                I_fit_met = self.calc_spectrum(self.fit_e_vals, 0, popt_2[0], popt_2[1], edc_k, spline_func)
                RSS_met[a] = np.sum(((I_edc - I_fit_met) / sigma_arr)**2)
                
                self.fit_results_gap.append({
                    'x': self.fit_e_vals, 'y_ori': I_ori, 'y_data': I_edc, 'y_fit': I_fit_gap, 
                    'popt': popt_1.copy(), 'orig_popt': popt_1.copy(), 'sigma': sigma_arr
                })
                self.fit_results_metal.append({'y_fit': I_fit_met})
                
            self.I_recon_gap_plus_bg = self.I_recon_gap + self.I_fit_bg
            self.I_diff = self.I_recon_gap_plus_bg - self.I_fit_raw
            
            n_points = len(self.fit_e_vals)
            # Degrees of freedom
            P_gap = 3
            df1 = 1
            df2 = max(1, n_points - P_gap)

            # Avoid division by zero in denominator (RSS_gap/df2)
            denom = RSS_gap / df2
            eps = np.finfo(float).eps
            denom = np.where(denom <= 0, eps, denom)

            f_stats = ((RSS_met - RSS_gap) / df1) / denom
            f_stats = np.maximum(f_stats, 0.0)

            # Survival function (upper tail) of F-distribution
            p_vals = stats.f.sf(f_stats, df1, df2)
            p_vals = np.clip(p_vals, 0.0, 1.0)
            
            self.final_stats = {
                'delta_fit': delta_fit, 'delta_err': delta_err,
                'gamma_fit': gamma_fit, 'gamma_err': gamma_err,
                'RSS_gap': RSS_gap, 'RSS_met': RSS_met, 'p_vals': p_vals,
                'delta_point_valid': delta_point_valid,
            }
            
            self.after(0, self._fitting_done)
        except Exception as e:
            self.after(0, lambda: self.lbl_fit_status.config(text=f"Fitting Failed: {str(e)}", foreground="red"))
            self.after(0, lambda: self.btn_fit.config(state=tk.NORMAL))

    def _fitting_done(self):
        self.btn_fit.config(state=tk.NORMAL)
        self.btn_inspect_fits.config(state=tk.NORMAL)
        self.btn_save_res.config(state=tk.NORMAL)
        self.lbl_fit_status.config(text="Dual Model Fitting & F-Test Completed!", foreground="green")
        
        self.cb_view['values'] = [
            "Full Raw Spectrum",
            "Shirley ROI: Raw", 
            "Shirley ROI: Background", 
            "Shirley ROI: Processed",
            "Fit ROI: Raw Spectrum",
            "Fit ROI: Shirley Background",
            "Fit ROI: Processed (Signal)",
            "Fit ROI: Reconstructed 2D (Gap Model)",
            "Fit ROI: Reconstructed 2D + Background",
            "Fit ROI: Difference (Recon+BG - Raw)",
            "Fitted Delta (Δ)", 
            "Fitted Gamma (Γ)", 
            "F-Test: RSS Comparison",
            "F-Test: P-Value"
        ]
        self.show_mode.set("Fitted Delta (Δ)")
        self._update_plot()

    # =============================================================================
    # --- Helper: momentum points used in Δ plots & inverse-variance averaging ---
    # =============================================================================
    def _delta_point_valid_mask(self):
        if not self.final_stats or not self.fit_k_points:
            return None
        m = self.final_stats.get('delta_point_valid')
        n = len(self.fit_k_points)
        if m is None:
            return np.ones(n, dtype=bool)
        m = np.asarray(m, dtype=bool)
        if m.size != n:
            return np.ones(n, dtype=bool)
        return m

    def _get_weighted_delta(self):
        """MMWA combination of per-EDC gaps (see ``arpes_physics.mmwa_combine``)."""
        if not self.final_stats:
            return None

        k_vals = np.array(self.fit_k_points)
        delta_vals = np.array(self.final_stats['delta_fit'])
        err_vals = np.array(self.final_stats['delta_err'])
        gamma_vals = np.array(self.final_stats.get('gamma_fit', []))
        gamma_errs = np.array(self.final_stats.get('gamma_err', []))
        valid_k = self._delta_point_valid_mask()
        if valid_k is None:
            return None

        try:
            n_sigma = float(self.ent_err_mult.get())
        except Exception:
            n_sigma = 2.0
        try:
            err_cap_mult = float(self.ent_delta_err_cap_mult.get())
        except Exception:
            err_cap_mult = 3.0

        return mmwa_combine(
            k_vals, delta_vals, err_vals, valid_k,
            k_f=self.kF_actual,
            n_sigma=n_sigma,
            err_cap_mult=err_cap_mult,
            gamma_vals=gamma_vals,
            gamma_errs=gamma_errs,
        )

    # =============================================================================
    # --- Plotting & Visualization ---
    # =============================================================================
    def _apply_axis_limits(self, ax, plot_type):
        try:
            x_min = self.ent_lim_k_min.get().strip()
            x_max = self.ent_lim_k_max.get().strip()
            if x_min and x_max: ax.set_xlim(float(x_min), float(x_max))
        except ValueError: pass

        if plot_type == 'delta':
            try:
                y_min = self.ent_lim_d_min.get().strip()
                y_max = self.ent_lim_d_max.get().strip()
                if y_min and y_max: ax.set_ylim(float(y_min), float(y_max))
            except ValueError: pass
        elif plot_type == 'gamma':
            try:
                y_min = self.ent_lim_g_min.get().strip()
                y_max = self.ent_lim_g_max.get().strip()
                if y_min and y_max: ax.set_ylim(float(y_min), float(y_max))
            except ValueError: pass

    def _update_plot(self, preserve_limits=False):
        mode = self.show_mode.get()
        self.fig.clf()
        
        if "Spectrum" in mode or "ROI" in mode:
            self.ax = self.fig.add_subplot(111)
            self.divider = make_axes_locatable(self.ax)
            self.cax = self.divider.append_axes("right", size="5%", pad=0.05)
            
            plot_I, plot_k, plot_e = None, None, None
            title_text = ""
            vmin_global, vmax_global = 0, 1
            cmap_to_use = get_intensity_cmap()

            if mode == "Full Raw Spectrum":
                if self.I_raw is None: return
                plot_I, plot_k, plot_e = self.I_raw, self.k_raw, self.e_raw
                title_text = f'Raw Band Spectrum (T = {self.T} K)'
                vmin_global = np.nanmin(self.I_raw)
                vmax_global = np.nanmax(self.I_raw)
            elif "Fit ROI" in mode:
                if self.fit_k_vals is None: return
                plot_k, plot_e = self.fit_k_vals, self.fit_e_vals
                vmin_global = np.nanmin(self.I_fit_raw)
                vmax_global = np.nanmax(self.I_fit_raw)
                
                if "Raw Spectrum" in mode: 
                    plot_I, title_text = self.I_fit_raw, f'Fit ROI: Raw Data (T = {self.T} K)'
                elif "Shirley Background" in mode: 
                    plot_I, title_text = self.I_fit_bg, f'Fit ROI: Background (T = {self.T} K)'
                elif "Processed" in mode: 
                    plot_I, title_text = self.I_fit_proc, f'Fit ROI: Processed Signal (T = {self.T} K)'
                elif "Reconstructed 2D (Gap" in mode:
                    plot_I, title_text = self.I_recon_gap, f'Reconstructed 2D Spectrum (Fit Model only)'
                elif "Reconstructed 2D +" in mode:
                    plot_I, title_text = self.I_recon_gap_plus_bg, f'Reconstructed Spectrum + Background'
                elif "Difference" in mode:
                    plot_I, title_text = self.I_diff, f'Fit ROI: Difference (Recon+BG - Raw)'
                    cmap_to_use = get_diverging_cmap()
                    abs_max = np.nanpercentile(np.abs(plot_I), 98) if plot_I is not None else 1.0
                    if abs_max == 0 or np.isnan(abs_max): abs_max = 1e-6
                    vmin_global, vmax_global = -abs_max, abs_max
            elif "Shirley ROI" in mode:
                if self.I_proc is None: return
                plot_k, plot_e = self.k_proc, self.e_proc
                vmin_global = np.nanmin(self.I_raw_roi)
                vmax_global = np.nanmax(self.I_raw_roi)
                
                if "Raw" in mode: 
                    plot_I, title_text = self.I_raw_roi, f'Shirley ROI: Raw (T = {self.T} K)'
                elif "Background" in mode: 
                    plot_I, title_text = self.I_shirley_bg, f'Shirley ROI: Background (T = {self.T} K)'
                else: 
                    plot_I, title_text = self.I_proc, f'Shirley ROI: Processed (T = {self.T} K)'

            if plot_I is None: return

            extent = [plot_k[0], plot_k[-1], plot_e[0]*1000, plot_e[-1]*1000]
            im = imshow_intensity(self.ax, plot_I, extent, vmin=vmin_global, vmax=vmax_global, cmap=cmap_to_use)
            add_intensity_colorbar(self.fig, im, self.cax)
            set_axis_labels(
                self.ax,
                xlabel=fr'Momentum $k$ ($\mathrm{{\AA}}^{{-1}}$)',
                ylabel='Energy $E$ (meV)',
                title=title_text,
            )
            apply_style(self.ax)
            self.fig.tight_layout()

        elif mode == "Fitted Delta (Δ)":
            if not self.final_stats: return
            
            ax = self.fig.add_subplot(111)
            k_vals = np.array(self.fit_k_points)
            valid_k = self._delta_point_valid_mask()
            if valid_k is None: return
            
            w_res = self._get_weighted_delta()
            if w_res is None: return
            
            kF = w_res.get('kF', None)
            delta_kf = w_res.get('delta_kf', w_res.get('delta_mid', np.nan))
            err_kf = w_res.get('err_kf', w_res.get('err_mid', np.nan))
            sel_k = w_res['sel_k']
            delta_best, error_best = w_res['delta_best'], w_res['error_best']
            chi2_nu = w_res['chi2_nu']

            # Show delta at kF explicitly (not the chosen search-center)
            self.var_delta_kf.set(f"{delta_kf*1000:.2f} \u00B1 {err_kf*1000:.2f} meV")
            self.var_delta_best.set(f"{delta_best*1000:.2f} \u00B1 {error_best*1000:.2f} meV")

            delta_vals_meV = self.final_stats['delta_fit'] * 1000
            err_vals_meV = self.final_stats['delta_err'] * 1000
            
            shade_region(ax, sel_k[0], sel_k[-1])
            
            if np.any(valid_k):
                ax.errorbar(k_vals[valid_k], delta_vals_meV[valid_k], yerr=err_vals_meV[valid_k],
                            **errorbar_kwargs(SERIES["gap_model"], marker='o', linestyle='none'),
                            label=r"Fitted $\Delta$")
            ax.axvline(kF, color=SERIES["reference"], linestyle='--', linewidth=PRL_LINEWIDTH, label=fr"$k_F = {kF:.3f}$")
            ax.axhline(delta_best * 1000, color=SERIES["weighted"], linestyle='-.', linewidth=PRL_LINEWIDTH_THICK,
                       label=r"Weighted $\Delta_{\mathrm{best}}$")
            
            apply_style(ax)
            set_axis_labels(
                ax,
                xlabel=fr'Momentum $k$ ($\mathrm{{\AA}}^{{-1}}$)',
                ylabel=r'Fitted $\Delta$ (meV)',
                title='Fitted Superconducting Gap & Averaging',
            )
            
            ax2 = ax.twinx()
            if np.any(valid_k):
                ax2.plot(k_vals[valid_k], err_vals_meV[valid_k], color=SERIES["error"],
                         linestyle='--', linewidth=PRL_LINEWIDTH, label=r"$|\sigma_\Delta|$")
            ax2.set_ylabel(r'Error $\Delta$ (meV)', fontsize=PRL_LABEL_SIZE, color=SERIES["error"])
            apply_twinx_style(ax2, color=SERIES["error"])
            
            res_str = (
                f"Interval: [{sel_k[0]:.3f}, {sel_k[-1]:.3f}]\n"
                f"$\\Delta_F$ = {delta_kf*1000:.2f} $\\pm$ {err_kf*1000:.2f} meV\n"
                f"$\\Delta_{{best}}$ = {delta_best*1000:.2f} $\\pm$ {error_best*1000:.2f} meV\n"
                f"$\\chi^2_\\nu$ = {chi2_nu:.2f}"
            )
            
            handles, labels = ax.get_legend_handles_labels()
            proxy = mpatches.Rectangle((0,0), 1, 1, fill=False, edgecolor='none', visible=False)
            handles.append(proxy)
            labels.append(res_str)
            
            ax.legend(handles, labels, loc='best', **legend_kwargs(handlelength=1.2, labelspacing=0.3))
            self._apply_axis_limits(ax, 'delta')
            self.fig.tight_layout()

        elif mode == "Fitted Gamma (Γ)":
            ax = self.fig.add_subplot(111)
            valid_k = self._delta_point_valid_mask()
            if valid_k is None: return
            k_pts = np.array(self.fit_k_points)
            g = np.asarray(self.final_stats['gamma_fit'], dtype=float)
            ge = np.asarray(self.final_stats['gamma_err'], dtype=float)
            if np.any(valid_k):
                ax.errorbar(k_pts[valid_k], g[valid_k], yerr=ge[valid_k],
                            **errorbar_kwargs(SERIES["gamma_weighted"], marker='s', linestyle='none'),
                            label=r"Fitted $\Gamma$")
            
            if self.kF_actual is not None:
                ax.axvline(self.kF_actual, color=SERIES["reference"], linestyle='--',
                           linewidth=PRL_LINEWIDTH, label=fr"$k_F = {self.kF_actual:.3f}$")
                
            apply_style(ax)
            set_axis_labels(
                ax,
                xlabel=fr'Momentum $k$ ($\mathrm{{\AA}}^{{-1}}$)',
                ylabel=r'$\Gamma$ (eV)',
                title='Fitted Scattering Rate',
            )
            
            ax.legend(loc='best', **legend_kwargs(handlelength=1.2, labelspacing=0.3))
            self._apply_axis_limits(ax, 'gamma')
            self.fig.tight_layout()

        elif mode == "F-Test: RSS Comparison":
            ax = self.fig.add_subplot(111)
            valid_k = self._delta_point_valid_mask()
            if valid_k is None:
                return
            k_all = np.asarray(self.fit_k_points, dtype=float)
            rss_g = np.asarray(self.final_stats['RSS_gap'], dtype=float)
            rss_m = np.asarray(self.final_stats['RSS_met'], dtype=float)
            vk = np.asarray(valid_k, dtype=bool)
            if np.any(vk):
                order = np.argsort(k_all[vk])
                k_sel = k_all[vk][order]
                ax.plot(k_sel, rss_g[vk][order], **plot_curve_kwargs(SERIES["gap_model"], marker='o', linestyle='-'),
                        label=r'Gap Model ($\Delta$ free)')
                ax.plot(k_sel, rss_m[vk][order], **plot_curve_kwargs(SERIES["metal_model"], marker='s', linestyle='--'),
                        label=r'Metal Model ($\Delta=0$)')
            
            if self.kF_actual is not None:
                ax.axvline(self.kF_actual, color=SERIES["reference"], linestyle='--',
                           linewidth=PRL_LINEWIDTH, label=fr"$k_F = {self.kF_actual:.3f}$")
                
            apply_style(ax, grid=True)
            set_axis_labels(
                ax,
                xlabel=fr'Momentum $k$ ($\mathrm{{\AA}}^{{-1}}$)',
                ylabel='Residual Sum of Squares (RSS)',
                title='Goodness of Fit Comparison',
            )
            
            ax.legend(loc='best', **legend_kwargs(handlelength=1.2, labelspacing=0.3))
            self.fig.tight_layout()

        elif mode == "F-Test: P-Value":
            ax = self.fig.add_subplot(111)
            valid_k = self._delta_point_valid_mask()
            if valid_k is None:
                return
            k_all = np.asarray(self.fit_k_points, dtype=float)
            p_all = np.asarray(self.final_stats['p_vals'], dtype=float)
            vk = np.asarray(valid_k, dtype=bool)
            
            try: p_min = float(self.ent_lim_p_min.get())
            except ValueError: p_min = 1e-10
            try: thresh = float(self.ent_p_thresh.get())
            except ValueError: thresh = 0.01
            
            if np.any(vk):
                order = np.argsort(k_all[vk])
                k_sel = k_all[vk][order]
                p_sel = np.clip(p_all[vk][order], np.finfo(float).tiny, 1.0)
                ax.semilogy(k_sel, p_sel, **plot_curve_kwargs(SERIES["pvalue"], marker='D', linestyle='-'),
                            label='F-test $p$-value')
            ax.axhline(y=thresh, color=SERIES["reference"], linestyle='--', linewidth=PRL_LINEWIDTH_THICK,
                       label=f'Threshold ({thresh})')
            
            if self.kF_actual is not None:
                ax.axvline(self.kF_actual, color=SERIES["reference"], linestyle='--',
                           linewidth=PRL_LINEWIDTH, label=fr"$k_F = {self.kF_actual:.3f}$")
                
            shade_significant_pvalue(ax, k_all, p_min, thresh)
            shade_insignificant_pvalue(ax, k_all, thresh, 1.5)
            ax.set_ylim(bottom=p_min, top=1.5)
            
            apply_style(ax, grid=True)
            set_axis_labels(
                ax,
                xlabel=fr'Momentum $k$ ($\mathrm{{\AA}}^{{-1}}$)',
                ylabel='$p$-value (log scale)',
                title='Statistical Significance (F-Test)',
            )
            
            ax.legend(loc='best', **legend_kwargs(handlelength=1.2, labelspacing=0.3))
            self.fig.tight_layout()

        self.canvas.draw()

    # ================= Inspection Tool =================
    def open_fit_inspector(self):
        if not self.fit_results_gap: return messagebox.showwarning("Warning", "No fits available to inspect.")
            
        top = tk.Toplevel(self.winfo_toplevel())
        top.title("SC Gap Fit Inspector (Gap Model Adjustment)")
        top.geometry("1100x800")
        current_idx = [0] 
        
        ctrl_frame = ttk.Frame(top, padding=5)
        ctrl_frame.pack(side=tk.TOP, fill=tk.X)
        
        ttk.Label(ctrl_frame, text="Go to k (Å⁻¹):").pack(side=tk.LEFT)
        ent_goto = ttk.Entry(ctrl_frame, width=8); ent_goto.pack(side=tk.LEFT, padx=2)
        
        def goto_val():
            try:
                target = float(ent_goto.get())
                idx = np.argmin(np.abs(np.array(self.fit_k_points) - target))
                current_idx[0] = idx
                update_plot()
            except: pass
        ttk.Button(ctrl_frame, text="Go", command=goto_val).pack(side=tk.LEFT)
        
        slider_main_frame = ttk.LabelFrame(top, text="Dynamic Fit Adjustment (Gap Model)", padding=5)
        slider_main_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=5)
        
        res_var = tk.StringVar(value="Reduced Chi-Squared (Gap): N/A")
        ttk.Label(slider_main_frame, textvariable=res_var, font=("Arial", 10, "bold"), foreground="blue").pack(side=tk.TOP, pady=2)
        
        sliders_inner_frame = ttk.Frame(slider_main_frame)
        sliders_inner_frame.pack(side=tk.TOP, fill=tk.X)
        
        nav_frame = ttk.Frame(top, padding=5)
        nav_frame.pack(side=tk.BOTTOM, fill=tk.X)
        btn_prev = ttk.Button(nav_frame, text="<< Prev k-slice", command=lambda: update_plot(step=-1))
        btn_prev.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=5)
        btn_next = ttk.Button(nav_frame, text="Next k-slice >>", command=lambda: update_plot(step=1))
        btn_next.pack(side=tk.RIGHT, expand=True, fill=tk.X, padx=5)
        
        plot_frame = ttk.Frame(top)
        plot_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        fig, ax = plt.subplots(figsize=gui_figsize())
        
        canvas = FigureCanvasTkAgg(fig, master=plot_frame)
        canvas.draw()
        
        toolbar = NavigationToolbar2Tk(canvas, plot_frame)
        toolbar.update()
        toolbar.pack(side=tk.BOTTOM, fill=tk.X)
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        param_vars = []
        line_fit_gap = None

        def clear_sliders():
            for w in sliders_inner_frame.winfo_children():
                w.destroy()
            param_vars.clear()

        def on_slider_change(*args):
            if not line_fit_gap: return
            idx = current_idx[0]
            data_g = self.fit_results_gap[idx]
            data_m = self.fit_results_metal[idx]
            edc_k = self.fit_k_points[idx]
            
            popt_new = [v.get() for v in param_vars]
            
            spline_func = self.controller.step1_module.spline_func
            y_fit_new = self.calc_spectrum(data_g['x'], popt_new[0], popt_new[1], popt_new[2], edc_k, spline_func)
            
            data_g['popt'] = popt_new
            data_g['y_fit'] = y_fit_new
            
            line_fit_gap[0].set_ydata(y_fit_new)
            
            N = len(data_g['y_data'])
            P_gap = len(popt_new)
            P_met = P_gap - 1
            
            chi2_gap_new = np.sum(((data_g['y_data'] - y_fit_new) / data_g['sigma'])**2) / (N - P_gap)
            chi2_met = np.sum(((data_g['y_data'] - data_m['y_fit']) / data_g['sigma'])**2) / (N - P_met)
            res_var.set(f"Reduced Chi-Sq (Gap): {chi2_gap_new:.4f}   |   (Metal): {chi2_met:.4f}")
            
            canvas.draw_idle()

        def build_sliders(popt, names, orig_popt):
            clear_sliders()
            for i, (val, name, orig_val) in enumerate(zip(popt, names, orig_popt)):
                row = i
                col_base = 0
                
                ttk.Label(sliders_inner_frame, text=f"{name}:").grid(row=row, column=col_base, sticky=tk.E, padx=(5,2), pady=2)
                var = tk.DoubleVar(value=val)
                param_vars.append(var)
                
                v_min, v_max = orig_val * 0.5, orig_val * 1.5
                if v_min > v_max: v_min, v_max = v_max, v_min
                if v_min == v_max: v_min, v_max = 0.0, 0.01 
                
                s = ttk.Scale(sliders_inner_frame, from_=v_min, to=v_max, variable=var, command=on_slider_change)
                s.grid(row=row, column=col_base+1, sticky=tk.EW, padx=2)
                
                val_lbl = ttk.Label(sliders_inner_frame, text=f"{val:.4e}", width=10)
                val_lbl.grid(row=row, column=col_base+2, sticky=tk.W)
                
                def make_reset_cmd(v=var, orig=orig_val, l=val_lbl):
                    def cmd():
                        v.set(orig); l.config(text=f"{orig:.4e}"); on_slider_change()
                    return cmd
                    
                btn_reset = ttk.Button(sliders_inner_frame, text="Reset", width=5, command=make_reset_cmd())
                btn_reset.grid(row=row, column=col_base+3, padx=(0, 5))
                
                def update_lbl(event_val, l=val_lbl, v=var):
                    l.config(text=f"{v.get():.4e}")
                
                s.config(command=lambda e, l=val_lbl, v=var: (update_lbl(e, l, v), on_slider_change()))
            
            sliders_inner_frame.columnconfigure(1, weight=1)

        def update_plot(step=0):
            nonlocal line_fit_gap
            current_idx[0] = max(0, min(len(self.fit_results_gap) - 1, current_idx[0] + step))
            idx = current_idx[0]
            data_g = self.fit_results_gap[idx]
            data_m = self.fit_results_metal[idx]
            edc_k = self.fit_k_points[idx]
            
            ax.clear()
            x_plot = data_g['x'] * 1000
            ax.plot(x_plot, data_g['y_ori'], color=SERIES["data_bg"], linestyle='--', linewidth=PRL_LINEWIDTH, label='Original', zorder=1)
            ax.plot(x_plot, data_g['y_data'], label='Experiment', **plot_data_points_kwargs())
            line_fit_gap = ax.plot(x_plot, data_g['y_fit'], label=r'Gap Model ($\Delta$ free)',
                                   **plot_fit_line_kwargs(SERIES["gap_model"]))
            ax.plot(x_plot, data_m['y_fit'], label=r'Metal Model ($\Delta=0$)',
                    **plot_fit_line_kwargs(SERIES["metal_model"], linestyle='--'))
            
            apply_style(ax)
            set_axis_labels(
                ax,
                xlabel='Energy (meV)',
                ylabel='ARPES Intensity (a.u.)',
                title=fr"Dual Model Fit Comparison | $k = {edc_k:.4f}$ $\mathrm{{\AA}}^{{-1}}$",
            )
            
            N = len(data_g['y_data'])
            P_gap = len(data_g['popt'])
            chi2_gap = np.sum(((data_g['y_data'] - data_g['y_fit']) / data_g['sigma'])**2) / (N - P_gap)
            P_met = P_gap - 1
            chi2_met = np.sum(((data_g['y_data'] - data_m['y_fit']) / data_g['sigma'])**2) / (N - P_met)
            res_var.set(f"Reduced Chi-Sq (Gap): {chi2_gap:.4f}   |   (Metal): {chi2_met:.4f}")
            
            ax.legend(loc="best", **legend_kwargs(handlelength=1.2, labelspacing=0.3))
            fig.tight_layout()
            canvas.draw()
            
            names = ["Delta (eV)", "Gamma (eV)", "Amplitude"]
            build_sliders(data_g['popt'], names, data_g['orig_popt'])

        update_plot() 

    # =============================================================================
    # --- Step 5: Save Results for Step 3 ---
    # =============================================================================
    def save_current_result(self):
        if not self.final_stats: return messagebox.showwarning("Warning", "No fit results to save!")
        
        filename = self.file_path.split("/")[-1] if self.file_path else "Unknown"
        T_val = self.T
        key = f"{filename} (T={T_val}K)"
        
        if key in self.saved_results:
            ans = messagebox.askyesno("Overwrite?", f"Result for {key} already exists. Overwrite?")
            if not ans: return
            
        w_res = self._get_weighted_delta()
        if w_res is None:
            return messagebox.showerror(
                "Error",
                "Could not calculate weighted Δ: no momentum points pass the validity filter "
                "(successful gap fit, Δ ≥ fit uncertainty, and within min(err among valid k)×cap).",
            )
        
        self.saved_results[key] = {
            'filename': filename,
            'Temperature': T_val,
            'k_points': self.fit_k_points,
            'final_stats': self.final_stats,
            'weighted_res': w_res,
            'kF': self.kF_actual
        }
        
        if key not in self.listbox_saved.get(0, tk.END):
            self.listbox_saved.insert(tk.END, key)
            
        messagebox.showinfo("Saved", f"Results for {key} saved successfully!")
        
        self.btn_next_step.config(state=tk.NORMAL)

    def clear_selected_result(self):
        sel_idx = self.listbox_saved.curselection()
        if not sel_idx: return messagebox.showwarning("Warning", "Please select a result to clear.")
        
        key = self.listbox_saved.get(sel_idx)
        if key in self.saved_results:
            del self.saved_results[key]
            
        self.listbox_saved.delete(sel_idx)
        
        if self.listbox_saved.size() == 0:
            self.btn_next_step.config(state=tk.DISABLED)
            
    def export_all_results(self):
        # Check if there are saved results in memory
        if not hasattr(self, 'saved_results') or not self.saved_results:
            messagebox.showwarning("Warning", "No results to export. Please fit and save results first.")
            return
            
        # Ask user for a parent directory where the 'result' folder will be created
        parent_dir = filedialog.askdirectory(title="Select Parent Directory to Create 'result' Folder")
        if not parent_dir: 
            return

        try:
            # Create a dedicated 'result' folder inside the selected parent directory
            export_dir = os.path.join(parent_dir, 'result')
            os.makedirs(export_dir, exist_ok=True)
            
            for key, res in self.saved_results.items():
                # 1. Extract data from the nested structure as defined in your code
                T_val = res.get('Temperature', 0)
                k_vals = np.array(res.get('k_points', []))
                kF_val = res.get('kF', self.kF_actual)
                kF_str = f"{kF_val:.4f}" if kF_val is not None else "N/A"
                
                # Inner level: final_stats dictionary
                stats = res.get('final_stats', {})
                delta_vals = np.array(stats.get('delta_fit', []))
                err_vals = np.array(stats.get('delta_err', []))
                
                gamma_vals = np.array(stats.get('gamma_fit', []))
                gamma_err_vals = np.array(stats.get('gamma_err', []))
                
                rss_gap = np.array(stats.get('RSS_gap', []))
                rss_met = np.array(stats.get('RSS_met', []))
                p_vals = np.array(stats.get('p_vals', []))
                
                n_pts = len(k_vals)
                if n_pts == 0:
                    continue # Skip if no momentum data
                
                # 2. Ensure all arrays are of the same length as k_vals
                # This ensures matrix alignment for np.column_stack
                if gamma_vals.ndim == 0 or len(gamma_vals) != n_pts: gamma_vals = np.full(n_pts, gamma_vals)
                if gamma_err_vals.ndim == 0 or len(gamma_err_vals) != n_pts: gamma_err_vals = np.full(n_pts, gamma_err_vals)
                if rss_gap.ndim == 0 or len(rss_gap) != n_pts: rss_gap = np.full(n_pts, rss_gap)
                if rss_met.ndim == 0 or len(rss_met) != n_pts: rss_met = np.full(n_pts, rss_met)
                if p_vals.ndim == 0 or len(p_vals) != n_pts: p_vals = np.full(n_pts, p_vals)
                
                # 3. Define the filename including the temperature
                filename = f"fit_results_{T_val}K.txt"
                file_path = os.path.join(export_dir, filename)
                
                dpv = stats.get('delta_point_valid')
                if dpv is None:
                    dpv_col = np.ones(n_pts, dtype=float)
                else:
                    dpv_col = np.asarray(dpv, dtype=float).ravel()
                    if dpv_col.size != n_pts:
                        dpv_col = np.ones(n_pts, dtype=float)

                # 4. Construct the data matrix
                # Step 3 expects: 0:k, 1:delta, 2:err, 3:gamma, 4:gamma_err, 5:RSS_gap, 6:RSS_met, 7:p_val [, 8:delta_point_valid]
                export_data = np.column_stack((
                    k_vals, delta_vals, err_vals, gamma_vals, gamma_err_vals, rss_gap, rss_met, p_vals, dpv_col,
                ))
                
                # 5. Write file with a 4-line header (updated column names)
                weighted = res.get('weighted_res', None)
                if weighted is not None:
                    w_delta = weighted.get('delta_best', np.nan)
                    w_err = weighted.get('error_best', np.nan)
                    w_gamma = weighted.get('gamma_best', np.nan)
                    w_g_err = weighted.get('gamma_err', np.nan)
                else:
                    w_delta = np.nan
                    w_err = np.nan
                    w_gamma = np.nan
                    w_g_err = np.nan

                header_line2 = (
                    f"Temperature: {T_val} K, kF: {kF_str}, WeightedDelta: {w_delta:.8e}, WeightedErr: {w_err:.8e}, "
                    f"WeightedGamma: {w_gamma:.8e}, WeightedGammaErr: {w_g_err:.8e}"
                )
                if weighted is not None:
                    sk = weighted.get('sel_k')
                    if sk is not None and len(np.asarray(sk, dtype=float).ravel()) > 0:
                        ska = np.asarray(sk, dtype=float).ravel()
                        header_line2 += ", WeightedSelK: " + ";".join(f"{x:.10e}" for x in ska)

                header_str = (
                    "Exported Fit Results\n" +
                    header_line2 + "\n" +
                    "--------------------------------------------------------------------------------\n" +
                    "k_vals\tdelta_fit\tdelta_err\tgamma_fit\tgamma_err\tRSS_gap\tRSS_met\tp_vals\tdelta_point_valid"
                )

                np.savetxt(file_path, export_data, header=header_str, comments='', delimiter='\t', fmt='%.12e')
                
            messagebox.showinfo("Success", f"Successfully exported {len(self.saved_results)} files to:\n{export_dir}")
            
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export data: {str(e)}")
            
    def go_to_step_3(self):
        if self.controller and hasattr(self.controller, 'notebook') and hasattr(self.controller, 'step3_module'):
            self.controller.notebook.select(self.controller.step3_module)


Step2_GapFitting = Step2GapFitting


if __name__ == "__main__":
    root = tk.Tk()
    root.title("ARPES Tool - Step 2")
    root.geometry("1350x850")
    
    class MockController:
        class MockStep1:
            @staticmethod
            def spline_func(k):
                return -1.0 * (k - 0.05)**2 + 0.01 
        step1_module = MockStep1()
        
        class MockStep3:
            pass
        step3_module = MockStep3()
        
        def __init__(self):
            self.notebook = None

    controller = MockController()
    notebook = ttk.Notebook(root)
    controller.notebook = notebook
    notebook.pack(fill=tk.BOTH, expand=True)
    
    tab_2_container = ttk.Frame(notebook)
    notebook.add(tab_2_container, text="Step 2: SC Gap Fitting & F-Test ")
    
    app_step2 = Step2_GapFitting(tab_2_container, controller=controller)
    app_step2.pack(fill=tk.BOTH, expand=True)
    
    root.mainloop()