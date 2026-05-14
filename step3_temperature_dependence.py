import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

class Step3_TemperatureDependence(ttk.Frame):
    def __init__(self, parent, controller=None, **kwargs):
        super().__init__(parent, **kwargs)
        self.controller = controller 
        
        self.temp_data = [] 
        self.extracted_physics = [] 
        self.Tc_estimate = None

        self.show_mode = tk.StringVar(value="RSS Comparison vs T")
        
        self._build_ui()

    # =============================================================================
    # --- Publication Ready Style Helper ---
    # =============================================================================
    def _set_scientific_style(self, ax):
        ax.tick_params(direction='in', length=6, width=1.5, colors='k', top=True, right=True, labelsize=12)
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
        ax.grid(True, linestyle='--', alpha=0.6)

    def _apply_gapless_shading(self, ax, T_max):
        """在图表上绘制代表 Gapless (Normal State) 的灰色区域"""
        if self.Tc_estimate is not None and self.Tc_estimate <= T_max:
            ax.axvspan(self.Tc_estimate, T_max + 50, color='gray', alpha=0.3, label='Gapless Region')

    @staticmethod
    def _finite_sorted_series(T, y, err, mask):
        """Return (T, y, err) at finite points in mask, sorted by T (avoids NaN gaps in lines)."""
        idx = np.where(mask)[0]
        if idx.size == 0:
            return None, None, None
        idx = idx[np.argsort(T[idx])]
        return T[idx], y[idx], err[idx]

    # =============================================================================
    # --- UI Building ---
    # =============================================================================
    def _build_ui(self):
        self.left_panel = ttk.Frame(self, width=320)
        self.left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        self.left_panel.pack_propagate(False)

        self.right_panel = ttk.Frame(self)
        self.right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self._build_left_panel()
        self._build_plot_area()

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

        # 3. Visualization Mode
        frame_vis = ttk.LabelFrame(self.left_panel, text="3. Visualization Mode", padding=5)
        frame_vis.pack(fill=tk.X, pady=5)
        modes = ["RSS Comparison vs T", "P-value vs T", "SC Gap (Delta) vs T", "Gamma vs T"]
        for m in modes:
            ttk.Radiobutton(frame_vis, text=m, variable=self.show_mode, value=m, command=self._update_plot).pack(anchor=tk.W, pady=2)

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

        # Right-axis |σ_Δ| limits (Δ vs T dual y-axis)
        f_sd = ttk.Frame(frame_lim); f_sd.pack(fill=tk.X, pady=2)
        ttk.Label(f_sd, text="|σ_Δ| (meV):", width=12).pack(side=tk.LEFT)
        self.ent_sig_d_min = ttk.Entry(f_sd, width=6); self.ent_sig_d_min.insert(0, "auto"); self.ent_sig_d_min.pack(side=tk.LEFT, padx=1)
        ttk.Label(f_sd, text="to").pack(side=tk.LEFT)
        self.ent_sig_d_max = ttk.Entry(f_sd, width=6); self.ent_sig_d_max.insert(0, "auto"); self.ent_sig_d_max.pack(side=tk.LEFT, padx=1)

        # Gamma limits
        f_g = ttk.Frame(frame_lim); f_g.pack(fill=tk.X, pady=2)
        ttk.Label(f_g, text="Gamma (meV):", width=12).pack(side=tk.LEFT)
        self.ent_g_min = ttk.Entry(f_g, width=6); self.ent_g_min.insert(0, "auto"); self.ent_g_min.pack(side=tk.LEFT, padx=1)
        ttk.Label(f_g, text="to").pack(side=tk.LEFT); 
        self.ent_g_max = ttk.Entry(f_g, width=6); self.ent_g_max.insert(0, "auto"); self.ent_g_max.pack(side=tk.LEFT, padx=1)

        ttk.Button(frame_lim, text="Apply Limits", command=self._update_plot).pack(fill=tk.X, pady=5)

    def _build_plot_area(self):
        self.fig = plt.Figure(figsize=(8, 6), dpi=100)
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
                
                # 读取并解析文件头部，获取 T 和 kF
                with open(fpath, 'r') as f:
                    lines = [f.readline() for _ in range(4)]
                    # 第二行通常包含: Temperature: 14.0 K, kF: 0.1234, WeightedDelta: ..., WeightedErr: ...
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
                
                # 如果没读到 T，使用文件名 fallback
                if T_val is None:
                    T_val = float(fname.replace('fit_results_', '').replace('K.txt', ''))
                
                # 读取矩阵数据
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
                    'kF': res.get('kF', None),  # 直接从内存加载保存的 kF
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
    def _update_plot(self):
        if not self.extracted_physics: return
        self.fig.clf()
        
        mode = self.show_mode.get()
        T_arr = np.array([p['T'] for p in self.extracted_physics])
        T_max = np.max(T_arr)
        
        ax = self.fig.add_subplot(111)
        
        if mode == "RSS Comparison vs T":
            rss_g = [p['rss_gap_mean'] for p in self.extracted_physics]
            rss_m = [p['rss_met_mean'] for p in self.extracted_physics]
            
            ax.plot(T_arr, rss_g, '-o', color=[0, 0.2, 0.6], markerfacecolor=[0, 0.2, 0.6], 
                     linewidth=1.8, label='RSS: Gap Model')
            ax.plot(T_arr, rss_m, '--s', color=[0.7, 0, 0], markerfacecolor=[0.7, 0, 0], 
                     linewidth=1.8, label='RSS: Zero Gap Model')
            ax.set_xlabel('Temperature $T$ (K)', fontsize=16)
            ax.set_ylabel('Residual Sum of Squares (RSS)', fontsize=16)
            ax.set_title('Model Comparison', fontsize=18)
            self._apply_gapless_shading(ax, T_max)
            ax.legend(loc='best', fontsize=12)
            self._set_scientific_style(ax)

        elif mode == "P-value vs T":
            p_vals = np.array([p['sp_p_val'] for p in self.extracted_physics], dtype=float)
            min_pos = np.finfo(float).tiny
            log_p = np.log10(np.clip(p_vals, min_pos, 1.0))
            
            ax.plot(T_arr, log_p, '-D', color=[0.3, 0, 0.5], markerfacecolor=[0.3, 0, 0.5], 
                     linewidth=1.8, markersize=8)
            
            try:
                thresh = float(self.ent_p_thresh.get())
                ax.axhline(np.log10(thresh), color='k', linestyle=':', linewidth=2, label=f'Threshold (p = {thresh})')
            except: pass
            
            ax.set_xlabel('Temperature $T$ (K)', fontsize=16)
            ax.set_ylabel(r'$\log_{10}(P$-value$)$', fontsize=16)
            ax.set_title('Significance of Superconducting Gap', fontsize=18)
            self._apply_gapless_shading(ax, T_max)
            ax.legend(loc='best', fontsize=12)
            self._set_scientific_style(ax)
            
        elif mode == "SC Gap (Delta) vs T":
            sp_d = np.array([p['sp_delta'] for p in self.extracted_physics], dtype=float) * 1000
            sp_e = np.abs(np.array([p['sp_err'] for p in self.extracted_physics], dtype=float)) * 1000
            w_d = np.array([p['w_delta'] for p in self.extracted_physics], dtype=float) * 1000
            w_e = np.abs(np.array([p['w_err'] for p in self.extracted_physics], dtype=float)) * 1000

            m_sp = np.isfinite(sp_d) & np.isfinite(sp_e)
            m_w = np.isfinite(w_d) & np.isfinite(w_e)
            T_sp, sp_ds, sp_es = self._finite_sorted_series(T_arr, sp_d, sp_e, m_sp)
            T_w, w_ds, w_es = self._finite_sorted_series(T_arr, w_d, w_e, m_w)

            c_sp = [0, 0.4470, 0.7410]
            c_w = [0.8500, 0.3250, 0.0980]

            if T_sp is not None:
                ax.errorbar(T_sp, sp_ds, yerr=sp_es, fmt='-o', color=c_sp, markerfacecolor=c_sp, markeredgecolor='k',
                        linewidth=2, capsize=4, capthick=1.5, elinewidth=1.5,
                        label=r'Single Point $\Delta$ (at target $k$)')
            if T_w is not None:
                ax.errorbar(T_w, w_ds, yerr=w_es, fmt='--s', color=c_w, markerfacecolor=c_w, markeredgecolor='k',
                        linewidth=2, capsize=4, capthick=1.5, elinewidth=1.5,
                        label=r'Inverse-Variance Weighted $\Delta$')

            ax.set_xlabel('Temperature $T$ (K)', fontsize=16)
            ax.set_ylabel(r'Superconducting Gap $\Delta$ (meV)', fontsize=16)
            ax.set_title(r'Temperature Dependence of SC Gap', fontsize=18)
            self._apply_gapless_shading(ax, T_max)
            self._set_scientific_style(ax)

            ax2 = ax.twinx()
            if T_sp is not None:
                ax2.plot(T_sp, sp_es, ':o', color=c_sp, alpha=0.9, linewidth=1.8, markersize=6,
                         markerfacecolor='white', markeredgewidth=1.4, markeredgecolor=c_sp,
                         label=r'Single Point $|\sigma_\Delta|$')
            if T_w is not None:
                ax2.plot(T_w, w_es, ':s', color=c_w, alpha=0.9, linewidth=1.8, markersize=6,
                         markerfacecolor='white', markeredgewidth=1.4, markeredgecolor=c_w,
                         label=r'Weighted $|\sigma_\Delta|$')
            ax2.set_ylabel(r'Absolute uncertainty $|\sigma_\Delta|$ (meV)', fontsize=16, color='dimgray')
            ax2.tick_params(axis='y', direction='in', length=6, width=1.5, colors='dimgray', labelsize=12)
            ax2.spines['right'].set_color('dimgray')
            ax2.spines['right'].set_linewidth(1.5)
            ax2.grid(False)

            h1, l1 = ax.get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels()
            ax.legend(h1 + h2, l1 + l2, loc='best', fontsize=10)
            
            # Apply Y Limits (left axis: Δ only)
            try: ymin = float(self.ent_d_min.get()) if self.ent_d_min.get() != 'auto' else None
            except Exception: ymin = None
            try: ymax = float(self.ent_d_max.get()) if self.ent_d_max.get() != 'auto' else None
            except Exception: ymax = None
            ax.set_ylim(bottom=ymin, top=ymax)
            try:
                symin = float(self.ent_sig_d_min.get()) if self.ent_sig_d_min.get().strip() != 'auto' else None
            except Exception:
                symin = None
            try:
                symax = float(self.ent_sig_d_max.get()) if self.ent_sig_d_max.get().strip() != 'auto' else None
            except Exception:
                symax = None
            ax2.set_ylim(bottom=symin, top=symax)

        elif mode == "Gamma vs T":
            sp_g = np.array([p['sp_gamma'] for p in self.extracted_physics], dtype=float) * 1000
            sp_ge = np.abs(np.array([p['sp_g_err'] for p in self.extracted_physics], dtype=float)) * 1000
            w_g = np.array([p['w_gamma'] for p in self.extracted_physics], dtype=float) * 1000
            w_ge = np.abs(np.array([p['w_g_err'] for p in self.extracted_physics], dtype=float)) * 1000

            m_sp = np.isfinite(sp_g) & np.isfinite(sp_ge)
            m_w = np.isfinite(w_g) & np.isfinite(w_ge)
            T_sp, sp_gs, sp_ges = self._finite_sorted_series(T_arr, sp_g, sp_ge, m_sp)
            T_w, w_gs, w_ges = self._finite_sorted_series(T_arr, w_g, w_ge, m_w)

            c_sp = [0, 0.4470, 0.7410]
            c_w = [0.4660, 0.6740, 0.1880]

            if T_sp is not None:
                ax.errorbar(T_sp, sp_gs, yerr=sp_ges, fmt='-o', color=c_sp,
                            markerfacecolor=c_sp, markeredgecolor='k',
                            linewidth=2, capsize=4, capthick=1.5, elinewidth=1.5,
                            label='Single Point (at target $k$)')
            if T_w is not None:
                ax.errorbar(T_w, w_gs, yerr=w_ges, fmt='--s', color=c_w,
                            markerfacecolor=c_w, markeredgecolor='k',
                            linewidth=2, capsize=4, capthick=1.5, elinewidth=1.5,
                            label='Inverse-Variance Weighted')
            
            ax.set_xlabel('Temperature $T$ (K)', fontsize=16)
            ax.set_ylabel(r'Scattering Rate $\Gamma$ (meV)', fontsize=16)
            ax.set_title(r'Temperature Dependence of Scattering Rate', fontsize=18)
            self._apply_gapless_shading(ax, T_max)
            ax.legend(loc='best', fontsize=12)
            self._set_scientific_style(ax)
            
            # Apply Y Limits
            try: ymin = float(self.ent_g_min.get()) if self.ent_g_min.get() != 'auto' else None
            except: ymin = None
            try: ymax = float(self.ent_g_max.get()) if self.ent_g_max.get() != 'auto' else None
            except: ymax = None
            ax.set_ylim(bottom=ymin, top=ymax)
            
        # Apply X (Temperature) Limits across all modes
        try: tmin = float(self.ent_t_min.get()) if self.ent_t_min.get() != 'auto' else None
        except: tmin = None
        try: tmax = float(self.ent_t_max.get()) if self.ent_t_max.get() != 'auto' else None
        except: tmax = None
        if tmin is not None or tmax is not None:
            ax.set_xlim(left=tmin, right=tmax)
        else:
            ax.set_xlim(left=np.min(T_arr) - 2, right=T_max + 10)
                
        self.fig.tight_layout()
        self.canvas.draw()

if __name__ == "__main__":
    root = tk.Tk()
    root.title("Step 3 Test")
    root.geometry("1000x700")
    app = Step3_TemperatureDependence(root)
    app.pack(fill=tk.BOTH, expand=True)
    root.mainloop()