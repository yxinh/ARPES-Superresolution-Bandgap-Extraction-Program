# ARPES Superconducting-Gap Extraction

Python analysis suite that accompanies the RSI methods paper on
**multi-momentum weighted-average (MMWA)** extraction of the superconducting
gap from ARPES energy-distribution curves.

The workflow is a three-tab Tkinter application:

1. **Band extraction** — load a high-temperature (normal-state) map, optional
   Shirley background, EDC/MDC peak tracking, and a spline for \(\xi_k\).
2. **Gap fitting** — Dynes + Fermi–Dirac + Gaussian-resolution fits at every
   momentum in a window around \(k_F\), nested \(F\)-test against a gapless
   model, then MMWA combination of the valid \(\Delta(k)\) points.
3. **Temperature dependence** — batch-load Step-2 exports, apply the
   chemical-potential correction \(\Delta_{\mathrm{corr}}^2=\Delta_{\mathrm{app}}^2-\xi_\mu^2\),
   and compare MMWA with single-\(k_F\) (SKF) results against a BCS interpolation.

## Requirements

- Python 3.9 or later
- NumPy, SciPy, pandas, Matplotlib
- Tkinter (included with most Python distributions)

## Installation

```bash
git clone https://github.com/yxinh/ARPES-Superresolution-Bandgap-Extraction-Program.git
cd ARPES-Superresolution-Bandgap-Fitting-Program

conda env create -f environment.yml
conda activate arpes-fitting
```

Alternatively: `pip install -r requirements.txt`.

## Usage

```bash
python MainApp.py
```

Typical sequence:

1. In Step 1, load the high-\(T\) `.dat` map, subtract background, extract the
   band, and fit the spline (this spline is passed to Step 2).
2. In Step 2, load each low-\(T\) map, run the gap / gapless fits, inspect the
   MMWA window, and **Save** / **Export All Results**. Export files are written
   as `result/fit_results_<T>K.txt`.
3. In Step 3, load that folder (or pull saved results from Step 2), set the
   BCS-fit and \(\mu\)-drift windows, and export the temperature panels.

Each Step file can also be launched on its own for debugging
(`python step1_band_extraction.py`, …).

## Repository layout

| File | Role |
| --- | --- |
| `MainApp.py` | Three-tab entry point |
| `arpes_physics.py` | Shared kernels: `.dat` loader, Shirley, Dynes photocurrent, MMWA, \(\mu\) correction, BCS interpolation |
| `gui_common.py` | Shared Tkinter helpers |
| `prl_plot_style.py` | Publication figure style (AIP RSI / APS) |
| `step1_band_extraction.py` | Normal-state band extraction |
| `step2_sc_gap_fitting.py` | Per-EDC fits, \(F\)-test, MMWA |
| `step3_temperature_dependence.py` | \(\Delta(T)\), \(\Gamma(T)\), and BCS comparison |

Symbols in `arpes_physics.py` match the manuscript: `xi_k`, `delta`, `gamma`,
`delta_app`, `delta_corr`, `delta_best`.

Exported text files keep a four-line header plus tab-separated columns
`k`, `delta_fit`, `delta_err`, `gamma_fit`, `gamma_err`, `RSS_gap`, `RSS_met`,
`p_vals`, `delta_point_valid`. Do not change that layout if you want Step 3
to read existing results.

## Citation

If you use this code, please cite the accompanying RSI paper and this repository:

```bibtex
@misc{yang2026arpes_mmwa,
  author       = {Yang, Xinhao},
  title        = {ARPES Superconducting-Gap Extraction},
  year         = {2026},
  publisher    = {GitHub},
  howpublished = {\url{https://github.com/yxinh/ARPES-Superresolution-Bandgap-Extraction-Program}}
}
```

## License

MIT. See `LICENSE`.
