"""Publication figure style for matplotlib (AIP RSI / APS).

RSI/AIP: https://publishing.aip.org/resources/researchers/author-instructions/
APS:     https://journals.aps.org/authors/style-basics
"""

import matplotlib as mpl
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# --- Column widths (inches) ---
# AIP RSI: 8.5 cm (single) / 17 cm (double). APS PRL is 8.6 / 17.2 cm.
PRL_SINGLE_COL = 3.35
PRL_1P5_COL = 4.49
PRL_DOUBLE_COL = 6.69

PRL_DPI = 600

# Line / marker sizes in points (absolute in vector export)
PRL_MIN_LINE_PT = 0.5
PRL_LINEWIDTH = 1.2
PRL_LINEWIDTH_THICK = 1.5
PRL_MARKERSIZE = 4.0
PRL_MARKERSIZE_LARGE = 5.0

PRL_TICK_LENGTH = 4.0
PRL_TICK_WIDTH = 1.0
PRL_SPINE_WIDTH = 1.0

PRL_FONT_SIZE = 10
PRL_LABEL_SIZE = 11
PRL_TITLE_SIZE = 12
PRL_LEGEND_SIZE = 9

# Okabe-Ito palette — colorblind-safe, grayscale-distinguishable
COLORS = {
    "black": "#000000",
    "blue": "#0072B2",
    "red": "#D55E00",
    "green": "#009E73",
    "orange": "#E69F00",
    "purple": "#CC79A7",
    "cyan": "#56B4E9",
    "gray": "#666666",
    "lightgray": "#BBBBBB",
    "darkgray": "#333333",
    "magenta": "#CC00CC",
    "white": "#FFFFFF",
    "gold": "#FFC20A",
    "lime": "#78D949",
}

# Semantic roles — same meaning across all figures in the pipeline
SERIES = {
    "data": COLORS["darkgray"],
    "data_bg": COLORS["lightgray"],
    "gap_model": COLORS["blue"],
    "metal_model": COLORS["red"],
    "fit": COLORS["blue"],
    "fit_alt": COLORS["red"],
    "background": COLORS["orange"],
    "signal": COLORS["green"],
    "highlight": COLORS["gold"],
    "selected": COLORS["red"],
    "extracted": COLORS["lime"],
    "spline": COLORS["cyan"],
    "reference": COLORS["black"],
    "error": COLORS["orange"],
    "significant": COLORS["green"],
    "insignificant": COLORS["red"],
    "pvalue": COLORS["purple"],
    "weighted": COLORS["red"],
    "single_point": COLORS["blue"],
    "gamma_weighted": COLORS["green"],
    "uncertainty": COLORS["gray"],
    "region": COLORS["lightgray"],
}

# ARPES-style intensity colormap: black → indigo → magenta → gold → white
_ARPES_INTENSITY = LinearSegmentedColormap.from_list(
    "arpes_intensity",
    ["#000000", "#1a0a4e", "#6a0572", "#c43a31", "#f0a202", "#fff8e7"],
    N=256,
)
# Symmetric diverging map for difference / residual panels
_ARPES_DIVERGING = LinearSegmentedColormap.from_list(
    "arpes_diverging",
    ["#2166ac", "#67a9cf", "#f7f7f7", "#ef8a62", "#b2182b"],
    N=256,
)


def get_intensity_cmap():
    return _ARPES_INTENSITY


def get_diverging_cmap():
    return _ARPES_DIVERGING


def configure_matplotlib():
    """Apply global rcParams once at import."""
    mpl.rcParams.update(
        {
            "savefig.dpi": PRL_DPI,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": PRL_FONT_SIZE,
            "axes.labelsize": PRL_LABEL_SIZE,
            "axes.titlesize": PRL_TITLE_SIZE,
            "axes.titlepad": 6,
            "axes.labelpad": 4,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "xtick.labelsize": PRL_FONT_SIZE,
            "ytick.labelsize": PRL_FONT_SIZE,
            "legend.fontsize": PRL_LEGEND_SIZE,
            "lines.linewidth": PRL_LINEWIDTH,
            "lines.markersize": PRL_MARKERSIZE,
            "axes.linewidth": PRL_SPINE_WIDTH,
            "xtick.major.width": PRL_TICK_WIDTH,
            "ytick.major.width": PRL_TICK_WIDTH,
            "xtick.major.size": PRL_TICK_LENGTH,
            "ytick.major.size": PRL_TICK_LENGTH,
            "xtick.minor.size": 2.0,
            "ytick.minor.size": 2.0,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "grid.linestyle": ":",
            "grid.alpha": 0.45,
            "grid.linewidth": 0.5,
            "grid.color": "#CCCCCC",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "svg.hashsalt": "arpes_fig",
        }
    )


def figsize(columns=1, aspect=0.75):
    widths = {1: PRL_SINGLE_COL, 1.5: PRL_1P5_COL, 2: PRL_DOUBLE_COL}
    w = widths.get(columns, PRL_SINGLE_COL)
    return (w, w * aspect)


def gui_figsize(aspect=0.75):
    w = PRL_SINGLE_COL * 2.0
    return (w, w * aspect)


# Step 3 temperature panels — identical layout for 2×2 composite figures
STEP3_PANEL_ASPECT = 0.75  # width : height = 4 : 3


def step3_panel_figsize(for_gui=False):
    """Fixed 4:3 panel size; use ``for_gui=True`` for on-screen preview."""
    if for_gui:
        return gui_figsize(aspect=STEP3_PANEL_ASPECT)
    return figsize(columns=1, aspect=STEP3_PANEL_ASPECT)


STEP3_PANEL_MARGINS = dict(left=0.16, right=0.84, bottom=0.14, top=0.88)


def step3_finalize_axes(ax, ax2=None):
    """PRL tick/spine styling for one Step 3 panel."""
    finalize_publication_axes(ax, ax2=ax2, grid=True)


def step3_apply_panel_layout(fig, ax, ax2=None):
    """Same axes box and margins on every standalone Step 3 panel."""
    step3_finalize_axes(ax, ax2=ax2)
    fig.subplots_adjust(**STEP3_PANEL_MARGINS)


def step3_legend_kwargs(**overrides):
    """Compact legend for RSI single-column panels."""
    kw = dict(
        loc="upper left",
        fontsize=8,
        handlelength=1.6,
        handletextpad=0.45,
        borderpad=0.30,
        labelspacing=0.25,
    )
    kw.update(overrides)
    return legend_kwargs(**kw)


def step3_save_panel(fig, path):
    """Save panel at fixed size — do not crop with bbox='tight'."""
    fig.savefig(path, dpi=PRL_DPI, bbox_inches=None, pad_inches=0.02)


def apply_style(ax, grid=False, minor_ticks=True):
    """Tick, spine, and optional grid styling."""
    ax.tick_params(
        direction="in",
        length=PRL_TICK_LENGTH,
        width=PRL_TICK_WIDTH,
        colors="k",
        top=True,
        right=True,
        labelsize=PRL_FONT_SIZE,
    )
    if minor_ticks:
        try:
            ax.minorticks_on()
            ax.tick_params(which="minor", direction="in", length=2, width=0.6, top=True, right=True)
        except Exception:
            pass
    for spine in ax.spines.values():
        spine.set_linewidth(PRL_SPINE_WIDTH)
        spine.set_color("k")
    ax.set_facecolor("white")
    if grid:
        ax.set_axisbelow(True)
        ax.grid(True, which="major", linestyle=":", alpha=0.45, linewidth=0.5, color="#CCCCCC")


def apply_twinx_style(ax2, color="k"):
    ax2.tick_params(
        axis="y",
        direction="in",
        length=PRL_TICK_LENGTH,
        width=PRL_TICK_WIDTH,
        colors=color,
        labelsize=PRL_FONT_SIZE,
    )
    ax2.spines["right"].set_linewidth(PRL_SPINE_WIDTH)
    ax2.spines["right"].set_color(color)


def set_axis_labels(ax, xlabel=None, ylabel=None, title=None):
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=PRL_LABEL_SIZE)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=PRL_LABEL_SIZE)
    if title:
        ax.set_title(title, fontsize=PRL_TITLE_SIZE, pad=6)


def style_colorbar(cbar, label=None):
    """Publication-style colorbar ticks and outline."""
    cbar.ax.tick_params(direction="out", length=3, width=PRL_TICK_WIDTH, labelsize=PRL_FONT_SIZE)
    cbar.outline.set_linewidth(PRL_SPINE_WIDTH)
    cbar.outline.set_edgecolor("k")
    if label:
        cbar.set_label(label, fontsize=PRL_LABEL_SIZE, labelpad=6)


def add_intensity_colorbar(fig, im, cax, label="Intensity (a.u.)"):
    cbar = fig.colorbar(im, cax=cax)
    style_colorbar(cbar, label=label)
    return cbar


def imshow_intensity(ax, data, extent, vmin=None, vmax=None, cmap=None, aspect="auto"):
    """ARPES-style intensity heatmap."""
    kw = dict(
        aspect=aspect,
        origin="lower",
        extent=extent,
        interpolation="nearest",
        cmap=cmap or get_intensity_cmap(),
    )
    if vmin is not None:
        kw["vmin"] = vmin
    if vmax is not None:
        kw["vmax"] = vmax
    return ax.imshow(data, **kw)


def imshow_diverging(ax, data, extent, vlim, cmap=None, aspect="auto"):
    """Symmetric diverging map for difference / residual panels."""
    return ax.imshow(
        data,
        aspect=aspect,
        origin="lower",
        extent=extent,
        cmap=cmap or get_diverging_cmap(),
        vmin=-abs(vlim),
        vmax=abs(vlim),
        interpolation="nearest",
    )


def scatter_on_heatmap(ax, x, y, role="extracted", label=None):
    """High-contrast markers for overlay on dark ARPES maps."""
    color = SERIES.get(role, COLORS["gold"])
    ax.scatter(
        x, y,
        c=color,
        s=(PRL_MARKERSIZE + 1) ** 2,
        marker="o",
        edgecolors=COLORS["black"],
        linewidths=0.6,
        zorder=5,
        label=label,
    )


def legend_kwargs(**overrides):
    kw = dict(
        frameon=True,
        fancybox=False,
        edgecolor="k",
        facecolor="white",
        framealpha=1.0,
        handlelength=2.0,
        handletextpad=0.6,
        borderpad=0.4,
        labelspacing=0.35,
        fontsize=PRL_LEGEND_SIZE,
    )
    kw.update(overrides)
    return kw


def errorbar_kwargs(color, marker="o", linestyle="-", markersize=None):
    fmt = marker if linestyle == "none" else f"{linestyle}{marker}"
    ms = PRL_MARKERSIZE if markersize is None else markersize
    return dict(
        fmt=fmt,
        color=color,
        markerfacecolor=color,
        markeredgecolor="k",
        markeredgewidth=0.5,
        linewidth=PRL_LINEWIDTH,
        markersize=ms,
        capsize=3.5,
        capthick=PRL_TICK_WIDTH,
        elinewidth=PRL_TICK_WIDTH,
        zorder=4,
    )


def plot_curve_kwargs(color, marker="o", linestyle="-", markersize=None):
    ms = PRL_MARKERSIZE if markersize is None else markersize
    return dict(
        color=color,
        linestyle=linestyle,
        linewidth=PRL_LINEWIDTH,
        marker=marker,
        markersize=ms,
        markerfacecolor=color,
        markeredgecolor="k",
        markeredgewidth=0.5,
        zorder=3,
    )


def reference_line_kwargs(linestyle="--"):
    """Dashed reference / threshold line (PRL linewidth)."""
    return dict(
        color=SERIES["reference"],
        linestyle=linestyle,
        linewidth=PRL_LINEWIDTH,
        zorder=2,
    )


def finalize_publication_axes(ax, ax2=None, grid=False):
    """Apply PRL tick/spine/label sizing after all artists are drawn."""
    apply_style(ax, grid=grid)
    if ax.xaxis.label.get_text():
        ax.xaxis.label.set_fontsize(PRL_LABEL_SIZE)
    if ax.yaxis.label.get_text():
        ax.yaxis.label.set_fontsize(PRL_LABEL_SIZE)
    if ax.title.get_text():
        ax.title.set_fontsize(PRL_TITLE_SIZE)
    if ax2 is not None:
        apply_twinx_style(ax2, color=SERIES["uncertainty"])
        if ax2.yaxis.label.get_text():
            ax2.yaxis.label.set_fontsize(PRL_LABEL_SIZE)
        ax2.tick_params(axis="y", labelsize=PRL_FONT_SIZE, top=False)
        ax2.grid(False)


def plot_fit_line_kwargs(color, linestyle="-"):
    return dict(color=color, linestyle=linestyle, linewidth=PRL_LINEWIDTH_THICK, zorder=4)


def plot_data_points_kwargs(color=None):
    c = color or SERIES["data"]
    return dict(
        linestyle="None",
        marker="o",
        markersize=PRL_MARKERSIZE,
        color=c,
        markerfacecolor=c,
        markeredgecolor="k",
        markeredgewidth=0.5,
        zorder=3,
    )


# Opaque region fills — safe for Adobe Illustrator SVG (no hatch, no alpha)
REGION_GAPLESS = "#E0E0E0"
REGION_BAND = "#EEEEEE"
REGION_SIGNIFICANT = "#E3EFE8"
REGION_INSIGNIFICANT = "#F5E8E3"


def _ai_safe_span(ax, x0, x1, facecolor, label=None, zorder=-10):
    """Vertical band: solid opaque fill for PDF/SVG/Illustrator export."""
    ax.set_axisbelow(True)
    return ax.axvspan(
        x0, x1,
        facecolor=facecolor,
        edgecolor="none",
        linewidth=0,
        alpha=1.0,
        label=label,
        zorder=zorder,
    )


def _ai_safe_fill_between(ax, x, y0, y1, facecolor, label=None, zorder=0):
    """Horizontal band: solid opaque fill for PDF/SVG/Illustrator export."""
    return ax.fill_between(
        x, y0, y1,
        facecolor=facecolor,
        edgecolor="none",
        linewidth=0,
        alpha=1.0,
        label=label,
        zorder=zorder,
    )


def shade_region(ax, x0, x1, label=None, hatch=None):
    """Highlight band on momentum axis (opaque, AI-safe)."""
    return _ai_safe_span(ax, x0, x1, REGION_BAND, label=label)


def shade_gapless(ax, t_start, t_end, label="Gapless region"):
    """Normal-state region above T_c (opaque, AI-safe)."""
    return _ai_safe_span(ax, t_start, t_end, REGION_GAPLESS, label=label)


def shade_significant_pvalue(ax, x, y_bottom, y_top, label="Gap significant"):
    """F-test significant zone (opaque, AI-safe)."""
    return _ai_safe_fill_between(ax, x, y_bottom, y_top, REGION_SIGNIFICANT, label=label)


def shade_insignificant_pvalue(ax, x, y_bottom, y_top, label="Gap not significant"):
    """F-test insignificant zone (opaque, AI-safe)."""
    return _ai_safe_fill_between(ax, x, y_bottom, y_top, REGION_INSIGNIFICANT, label=label)


# Semi-transparent twin-axis bars (Illustrator-friendly: no hatch)
PRL_BAR_ALPHA = 0.38
PRL_BAR_EDGE_LW = 0.6
# Overflow / failed-fit bars: same geometry as in-range bars, light-gray fill
OVERFLOW_BAR_FACE = "#D0D0D0"
OVERFLOW_BAR_EDGE = "#8A8A8A"
OVERFLOW_BAR_LW = 0.6
# Each bar width ≈ 0.8 × characteristic T spacing (half of previous 1.6 setting)
PRL_BAR_WIDTH_FRAC = 0.80
PRL_BAR_OFFSET_FRAC = 0.10


def temperature_bar_spacing(T_arr):
    """
    Characteristic temperature slot width (K) for bar sizing.

    Uses average spacing (span / (n-1)) so bars stay visually thick even when
    some temperature pairs are closely spaced; falls back to median ΔT if needed.
    """
    T = np.unique(np.asarray(T_arr, dtype=float))
    if T.size == 0:
        return 6.0
    if T.size == 1:
        return 6.0
    T = np.sort(T)
    dT = np.diff(T)
    positive = dT[dT > 0]
    if positive.size == 0:
        return 6.0
    avg_slot = (T[-1] - T[0]) / (T.size - 1)
    med_slot = float(np.median(positive))
    return max(avg_slot, med_slot, float(np.min(positive)))


def bar_width_from_temperatures(T_arr):
    """Backward-compatible alias: returns spacing between T points."""
    return temperature_bar_spacing(T_arr)


def plot_offset_uncertainty_bars(
    ax2,
    T_arr,
    values,
    color,
    label,
    side="left",
    group_width=None,
    alpha=PRL_BAR_ALPHA,
    style="fill",
):
    """
    Offset vertical bars on a twin axis for comparing two uncertainty series at each T.

    side: 'left' shifts bars slightly below T; 'right' shifts above T.
    style: 'fill' (default colored bars) or 'frame' (light-gray filled overflow bars).
    """
    T = np.asarray(T_arr, dtype=float)
    H = np.asarray(values, dtype=float)
    if T.size == 0:
        return None
    spacing = group_width if group_width is not None else temperature_bar_spacing(T)
    bar_w = spacing * PRL_BAR_WIDTH_FRAC
    shift = spacing * PRL_BAR_OFFSET_FRAC * (-1 if side == "left" else 1)
    if style == "frame":
        # Light-gray filled bars (not empty frames, not background shading)
        return ax2.bar(
            T + shift,
            H,
            width=bar_w,
            facecolor=OVERFLOW_BAR_FACE,
            edgecolor=OVERFLOW_BAR_EDGE,
            linewidth=OVERFLOW_BAR_LW,
            alpha=1.0,
            label=label,
            align="center",
            zorder=2,
        )
    return ax2.bar(
        T + shift,
        H,
        width=bar_w,
        color=color,
        edgecolor=COLORS["black"],
        linewidth=PRL_BAR_EDGE_LW,
        alpha=alpha,
        label=label,
        align="center",
        zorder=5,
    )


configure_matplotlib()
