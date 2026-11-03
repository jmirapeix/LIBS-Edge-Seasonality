"""
mtp_v51.py -- Edge-referenced analysis of 2D LIBS Mg/Ca maps of limpet shells
==============================================================================

Code accompanying the paper

    Mirapeix, J., Arniz-Mateos, R., Garcia-Escarzaga, A., Gutierrez-Zugasti, I.,
    Lopez-Higuera, J.M., Cobo, A. "Edge-referenced analysis of 2D LIBS Mg/Ca
    maps for automated seasonality estimation in limpet shells".

Given a 2D Mg/Ca map of a limpet shell section (Calibration-Free LIBS, 50 um
pixels), the script estimates the season of collection automatically:

  1. Final growth edge.  The shell mask is cleaned and the final growth edge is
     delimited from the morphology of the section: the left (ventral) margin is
     followed down to the end of the descending bend of the lower contour, plus
     a 3-pixel margin (Section 2.4.1 of the paper).
  2. Edge-referenced map.  Every pixel is assigned its distance d to that edge,
     and the map is "unrolled" into I(s, d): one row per tangential position s
     (image row), one column per 1-pixel distance bin (Section 2.4.1).
  3. Mg/Ca(d) profile and dynamic calibration.  The median profile is computed
     and the calibration interval (the most recent complete seasonal cycle
     next to the edge) is detected automatically; its 15th/85th percentiles
     are the sample-specific low/high references (Section 2.4.2).
  4. Season estimate.  The Mg/Ca level at the edge (relative to the references)
     and its trend towards the edge give the season: level <= 0.25 winter,
     >= 0.75 summer, otherwise spring (increasing) or autumn (decreasing)
     (Section 2.4.3).
  5. Uncertainty.  Block bootstrap over the rows of the edge-referenced map
     (5-row blocks, 200 resamplings), sensitivity to the first edge pixels
     and an edge-quality index (Section 2.4.4).

Usage
-----
Reproduce all the results of the paper (Tables 4-5, figures, metrics)::

    python mtp_v51.py --batch                    # uses ./data, writes ./outputs
    python mtp_v51.py --batch --boot 50          # quicker test run
    python mtp_v51.py --batch --only LAN96 LAN230

Analyse a single map (e.g. a new specimen)::

    python mtp_v51.py path/to/map.mat --edge-right   # edge on the right: flipped
    python mtp_v51.py                                # file dialog + questions

Input .mat files must contain the Mg/Ca map (variable ``MgCa_map``) and,
optionally, a boolean ``mask`` of valid shell pixels (see load_sample()).

The processing functions come from the previous version of the script
(mtp_v50v7.py) without changes to their logic; only the code needed for the
paper has been kept.  New in v51: morphological edge delimitation, batch mode,
block bootstrap, edge sensitivity, edge-quality index and evaluation metrics.

License: MIT (code).  The Mg/Ca maps in ./data are distributed under CC BY 4.0.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import warnings
from concurrent.futures import ProcessPoolExecutor
from datetime import date
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")                    # figures are written to files
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from scipy.io import loadmat
from scipy.ndimage import distance_transform_edt, binary_fill_holes, median_filter, convolve
from scipy.signal import find_peaks
from scipy.stats import binom
from skimage.morphology import remove_small_objects, binary_opening, binary_closing, disk

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent

# ============================================================================
# 1. SPECIMENS AND PARAMETERS
# ============================================================================
# One entry per specimen analysed in the paper (Tables 4 and 5):
#   id         specimen code used in the paper
#   file       .mat file in the data folder
#   group      "modern" (Langre Beach, known collection date) or "archaeological"
#              (El Toral III cave)
#   edge_left  True if the collection edge is already on the LEFT of the stored
#              map; otherwise the map is flipped horizontally.  The orientation
#              is fixed here on purpose: the metadata stored in some old .mat
#              files (apex_right_edge_left) is not reliable.
#   known      season of collection of the modern specimens (ground truth)
#   date       collection date of the modern specimens (used for the circular error)
#   vs         estimate obtained with the Virtual Sampling method (Mirapeix et al., 2025a)
#   d18o       estimate obtained from delta-18O values (archaeological specimens)

SAMPLES = [
    # id,        file,                group,  edge_left, known,    date,          vs,        d18o
    ("LAN96",   "LAN96_map.mat",     "modern", False, "autumn", "2011-11-12", "autumn", None),
    ("LAN98",   "LAN98_map.mat",     "modern", False, "autumn", "2011-11-12", "autumn", None),
    ("LAN100",  "LAN100_map.mat",    "modern", False, "autumn", "2011-11-12", "autumn", None),
    ("LAN105",  "LAN105.mat",        "modern", False, "autumn", "2011-11-12", "autumn", None),
    ("LAN109",  "LAN109_map.mat",    "modern", False, "autumn", "2011-11-12", "autumn", None),
    ("LAN122",  "LAN122.mat",        "modern", True,  "autumn", "2011-11-12", "summer", None),
    ("LAN176",  "LAN176_map.mat",    "modern", False, "winter", "2012-02-23", "winter", None),
    ("LAN177",  "LAN177_map.mat",    "modern", False, "winter", "2012-02-23", "winter", None),
    ("LAN178",  "LAN178_map.mat",    "modern", False, "winter", "2012-02-23", "winter", None),
    ("LAN182",  "LAN182_map.mat",    "modern", False, "winter", "2012-02-23", "winter", None),
    ("LAN183",  "LAN183_map.mat",    "modern", False, "winter", "2012-02-23", "winter", None),
    ("LAN184",  "LAN184_map.mat",    "modern", False, "winter", "2012-02-23", "summer", None),
    ("LAN229",  "LAN229_map.mat",    "modern", False, "spring", "2012-06-03", "unclear", None),
    ("LAN230",  "LAN230.mat",        "modern", False, "spring", "2012-06-03", "unclear", None),
    ("LAN231",  "LAN231_map.mat",    "modern", False, "spring", "2012-06-03", "spring", None),
    ("LAN235",  "LAN235_map.mat",    "modern", False, "spring", "2012-06-03", "spring", None),
    ("LAN236",  "LAN236_map.mat",    "modern", False, "spring", "2012-06-03", "unclear", None),
    ("LAN237",  "LAN237_map.mat",    "modern", False, "spring", "2012-06-03", "unclear", None),
    ("LAN254",  "LAN254.mat",        "modern", False, "summer", "2012-08-05", "summer", None),
    ("LAN255",  "LAN255.mat",        "modern", False, "summer", "2012-08-05", "summer", None),
    ("LAN256",  "LAN256_map.mat",    "modern", False, "summer", "2012-08-05", "summer", None),
    ("LAN258",  "LAN258_map.mat",    "modern", False, "summer", "2012-08-05", "spring", None),
    ("LAN259",  "LAN259_map.mat",    "modern", False, "summer", "2012-08-05", "summer", None),
    ("LAN261",  "LAN261.mat",        "modern", False, "summer", "2012-08-05", "spring", None),
    ("TOR.13C.9.PV", "TOR13C9.mat",     "archaeological", False, None, None, "autumn", "autumn"),
    ("TOR.22.3.PV",  "TOR223_map.mat",  "archaeological", False, None, None, "winter", "autumn"),
    ("TOR.19.3.PV",  "TOR19-3_map.mat", "archaeological", True,  None, None, "spring", "spring"),
    ("TOR.21.3.PV",  "TOR21-3_map.mat", "archaeological", True,  None, None, "winter/spring", "winter/spring"),
    ("TOR.20.5.PV",  "TOR20-5_map.mat", "archaeological", True,  None, None, "winter", "autumn"),
    ("TOR.17.3.PV",  "TOR!7-3.mat",     "archaeological", False, None, None, "autumn", "autumn"),
]


# Processing parameters.  ONE fixed set is used for all the specimens (modern and
# archaeological); none of them is tuned per specimen.
P = dict(
    # --- shell mask and row-wise edge (Section 2.4.1) ---
    min_size_mask=20, morph_radius=1, top_margin_rows=3,
    median_size_edge=7, min_row_width=8, x_max_edge=40,
    lower_edge_extension_x=45, lower_edge_extension_max_rise=8,
    lower_edge_extension_max_step_y=2,
    # --- removal of isolated high-Mg/Ca cells in the edge-referenced map ---
    isolated_high_percentile=97.5, isolated_min_high_neighbors=2,
    # --- dynamic calibration interval (Section 2.4.2) ---
    dyn_smooth_window=5, dyn_upper_percentile=75, dyn_high_zone_fraction=0.62,
    dyn_min_high_zone_width=3, dyn_margin_points=2,
    dyn_min_prominence_abs=0.30, dyn_prominence_fraction=0.15,
    dyn_min_extrema_distance_points=2,
    # --- season estimate (Section 2.4.3) ---
    edge_band_points=2,          # edge value = median of the first 2 distance bins
    low_ref_percentile=15,       # low/high references = P15/P85 of the Mg/Ca values
    high_ref_percentile=85,      #   within the calibration interval
    trend_points=8,
    # --- uncertainty and sensitivity (Section 2.4.4) ---
    edge_guard_px=0,             # pixels excluded next to the edge (0 = as published)
    sensitivity_guards=(0, 1, 2),
    boot_block_rows=5,           # rows per bootstrap block
    boot_reps=200,
    boot_seed=12345,
    quality_d_px=10,             # edge-quality index computed over d < quality_d_px
    quality_edge_rows_d=3,       # only rows that reach d < this value (touch the edge)
    pixel_size_mm=0.05,
    # Final-growth-edge detection, per group:
    #   "bend"  : geometric end of the lower bend of the contour (+ small margin)
    #   "v50v7" : row-wise edge + lower-boundary extension of mtp_v50v7
    edge_method={"modern": "bend", "archaeological": "bend"},
    bend_tol_deg=5.0,            # descent considered finished when slope >= min(base,0) - tol
    bend_persist_px=8,           # ... during this many consecutive columns
    bend_base_from_px=60,        # far-field lower-contour direction fitted over
    bend_base_to_px=160,         #   [x_tip + from, x_tip + to]
    bend_margin_px=3,            # advance "a little more" after the end of the bend
    sst_min_doy=45,              # SST minimum at El Bocal (annual harmonic fit 1999-2012: min doy 45, max doy 228)
)

# Optional manual end of the final growth edge (x pixel) per specimen, e.g.
# {"LAN229": 50}.  Empty in the paper: the edge is always delimited automatically.
EDGE_X_END_OVERRIDE = {}

SEASONS = ["winter", "spring", "summer", "autumn"]
# Position of the centre of each season in the annual cycle (0 = SST minimum),
# used for the circular error in months.
SEASON_CENTRE = {"winter": 0.0, "spring": 0.25, "summer": 0.5, "autumn": 0.75}


# ============================================================================
# 2. DATA LOADING AND ORIENTATION
# ============================================================================

def matlab_scalar_to_string(value, default=""):
    """
    Converts simple MATLAB strings loaded with scipy.io.loadmat to Python str.
    """
    if value is None:
        return default
    try:
        arr = np.asarray(value).squeeze()
        if arr.size == 1:
            item = arr.item()
            return str(item)
        return str(arr)
    except Exception:
        return default


def load_sample(mat_file):
    """Read a 2D Mg/Ca map from a .mat file.

    Required variable: ``MgCa_map`` (2D array, Mg/Ca in mmol/mol, NaN outside
    the shell).  Optional: ``mask`` (boolean array of valid shell pixels; if
    absent, all non-NaN pixels are used), ``XData``/``YData`` and some metadata
    that are not used by the analysis.  Array rows are the y coordinate and
    columns the x coordinate; maps are displayed with origin='lower'.
    """
    data = loadmat(mat_file)

    mgca = data["MgCa_map"].astype(float)

    if "mask" in data:
        mask = data["mask"].astype(bool)
    else:
        mask = ~np.isnan(mgca)

    xdata = data.get("XData", None)
    ydata = data.get("YData", None)

    sample_id = data.get("sample_id", None)
    if sample_id is not None:
        try:
            sample_id = str(sample_id.squeeze())
        except Exception:
            sample_id = "unknown"
    else:
        sample_id = "unknown"

    orientation = matlab_scalar_to_string(data.get("orientation", None), default="")
    edge_is_left = data.get("edge_is_left", None)

    return {
        "MgCa_map": mgca,
        "mask": mask,
        "XData": xdata,
        "YData": ydata,
        "sample_id": sample_id,
        "orientation": orientation,
        "edge_is_left": edge_is_left,
    }


def flip_map_to_edge_left(mgca, mask=None, xdata=None):
    """Mirror the map horizontally so that the collection edge is on the left.
    All the subsequent steps assume this orientation."""
    mgca_flipped = np.fliplr(mgca)

    if mask is not None:
        mask_flipped = np.fliplr(mask)
    else:
        mask_flipped = None

    return mgca_flipped, mask_flipped, xdata


def select_mat_file(initial_dir=None):
    """Open a file dialog to choose the .mat file to analyse (single-specimen
    mode).  If no graphical window can be opened, the path is asked for in the
    console."""
    if initial_dir is None:
        initial_dir = Path.cwd()
    else:
        initial_dir = Path(initial_dir)

    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)

        file_path = filedialog.askopenfilename(
            title="Select the .mat file with the Mg/Ca map",
            initialdir=str(initial_dir),
            filetypes=[("MATLAB files", "*.mat"), ("All files", "*.*")]
        )

        root.destroy()

        if file_path:
            return file_path

    except Exception as exc:
        print(f"The file dialog could not be opened: {exc}")

    file_path = input("Path of the .mat file to analyse: ").strip().strip('\"')
    if not file_path:
        raise ValueError("No .mat file was selected.")

    return file_path



# ============================================================================
# 3. SHELL MASK AND FINAL GROWTH EDGE
# ============================================================================
# The final growth edge (the shell margin at the time of collection) is the
# reference of the whole method.  It is built in two parts (Section 2.4.1):
#   (a) the left margin of the section, detected row by row (x_edge(y)) and
#       limited to the rows where it behaves as a margin (auto_detect_ymax...);
#   (b) its continuation along the lower contour of the section, which is
#       followed only until the end of the descending bend that follows the
#       shell tip (bend_end_x), plus a 3-pixel margin.

def clean_mask(mask, min_size=20, morph_radius=1):
    """Clean the shell mask: fill holes, remove small detached objects and
    smooth the contour with a morphological opening + closing."""
    mask_clean = mask.copy().astype(bool)

    mask_clean = binary_fill_holes(mask_clean)
    mask_clean = remove_small_objects(mask_clean, min_size=min_size)

    selem = disk(morph_radius)
    mask_clean = binary_opening(mask_clean, selem)
    mask_clean = binary_closing(mask_clean, selem)

    return mask_clean


def exclude_pixels_near_top_boundary(mask, top_margin_rows=3):
    """Exclude the last `top_margin_rows` pixels of each column at the upper
    boundary of the section, where partial ablation of the surface can bias
    the Mg/Ca values.  This mask is used for the analysis, not for the edge."""
    mask_out = mask.copy()
    ny, nx = mask.shape

    for x in range(nx):
        ys = np.where(mask[:, x])[0]
        if ys.size == 0:
            continue

        y_top = ys.max()
        y_exclude_min = max(y_top - top_margin_rows + 1, 0)
        mask_out[y_exclude_min:y_top + 1, x] = False

    return mask_out


def detect_left_edge_rowwise_robust(mask, median_size=7, min_row_width=8):
    """Left margin of the shell section, x_edge(y): first valid pixel of each
    row (rows narrower than `min_row_width` are ignored).  Gaps are
    interpolated and the curve is median-filtered.  Returns the interpolated
    and the filtered curves (NaN where no margin is detected)."""
    ny, nx = mask.shape
    edge_x = np.full(ny, np.nan)
    row_width = np.sum(mask, axis=1)

    for y in range(ny):
        if row_width[y] >= min_row_width:
            xs = np.where(mask[y, :])[0]
            if xs.size > 0:
                edge_x[y] = xs.min()

    valid = ~np.isnan(edge_x)
    y_valid = np.where(valid)[0]

    if y_valid.size < 5:
        raise ValueError("Not enough valid rows to detect the shell edge.")

    all_y = np.arange(ny)

    # Interpolate only gaps BETWEEN valid edge rows.
    # Do not extrapolate beyond the first/last row where a real shell edge
    # has been detected. np.interp otherwise extends the nearest value
    # constantly outside the valid interval, which can create artificial
    # vertical edge segments in cropped maps.
    edge_interp = edge_x.copy()

    y_first = int(y_valid[0])
    y_last = int(y_valid[-1])

    inside_valid_span = (all_y >= y_first) & (all_y <= y_last)
    missing_inside = (~valid) & inside_valid_span

    if np.any(missing_inside):
        edge_interp[missing_inside] = np.interp(
            all_y[missing_inside],
            y_valid,
            edge_x[valid]
        )

    edge_interp[all_y < y_first] = np.nan
    edge_interp[all_y > y_last] = np.nan

    # Median-filter only the finite span. Filtering an array containing NaNs
    # directly can propagate artifacts near the endpoints.
    edge_med = np.full_like(edge_interp, np.nan, dtype=float)
    finite_span = np.where(np.isfinite(edge_interp))[0]
    if finite_span.size > 0:
        i0 = int(finite_span[0])
        i1 = int(finite_span[-1]) + 1
        edge_med[i0:i1] = median_filter(
            edge_interp[i0:i1],
            size=median_size,
            mode='nearest'
        )

    return edge_interp, edge_med


def auto_detect_ymax_from_edge_trend(edge_curve, x_max=None, y_min=None,
                                     min_branch_rows=6,
                                     min_negative_slope=0.15,
                                     min_slope_difference=0.25,
                                     min_fraction_valid_span=0.35,
                                     monotonic_window=5,
                                     monotonic_tol=0.03):
    """
    Find automatically the last row up to which the row-wise left margin
    x_edge(y) represents the final growth edge.

    Near the shell tip x_edge(y) decreases clearly with y; above that point
    the left margin turns into the outer surface of the shell and x_edge(y)
    becomes flat or increasing.  The change is located with a two-line
    (piecewise linear) fit of x(y); a second, local criterion based on the
    monotonic evolution of x(y) is used when the piecewise cut is too early
    (cropped maps or irregular contours).

    Returns:
        y_max_auto : last row of the operative edge (None if not found)
        diagnostics: dictionary with diagnostic information
    """
    edge = np.asarray(edge_curve, dtype=float)
    y_all = np.arange(len(edge))

    valid = ~np.isnan(edge)
    if x_max is not None:
        valid &= edge <= x_max
    if y_min is not None:
        valid &= y_all >= y_min

    y = y_all[valid]
    x = edge[valid]

    diagnostics = {
        "method": "hybrid_piecewise_monotonic_edge_trend",
        "status": "not_computed",
        "y_max_auto": None,
        "slope_lower": np.nan,
        "slope_upper": np.nan,
        "sse": np.nan,
        "candidate_piecewise": None,
        "candidate_monotonic": None,
    }

    if len(y) < 2 * min_branch_rows + 1:
        diagnostics["status"] = "too_few_valid_points"
        return None, diagnostics

    # --------------------------------------------------
    # 1) Main candidate: two-line (piecewise linear) fit of x(y)
    # --------------------------------------------------
    best = None
    for k in range(min_branch_rows, len(y) - min_branch_rows):
        y1, x1 = y[:k + 1], x[:k + 1]
        y2, x2 = y[k + 1:], x[k + 1:]

        try:
            p1 = np.polyfit(y1, x1, 1)
            p2 = np.polyfit(y2, x2, 1)
        except Exception:
            continue

        slope1 = p1[0]
        slope2 = p2[0]

        if slope1 > -min_negative_slope:
            continue
        if abs(slope2 - slope1) < min_slope_difference:
            continue

        fit1 = np.polyval(p1, y1)
        fit2 = np.polyval(p2, y2)
        sse = np.sum((x1 - fit1) ** 2) + np.sum((x2 - fit2) ** 2)

        if best is None or sse < best["sse"]:
            best = {
                "k": k,
                "y_cut": int(y[k]),
                "slope_lower": float(slope1),
                "slope_upper": float(slope2),
                "sse": float(sse),
            }

    # --------------------------------------------------
    # 2) Alternative candidate: local change of monotonicity
    # --------------------------------------------------
    x_smooth = median_filter(x, size=5, mode='nearest')
    dxdy = np.gradient(x_smooth, y)
    candidate_monotonic = None

    for i in range(min_branch_rows, len(y) - min_branch_rows):
        before = np.nanmedian(dxdy[max(0, i - monotonic_window):i])
        after = np.nanmedian(dxdy[i:min(len(y), i + monotonic_window)])

        # Look for the point where x(y) stops decreasing clearly and becomes
        # almost flat or increasing.
        if before < -min_negative_slope and after > -monotonic_tol:
            candidate_monotonic = {
                "y_cut": int(y[i]),
                "slope_lower": float(before),
                "slope_upper": float(after),
            }
            break

    if best is not None:
        diagnostics["candidate_piecewise"] = int(best["y_cut"])
    if candidate_monotonic is not None:
        diagnostics["candidate_monotonic"] = int(candidate_monotonic["y_cut"])

    # --------------------------------------------------
    # 3) Hybrid selection
    # --------------------------------------------------
    if best is not None:
        span_total = float(y[-1] - y[0]) if len(y) > 1 else 0.0
        span_piecewise = float(best["y_cut"] - y[0])
        frac_piecewise = span_piecewise / span_total if span_total > 0 else 0.0

        use_monotonic = False
        if candidate_monotonic is not None:
            # If the piecewise cut is very early, prefer the local cut as long
            # as it extends the operative edge reasonably.
            if frac_piecewise < min_fraction_valid_span and candidate_monotonic["y_cut"] > best["y_cut"]:
                use_monotonic = True

        if use_monotonic:
            diagnostics.update({
                "status": "hybrid_monotonic_override",
                "y_max_auto": candidate_monotonic["y_cut"],
                "slope_lower": candidate_monotonic["slope_lower"],
                "slope_upper": candidate_monotonic["slope_upper"],
                "sse": best["sse"],
            })
            return candidate_monotonic["y_cut"], diagnostics
        else:
            diagnostics.update({
                "status": "ok_piecewise",
                "y_max_auto": best["y_cut"],
                "slope_lower": best["slope_lower"],
                "slope_upper": best["slope_upper"],
                "sse": best["sse"],
            })
            return best["y_cut"], diagnostics

    if candidate_monotonic is not None:
        diagnostics.update({
            "status": "fallback_monotonic",
            "y_max_auto": candidate_monotonic["y_cut"],
            "slope_lower": candidate_monotonic["slope_lower"],
            "slope_upper": candidate_monotonic["slope_upper"],
            "sse": np.nan,
        })
        return candidate_monotonic["y_cut"], diagnostics

    diagnostics["status"] = "no_clear_change_found"
    return None, diagnostics


def restrict_edge_curve(edge_curve, x_max=None, y_min=None, y_max=None):
    """Keep only the part of x_edge(y) with x <= x_max and y_min <= y <= y_max."""
    edge_out = edge_curve.copy()
    ny = len(edge_out)
    y = np.arange(ny)

    if x_max is not None:
        edge_out[edge_out > x_max] = np.nan

    if y_min is not None:
        edge_out[y < y_min] = np.nan

    if y_max is not None:
        edge_out[y > y_max] = np.nan

    return edge_out


def detect_lower_boundary_extension(mask, edge_curve,
                                    max_extension_x=45,
                                    max_vertical_rise=8,
                                    max_step_y=2,
                                    min_points=4):
    """
    Extend the detected final-growth edge along the LOWER shell boundary.

    The row-wise detector describes the left/upper part of the edge well, but
    for strongly oblique shells the physical final edge can continue almost
    horizontally along the lower contour.  That continuation cannot be
    represented reliably by x(y) alone, so this routine returns an independent
    set of (x, y) edge pixels obtained from the lower boundary of the mask.

    The extension starts from the lowest valid row of ``edge_curve`` and moves
    toward positive x while the lower boundary remains smooth and does not rise
    by more than ``max_vertical_rise`` pixels.
    """
    mask = np.asarray(mask, dtype=bool)
    edge = np.asarray(edge_curve, dtype=float)
    ny, nx = mask.shape

    valid_rows = np.where(np.isfinite(edge))[0]
    diag = {
        "status": "not_extended",
        "x_start": np.nan,
        "y_start": np.nan,
        "x_end": np.nan,
        "y_end": np.nan,
        "n_points": 0,
    }
    if valid_rows.size == 0:
        return np.empty((0, 2), dtype=int), diag

    # Use the lowest detected point as the anchor of the lower branch.
    y0 = int(valid_rows.min())
    x0 = int(round(edge[y0]))
    if not (0 <= x0 < nx):
        return np.empty((0, 2), dtype=int), diag

    # Snap the anchor to the actual lower mask boundary at x0 if possible.
    ys0 = np.where(mask[:, x0])[0]
    if ys0.size:
        y0 = int(ys0.min())

    diag["x_start"] = x0
    diag["y_start"] = y0

    pts = [(x0, y0)]
    y_prev = y0
    x_stop = min(nx - 1, x0 + int(max_extension_x))

    for xx in range(x0 + 1, x_stop + 1):
        ys = np.where(mask[:, xx])[0]
        if ys.size == 0:
            break
        yy = int(ys.min())

        # Keep only a smooth continuation of the lower contour.
        if abs(yy - y_prev) > int(max_step_y):
            break
        if yy - y0 > int(max_vertical_rise):
            break

        pts.append((xx, yy))
        y_prev = yy

    pts = np.asarray(pts, dtype=int)
    if len(pts) < int(min_points):
        return np.empty((0, 2), dtype=int), diag

    diag.update({
        "status": "extended_lower_boundary",
        "x_end": int(pts[-1, 0]),
        "y_end": int(pts[-1, 1]),
        "n_points": int(len(pts)),
    })
    return pts, diag


def build_edge_mask_from_curve(edge_x_curve, shape):
    """Rasterise the row-wise edge curve x_edge(y) into a boolean edge mask."""
    ny, nx = shape
    edge_mask = np.zeros((ny, nx), dtype=bool)

    for y in range(ny):
        x = edge_x_curve[y]

        if np.isnan(x):
            continue

        x = int(round(x))

        if 0 <= x < nx:
            edge_mask[y, x] = True

    return edge_mask


def add_edge_points_to_mask(edge_mask, edge_points_xy):
    """Add an arbitrary list of (x, y) edge pixels to an existing edge mask."""
    out = np.array(edge_mask, copy=True, dtype=bool)
    if edge_points_xy is None:
        return out
    pts = np.asarray(edge_points_xy)
    if pts.size == 0:
        return out
    ny, nx = out.shape
    for x, y in pts:
        x = int(x); y = int(y)
        if 0 <= x < nx and 0 <= y < ny:
            out[y, x] = True
    return out


def lower_contour(mask):
    """Lower boundary y_low(x) of the shell mask (origin='lower' convention)."""
    xs, ys = [], []
    for x in range(mask.shape[1]):
        r = np.where(mask[:, x])[0]
        if r.size:
            xs.append(x); ys.append(r.min())
    return np.asarray(xs), np.asarray(ys, float)


def bend_end_x(mask):
    """Geometric end of the final growth edge.

    The shell tip is followed by a curved, descending lower contour that
    eventually becomes straight (parallel to the far-field lower contour).  The
    edge is taken to end where this descent has finished, i.e. where the local
    slope angle of the smoothed lower contour reaches min(base angle, 0) - tol
    for `bend_persist_px` consecutive columns, plus a small margin.

    This replaces the fixed-length extension used in previous versions, which
    could follow the lower contour too far in oblique sections."""
    from scipy.ndimage import median_filter, uniform_filter1d
    xs, ys = lower_contour(mask)
    x_tip = xs.min()                               # leftmost column of the shell = tip
    # Smoothed lower contour and its local slope angle (degrees) at each column
    ysm = uniform_filter1d(median_filter(ys, 5, mode="nearest"), 9, mode="nearest")
    th = np.degrees(np.arctan(np.gradient(ysm, xs)))
    # Reference ("far-field") direction of the lower contour, fitted far from the tip
    b = (xs >= x_tip + P["bend_base_from_px"]) & (xs <= x_tip + P["bend_base_to_px"])
    if b.sum() < 10:
        b = xs >= x_tip + 0.5 * (xs.max() - x_tip)
    th_base = np.degrees(np.arctan(np.polyfit(xs[b], ysm[b], 1)[0]))
    # Start the search at the steepest point of the bend near the tip ...
    near = xs <= x_tip + P["bend_base_from_px"]
    i0 = int(np.argmax(np.abs(th[near] - th_base)))
    # ... and stop where the contour no longer descends (slope back to the
    # far-field direction, within a tolerance) for several consecutive columns.
    ok = th >= min(th_base, 0.0) - P["bend_tol_deg"]
    k = P["bend_persist_px"]
    xe = next((xs[i] for i in range(i0, len(xs) - k) if ok[i:i + k].all()),
              x_tip + P["bend_base_from_px"])
    return int(xe + P["bend_margin_px"])



# ============================================================================
# 4. EDGE-REFERENCED (UNROLLED) MAP
# ============================================================================
# Each shell pixel is assigned its Euclidean distance d to the final growth
# edge, and the map is reorganised into I(s, d): rows s are the original image
# rows (tangential position) and columns are 1-pixel distance bins.  In this
# representation the most recent material is always at d = 0, whatever the
# shape of the section, and every row behaves as an automatically defined
# "trajectory" perpendicular to the growth lines (Section 2.4.1, Fig. 5).

def compute_distance_from_edge(mask, edge_mask, dx=1.0, dy=1.0):
    """Euclidean distance (pixels) from every shell pixel to the nearest pixel
    of the final growth edge (NaN outside the shell)."""
    if not np.any(edge_mask):
        raise ValueError("The edge mask is empty.")

    not_edge = ~edge_mask
    dist = distance_transform_edt(not_edge, sampling=(dy, dx)).astype(float)
    dist[~mask] = np.nan
    return dist


def restrict_analysis_region(mask, dist, max_distance):
    """Shell pixels within `max_distance` of the edge (the whole section in the paper)."""
    return mask & (dist <= max_distance)


def build_unrolled_map_distance_binned_by_row(mgca, analysis_mask, dist, max_distance=None,
                                              distance_bin_width=1.0):
    """
    Build a hybrid unrolled map I(s,d) using:
      - d: Euclidean distance to the detected edge, binned in pixels;
      - s: original image row coordinate.

    This avoids the previous row-wise horizontal unrolling x = edge_x(y) + d,
    which can fail for oblique shells, and also avoids the sparse nearest-edge-point
    representation. For each original row and distance bin, all valid pixels are
    averaged.
    """
    ny, nx = mgca.shape

    valid = analysis_mask & ~np.isnan(dist) & ~np.isnan(mgca)
    if not np.any(valid):
        raise ValueError("No valid pixels available for distance-binned unrolling.")

    if max_distance is None:
        dmax = np.nanmax(dist[valid])
    else:
        dmax = max_distance

    nd = int(np.floor(dmax / distance_bin_width)) + 1
    unrolled_map = np.full((ny, nd), np.nan)
    unrolled_mask = np.zeros((ny, nd), dtype=bool)

    for y in range(ny):
        row_valid = valid[y, :]
        if not np.any(row_valid):
            continue

        d_vals = dist[y, row_valid]
        z_vals = mgca[y, row_valid]

        d_bins = np.floor(d_vals / distance_bin_width).astype(int)
        keep = (d_bins >= 0) & (d_bins < nd)
        d_bins = d_bins[keep]
        z_vals = z_vals[keep]

        if d_bins.size == 0:
            continue

        sums = np.bincount(d_bins, weights=z_vals, minlength=nd)
        counts = np.bincount(d_bins, minlength=nd)
        used = counts > 0

        unrolled_map[y, used] = sums[used] / counts[used]
        unrolled_mask[y, used] = True

    d_coords = np.arange(nd) * distance_bin_width
    s_coords = np.arange(ny)

    return unrolled_map, unrolled_mask, d_coords, s_coords


def remove_isolated_high_pixels_unrolled(unrolled_map, unrolled_mask,
                                         high_percentile=97.5,
                                         min_high_neighbors=2,
                                         neighborhood_size=3):
    """
    Remove isolated high-Mg/Ca cells from the distance-binned unrolled map.

    Rationale:
    very bright isolated cells close to the edge can be caused by local plasma/
    edge effects or poorly supported bins. They can strongly bias edge mean/P75
    profiles and therefore the season estimate.

    A cell is removed only if:
    - it is above the global high_percentile of valid unrolled-map values; and
    - it does not have enough neighbouring high-Mg/Ca cells in a local window.

    This preserves coherent high-Mg/Ca bands while suppressing isolated speckles.
    """
    z = np.array(unrolled_map, copy=True)
    m = np.array(unrolled_mask, copy=True).astype(bool)

    vals = z[m & ~np.isnan(z)]
    if vals.size < 20:
        return z, m, {
            "enabled": True,
            "status": "not_enough_values",
            "threshold": np.nan,
            "removed": 0,
            "valid_before": int(vals.size),
            "valid_after": int(vals.size),
        }

    thr = float(np.nanpercentile(vals, high_percentile))
    high = (z >= thr) & m & ~np.isnan(z)

    kernel = np.ones((neighborhood_size, neighborhood_size), dtype=int)
    neigh = convolve(high.astype(int), kernel, mode="constant", cval=0)

    # neigh includes the central high cell. min_high_neighbors=2 means at least
    # one neighbouring high cell is required.
    isolated = high & (neigh <= min_high_neighbors)

    removed = int(np.sum(isolated))
    z[isolated] = np.nan
    m[isolated] = False

    valid_after = int(np.sum(m & ~np.isnan(z)))
    return z, m, {
        "enabled": True,
        "status": "ok",
        "threshold": thr,
        "removed": removed,
        "valid_before": int(vals.size),
        "valid_after": valid_after,
        "high_percentile": float(high_percentile),
        "min_high_neighbors": int(min_high_neighbors),
    }


def compute_unrolled_statistics(unrolled_map, unrolled_mask):
    """Statistics of the edge-referenced map for each distance bin d (over all
    the rows s): mean, median, SD, percentiles and number of valid cells.
    The median profile Mg/Ca(d) is the one used for the season estimate."""
    nd = unrolled_map.shape[1]

    mean_d = np.full(nd, np.nan)
    median_d = np.full(nd, np.nan)
    std_d = np.full(nd, np.nan)
    p20_d = np.full(nd, np.nan)
    p25_d = np.full(nd, np.nan)
    p75_d = np.full(nd, np.nan)
    p80_d = np.full(nd, np.nan)
    count_d = np.zeros(nd, dtype=int)

    for d in range(nd):
        vals = unrolled_map[:, d]
        vals = vals[unrolled_mask[:, d] & ~np.isnan(vals)]

        if vals.size > 0:
            mean_d[d] = np.mean(vals)
            median_d[d] = np.median(vals)
            std_d[d] = np.std(vals)
            p20_d[d] = np.percentile(vals, 20)
            p25_d[d] = np.percentile(vals, 25)
            p75_d[d] = np.percentile(vals, 75)
            p80_d[d] = np.percentile(vals, 80)
            count_d[d] = vals.size

    return {
        "mean_d": mean_d,
        "median_d": median_d,
        "std_d": std_d,
        "p20_d": p20_d,
        "p25_d": p25_d,
        "p75_d": p75_d,
        "p80_d": p80_d,
        "count_d": count_d
    }



# ============================================================================
# 5. DYNAMIC CALIBRATION INTERVAL
# ============================================================================
# Mg/Ca values are not comparable between specimens (individual and
# environmental effects), so the season is not estimated from absolute values
# but relative to sample-specific references.  These are obtained from the
# "calibration interval": the part of the edge-referenced profile closest to
# the edge that covers the most recent complete seasonal cycle.  Its end is
# detected automatically (no fixed distance) as the shortest interval from the
# edge containing two high-Mg/Ca (summer-like) zones of the 2D map; if they are
# not found, three alternating extrema of the profile (max-min-max or
# min-max-min) are used instead (Section 2.4.2).

def extract_compact_profile(mean_d, std_d=None):
    """
    Extract the valid part of a profile and re-index it so that it starts at
    d = 0, without the gaps of the original index (d_orig keeps the original
    distance of each point).
    """
    mean_d = np.asarray(mean_d, dtype=float)

    valid = ~np.isnan(mean_d)
    if np.sum(valid) == 0:
        return None

    d_orig = np.where(valid)[0]
    y = mean_d[valid]

    d_local = d_orig - d_orig[0]

    result = {
        "d_orig": d_orig,
        "d_local": d_local,
        "y": y
    }

    if std_d is not None:
        std_d = np.asarray(std_d, dtype=float)
        result["std"] = std_d[valid]

    return result


def smooth_compact_series(y, window=5):
    """
    Light moving-average smoothing used to detect extrema in Mg/Ca profiles.
    Keeps the length of the series (reflective padding).
    """
    y = np.asarray(y, dtype=float)
    if len(y) < 3 or window <= 1:
        return y.copy()

    window = int(window)
    if window % 2 == 0:
        window += 1
    window = min(window, len(y) if len(y) % 2 == 1 else len(y) - 1)
    if window < 3:
        return y.copy()

    pad = window // 2
    y_pad = np.pad(y, pad_width=pad, mode='reflect')
    kernel = np.ones(window) / window
    return np.convolve(y_pad, kernel, mode='valid')


def clean_alternating_extrema(extrema):
    """
    Receive a list of extrema sorted by position and remove consecutive extrema
    of the same type, keeping the one with the largest prominence, so that
    maxima and minima alternate.
    """
    if not extrema:
        return []

    cleaned = [extrema[0]]
    for ext in extrema[1:]:
        last = cleaned[-1]
        if ext["type"] != last["type"]:
            cleaned.append(ext)
        else:
            if ext.get("prominence", 0) > last.get("prominence", 0):
                cleaned[-1] = ext
    return cleaned


def _endpoint_extrema(y_s, prominence):
    """
    Adds endpoint extrema candidates. scipy.signal.find_peaks does not detect
    endpoints, but in our profiles the shell edge at d=0 can itself be part of
    the last seasonal maximum or minimum.
    """
    extras = []
    n = len(y_s)
    if n < 3:
        return extras

    # Edge endpoint
    if y_s[0] > y_s[1] and (y_s[0] - np.nanmin(y_s[:min(n, 5)])) >= 0.5 * prominence:
        extras.append({"idx": 0, "type": "max", "prominence": float(y_s[0] - y_s[1])})
    if y_s[0] < y_s[1] and (np.nanmax(y_s[:min(n, 5)]) - y_s[0]) >= 0.5 * prominence:
        extras.append({"idx": 0, "type": "min", "prominence": float(y_s[1] - y_s[0])})

    # Last endpoint, useful only for diagnostics / possible fallback
    if y_s[-1] > y_s[-2] and (y_s[-1] - np.nanmin(y_s[max(0, n-5):])) >= 0.5 * prominence:
        extras.append({"idx": n - 1, "type": "max", "prominence": float(y_s[-1] - y_s[-2])})
    if y_s[-1] < y_s[-2] and (np.nanmax(y_s[max(0, n-5):]) - y_s[-1]) >= 0.5 * prominence:
        extras.append({"idx": n - 1, "type": "min", "prominence": float(y_s[-2] - y_s[-1])})

    return extras


def _detect_profile_extrema(d, d_orig, y_s, prominence, min_extrema_distance_points):
    peaks, peak_props = find_peaks(
        y_s,
        prominence=prominence,
        distance=min_extrema_distance_points
    )
    troughs, trough_props = find_peaks(
        -y_s,
        prominence=prominence,
        distance=min_extrema_distance_points
    )

    extrema = []
    for idx_i, prom in zip(peaks, peak_props.get("prominences", [])):
        extrema.append({
            "idx": int(idx_i),
            "d": float(d[idx_i]),
            "d_orig": float(d_orig[idx_i]),
            "value": float(y_s[idx_i]),
            "type": "max",
            "prominence": float(prom)
        })
    for idx_i, prom in zip(troughs, trough_props.get("prominences", [])):
        extrema.append({
            "idx": int(idx_i),
            "d": float(d[idx_i]),
            "d_orig": float(d_orig[idx_i]),
            "value": float(y_s[idx_i]),
            "type": "min",
            "prominence": float(prom)
        })

    for ext in _endpoint_extrema(y_s, prominence):
        idx_i = ext["idx"]
        extrema.append({
            "idx": int(idx_i),
            "d": float(d[idx_i]),
            "d_orig": float(d_orig[idx_i]),
            "value": float(y_s[idx_i]),
            "type": ext["type"],
            "prominence": float(ext.get("prominence", prominence))
        })

    # Remove duplicated endpoint/internal candidates at the same index and type.
    unique = {}
    for ext in extrema:
        key = (ext["idx"], ext["type"])
        if key not in unique or ext.get("prominence", 0) > unique[key].get("prominence", 0):
            unique[key] = ext

    extrema = sorted(unique.values(), key=lambda e: e["d"])
    return clean_alternating_extrema(extrema)


def _find_first_alternating_triplet(extrema):
    if len(extrema) < 3:
        return None

    for i in range(len(extrema) - 2):
        triad = extrema[i:i+3]
        types = [e["type"] for e in triad]
        if types in (["max", "min", "max"], ["min", "max", "min"]):
            return triad
    return None


def _update_calibration_diag(d, d_orig, y, y_s, diagnostics, status, n_use,
                             extrema=None, prominence=np.nan, smooth_window=None):
    n_use = int(np.clip(n_use, 1, len(y)))
    y_cal = y[:n_use]
    y_s_cal = y_s[:n_use]

    # Use the measured profile for the actual range, not only the smoothed curve.
    y_min = float(np.nanmin(y_cal))
    y_max = float(np.nanmax(y_cal))
    amp = y_max - y_min

    edge_level = np.nan
    if amp > 0:
        edge_level = float(np.clip((y_cal[0] - y_min) / amp, 0, 1))

    diagnostics.update({
        "status": status,
        "n_points_calibration": int(n_use),
        "d_end": float(d[n_use - 1]),
        "d_end_orig": float(d_orig[n_use - 1]),
        "profile_min": y_min,
        "profile_max": y_max,
        "profile_amplitude": float(amp),
        "edge_level_normalized_dynamic": edge_level,
        "prominence_used": float(prominence) if not np.isnan(prominence) else np.nan,
        "smooth_window": int(smooth_window) if smooth_window is not None else None,
        "calibration_mean_smoothed_min": float(np.nanmin(y_s_cal)),
        "calibration_mean_smoothed_max": float(np.nanmax(y_s_cal)),
    })

    if extrema is not None:
        diagnostics["extrema_used_for_window"] = extrema

    return int(n_use), diagnostics


def _segments_from_labels(labels, min_len=2):
    """
    Convert a sequence of labels ('high', 'low', 'mid') into contiguous segments.
    Very short high/low segments are ignored to reduce sensitivity to single-point noise.
    """
    segments = []
    n = len(labels)
    i = 0
    while i < n:
        lab = labels[i]
        j = i + 1
        while j < n and labels[j] == lab:
            j += 1
        if lab in ("high", "low") and (j - i) >= min_len:
            segments.append({"type": lab, "start": i, "end": j - 1})
        i = j
    return segments


def _merge_same_type_segments(segments, max_gap=2):
    """
    Merge neighbouring segments of the same type if they are separated by a short gap.
    """
    if not segments:
        return []
    merged = [segments[0].copy()]
    for seg in segments[1:]:
        last = merged[-1]
        gap = seg["start"] - last["end"] - 1
        if seg["type"] == last["type"] and gap <= max_gap:
            last["end"] = seg["end"]
        else:
            merged.append(seg.copy())
    return merged


def _find_first_alternating_zone_triplet(d, d_orig, y_s,
                                         high_fraction=0.65,
                                         low_fraction=0.35,
                                         min_segment_len=2,
                                         max_gap=2):
    """
    Detect a dynamic calibration interval using value zones rather than fixed distance.

    The profile is converted into high/low Mg/Ca zones using thresholds derived from
    the observed local range. The selected window ends at the third alternating zone,
    i.e. high-low-high or low-high-low. This is intended to capture either two maxima
    and one minimum, or two minima and one maximum, without imposing a distance such
    as 80 pixels.
    """
    y_min = float(np.nanmin(y_s))
    y_max = float(np.nanmax(y_s))
    amp = y_max - y_min
    if amp <= 0:
        return None, []

    high_thr = y_min + high_fraction * amp
    low_thr = y_min + low_fraction * amp

    labels = []
    for v in y_s:
        if v >= high_thr:
            labels.append("high")
        elif v <= low_thr:
            labels.append("low")
        else:
            labels.append("mid")

    segments = _segments_from_labels(labels, min_len=min_segment_len)
    segments = _merge_same_type_segments(segments, max_gap=max_gap)

    if len(segments) < 3:
        return None, segments

    for i in range(len(segments) - 2):
        triad = segments[i:i+3]
        types = [seg["type"] for seg in triad]
        if types in (["high", "low", "high"], ["low", "high", "low"]):
            # Convert zones to a structure similar to extrema for diagnostics.
            zone_extrema = []
            for seg in triad:
                sl = slice(seg["start"], seg["end"] + 1)
                if seg["type"] == "high":
                    idx_rel = int(np.nanargmax(y_s[sl])) + seg["start"]
                    ext_type = "max"
                else:
                    idx_rel = int(np.nanargmin(y_s[sl])) + seg["start"]
                    ext_type = "min"
                zone_extrema.append({
                    "idx": int(idx_rel),
                    "d": float(d[idx_rel]),
                    "d_orig": float(d_orig[idx_rel]),
                    "value": float(y_s[idx_rel]),
                    "type": ext_type,
                    "prominence": float(amp),
                    "zone_start_idx": int(seg["start"]),
                    "zone_end_idx": int(seg["end"]),
                    "zone_type": seg["type"]
                })
            end_idx = triad[-1]["end"]
            return int(end_idx), zone_extrema

    return None, segments


def estimate_dynamic_calibration_window(mean_d,
                                        std_d=None,
                                        smooth_window=5,
                                        min_prominence_abs=0.30,
                                        prominence_fraction=0.15,
                                        min_extrema_distance_points=2,
                                        min_points=4,
                                        margin_points=2,
                                        max_search_distance=None,
                                        fallback_max_points=None):
    """
    Automatically estimates a calibration window from the Mg/Ca profile.

    No fixed fallback distance is used. The preferred criterion is the shortest
    edge-referenced interval containing three alternating extrema: max-min-max or
    min-max-min. If peak detection fails, the function tries a value-zone criterion
    based on high/low Mg/Ca regions. If neither criterion finds a valid seasonal
    pattern, no dynamic calibration window is returned.
    """
    prof = extract_compact_profile(mean_d, std_d)
    diagnostics = {
        "status": "not_computed",
        "n_points": 0,
        "n_extrema": 0,
        "n_points_calibration": None,
        "d_end": None,
        "d_end_orig": None,
        "profile_min": np.nan,
        "profile_max": np.nan,
        "profile_amplitude": np.nan,
        "edge_level_normalized_dynamic": np.nan,
        "extrema": [],
        "extrema_used_for_window": [],
        "zone_segments": []
    }

    if prof is None:
        diagnostics["status"] = "no_valid_profile"
        return None, diagnostics

    d = prof["d_local"].astype(float)
    y = prof["y"].astype(float)
    d_orig = prof["d_orig"].astype(float)

    if max_search_distance is not None:
        keep = d <= max_search_distance
        d = d[keep]
        y = y[keep]
        d_orig = d_orig[keep]

    diagnostics["n_points"] = len(y)

    if len(y) < min_points:
        diagnostics["status"] = "too_few_points"
        return None, diagnostics

    y_s = smooth_compact_series(y, window=smooth_window)
    y_range = float(np.nanmax(y_s) - np.nanmin(y_s))

    if y_range <= 0:
        diagnostics["status"] = "flat_profile"
        return None, diagnostics

    # 1) Try extrema-based detection with increasingly relaxed prominence.
    prominence_candidates = [
        max(float(min_prominence_abs), float(prominence_fraction) * y_range),
        max(0.15, 0.10 * y_range),
        max(0.08, 0.06 * y_range),
        max(0.04, 0.03 * y_range),
    ]

    extrema = []
    triad = None
    prominence_used = prominence_candidates[0]
    for prominence in prominence_candidates:
        extrema_try = _detect_profile_extrema(
            d, d_orig, y_s,
            prominence=prominence,
            min_extrema_distance_points=max(1, min_extrema_distance_points)
        )
        triad_try = _find_first_alternating_triplet(extrema_try)
        if triad_try is not None:
            extrema = extrema_try
            triad = triad_try
            prominence_used = prominence
            break
        if len(extrema_try) > len(extrema):
            extrema = extrema_try
            prominence_used = prominence

    diagnostics["n_extrema"] = len(extrema)
    diagnostics["extrema"] = extrema

    if triad is not None:
        e3 = triad[-1]
        end_idx = min(e3["idx"] + margin_points, len(y) - 1)
        n_use = max(end_idx + 1, min_points)
        n_use = min(n_use, len(y))
        return _update_calibration_diag(
            d, d_orig, y, y_s, diagnostics,
            status="ok_alternating_extrema",
            n_use=n_use,
            extrema=triad,
            prominence=prominence_used,
            smooth_window=smooth_window
        )

    # 2) Try value-zone detection: high-low-high or low-high-low regions.
    # This is often more robust for broad increments that do not form sharp peaks.
    zone_settings = [
        (0.70, 0.30, 2),
        (0.65, 0.35, 2),
        (0.60, 0.40, 2),
        (0.58, 0.42, 1),
    ]
    for high_f, low_f, min_seg_len in zone_settings:
        end_idx, zone_info = _find_first_alternating_zone_triplet(
            d, d_orig, y_s,
            high_fraction=high_f,
            low_fraction=low_f,
            min_segment_len=min_seg_len,
            max_gap=2
        )
        diagnostics["zone_segments"] = zone_info
        if end_idx is not None:
            end_idx = min(end_idx + margin_points, len(y) - 1)
            n_use = max(end_idx + 1, min_points)
            return _update_calibration_diag(
                d, d_orig, y, y_s, diagnostics,
                status="ok_alternating_value_zones",
                n_use=n_use,
                extrema=zone_info,
                prominence=np.nan,
                smooth_window=smooth_window
            )

    # 3) No fixed fallback. Do not use the whole shell and do not invent a window.
    diagnostics["status"] = "no_reliable_dynamic_pattern"
    diagnostics["n_points_calibration"] = None
    diagnostics["d_end"] = None
    diagnostics["d_end_orig"] = None
    return None, diagnostics


def _column_profile_from_unrolled(unrolled_map, unrolled_mask, percentile=75, min_valid_pixels=3):
    """
    Builds a robust column-wise profile from the unrolled 2D map.

    The mean Mg/Ca(d) profile can hide diagonal/high-Mg increments because values
    are averaged over the tangential coordinate. For calibration-window detection,
    an upper percentile profile or high-zone score is often more sensitive to
    summer-like high Mg/Ca bands visible in the 2D map.
    """
    nd = unrolled_map.shape[1]
    prof = np.full(nd, np.nan)
    counts = np.zeros(nd, dtype=int)

    for d in range(nd):
        vals = unrolled_map[:, d]
        vals = vals[unrolled_mask[:, d] & ~np.isnan(vals)]
        counts[d] = vals.size
        if vals.size >= min_valid_pixels:
            prof[d] = np.nanpercentile(vals, percentile)

    return prof, counts


def _find_first_two_high_zones_from_profile(y_score,
                                            smooth_window=5,
                                            high_fraction=0.62,
                                            min_segment_len=3,
                                            max_gap=2,
                                            margin_points=2):
    """
    Finds the shortest interval containing two high-Mg/Ca zones.

    This is intended to capture two summer-like increments without imposing a
    fixed pixel distance. The detector works on a score profile, normally an
    upper-percentile Mg/Ca profile extracted from the unrolled 2D map.
    """
    y_score = np.asarray(y_score, dtype=float)
    valid = ~np.isnan(y_score)

    if np.sum(valid) < max(6, min_segment_len * 2):
        return None, [], None

    d_orig = np.where(valid)[0]
    y = y_score[valid]
    d_local = d_orig - d_orig[0]

    y_s = smooth_compact_series(y, window=smooth_window)
    y_min = float(np.nanmin(y_s))
    y_max = float(np.nanmax(y_s))
    amp = y_max - y_min
    if amp <= 0:
        return None, [], {"status": "flat_score_profile"}

    high_thr = y_min + high_fraction * amp

    labels = []
    for v in y_s:
        labels.append("high" if v >= high_thr else "mid")

    segments = _segments_from_labels(labels, min_len=min_segment_len)
    # _segments_from_labels keeps only high/low; for this call it will keep high.
    segments = [seg for seg in segments if seg["type"] == "high"]
    segments = _merge_same_type_segments(segments, max_gap=max_gap)

    if len(segments) < 2:
        diag = {
            "status": "fewer_than_two_high_zones",
            "high_thr": float(high_thr),
            "score_min": y_min,
            "score_max": y_max,
            "score_amplitude": float(amp),
            "high_segments": segments,
        }
        return None, segments, diag

    # End after the second high zone, not after the whole map.
    second = segments[1]
    end_local_idx = min(second["end"] + margin_points, len(y) - 1)
    end_orig_idx = int(d_orig[end_local_idx])

    zone_info = []
    for seg in segments[:2]:
        sl = slice(seg["start"], seg["end"] + 1)
        idx_rel = int(np.nanargmax(y_s[sl])) + seg["start"]
        zone_info.append({
            "idx": int(idx_rel),
            "d": float(d_local[idx_rel]),
            "d_orig": float(d_orig[idx_rel]),
            "value": float(y_s[idx_rel]),
            "type": "max",
            "prominence": float(amp),
            "zone_start_idx": int(seg["start"]),
            "zone_end_idx": int(seg["end"]),
            "zone_type": "high"
        })

    diag = {
        "status": "ok_two_high_zones_2d_score",
        "high_thr": float(high_thr),
        "score_min": y_min,
        "score_max": y_max,
        "score_amplitude": float(amp),
        "high_segments": segments,
        "score_d_orig": d_orig,
        "score_d_local": d_local,
        "score_y": y,
        "score_y_smooth": y_s,
        "score_profile_name": "upper-percentile column Mg/Ca profile",
    }

    return end_orig_idx, zone_info, diag


def _high_fraction_score_from_unrolled(unrolled_map, unrolled_mask,
                                      high_value_percentile=75,
                                      min_valid_pixels=3):
    """
    Computes a column-wise score for high-Mg/Ca bands in the unrolled 2D map.

    For each distance column d, the score is the fraction of valid pixels whose
    Mg/Ca value is above a robust global high-value threshold. This is often
    more sensitive to diagonal/high-Mg seasonal bands than the column mean.
    """
    vals_all = unrolled_map[unrolled_mask & ~np.isnan(unrolled_map)]
    nd = unrolled_map.shape[1]
    score = np.full(nd, np.nan)
    counts = np.zeros(nd, dtype=int)

    if vals_all.size == 0:
        return score, counts, np.nan

    high_value_thr = float(np.nanpercentile(vals_all, high_value_percentile))

    for d in range(nd):
        vals = unrolled_map[:, d]
        vals = vals[unrolled_mask[:, d] & ~np.isnan(vals)]
        counts[d] = vals.size
        if vals.size >= min_valid_pixels:
            score[d] = float(np.mean(vals >= high_value_thr))

    return score, counts, high_value_thr


def _find_first_two_high_zones_from_score(score_profile,
                                          smooth_window=5,
                                          score_fraction=0.45,
                                          min_score_abs=0.12,
                                          min_segment_len=2,
                                          max_gap=2,
                                          margin_points=2):
    """
    Finds the shortest edge-referenced interval containing two high-Mg/Ca zones
    in a column-wise high-band score. No fixed distance is imposed.
    """
    score_profile = np.asarray(score_profile, dtype=float)
    valid = ~np.isnan(score_profile)

    if np.sum(valid) < max(6, min_segment_len * 2):
        return None, [], {"status": "too_few_valid_score_points"}

    d_orig = np.where(valid)[0]
    score = score_profile[valid]
    d_local = d_orig - d_orig[0]
    score_s = smooth_compact_series(score, window=smooth_window)

    s_min = float(np.nanmin(score_s))
    s_max = float(np.nanmax(score_s))
    amp = s_max - s_min
    if amp <= 0:
        return None, [], {"status": "flat_high_fraction_score"}

    thr = max(float(min_score_abs), s_min + float(score_fraction) * amp)
    labels = ["high" if v >= thr else "mid" for v in score_s]
    segments = _segments_from_labels(labels, min_len=min_segment_len)
    segments = [seg for seg in segments if seg["type"] == "high"]
    segments = _merge_same_type_segments(segments, max_gap=max_gap)

    if len(segments) < 2:
        return None, segments, {
            "status": "fewer_than_two_high_fraction_zones",
            "score_threshold": float(thr),
            "score_min": s_min,
            "score_max": s_max,
            "score_amplitude": float(amp),
            "high_segments": segments,
            "score_d_orig": d_orig,
            "score_d_local": d_local,
            "score_y": score,
            "score_y_smooth": score_s,
        }

    second = segments[1]
    end_local_idx = min(second["end"] + margin_points, len(score_s) - 1)
    end_orig_idx = int(d_orig[end_local_idx])

    zone_info = []
    for seg in segments[:2]:
        sl = slice(seg["start"], seg["end"] + 1)
        idx_rel = int(np.nanargmax(score_s[sl])) + seg["start"]
        zone_info.append({
            "idx": int(idx_rel),
            "d": float(d_local[idx_rel]),
            "d_orig": float(d_orig[idx_rel]),
            "value": float(score_s[idx_rel]),
            "type": "max",
            "prominence": float(amp),
            "zone_start_idx": int(seg["start"]),
            "zone_end_idx": int(seg["end"]),
            "zone_type": "high_fraction"
        })

    diag = {
        "status": "ok_two_high_fraction_zones_2d",
        "score_threshold": float(thr),
        "score_min": s_min,
        "score_max": s_max,
        "score_amplitude": float(amp),
        "high_segments": segments,
        "score_d_orig": d_orig,
        "score_d_local": d_local,
        "score_y": score,
        "score_y_smooth": score_s,
        "score_profile_name": "column high-Mg/Ca pixel fraction",
    }
    return end_orig_idx, zone_info, diag


def estimate_dynamic_calibration_window_from_unrolled(unrolled_map,
                                                      unrolled_mask,
                                                      mean_d,
                                                      std_d=None,
                                                      smooth_window=5,
                                                      upper_percentile=75,
                                                      high_fraction=0.62,
                                                      min_segment_len=3,
                                                      max_gap=2,
                                                      margin_points=2,
                                                      min_valid_pixels=3,
                                                      fallback_to_mean_profile=True,
                                                      **mean_profile_kwargs):
    """
    Dynamic calibration interval of the paper (Section 2.4.2).

    Returns (n_points_calibration, diagnostics); diagnostics["d_end_orig"] is
    the distance (pixels) at which the calibration interval ends.

    Dynamic calibration using high-Mg/Ca bands in the 2D unrolled map.

    It first tries to detect the shortest edge-referenced interval containing two
    high-Mg/Ca zones in an upper-percentile column profile. This is more sensitive
    to diagonal/summer increments than the simple mean Mg/Ca(d) profile.

    If two high zones are not found, it can fall back to the previous mean-profile
    alternating-extrema criterion. No fixed distance is imposed.
    """
    # First try a true 2D high-band score: fraction of high-Mg/Ca pixels
    # per distance column. This usually detects broad/diagonal summer increments
    # more robustly than a mean or even an upper-percentile profile.
    high_score_profile, valid_counts, high_value_thr = _high_fraction_score_from_unrolled(
        unrolled_map,
        unrolled_mask,
        high_value_percentile=upper_percentile,
        min_valid_pixels=min_valid_pixels
    )

    end_orig_idx, zone_info, score_diag = None, [], None
    # The detection is attempted with progressively less strict thresholds.
    for score_fraction, min_score_abs, seg_len in [
        (0.45, 0.12, min_segment_len),
        (0.38, 0.10, max(2, min_segment_len - 1)),
        (0.32, 0.08, max(1, min_segment_len - 1)),
    ]:
        end_try, zones_try, diag_try = _find_first_two_high_zones_from_score(
            high_score_profile,
            smooth_window=smooth_window,
            score_fraction=score_fraction,
            min_score_abs=min_score_abs,
            min_segment_len=seg_len,
            max_gap=max_gap,
            margin_points=margin_points
        )
        if diag_try is not None:
            diag_try["high_value_threshold_mgca"] = high_value_thr
        if end_try is not None:
            end_orig_idx, zone_info, score_diag = end_try, zones_try, diag_try
            break
        if score_diag is None:
            score_diag = diag_try

    # If high-fraction scoring fails, try the previous upper-percentile profile.
    if end_orig_idx is None:
        score_profile, valid_counts = _column_profile_from_unrolled(
            unrolled_map,
            unrolled_mask,
            percentile=upper_percentile,
            min_valid_pixels=min_valid_pixels
        )

        end_orig_idx, zone_info, score_diag_p = _find_first_two_high_zones_from_profile(
            score_profile,
            smooth_window=smooth_window,
            high_fraction=high_fraction,
            min_segment_len=min_segment_len,
            max_gap=max_gap,
            margin_points=margin_points
        )
        if score_diag_p is not None:
            score_diag_p["score_profile_name"] = "upper-percentile column Mg/Ca profile"
        if end_orig_idx is not None:
            score_diag = score_diag_p

    prof = extract_compact_profile(mean_d, std_d)
    if prof is None:
        diagnostics = {"status": "no_valid_mean_profile"}
        return None, diagnostics

    d = prof["d_local"].astype(float)
    y = prof["y"].astype(float)
    d_orig = prof["d_orig"].astype(float)
    y_s = smooth_compact_series(y, window=smooth_window)

    if end_orig_idx is not None:
        # Convert original d index into number of compact mean-profile points.
        idx_candidates = np.where(d_orig <= end_orig_idx)[0]
        if idx_candidates.size == 0:
            n_use = None
        else:
            n_use = int(idx_candidates[-1] + 1)

        if n_use is not None and n_use >= 4:
            diagnostics = {
                "status": "ok_two_high_zones_2d_score",
                "n_points": int(len(y)),
                "n_extrema": len(zone_info),
                "extrema": zone_info,
                "extrema_used_for_window": zone_info,
                "zone_segments": zone_info,
                "column_score_valid_counts": valid_counts,
            }
            if score_diag is not None:
                diagnostics.update({k: v for k, v in score_diag.items()
                                    if k not in ("score_d_orig", "score_d_local", "score_y", "score_y_smooth")})
            return _update_calibration_diag(
                d, d_orig, y, y_s, diagnostics,
                status="ok_two_high_zones_2d_score",
                n_use=n_use,
                extrema=zone_info,
                prominence=np.nan,
                smooth_window=smooth_window
            )

    if fallback_to_mean_profile:
        n_use, diag = estimate_dynamic_calibration_window(
            mean_d,
            std_d=std_d,
            smooth_window=smooth_window,
            **mean_profile_kwargs
        )
        diag["status"] = "fallback_mean_profile_" + str(diag.get("status"))
        diag["two_high_zone_score_status"] = None if score_diag is None else score_diag.get("status")
        return n_use, diag

    diagnostics = {
        "status": "no_two_high_zones_2d_score",
        "n_points": int(len(y)),
        "n_extrema": 0,
        "n_points_calibration": None,
        "d_end": None,
        "d_end_orig": None,
        "profile_min": np.nan,
        "profile_max": np.nan,
        "profile_amplitude": np.nan,
        "edge_level_normalized_dynamic": np.nan,
        "extrema": [],
        "extrema_used_for_window": [],
        "zone_segments": [],
        "two_high_zone_score_status": None if score_diag is None else score_diag.get("status"),
    }
    return None, diagnostics



# ============================================================================
# 6. SEASON-OF-COLLECTION ESTIMATE (LEVEL/TREND CRITERION)
# ============================================================================
# Section 2.4.3 of the paper.  With the low (P15) and high (P85) references of
# the calibration interval, the Mg/Ca value at the edge is expressed as a
# relative level between 0 and 1:
#     level <= 0.25  -> winter   (edge close to the seasonal minimum)
#     level >= 0.75  -> summer   (edge close to the seasonal maximum)
#     otherwise      -> spring if Mg/Ca increases towards the edge,
#                       autumn if it decreases towards the edge.
# The criterion is equivalent to the quartile method used for delta-18O values
# (Mannino et al., 2003, 2007).

def _edge_value_from_unrolled(unrolled_map, unrolled_mask, first_col, edge_band_points=2):
    """
    Robust edge Mg/Ca value from the first valid distance columns of the unrolled map.
    """
    nd = unrolled_map.shape[1]
    first_col = int(np.clip(first_col, 0, nd - 1))
    last_col = int(min(first_col + edge_band_points, nd))
    vals = unrolled_map[:, first_col:last_col]
    msk = unrolled_mask[:, first_col:last_col]
    vals = vals[msk & ~np.isnan(vals)]
    if vals.size == 0:
        return np.nan, 0
    return float(np.nanmedian(vals)), int(vals.size)


def estimate_primary_season_from_dynamic_cycle(unrolled_map, unrolled_mask, mean_d,
                                               dynamic_diag,
                                               edge_band_points=2,
                                               low_ref_percentile=15,
                                               high_ref_percentile=85,
                                               trend_points=8,
                                               near_edge_points=12,
                                               immediate_trend_points=6,
                                               contextual_trend_min_points=10,
                                               contextual_trend_fraction=0.65,
                                               broad_trend_min_points=18,
                                               broad_trend_fraction=1.00):
    """
    Season of collection from the edge-referenced map (primary criterion of
    the paper).

    Inputs: the edge-referenced map, the median profile Mg/Ca(d) (passed as
    `mean_d`) and the diagnostics of the dynamic calibration interval.
    Output: dictionary with the season ("season_label"), the edge value and
    references, the relative edge level ("edge_level_dynamic"), the trend
    towards the edge ("trend_label") and the method used to obtain it
    ("trend_method"), plus many diagnostic values.

    The level is evaluated first.  For intermediate levels the direction of the
    Mg/Ca variation towards the edge is evaluated at several scales, so that a
    small local fluctuation next to the border does not determine the result:
    a multi-scale local slope (8, 12 and 16 bins) when its windows agree; else
    the slope over the calibration interval when it shows a clear fall or rise
    towards the edge; else the contrast between the edge and the immediate or
    contextual interior.

    Note on an earlier implementation, which could be too sensitive to the first 2-6 distance bins.
    In some archaeological samples, a local dip very close to the edge made the
    edge value slightly higher than the immediate interior reference, producing
    a spring estimate even when the broader edge-referenced profile showed a
    decreasing Mg/Ca trend toward the shell edge.

    A broad calibration-window slope was therefore added:
      - positive Mg/Ca(d) slope as d increases inward means that Mg/Ca decreases
        toward the edge -> autumn-like;
      - negative Mg/Ca(d) slope as d increases inward means that Mg/Ca increases
        toward the edge -> spring-like.

    For intermediate edge levels, this broad trend is allowed to override a weak
    immediate edge contrast when it is supported by a clear drop from an interior
    high-Mg/Ca band or rise from an interior low-Mg/Ca band.
    """
    if dynamic_diag is None:
        return None

    n_use = dynamic_diag.get("n_points_calibration", None)
    if n_use is None:
        return None

    prof = extract_compact_profile(mean_d)
    if prof is None or len(prof["y"]) < 4:
        return None

    d_orig = prof["d_orig"].astype(int)
    y = prof["y"].astype(float)
    d = prof["d_local"].astype(float)
    first_col = int(d_orig[0])

    n_use = int(min(max(n_use, 1), len(d_orig)))
    d_end_orig = int(d_orig[n_use - 1])

    cal_vals = unrolled_map[:, first_col:d_end_orig + 1]
    cal_msk = unrolled_mask[:, first_col:d_end_orig + 1]
    cal_vals = cal_vals[cal_msk & ~np.isnan(cal_vals)]
    if cal_vals.size < 6:
        return None

    # Sample-specific references: P15 and P85 of ALL the map cells (all rows s)
    # within the calibration interval.
    mgca_min_ref = float(np.nanpercentile(cal_vals, low_ref_percentile))
    mgca_max_ref = float(np.nanpercentile(cal_vals, high_ref_percentile))
    amp = mgca_max_ref - mgca_min_ref
    if not np.isfinite(amp) or amp <= 0:
        return None

    edge_value, edge_n = _edge_value_from_unrolled(
        unrolled_map, unrolled_mask, first_col, edge_band_points=edge_band_points
    )
    if not np.isfinite(edge_value):
        edge_value = float(y[0])
        edge_n = 0

    # Relative edge level: 0 = low reference, 1 = high reference.
    edge_level = float(np.clip((edge_value - mgca_min_ref) / amp, 0, 1))

    # Use the robust map-derived edge_value for the LEVEL criterion,
    # but use the actual edge-referenced mean-profile value for TREND estimation.
    # This avoids mixing two different statistics when deciding whether Mg/Ca
    # increases or decreases toward d=0 (relevant for cases such as TOR19-3).
    edge_trend_value = float(y[0])

    # Thresholds scale with the dynamic amplitude, with a small absolute floor.
    trend_thr = max(0.10, 0.04 * amp)
    strong_trend_thr = max(2.0 * trend_thr, 0.25)
    # Slope is in Mg/Ca units per distance pixel. This is deliberately small,
    # because the broad trend is accumulated over many distance bins.
    broad_slope_thr = max(0.006, 0.004 * amp)
    broad_drop_thr = max(0.25, 0.18 * amp)

    k_local = min(int(near_edge_points), len(y))
    y_local = y[:k_local]
    d_local = d[:k_local]

    near_edge_max = np.nan
    near_edge_min = np.nan
    near_edge_max_d = np.nan
    near_edge_min_d = np.nan
    near_edge_drop_from_max = np.nan
    near_edge_rise_from_min = np.nan

    slope_space = np.nan
    delta_space = np.nan
    delta_time = np.nan
    interior_ref_for_trend = np.nan

    immediate_ref = np.nan
    immediate_delta_time = np.nan
    immediate_slope_space = np.nan

    contextual_ref_for_trend = np.nan
    contextual_delta_time = np.nan
    contextual_slope_space = np.nan
    contextual_start_d = np.nan
    contextual_stop_d = np.nan

    broad_slope_space = np.nan
    broad_trend_time = np.nan
    broad_start_d = np.nan
    broad_stop_d = np.nan
    broad_max = np.nan
    broad_min = np.nan
    broad_max_d = np.nan
    broad_min_d = np.nan
    broad_drop_from_max = np.nan
    broad_rise_from_min = np.nan
    broad_trend_label = "unknown"

    trend_time = np.nan
    trend_method = "unknown"

    # Multi-scale LOCAL near-edge trend.
    # The broad calibration-window slope can be dominated by a later seasonal
    # maximum/minimum and therefore describe the whole cycle rather than the
    # direction of the FINAL branch approaching d=0.  We estimate the slope
    # over several nested near-edge windows and use the median when their signs
    # are consistent.
    local_multiscale_slope_space = np.nan
    local_multiscale_trend_time = np.nan
    local_multiscale_ref = np.nan
    local_multiscale_ref_d = np.nan
    local_multiscale_agreement = False

    if k_local >= 3:
        # ------------------------------------------------------------
        # 1) Immediate contrast: edge vs first interior bins.
        # ------------------------------------------------------------
        k_imm = min(int(immediate_trend_points), len(y))
        if k_imm >= 3:
            immediate_ref = float(np.nanmedian(y[1:k_imm]))
            immediate_delta_time = float(edge_trend_value - immediate_ref)
            immediate_slope_space = float(np.polyfit(d[:k_imm], y[:k_imm], 1)[0])

        # ------------------------------------------------------------
        # 2) Contextual near-edge contrast within the dynamic calibration window.
        # ------------------------------------------------------------
        k_context = min(int(n_use), len(y))
        if k_context >= int(contextual_trend_min_points):
            start_idx = max(int(immediate_trend_points), 3)
            stop_idx = max(start_idx + 3, int(np.ceil(contextual_trend_fraction * k_context)))
            stop_idx = min(stop_idx, k_context)

            if stop_idx > start_idx:
                contextual_ref_for_trend = float(np.nanmedian(y[start_idx:stop_idx]))
                contextual_delta_time = float(edge_trend_value - contextual_ref_for_trend)
                contextual_start_d = float(d[start_idx])
                contextual_stop_d = float(d[stop_idx - 1])
                contextual_slope_space = float(np.polyfit(d[:stop_idx], y[:stop_idx], 1)[0])

        # ------------------------------------------------------------
        # 3) Broad calibration-window slope.
        #    This criterion is used to avoid decisions based only on
        #    a very local dip/spike close to d=0.
        # ------------------------------------------------------------
        k_broad = min(int(n_use), len(y))
        if k_broad >= int(broad_trend_min_points):
            stop_broad = max(int(broad_trend_min_points), int(np.ceil(broad_trend_fraction * k_broad)))
            stop_broad = min(stop_broad, k_broad)
            if stop_broad >= int(broad_trend_min_points):
                # Smooth only for the broad slope/diagnostic extrema; references
                # and edge values still use measured profile/map values.
                y_broad_s = smooth_compact_series(y[:stop_broad], window=7)
                broad_slope_space = float(np.polyfit(d[:stop_broad], y_broad_s, 1)[0])
                broad_trend_time = float(-broad_slope_space * max(d[stop_broad - 1] - d[0], 1.0))
                broad_start_d = float(d[0])
                broad_stop_d = float(d[stop_broad - 1])

                i_bmax = int(np.nanargmax(y_broad_s))
                i_bmin = int(np.nanargmin(y_broad_s))
                broad_max = float(y_broad_s[i_bmax])
                broad_min = float(y_broad_s[i_bmin])
                broad_max_d = float(d[i_bmax])
                broad_min_d = float(d[i_bmin])
                broad_drop_from_max = float(broad_max - edge_trend_value)
                broad_rise_from_min = float(edge_trend_value - broad_min)

                if broad_slope_space > broad_slope_thr and broad_drop_from_max > broad_drop_thr:
                    broad_trend_label = "decreasing_toward_edge"
                elif broad_slope_space < -broad_slope_thr and broad_rise_from_min > broad_drop_thr:
                    broad_trend_label = "increasing_toward_edge"
                else:
                    broad_trend_label = "weak_or_mixed_broad_trend"

        # ------------------------------------------------------------
        # 4) Secondary local extrema diagnostics.
        # ------------------------------------------------------------
        y_inside = y_local[1:]
        d_inside = d_local[1:]
        if y_inside.size > 0 and np.any(np.isfinite(y_inside)):
            i_max = int(np.nanargmax(y_inside))
            i_min = int(np.nanargmin(y_inside))
            near_edge_max = float(y_inside[i_max])
            near_edge_min = float(y_inside[i_min])
            near_edge_max_d = float(d_inside[i_max])
            near_edge_min_d = float(d_inside[i_min])
            near_edge_drop_from_max = float(near_edge_max - edge_trend_value)
            near_edge_rise_from_min = float(edge_trend_value - near_edge_min)

        # ------------------------------------------------------------
        # 5) Multi-scale LOCAL slope over the final branch approaching d=0.
        #    Use nested windows of approximately 8, 12 and 16 points whenever
        #    available. Negative dy/dd means Mg/Ca increases toward the edge.
        # ------------------------------------------------------------
        slope_candidates = []
        window_candidates = []
        for kw in (8, 12, 16):
            kk = min(int(kw), len(y), int(n_use))
            if kk >= 5:
                ys = smooth_compact_series(y[:kk], window=5)
                ss = float(np.polyfit(d[:kk], ys, 1)[0])
                if np.isfinite(ss):
                    slope_candidates.append(ss)
                    window_candidates.append(kk)

        if len(slope_candidates) >= 2:
            signs = np.sign(slope_candidates)
            nonzero = signs != 0
            if np.any(nonzero):
                sign_vals = signs[nonzero]
                majority_sign = np.sign(np.sum(sign_vals))
                n_majority = int(np.sum(sign_vals == majority_sign)) if majority_sign != 0 else 0
                local_multiscale_agreement = n_majority >= max(2, int(np.ceil(0.67 * len(sign_vals))))

            if local_multiscale_agreement:
                local_multiscale_slope_space = float(np.nanmedian(slope_candidates))
                k_ref = int(np.median(window_candidates))
                k_ref = max(3, min(k_ref, len(y), int(n_use)))
                local_multiscale_ref_d = float(d[k_ref - 1])
                # Use a robust interior value near the end of the representative
                # window for the plotted trend reference.
                i0 = max(1, k_ref - 3)
                local_multiscale_ref = float(np.nanmedian(y[i0:k_ref]))
                local_multiscale_trend_time = float(
                    -local_multiscale_slope_space *
                    max(local_multiscale_ref_d - d[0], 1.0)
                )

        # ------------------------------------------------------------
        # 6) Combine indicators.
        #    Positive trend_time => increasing toward edge => spring.
        #    Negative trend_time => decreasing toward edge => autumn.
        # ------------------------------------------------------------
        imm_ok = np.isfinite(immediate_delta_time) and abs(immediate_delta_time) > trend_thr
        ctx_ok = np.isfinite(contextual_delta_time) and abs(contextual_delta_time) > trend_thr
        broad_autumn = broad_trend_label == "decreasing_toward_edge"
        broad_spring = broad_trend_label == "increasing_toward_edge"

        # Priority: a clear broad calibration-window slope can override a
        # weak immediate edge contrast in intermediate Mg/Ca cases. This is the
        # behaviour needed for cases where the first bins are affected by a local
        # dip/spike but the map/profile show a broader fall toward the border.
        local_ms_ok = (
            local_multiscale_agreement and
            np.isfinite(local_multiscale_trend_time) and
            abs(local_multiscale_trend_time) > trend_thr
        )

        if local_ms_ok:
            trend_time = float(local_multiscale_trend_time)
            trend_method = "local_multiscale_slope"
        elif broad_autumn and (not imm_ok or immediate_delta_time < strong_trend_thr):
            trend_time = float(broad_trend_time)  # negative by construction
            trend_method = "broad_calibration_slope_over_weak_immediate"
        elif broad_spring and (not imm_ok or immediate_delta_time > -strong_trend_thr):
            trend_time = float(broad_trend_time)  # positive by construction
            trend_method = "broad_calibration_slope_over_weak_immediate"
        elif imm_ok and ctx_ok:
            same_sign = np.sign(immediate_delta_time) == np.sign(contextual_delta_time)
            if same_sign:
                trend_time = float(0.55 * immediate_delta_time + 0.45 * contextual_delta_time)
                trend_method = "immediate_and_contextual_agree"
            else:
                if abs(immediate_delta_time) >= strong_trend_thr and abs(immediate_delta_time) >= abs(contextual_delta_time):
                    trend_time = float(immediate_delta_time)
                    trend_method = "strong_immediate_over_contextual"
                else:
                    trend_time = float(contextual_delta_time)
                    trend_method = "contextual_over_weak_immediate"
        elif ctx_ok:
            trend_time = float(contextual_delta_time)
            trend_method = "contextual_edge_vs_calibration_window"
        elif imm_ok:
            trend_time = float(immediate_delta_time)
            trend_method = "edge_vs_immediate_interior_median"
        else:
            # Last fallback: robust comparison with a slightly broader near-edge
            # interior region.
            k = min(trend_points, len(y))
            if k >= 3:
                slope_space = float(np.polyfit(d[:k], y[:k], 1)[0])
                delta_space = float(y[min(k - 1, 5)] - y[0])
                interior_start = max(2, min(3, k - 1))
                interior_stop = k
                interior_ref_for_trend = float(np.nanmedian(y[interior_start:interior_stop]))
                trend_time = float(edge_trend_value - interior_ref_for_trend)
                delta_time = -delta_space if np.isfinite(delta_space) else np.nan
                trend_method = "edge_vs_near_interior_median"

    if np.isfinite(trend_time):
        if trend_time > trend_thr:
            trend_label = "increasing_toward_edge"
        elif trend_time < -trend_thr:
            trend_label = "decreasing_toward_edge"
        else:
            trend_label = "stable_near_edge"
    else:
        trend_label = "unknown"

    # ---- Final decision: level first, then trend (Section 2.4.3) ----
    low_level_thr = 0.25
    high_level_thr = 0.75

    if edge_level <= low_level_thr:
        season, phase = "winter", 0.05
        reason = "edge Mg/Ca close to dynamic seasonal minimum"
    elif edge_level >= high_level_thr:
        season, phase = "summer", 0.55
        reason = "edge Mg/Ca close to dynamic seasonal maximum"
    else:
        if trend_time > trend_thr:
            season, phase = "spring", 0.35
            reason = "intermediate Mg/Ca level and increasing toward edge"
        elif trend_time < -trend_thr:
            season, phase = "autumn", 0.70
            reason = "intermediate Mg/Ca level and decreasing toward edge"
        else:
            season, phase = "transitional", 0.50
            reason = "intermediate Mg/Ca level and weak edge trend"

    # Heuristic reliability index (diagnostic only; not used in the paper, where
    # uncertainty is assessed with the block bootstrap).
    amp_score = 1.0 - np.exp(-amp / 1.0)
    n_score = min(1.0, cal_vals.size / 300.0)
    trend_score = 0.5 if not np.isfinite(trend_time) else min(1.0, abs(trend_time) / 0.15)
    reliability = float(np.clip(0.45 * amp_score + 0.35 * n_score + 0.20 * trend_score, 0, 1))

    return {
        "phase": float(phase),
        "season_label": season,
        "reason": reason,
        "edge_value": float(edge_value),
        "edge_n": int(edge_n),
        "mgca_min_ref": mgca_min_ref,
        "mgca_max_ref": mgca_max_ref,
        "dynamic_amplitude_ref": float(amp),
        "edge_level_dynamic": edge_level,
        "low_level_threshold": float(low_level_thr),
        "high_level_threshold": float(high_level_thr),
        "slope_space": float(slope_space) if np.isfinite(slope_space) else np.nan,
        "trend_time": float(trend_time) if np.isfinite(trend_time) else np.nan,
        "delta_time": float(delta_time) if np.isfinite(delta_time) else np.nan,
        "edge_ref_for_trend": float(edge_trend_value) if np.isfinite(edge_trend_value) else np.nan,
        "interior_ref_for_trend": float(interior_ref_for_trend) if np.isfinite(interior_ref_for_trend) else np.nan,
        "immediate_interior_ref_for_trend": float(immediate_ref) if np.isfinite(immediate_ref) else np.nan,
        "immediate_delta_time": float(immediate_delta_time) if np.isfinite(immediate_delta_time) else np.nan,
        "immediate_slope_space": float(immediate_slope_space) if np.isfinite(immediate_slope_space) else np.nan,
        "contextual_ref_for_trend": float(contextual_ref_for_trend) if np.isfinite(contextual_ref_for_trend) else np.nan,
        "contextual_delta_time": float(contextual_delta_time) if np.isfinite(contextual_delta_time) else np.nan,
        "contextual_slope_space": float(contextual_slope_space) if np.isfinite(contextual_slope_space) else np.nan,
        "contextual_start_d": float(contextual_start_d) if np.isfinite(contextual_start_d) else np.nan,
        "contextual_stop_d": float(contextual_stop_d) if np.isfinite(contextual_stop_d) else np.nan,
        "broad_slope_space": float(broad_slope_space) if np.isfinite(broad_slope_space) else np.nan,
        "broad_trend_time": float(broad_trend_time) if np.isfinite(broad_trend_time) else np.nan,
        "broad_trend_label": broad_trend_label,
        "broad_start_d": float(broad_start_d) if np.isfinite(broad_start_d) else np.nan,
        "broad_stop_d": float(broad_stop_d) if np.isfinite(broad_stop_d) else np.nan,
        "broad_max": float(broad_max) if np.isfinite(broad_max) else np.nan,
        "broad_max_d": float(broad_max_d) if np.isfinite(broad_max_d) else np.nan,
        "broad_min": float(broad_min) if np.isfinite(broad_min) else np.nan,
        "broad_min_d": float(broad_min_d) if np.isfinite(broad_min_d) else np.nan,
        "broad_drop_from_max": float(broad_drop_from_max) if np.isfinite(broad_drop_from_max) else np.nan,
        "broad_rise_from_min": float(broad_rise_from_min) if np.isfinite(broad_rise_from_min) else np.nan,
        "broad_slope_threshold": float(broad_slope_thr),
        "broad_drop_threshold": float(broad_drop_thr),
        "trend_label": trend_label,
        "trend_method": trend_method,
        "local_multiscale_slope_space": float(local_multiscale_slope_space) if np.isfinite(local_multiscale_slope_space) else np.nan,
        "local_multiscale_trend_time": float(local_multiscale_trend_time) if np.isfinite(local_multiscale_trend_time) else np.nan,
        "local_multiscale_ref": float(local_multiscale_ref) if np.isfinite(local_multiscale_ref) else np.nan,
        "local_multiscale_ref_d": float(local_multiscale_ref_d) if np.isfinite(local_multiscale_ref_d) else np.nan,
        "local_multiscale_agreement": bool(local_multiscale_agreement),
        "trend_threshold": float(trend_thr),
        "strong_trend_threshold": float(strong_trend_thr),
        "near_edge_max": float(near_edge_max) if np.isfinite(near_edge_max) else np.nan,
        "near_edge_min": float(near_edge_min) if np.isfinite(near_edge_min) else np.nan,
        "near_edge_max_d": float(near_edge_max_d) if np.isfinite(near_edge_max_d) else np.nan,
        "near_edge_min_d": float(near_edge_min_d) if np.isfinite(near_edge_min_d) else np.nan,
        "near_edge_drop_from_max": float(near_edge_drop_from_max) if np.isfinite(near_edge_drop_from_max) else np.nan,
        "near_edge_rise_from_min": float(near_edge_rise_from_min) if np.isfinite(near_edge_rise_from_min) else np.nan,
        "d_end_orig": float(d_end_orig),
        "n_points_calibration": int(n_use),
        "reliability_index": reliability,
    }



# ============================================================================
# 7. PROCESSING PIPELINE (ONE SPECIMEN)
# ============================================================================
# geometry()  -> final growth edge and edge-referenced map
# classify()  -> profile, calibration interval and season (one call = one estimate)
# bootstrap() -> season probabilities by resampling blocks of rows
# process()   -> everything for one specimen, including sensitivity and edge quality

@contextlib.contextmanager
def quiet():
    """Silence the diagnostic messages printed by the processing functions."""
    with contextlib.redirect_stdout(io.StringIO()):
        yield


def geometry(mat_path: Path, edge_left: bool, method="bend", x_end_override=None):
    """Load a map, delimit the final growth edge and build the edge-referenced map.

    method="bend" is the morphological delimitation used in the paper;
    method="v50v7" reproduces the fixed-length lower extension of the previous
    version of the script (kept only for comparison)."""
    s = load_sample(str(mat_path))
    mgca, mask = s["MgCa_map"], s["mask"]
    if not edge_left:
        mgca, mask, _ = flip_map_to_edge_left(mgca, mask)
    # Shell mask used for the geometry, and analysis mask (upper boundary excluded)
    mask_geom = clean_mask(mask, min_size=P["min_size_mask"], morph_radius=P["morph_radius"])
    mask_an = exclude_pixels_near_top_boundary(mask_geom, top_margin_rows=P["top_margin_rows"])
    # (a) Left margin x_edge(y), limited to the rows where it is the growth edge
    _, edge_med = detect_left_edge_rowwise_robust(mask_geom, median_size=P["median_size_edge"],
                                                    min_row_width=P["min_row_width"])
    y_max, _ = auto_detect_ymax_from_edge_trend(edge_med, x_max=P["x_max_edge"], y_min=None)
    edge_curve = restrict_edge_curve(edge_med.copy(), x_max=P["x_max_edge"], y_min=None, y_max=y_max)
    x_end = None
    if method == "v50v7":
        ext, _ = detect_lower_boundary_extension(
            mask_geom, edge_curve, max_extension_x=P["lower_edge_extension_x"],
            max_vertical_rise=P["lower_edge_extension_max_rise"],
            max_step_y=P["lower_edge_extension_max_step_y"])
    else:
        # follow the lower contour freely, then cut it at the end of the bend
        ext, _ = detect_lower_boundary_extension(
            mask_geom, edge_curve, max_extension_x=mask_geom.shape[1],
            max_vertical_rise=mask_geom.shape[0], max_step_y=3)
        x_end = x_end_override if x_end_override is not None else bend_end_x(mask_geom)
        edge_curve = edge_curve.copy()
        edge_curve[edge_curve > x_end] = np.nan
        if len(ext):
            ext = ext[ext[:, 0] <= x_end]
    # Edge mask -> distance to the edge -> edge-referenced map I(s, d)
    edge_mask = add_edge_points_to_mask(build_edge_mask_from_curve(edge_curve, mgca.shape), ext)
    dist = compute_distance_from_edge(mask_an, edge_mask, dx=1.0, dy=1.0)
    dmax = float(np.nanmax(dist[mask_an & np.isfinite(dist)]))
    region = restrict_analysis_region(mask_an, dist, max_distance=dmax)
    um, umask, _, _ = build_unrolled_map_distance_binned_by_row(
        mgca, region, dist, max_distance=dmax, distance_bin_width=1.0)
    # Remove isolated very high cells (local plasma/edge artefacts)
    um, umask, _ = remove_isolated_high_pixels_unrolled(
        um, umask, high_percentile=P["isolated_high_percentile"],
        min_high_neighbors=P["isolated_min_high_neighbors"], neighborhood_size=3)
    return dict(mgca=mgca, region=region, dist=dist, edge_curve=edge_curve,
                edge_ext=ext, um=um, umask=umask, x_end=x_end, edge_method=method)


def classify(um, umask, guard=0):
    """Season estimate (level/trend criterion) from an edge-referenced map.

    `guard` excludes the first `guard` distance bins next to the edge (used for
    the sensitivity analysis; 0 in the primary estimate).
    Returns (profile statistics, calibration diagnostics, season estimate)."""
    mk = umask.copy()
    if guard > 0:
        mk[:, :guard] = False
    st = compute_unrolled_statistics(um, mk)
    # 1) calibration interval (Section 2.4.2)
    _, diag = estimate_dynamic_calibration_window_from_unrolled(
        um, mk, st["mean_d"], std_d=st["std_d"], smooth_window=P["dyn_smooth_window"],
        upper_percentile=P["dyn_upper_percentile"], high_fraction=P["dyn_high_zone_fraction"],
        min_segment_len=P["dyn_min_high_zone_width"], max_gap=2,
        margin_points=P["dyn_margin_points"], min_valid_pixels=3, fallback_to_mean_profile=True,
        min_prominence_abs=P["dyn_min_prominence_abs"],
        prominence_fraction=P["dyn_prominence_fraction"],
        min_extrema_distance_points=P["dyn_min_extrema_distance_points"],
        max_search_distance=None, fallback_max_points=None)
    # 2) level/trend estimate on the MEDIAN profile Mg/Ca(d) (Section 2.4.3)
    prim = estimate_primary_season_from_dynamic_cycle(
        um, mk, st["median_d"], diag, edge_band_points=P["edge_band_points"],
        low_ref_percentile=P["low_ref_percentile"], high_ref_percentile=P["high_ref_percentile"],
        trend_points=P["trend_points"])
    return st, diag, prim


def bootstrap(um, umask, guard, reps, block, seed):
    """Block bootstrap over the rows of the edge-referenced map (Section 2.4.4).

    The rows s (tangential positions) are grouped into blocks of `block`
    contiguous rows, which preserves the correlation between neighbouring rows.
    In each of the `reps` resamplings, blocks are drawn with replacement until
    the original number of blocks is reached, and the COMPLETE classification
    (including the dynamic calibration) is repeated on the resampled map.
    Returns the proportion of resamplings assigned to each season.
    A fixed seed makes the result reproducible."""
    rng = np.random.default_rng(seed)
    rows = np.where(umask.any(axis=1))[0]
    if rows.size == 0:
        return {s: np.nan for s in SEASONS}
    starts = np.arange(rows.min(), rows.max() + 1, block)
    counts = {s: 0 for s in SEASONS + ["transitional", "none"]}
    for _ in range(reps):
        pick = rng.choice(starts, size=starts.size, replace=True)
        sel = np.concatenate([np.arange(a, min(a + block, um.shape[0])) for a in pick])
        try:
            _, _, p = classify(um[sel], umask[sel], guard)
            lab = p["season_label"] if p else "none"
        except Exception:
            lab = "none"
        counts[lab] += 1
    return {k: v / reps for k, v in counts.items()}


def edge_quality(um, umask):
    """Edge-quality index (Section 2.4.4): tangential heterogeneity near the
    edge, in mmol/mol.  For each row that reaches the edge (d < 3 px), the mean
    Mg/Ca over d < 10 px is computed; the index is the SD of these row means.
    High values indicate heterogeneous near-edge growth increments (threshold in
    the paper: 90th percentile of the modern specimens, ~1.0 mmol/mol)."""
    D, De = P["quality_d_px"], P["quality_edge_rows_d"]
    blk = np.where(umask[:, :D], um[:, :D], np.nan)
    touches = np.isfinite(blk[:, :De]).any(axis=1)
    rm = np.nanmean(blk[touches], axis=1)
    rm = rm[np.isfinite(rm)]
    return (float(np.std(rm)) if rm.size >= 4 else np.nan), int(rm.size)


def process(entry, data_dir: Path, reps: int):
    """Full analysis of one specimen (one entry of SAMPLES)."""
    sid, fname, group, edge_left, known, cdate, vs, d18o = entry
    with quiet():
        g = geometry(data_dir / fname, edge_left, P["edge_method"][group],
                     EDGE_X_END_OVERRIDE.get(sid))
        st, diag, prim = classify(g["um"], g["umask"], P["edge_guard_px"])
        # Sensitivity: repeat the estimate excluding 0, 1 and 2 bins next to the edge
        sens = {}
        for gg in P["sensitivity_guards"]:
            _, _, pp = classify(g["um"], g["umask"], gg)
            sens[gg] = pp["season_label"] if pp else "none"
        probs = bootstrap(g["um"], g["umask"], P["edge_guard_px"], reps,
                          P["boot_block_rows"], P["boot_seed"])
    q_sd, q_n = edge_quality(g["um"], g["umask"])
    return dict(id=sid, file=fname, group=group, edge_left=edge_left, known=known,
                date=cdate, vs=vs, d18o=d18o, geom=g, stats=st, diag=diag, prim=prim,
                sens=sens, probs=probs, q_sd=q_sd, q_n=q_n)



# ============================================================================
# 8. EVALUATION METRICS (MODERN SPECIMENS)
# ============================================================================
# Every metric is reported with its chance level (random assignment of one of
# the four seasons): exact agreement 25 %, correct or adjacent season 75 %,
# partial credit 0.5, circular error 3 months.  Binomial tests compare the
# exact and adjacent rates with their chance levels (Section 2.4.4).

def circ(a, b):
    """Circular distance between two phases of the annual cycle (0-1)."""
    d = (a - b) % 1.0
    return min(d, 1.0 - d)


def date_phase(dstr):
    """Phase of a date in the annual SST cycle (0 = SST minimum at El Bocal,
    day of year 45, i.e. mid-February)."""
    d = date.fromisoformat(dstr)
    doy = d.timetuple().tm_yday
    return ((doy - P["sst_min_doy"]) / 365.25) % 1.0


def season_steps(pred, true):
    """0 = same season, 1 = adjacent season, 2 = opposite season."""
    if pred not in SEASONS:
        return None
    k = (SEASONS.index(pred) - SEASONS.index(true)) % 4
    return min(k, 4 - k)


def evaluate(results, key="main"):
    """Performance metrics for the modern specimens, for the proposed method
    (key="main") or for the Virtual Sampling estimates (key="vs")."""
    mod = [r for r in results if r["group"] == "modern"]
    n = len(mod)
    steps, errs = [], []
    for r in mod:
        pred = r["prim"]["season_label"] if key == "main" else r["vs"]
        s = season_steps(pred, r["known"])
        steps.append(s)
        if pred in SEASONS:
            errs.append(12 * circ(SEASON_CENTRE[pred], date_phase(r["date"])))
    valid = [s for s in steps if s is not None]
    ex = sum(s == 0 for s in valid)
    adj = sum(s == 1 for s in valid)
    opp = sum(s == 2 for s in valid)
    nv = len(valid)
    # chance level of the circular error: random season (uniform over 4 centres)
    chance_err = np.mean([np.mean([12 * circ(SEASON_CENTRE[s], date_phase(r["date"]))
                                   for s in SEASONS]) for r in mod])
    return dict(
        n=n, n_classified=nv, exact=ex, adjacent=adj, opposite=opp,
        exact_rate=ex / nv if nv else np.nan,
        exact_chance=0.25, p_exact=float(binom.sf(ex - 1, nv, 0.25)) if nv else np.nan,
        within1_rate=(ex + adj) / nv if nv else np.nan, within1_chance=0.75,
        p_within1=float(binom.sf(ex + adj - 1, nv, 0.75)) if nv else np.nan,
        partial_credit=(ex + 0.5 * adj) / nv if nv else np.nan, partial_credit_chance=0.5,
        circ_error_months=float(np.mean(errs)) if errs else np.nan,
        circ_error_chance_months=float(chance_err),
    )



# ============================================================================
# 9. FIGURES
# ============================================================================
# Figures as in the paper: 190 mm wide, panels placed in millimetres, PDF/PNG/SVG
# at 600 dpi.  sample_figure() produces the per-specimen figures (Figs. 7-14,
# S1-S30), method_figure() Fig. 5, and the summary figures Figs. 6, 11 and 15.

MM = 1 / 25.4
COL1, COL2 = 90 * MM, 190 * MM
LEFT, RIGHT_CB = 18.0, 14.0          # mm: left margin (y labels), right margin (colour bar)
GAP_X, LAB_H, LAB_H2, TITLE_H = 14.0, 17.0, 10.0, 5.5  # horizontal gap; space for x label + letter; letter only
SEASON_COLORS = {"winter": "#2b6cb0", "spring": "#3c9a5f",
                 "summer": "#e08214", "autumn": "#8c510a",
                 "transitional": "#9e9e9e", "none": "#d0d0d0"}
MGCA_RANGE = (9, 15)
CMAP = "viridis"
FS_DELTA = 1.0                       # global font-size increment (pt)
FORMATS = ("pdf", "png", "svg")
DPI = 600


def set_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7 + FS_DELTA, "axes.labelsize": 7 + FS_DELTA,
        "axes.titlesize": 7.5 + FS_DELTA, "xtick.labelsize": 6.5 + FS_DELTA,
        "ytick.labelsize": 6.5 + FS_DELTA, "legend.fontsize": 6.5 + FS_DELTA,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "lines.linewidth": 0.9, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42, "svg.fonttype": "none",
    })


def save(fig, path_no_ext: Path):
    path_no_ext.parent.mkdir(parents=True, exist_ok=True)
    for ext in FORMATS:
        fig.savefig(f"{path_no_ext}.{ext}", dpi=DPI)
    plt.close(fig)


def panel_label(ax, letter, title=None):
    """Panel letter left-aligned under the axes (below the x-axis label, if any)."""
    from matplotlib.transforms import offset_copy
    has_xlabel = bool(ax.get_xlabel())
    dy = -(29 if has_xlabel else 17) * (1 + 0.1 * FS_DELTA)   # points below the axes
    tr = offset_copy(ax.transAxes, fig=ax.figure, x=0, y=dy, units="points")
    ax.text(0, 0, f"({letter})", transform=tr, ha="left", va="top",
            fontsize=8 + FS_DELTA, fontweight="bold")


class MMLayout:
    """Place axes on a figure using millimetres measured from the TOP-LEFT corner."""

    def __init__(self, width_mm, height_mm):
        self.W, self.H = width_mm, height_mm
        self.fig = plt.figure(figsize=(width_mm * MM, height_mm * MM))

    def axes(self, x, y, w, h, **kw):
        return self.fig.add_axes([x / self.W, 1 - (y + h) / self.H, w / self.W, h / self.H], **kw)

    def legend(self, handles, labels, x, y, ncol=3):
        self.fig.legend(handles, labels, loc="upper left", ncol=ncol, frameon=False,
                        bbox_to_anchor=(x / self.W, 1 - y / self.H), borderaxespad=0,
                        handlelength=1.6, columnspacing=1.2)


def _map_height(region, width_mm, hmin=12, hmax=45):
    rows = np.where(region.any(axis=1))[0]
    cols = np.where(region.any(axis=0))[0]
    ny, nx = rows.max() + 1, cols.max() + 1
    return float(np.clip(width_mm * ny / nx, hmin, hmax)), nx, ny


def _map_axes(ax, mgca, region, edge_curve, ext, show_dist=None, dist=None):
    z = np.where(region, mgca, np.nan)
    im = ax.imshow(z, origin="lower", cmap=CMAP, vmin=MGCA_RANGE[0], vmax=MGCA_RANGE[1],
                   interpolation="nearest", aspect="equal")
    ys = np.arange(len(edge_curve))
    ax.plot(edge_curve, ys, color="red", lw=1.2)
    if ext is not None and len(ext):
        ax.plot(ext[:, 0], ext[:, 1], color="red", lw=1.2)
    if show_dist is not None and dist is not None:
        cs = ax.contour(np.where(region, dist, np.nan), levels=show_dist, colors="white",
                        linewidths=0.5)
        ax.clabel(cs, fmt="%d", fontsize=6 + FS_DELTA)
    rows = np.where(region.any(axis=1))[0]
    cols = np.where(region.any(axis=0))[0]
    ax.set_xlim(-0.5, cols.max() + 0.5)
    ax.set_ylim(-0.5, rows.max() + 0.5)
    ax.set_anchor("C")
    ax.set_xlabel("x (pixel)")
    ax.set_ylabel("y (pixel)")
    return im


def _unrolled_axes(ax, um, umask, d_end=None):
    z = np.where(umask, um, np.nan)
    rows = np.where(np.isfinite(z).any(axis=1))[0]
    cols = np.where(np.isfinite(z).any(axis=0))[0]
    z = z[rows.min():rows.max() + 1, :cols.max() + 1]
    im = ax.imshow(z, origin="lower", cmap=CMAP, vmin=MGCA_RANGE[0], vmax=MGCA_RANGE[1],
                   interpolation="nearest", aspect="auto",
                   extent=(-0.5, z.shape[1] - 0.5, rows.min() - 0.5, rows.max() + 0.5))
    if d_end is not None and np.isfinite(d_end):
        ax.axvline(d_end, color="white", ls="--", lw=0.8)
    ax.set_xlabel("Distance to edge, d (pixel)")
    ax.set_ylabel("Tangential position, s (row)")
    return im


def _profile_axes(ax, st, prim, xmax=None, legend=True, zones=False, trend=False):
    y = st["median_d"].copy()
    y[st["count_d"] < 3] = np.nan
    d = np.arange(len(y))
    ax.fill_between(d, st["p25_d"], st["p75_d"], color="0.78", lw=0, label="IQR (all rows)")
    ax.plot(d, y, color="k", lw=0.9, label="Median Mg/Ca(d)")
    if prim:
        ax.axvspan(0, prim["d_end_orig"], color="#fde9c9", zorder=0, label="Calibration interval")
        lo, hi = prim["mgca_min_ref"], prim["mgca_max_ref"]
        ax.axhline(lo, color=SEASON_COLORS["winter"], ls="--", lw=0.8,
                   label=f"Low ref. (P{P['low_ref_percentile']})")
        ax.axhline(hi, color=SEASON_COLORS["summer"], ls="--", lw=0.8,
                   label=f"High ref. (P{P['high_ref_percentile']})")
        if zones:
            t_lo, t_hi = lo + 0.25 * (hi - lo), lo + 0.75 * (hi - lo)
            ax.axhline(t_lo, color=SEASON_COLORS["winter"], ls=":", lw=0.6)
            ax.axhline(t_hi, color=SEASON_COLORS["summer"], ls=":", lw=0.6)
            xt = (xmax if xmax else len(y)) * 0.99
            box = dict(boxstyle="square,pad=0.1", fc="white", ec="none", alpha=0.8)
            ax.text(xt, t_lo, "level 0.25 (winter below)", ha="right", va="top", fontsize=6 + FS_DELTA,
                    color=SEASON_COLORS["winter"], bbox=box, zorder=7)
            ax.text(xt, t_hi, "level 0.75 (summer above)", ha="right", va="bottom", fontsize=6 + FS_DELTA,
                    color=SEASON_COLORS["summer"], bbox=box, zorder=7)
        if trend:
            ref = _trend_reference(prim)
            if ref is not None:
                ax.annotate("", xy=(0, prim.get("edge_ref_for_trend", prim["edge_value"])),
                            xytext=ref, arrowprops=dict(arrowstyle="->", color="red", lw=0.9,
                                                        shrinkA=0, shrinkB=2))
                ax.plot(*ref, marker="s", ms=3, color="red", mfc="white", zorder=6,
                        label="Interior reference for trend")
        ax.plot([0], [prim["edge_value"]], marker="o", ms=4, color="red", zorder=6,
                label="Edge value")
    ax.set_xlim(-1, xmax if xmax else len(y))
    ax.set_ylim(np.nanpercentile(st["p25_d"], 1) - 0.4, np.nanpercentile(st["p75_d"], 99.5) + 0.4)
    ax.set_xlabel("Distance to edge, d (pixel)")
    ax.set_ylabel("Mg/Ca (mmol/mol)")
    if legend:
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.32), frameon=False, ncol=3,
                  borderaxespad=0, handlelength=1.6, columnspacing=1.0)


def _trend_reference(prim):
    """Interior point (d, Mg/Ca) against which the edge trend was evaluated."""
    m = prim.get("trend_method", "")
    cand = {
        "local_multiscale_slope": ("local_multiscale_ref_d", "local_multiscale_ref"),
        "contextual_edge_vs_calibration_window": ("contextual_stop_d", "contextual_ref_for_trend"),
        "contextual_over_weak_immediate": ("contextual_stop_d", "contextual_ref_for_trend"),
        "immediate_and_contextual_agree": ("contextual_stop_d", "contextual_ref_for_trend"),
    }
    if m in cand:
        kd, kv = cand[m]
        d, v = prim.get(kd, np.nan), prim.get(kv, np.nan)
        if np.isfinite(d) and np.isfinite(v):
            return (d, v)
    if m.startswith("broad"):
        if prim.get("trend_time", 0) < 0:
            return (prim.get("broad_max_d"), prim.get("broad_max"))
        return (prim.get("broad_min_d"), prim.get("broad_min"))
    v = prim.get("immediate_interior_ref_for_trend", np.nan)
    if np.isfinite(v):
        return (3.0, v)
    v = prim.get("interior_ref_for_trend", np.nan)
    return (5.0, v) if np.isfinite(v) else None


def _estimate_text(prim):
    if not prim:
        return "No estimate"
    tr = {"increasing_toward_edge": "increasing", "decreasing_toward_edge": "decreasing",
          "stable_near_edge": "stable"}.get(prim["trend_label"], prim["trend_label"])
    return (f"Estimate: {prim['season_label'].capitalize()}  |  "
            f"edge level {prim['edge_level_dynamic']:.2f}  |  trend to edge: {tr}")


def _boot_axes(ax, probs, ref=None, ref_label=None):
    vals = [probs.get(s, 0) for s in SEASONS]
    ax.barh(range(4), vals, color=[SEASON_COLORS[s] for s in SEASONS], height=0.7)
    ax.set_yticks(range(4))
    ax.set_yticklabels([s.capitalize() for s in SEASONS])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.5, 1])
    ax.set_xlabel("Bootstrap probability")
    for i, v in enumerate(vals):
        if v > 0.75:
            ax.text(v - 0.03, i, f"{v:.2f}", va="center", ha="right", fontsize=6 + FS_DELTA, color="white")
        else:
            ax.text(v + 0.03, i, f"{v:.2f}", va="center", fontsize=6 + FS_DELTA)
    if ref:
        for s in ref.split("/"):
            if s in SEASONS:
                ax.plot(-0.04, SEASONS.index(s), marker=">", color="k", ms=5, clip_on=False)


def _near_edge_xmax(st, prim):
    """Zoom window for the near-edge view: about half the calibration interval, 30-70 px."""
    dend = prim["d_end_orig"] if prim else 60
    return int(np.clip(0.5 * dend, 30, 70))


def _legend_from(ax):
    h, l = ax.get_legend_handles_labels()
    return h, l


def sample_figure(r, out: Path):
    g, st, prim = r["geom"], r["stats"], r["prim"]
    Wp = COL2 / MM * MM * 25.4 - LEFT - RIGHT_CB       # usable width (mm)
    Wl = 0.74 * Wp                                     # map / profile column
    Wr = Wp - Wl - GAP_X                               # bootstrap column
    ha, _, _ = _map_height(g["region"], Wl)
    hb, hc, hleg = 36.0, 48.0, 14.0
    y = 3.0
    ya = y; y += ha + LAB_H
    yb = y; y += hb + LAB_H2 + TITLE_H
    yc = y; y += hc + LAB_H
    H = y + hleg
    L = MMLayout(COL2 / MM, H)
    ax_a = L.axes(LEFT, ya, Wl, ha)
    ax_b = L.axes(LEFT, yb, Wl, hb)
    ax_c = L.axes(LEFT, yc, Wl, hc, sharex=ax_b)
    ax_d = L.axes(LEFT + Wl + GAP_X, yc, Wr, hc)
    cax = L.axes(LEFT + Wl + 3, ya, 3.0, (yb + hb) - ya)

    im = _map_axes(ax_a, g["mgca"], g["region"], g["edge_curve"], g["edge_ext"])
    _unrolled_axes(ax_b, g["um"], g["umask"], prim["d_end_orig"] if prim else None)
    ax_b.set_xlabel("")
    cb = L.fig.colorbar(im, cax=cax)
    cb.set_label("Mg/Ca (mmol/mol)")
    _profile_axes(ax_c, st, prim, legend=False, zones=True, trend=True)
    ax_c.set_xlim(-1, np.where(np.isfinite(st["median_d"]))[0].max() + 1)
    ax_c.set_title(_estimate_text(prim), loc="right", fontsize=6.5 + FS_DELTA, pad=3)
    ref = r["known"] if r["group"] == "modern" else r["d18o"]
    _boot_axes(ax_d, r["probs"], ref)
    ax_d.yaxis.tick_right()
    for ax, l in zip((ax_a, ax_b, ax_c, ax_d), "abcd"):
        panel_label(ax, l)
    L.legend(*_legend_from(ax_c), LEFT, yc + hc + LAB_H + 1)
    save(L.fig, out / "samples" / r["id"])


def method_figure(r, out: Path):
    g, st, prim = r["geom"], r["stats"], r["prim"]
    Wp = COL2 / MM * MM * 25.4 - LEFT - RIGHT_CB
    ha, _, _ = _map_height(g["region"], Wp)
    hcd, he, hleg = 42.0, 48.0, 14.0
    Wcd = (Wp - GAP_X) / 2
    We = 0.74 * Wp
    Wf = Wp - We - GAP_X
    y = 3.0
    ya = y; y += ha + LAB_H
    yb = y; y += ha + LAB_H
    yc = y; y += hcd + LAB_H + TITLE_H
    ye = y; y += he + LAB_H
    H = y + hleg
    L = MMLayout(COL2 / MM, H)
    ax_a = L.axes(LEFT, ya, Wp, ha)
    ax_b = L.axes(LEFT, yb, Wp, ha)
    ax_c = L.axes(LEFT, yc, Wcd, hcd)
    ax_d = L.axes(LEFT + Wcd + GAP_X, yc, Wcd, hcd)
    ax_e = L.axes(LEFT, ye, We, he)
    ax_f = L.axes(LEFT + We + GAP_X, ye, Wf, he)
    cax = L.axes(LEFT + Wp + 3, ya, 3.0, ha)

    im = _map_axes(ax_a, g["mgca"], g["region"], g["edge_curve"], g["edge_ext"])
    cb = L.fig.colorbar(im, cax=cax)
    cb.set_label("Mg/Ca\n(mmol/mol)", fontsize=6 + FS_DELTA)
    ax_b.imshow(np.where(g["region"], 1.0, np.nan), origin="lower", cmap="Greys", vmin=0, vmax=2.2,
                aspect="equal", interpolation="nearest")
    ax_b.plot(g["edge_curve"], np.arange(len(g["edge_curve"])), color="red", lw=1.2)
    if len(g["edge_ext"]):
        ax_b.plot(g["edge_ext"][:, 0], g["edge_ext"][:, 1], color="red", lw=1.2)
    cs = ax_b.contour(np.where(g["region"], g["dist"], np.nan), levels=np.arange(10, 400, 20),
                      colors="k", linewidths=0.4)
    ax_b.clabel(cs, fmt="%d", fontsize=6 + FS_DELTA)
    ax_b.set_xlim(ax_a.get_xlim()); ax_b.set_ylim(ax_a.get_ylim()); ax_b.set_anchor("C")
    ax_b.set_xlabel("x (pixel)"); ax_b.set_ylabel("y (pixel)")

    _unrolled_axes(ax_c, g["um"], g["umask"], prim["d_end_orig"] if prim else None)
    _profile_axes(ax_d, st, prim, legend=False)
    ax_d.set_xlim(-1, np.where(np.isfinite(st["median_d"]))[0].max() + 1)
    _profile_axes(ax_e, st, prim, xmax=_near_edge_xmax(st, prim), legend=False, zones=True,
                  trend=True)
    ax_e.set_title(_estimate_text(prim), loc="right", fontsize=6.5 + FS_DELTA, pad=3)
    _boot_axes(ax_f, r["probs"], r["known"])
    ax_f.yaxis.tick_right()
    for ax, l in zip((ax_a, ax_b, ax_c, ax_d, ax_e, ax_f), "abcdef"):
        panel_label(ax, l)
    L.legend(*_legend_from(ax_e), LEFT, ye + he + LAB_H + 1)
    save(L.fig, out / "summary" / f"Fig_method_{r['id']}")


def bootstrap_summary_figure(results, out: Path):
    groups = [("modern", "Modern specimens (Langre)"), ("archaeological", "Archaeological (El Toral III)")]
    fig, axes = plt.subplots(1, 2, figsize=(COL2, 70 * MM),
                             gridspec_kw={"width_ratios": [24, 7.5], "wspace": 0.08}, sharey=True)
    for ax, (grp, title) in zip(axes, groups):
        rs = [r for r in results if r["group"] == grp]
        for i, r in enumerate(rs):
            b = 0
            for s in SEASONS:
                v = r["probs"].get(s, 0)
                ax.bar(i, v, bottom=b, color=SEASON_COLORS[s], width=0.8, edgecolor="white", lw=0.3)
                b += v
            ref = r["known"] if grp == "modern" else r["d18o"]
            if ref:
                for k, s in enumerate(ref.split("/")):
                    if s in SEASONS:
                        ax.plot(i + (k - 0.5) * 0.3 * (len(ref.split("/")) - 1), 1.05, marker="v",
                                ms=4, color=SEASON_COLORS[s], clip_on=False)
            if grp == "modern" and r["prim"] and r["prim"]["season_label"] == r["known"]:
                ax.text(i, 1.11, "✓", ha="center", fontsize=6 + FS_DELTA)
        ax.set_xticks(range(len(rs)))
        ax.set_xticklabels([r["id"].replace(".PV", "") for r in rs], rotation=90)
        ax.set_ylim(0, 1.16)
        ax.set_title(title)
        ax.spines[["top", "right"]].set_visible(False)
        if grp == "modern":
            for x in (5.5, 11.5, 17.5):
                ax.axvline(x, color="k", lw=0.4)
    axes[0].set_ylabel("Bootstrap probability")
    handles = [plt.Rectangle((0, 0), 1, 1, color=SEASON_COLORS[s]) for s in SEASONS]
    fig.legend(handles, [s.capitalize() for s in SEASONS], ncol=4, loc="lower center",
               bbox_to_anchor=(0.5, -0.22), frameon=False)
    fig.text(0.5, -0.29, "▼ known season (modern) or δ¹⁸O estimate (archaeological); "
             "✓ primary estimate correct", ha="center", fontsize=6 + FS_DELTA)
    save(fig, out / "summary" / "Fig_bootstrap_all")


def confusion_figure(results, out: Path):
    mod = [r for r in results if r["group"] == "modern"]
    labs = SEASONS + ["transitional"]
    C = np.zeros((4, 5), int)
    for r in mod:
        p = r["prim"]["season_label"] if r["prim"] else "transitional"
        C[SEASONS.index(r["known"]), labs.index(p)] += 1
    C = C[:, :4] if C[:, 4].sum() == 0 else C
    fig, ax = plt.subplots(figsize=(COL1, 70 * MM))
    ax.imshow(C, cmap="Greys", vmin=0, vmax=max(6, C.max()))
    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            ax.text(j, i, C[i, j], ha="center", va="center",
                    color="white" if C[i, j] >= 4 else "black", fontsize=8 + FS_DELTA)
    ax.set_xticks(range(C.shape[1])); ax.set_xticklabels([l.capitalize() for l in labs[:C.shape[1]]])
    ax.set_yticks(range(4)); ax.set_yticklabels([s.capitalize() for s in SEASONS])
    ax.set_xlabel("Estimated season (primary criterion)"); ax.set_ylabel("Known season")
    save(fig, out / "summary" / "Fig_confusion_modern")


def sensitivity_figure(results, out: Path):
    rs = results
    variants = [f"Guard {g} px" for g in P["sensitivity_guards"]] + ["Bootstrap mode"]
    rows = ["Reference"] + variants
    cats = SEASONS + ["transitional", "none", "unknown"]
    cmap = ListedColormap([SEASON_COLORS[c] for c in SEASONS] + ["#9e9e9e", "#d0d0d0", "white"])
    norm = BoundaryNorm(np.arange(-0.5, len(cats) + 0.5), cmap.N)
    Z = np.full((len(rows), len(rs)), cats.index("unknown"))
    for j, r in enumerate(rs):
        ref = r["known"] if r["group"] == "modern" else (r["d18o"] or "").split("/")[0]
        Z[0, j] = cats.index(ref) if ref in cats else cats.index("unknown")
        for i, gg in enumerate(P["sensitivity_guards"]):
            Z[i + 1, j] = cats.index(r["sens"][gg])
        mode = max(SEASONS, key=lambda s: r["probs"].get(s, 0))
        Z[-1, j] = cats.index(mode)
    fig, ax = plt.subplots(figsize=(COL2, 45 * MM))
    ax.imshow(Z, cmap=cmap, norm=norm, aspect="auto")
    for j, r in enumerate(rs):
        for i in range(1, len(rows)):
            if Z[i, j] == Z[0, j]:
                ax.text(j, i, "✓", ha="center", va="center", fontsize=6 + FS_DELTA, color="white")
    ax.set_yticks(range(len(rows))); ax.set_yticklabels(rows)
    ax.set_xticks(range(len(rs)))
    ax.set_xticklabels([r["id"].replace(".PV", "") for r in rs], rotation=90)
    ax.set_xticks(np.arange(-0.5, len(rs)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(rows)), minor=True)
    ax.grid(which="minor", color="white", lw=0.8)
    ax.tick_params(which="minor", length=0)
    n_mod = sum(r["group"] == "modern" for r in rs)
    ax.axvline(n_mod - 0.5, color="k", lw=1.0)
    handles = [plt.Rectangle((0, 0), 1, 1, color=SEASON_COLORS[s]) for s in SEASONS]
    ax.legend(handles, [s.capitalize() for s in SEASONS], ncol=4, loc="upper center",
              bbox_to_anchor=(0.5, -0.55), frameon=False)
    save(fig, out / "summary" / "Fig_sensitivity")



# ============================================================================
# 10. OUTPUT TABLES
# ============================================================================

def write_tables(results, out: Path, q_thr):
    """Write results.csv (one row per specimen: estimate, edge level and trend,
    references, sensitivity, bootstrap probabilities and edge-quality index)
    and the median Mg/Ca(d) profile of each specimen (profiles/*.csv)."""
    out.mkdir(parents=True, exist_ok=True)
    fields = ["id", "group", "file", "edge_left", "edge_method", "edge_x_end", "known", "date", "vs", "d18o",
              "main", "edge_value", "low_ref", "high_ref", "edge_level", "trend_label",
              "trend_method", "calib_d_end"] + \
             [f"main_guard{g}" for g in P["sensitivity_guards"]] + \
             [f"p_{s}" for s in SEASONS] + ["p_max", "edge_quality_sd", "edge_quality_rows",
                                             "edge_quality_flag"]
    with open(out / "results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            p = r["prim"] or {}
            row = dict(id=r["id"], group=r["group"], file=r["file"], edge_left=r["edge_left"],
                       edge_method=r["geom"]["edge_method"], edge_x_end=r["geom"]["x_end"],
                       known=r["known"], date=r["date"], vs=r["vs"], d18o=r["d18o"],
                       main=p.get("season_label"), edge_value=round(p.get("edge_value", np.nan), 3),
                       low_ref=round(p.get("mgca_min_ref", np.nan), 3),
                       high_ref=round(p.get("mgca_max_ref", np.nan), 3),
                       edge_level=round(p.get("edge_level_dynamic", np.nan), 3),
                       trend_label=p.get("trend_label"), trend_method=p.get("trend_method"),
                       calib_d_end=p.get("d_end_orig"),
                       p_max=round(max(r["probs"].get(s, 0) for s in SEASONS), 3),
                       edge_quality_sd=round(r["q_sd"], 3) if np.isfinite(r["q_sd"]) else "",
                       edge_quality_rows=r["q_n"],
                       edge_quality_flag=bool(np.isfinite(r["q_sd"]) and r["q_sd"] > q_thr))
            for gg in P["sensitivity_guards"]:
                row[f"main_guard{gg}"] = r["sens"][gg]
            for s in SEASONS:
                row[f"p_{s}"] = round(r["probs"].get(s, 0), 3)
            w.writerow(row)
        # profiles
    pdir = out / "profiles"; pdir.mkdir(exist_ok=True)
    for r in results:
        st = r["stats"]
        with open(pdir / f"{r['id']}_profile.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["d_px", "d_mm", "n", "median", "mean", "p25", "p75", "std"])
            for i in range(len(st["median_d"])):
                if st["count_d"][i] > 0:
                    w.writerow([i, round(i * P["pixel_size_mm"], 3), int(st["count_d"][i]),
                                *[round(float(st[k][i]), 4) for k in
                                  ("median_d", "mean_d", "p25_d", "p75_d", "std_d")]])


# ============================================================================
# 11. COMMAND-LINE INTERFACE
# ============================================================================
def run_batch(args):
    """Process all the specimens of the paper and write figures, tables and metrics."""
    data, out = Path(args.data), Path(args.out)
    entries = [e for e in SAMPLES if not args.only or e[0] in args.only]
    set_style()

    print(f"Processing {len(entries)} specimens (bootstrap reps = {args.boot}) ...")
    # Specimens are independent, so they are processed in parallel.
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(process, e, data, args.boot) for e in entries]
        results = []
        for e, fu in zip(entries, futs):
            r = fu.result()
            results.append(r)
            p = r["prim"]
            print(f"  {r['id']:14s} estimate={p['season_label'] if p else 'none':12s} "
                  f"reference={r['known'] or r['d18o']}  "
                  f"p_max={max(r['probs'].get(s, 0) for s in SEASONS):.2f}")

    # Edge-quality threshold: 90th percentile of the index in the modern specimens.
    qm = [r["q_sd"] for r in results if r["group"] == "modern" and np.isfinite(r["q_sd"])]
    q_thr = float(np.percentile(qm, 90)) if qm else np.inf

    for r in results:
        sample_figure(r, out / "figures")
    mr = next((r for r in results if r["id"] == args.method_sample), None)
    if mr:
        method_figure(mr, out / "figures")
    bootstrap_summary_figure(results, out / "figures")
    if any(r["group"] == "modern" for r in results):
        confusion_figure(results, out / "figures")
    sensitivity_figure(results, out / "figures")
    write_tables(results, out, q_thr)

    metrics = {}
    if any(r["group"] == "modern" for r in results):
        metrics["primary"] = evaluate(results, "main")
        metrics["virtual_sampling"] = evaluate(results, "vs")
        for gg in P["sensitivity_guards"]:
            n_changed = sum(r["sens"][gg] != r["sens"][P["sensitivity_guards"][0]]
                            for r in results if r["group"] == "modern")
            metrics[f"labels_changed_guard{gg}_vs_guard{P['sensitivity_guards'][0]}"] = n_changed
        mod = [r for r in results if r["group"] == "modern"]
        metrics["bootstrap_mean_prob_true_season"] = float(np.mean([r["probs"][r["known"]] for r in mod]))
    metrics["edge_quality_threshold_P90_modern"] = q_thr
    metrics["parameters"] = {k: (list(v) if isinstance(v, tuple) else v) for k, v in P.items()}
    metrics["bootstrap_reps"] = args.boot
    with open(out / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, default=float)

    if "primary" in metrics:
        m = metrics["primary"]
        print("\nPrimary criterion (modern specimens):")
        print(f"  exact {m['exact']}/{m['n_classified']} ({100*m['exact_rate']:.1f} %, chance 25 %, p={m['p_exact']:.3f})")
        print(f"  correct or adjacent season {100*m['within1_rate']:.1f} % (chance 75 %, p={m['p_within1']:.3f})")
        print(f"  partial credit {m['partial_credit']:.3f} (chance 0.5)")
        print(f"  circular error {m['circ_error_months']:.2f} months (chance {m['circ_error_chance_months']:.2f})")
    print(f"\nOutputs written to {out}")


def run_single(args):
    """Analyse one .mat map (e.g. a new specimen) with the same parameters as the paper."""
    path = Path(args.file) if args.file else Path(select_mat_file(HERE))
    if args.edge_left is None:
        ans = input("Is the collection edge on the LEFT side of the map? [y/n]: ").strip().lower()
        edge_left = ans.startswith("y")
    else:
        edge_left = args.edge_left
    entry = (path.stem, path.name, "modern", edge_left, None, None, None, None)
    set_style()
    r = process(entry, path.parent, args.boot)
    p = r["prim"]
    out = Path(args.out)
    sample_figure(r, out / "figures")
    print(f"\nSpecimen: {path.name}")
    if p:
        print(f"  Season estimate:  {p['season_label']}")
        print(f"  Edge level:       {p.get('edge_level_dynamic', float('nan')):.2f} "
              f"(0 = low reference, 1 = high reference)")
        print(f"  Trend to edge:    {p.get('trend_label')}")
    print("  Bootstrap probabilities: " +
          ", ".join(f"{s} {r['probs'].get(s, 0):.2f}" for s in SEASONS))
    print("  Sensitivity (guard 0/1/2 px): " + " / ".join(r["sens"][g] for g in P["sensitivity_guards"]))
    print(f"  Edge-quality index: {r['q_sd']:.2f} mmol/mol (flagged in the paper if > ~1.0)")
    print(f"Figure written to {out / 'figures' / 'samples'}")


def main():
    ap = argparse.ArgumentParser(description="Edge-referenced analysis of 2D LIBS Mg/Ca maps.")
    ap.add_argument("file", nargs="?", help=".mat file to analyse (single-specimen mode)")
    ap.add_argument("--batch", action="store_true", help="process all the specimens of the paper")
    side = ap.add_mutually_exclusive_group()
    side.add_argument("--edge-left", dest="edge_left", action="store_true", default=None,
                      help="the collection edge is on the left of the map")
    side.add_argument("--edge-right", dest="edge_left", action="store_false",
                      help="the collection edge is on the right (the map is flipped)")
    ap.add_argument("--data", default=str(HERE / "data"), help="folder with the .mat files (batch mode)")
    ap.add_argument("--out", default=str(HERE / "outputs"), help="output folder")
    ap.add_argument("--boot", type=int, default=P["boot_reps"], help="bootstrap resamplings")
    ap.add_argument("--only", nargs="*", default=None, help="subset of specimen ids (batch mode)")
    ap.add_argument("--workers", type=int, default=None, help="parallel processes (batch mode)")
    ap.add_argument("--method-sample", default="LAN96", help="specimen used for the method figure")
    args = ap.parse_args()
    if args.batch:
        run_batch(args)
    else:
        run_single(args)


if __name__ == "__main__":
    main()
