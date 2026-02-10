import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib import colors as mcolors
from matplotlib import font_manager as fm
from io import BytesIO

plt.rcParams["font.family"] = "Calibri"

# Column names (must match your Excel headers)
PRICE_COL   = "Price ($/MWh)"
DEV_COL     = "Developer"
ISO_COL     = "ISO/RTO"
HUB_COL     = "Settlement Hub"
TECH_COL    = "Technology"
TERM_COL    = "Term (years)"
BID_COL     = "Bid IDs"          # internal only
PROJECT_COL = "Project Name"     # client-facing (not shown on chart in this version)

# Required columns
REQUIRED_COLS_DEV = [PRICE_COL, DEV_COL, ISO_COL, HUB_COL, TECH_COL, TERM_COL, BID_COL, PROJECT_COL]
REQUIRED_COLS_MKT = [PRICE_COL, ISO_COL, HUB_COL, TECH_COL, TERM_COL]

# Colors
ALL_TECH_COLOR = "#0EC477"
SOLAR_COLOR    = "#F68220"
WIND_COLOR     = "#00519B"
DOT_COLOR_DEFAULT = "#5B6670"

TECH_TO_BOX_COLOR = {
    "All Technologies": ALL_TECH_COLOR,
    "Solar": SOLAR_COLOR,
    "Wind": WIND_COLOR
}

LABEL_XOFFSET_DEFAULT = 0.06
FONT_SCALE_DEFAULT    = 0.50

# Term sparsity rule
SPARSE_TERMS_ALWAYS_ALL = {4, 7, 15, 20}
MIN_N_FOR_TERM_BOX      = 5


def fmt_currency(x, pos):
    s = f"${abs(x):,.2f}"
    return f"({s})" if x < 0 else s


def strip_text(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    return str(x).strip()


def normalize_colname(s: str) -> str:
    return strip_text(s).replace("\n", " ").replace("\r", " ").strip()


def normalize_dev_series(s: pd.Series) -> pd.Series:
    s = s.astype("string")
    s = s.str.strip()
    s = s.str.replace(r"\s+", " ", regex=True)
    return s


def term_equal(series: pd.Series, term_value: float, tol: float = 1e-6) -> np.ndarray:
    s = pd.to_numeric(series, errors="coerce")
    return np.isclose(s.to_numpy(dtype=float), float(term_value), atol=tol, rtol=0.0)


def save_png_transparent(fig):
    orig = [ax.get_facecolor() for ax in fig.get_axes()]
    for ax in fig.get_axes():
        ax.set_facecolor("none")
    fig.patch.set_alpha(0.0)

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=220, bbox_inches="tight", transparent=True)
    buf.seek(0)

    for ax, fc in zip(fig.get_axes(), orig):
        ax.set_facecolor(fc)
    fig.patch.set_alpha(1.0)
    return buf


def find_header_rows(raw_df: pd.DataFrame, required_cols: list[str]) -> list[int]:
    required_norm = {normalize_colname(c).lower() for c in required_cols}
    header_rows = []
    for i in range(len(raw_df)):
        row_vals = [normalize_colname(v).lower() for v in raw_df.iloc[i].tolist()]
        row_set = set([v for v in row_vals if v != ""])
        if required_norm.issubset(row_set):
            header_rows.append(i)
    return header_rows


def extract_tables_from_sheet(df_noheader: pd.DataFrame, sheet_name: str) -> pd.DataFrame:
    header_rows = sorted(
        set(find_header_rows(df_noheader, REQUIRED_COLS_MKT) + find_header_rows(df_noheader, REQUIRED_COLS_DEV))
    )
    if not header_rows:
        return pd.DataFrame()

    blocks = []
    header_rows_end = header_rows + [len(df_noheader)]

    for h_i, h_row in enumerate(header_rows):
        end_row = header_rows_end[h_i + 1]
        header_vals = [normalize_colname(v) for v in df_noheader.iloc[h_row].tolist()]

        colnames = []
        keep_idx = []
        for j, name in enumerate(header_vals):
            if name == "" or name.lower().startswith("unnamed"):
                continue
            colnames.append(name)
            keep_idx.append(j)

        if not colnames:
            continue

        data_block = df_noheader.iloc[h_row + 1:end_row, keep_idx].copy()
        data_block.columns = colnames
        data_block = data_block.dropna(how="all")
        if data_block.empty:
            continue

        blocks.append(data_block)

    if not blocks:
        return pd.DataFrame()

    out = pd.concat(blocks, ignore_index=True)
    out["_sheet"] = sheet_name
    return out


def load_all_tables_from_workbook(uploaded_file) -> pd.DataFrame:
    xls = pd.ExcelFile(uploaded_file)
    frames = []
    for sh in xls.sheet_names:
        df0 = pd.read_excel(xls, sheet_name=sh, header=None, dtype=object)
        extracted = extract_tables_from_sheet(df0, sh)
        if not extracted.empty:
            frames.append(extracted)

    if not frames:
        return pd.DataFrame()

    df_all = pd.concat(frames, ignore_index=True)
    df_all.columns = [normalize_colname(c) for c in df_all.columns]
    return df_all


def normalize_common_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if ISO_COL in df.columns:
        df[ISO_COL] = df[ISO_COL].astype("string").str.strip()
    if HUB_COL in df.columns:
        df[HUB_COL] = df[HUB_COL].astype("string").str.strip().str.upper()

    if PRICE_COL in df.columns:
        df[PRICE_COL] = pd.to_numeric(df[PRICE_COL], errors="coerce")
    if TERM_COL in df.columns:
        df[TERM_COL] = pd.to_numeric(df[TERM_COL], errors="coerce")

    def _norm_tech(x):
        x = str(x).strip()
        low = x.lower()
        if ("solar" in low) and ("bess" in low or "storage" in low):
            return "Solar"
        if "solar" in low:
            return "Solar"
        if "wind" in low:
            return "Wind"
        return x.title()

    if TECH_COL in df.columns:
        df[TECH_COL] = df[TECH_COL].map(_norm_tech)

    if HUB_COL in df.columns:
        df[HUB_COL] = df[HUB_COL].replace(r".*\bALBERTA\b.*", "ALBERTA", regex=True)

    return df


def clean_market_df(df_extracted: pd.DataFrame) -> pd.DataFrame:
    df = df_extracted.copy()
    missing = [c for c in REQUIRED_COLS_MKT if c not in df.columns]
    if missing:
        st.error(f"Missing required market columns after extraction: {missing}")
        st.stop()

    df = normalize_common_cols(df)
    df = df.dropna(subset=[PRICE_COL, ISO_COL, HUB_COL, TECH_COL, TERM_COL])
    df = df[df[ISO_COL].astype("string").str.len() > 0]
    df = df[df[HUB_COL].astype("string").str.len() > 0]
    return df


def clean_dev_df(df_extracted: pd.DataFrame) -> pd.DataFrame:
    df = df_extracted.copy()
    missing = [c for c in REQUIRED_COLS_DEV if c not in df.columns]
    if missing:
        st.error(f"Missing required developer columns after extraction: {missing}")
        st.stop()

    df = normalize_common_cols(df)

    df[DEV_COL] = normalize_dev_series(df[DEV_COL])
    df[BID_COL] = df[BID_COL].astype("string").str.strip()
    df[PROJECT_COL] = df[PROJECT_COL].astype("string").str.strip()

    df = df.dropna(subset=[DEV_COL, BID_COL, PROJECT_COL])
    df = df[df[DEV_COL].str.len() > 0]
    df = df[df[BID_COL].str.len() > 0]
    df = df[df[PROJECT_COL].str.len() > 0]

    df = df.dropna(subset=[PRICE_COL, ISO_COL, HUB_COL, TECH_COL, TERM_COL])
    return df


def compute_market_stats(series: pd.Series) -> dict:
    qs = series.quantile([0.10, 0.25, 0.50, 0.75, 0.90]).to_dict()
    return {
        "p10": float(qs.get(0.10, np.nan)),
        "p25": float(qs.get(0.25, np.nan)),
        "p50": float(qs.get(0.50, np.nan)),
        "p75": float(qs.get(0.75, np.nan)),
        "p90": float(qs.get(0.90, np.nan)),
    }


def compute_ylim(mkt_stats: dict, dev_vals: np.ndarray, extra_pad_abs: float = 1.0):
    lo = np.nanmin([mkt_stats["p10"], np.min(dev_vals)])
    hi = np.nanmax([mkt_stats["p90"], np.max(dev_vals)])

    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        lo, hi = 0.0, 1.0

    span = hi - lo
    pad = 0.06 * span + extra_pad_abs
    return lo - pad, hi + pad


def draw_box_and_dev_overlay(
    ax,
    *,
    df_dev: pd.DataFrame,
    mkt_stats: dict,
    box_color: str,
    font_scale: float,
    show_grid_y: bool,
    box_mode: str,
    dev_dot_color: str,
    label_xoffset: float = LABEL_XOFFSET_DEFAULT,
    dot_size: int = 65,
    dot_edgewidth: float = 0.7,
    fill_alpha: float = 0.18,
    edge_width: float = 2.0
):
    face = mcolors.to_rgba(box_color, fill_alpha)
    fs_tick = 10 * font_scale
    fs_lbl  = 9 * font_scale

    x0 = 0.0
    box_width = 0.90
    cap_half = box_width * (1 / 3)

    p10, p25, p50, p75, p90 = mkt_stats["p10"], mkt_stats["p25"], mkt_stats["p50"], mkt_stats["p75"], mkt_stats["p90"]

    # Determine box bounds and labels
    if box_mode == "P25-P75":
        y0, y1 = p25, p75
        whisk = False
        labels = [("P75", p75), ("P50", p50), ("P25", p25)]
    elif box_mode == "P25-P50":
        y0, y1 = p25, p50
        whisk = False
        labels = [("P50", p50), ("P25", p25)]
    else:  # Full
        y0, y1 = p25, p75
        whisk = True
        labels = [("P90", p90), ("P75", p75), ("P50", p50), ("P25", p25), ("P10", p10)]

    # Market box
    ax.add_patch(
        plt.Rectangle(
            (x0 - box_width / 2, y0),
            box_width, y1 - y0,
            facecolor=face,
            edgecolor=box_color,
            linewidth=edge_width
        )
    )

    # Whiskers only in Full mode
    if whisk:
        ax.vlines(x0, p75, p90, color=box_color, linewidth=edge_width)
        ax.hlines(p90, x0 - cap_half, x0 + cap_half, color=box_color, linewidth=edge_width)

        ax.vlines(x0, p25, p10, color=box_color, linewidth=edge_width)
        ax.hlines(p10, x0 - cap_half, x0 + cap_half, color=box_color, linewidth=edge_width)

    # Percentile labels (right of box)
    bbox_kw = dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.75)
    x_lbl = x0 + box_width / 2 + max(0.01, label_xoffset)
    for lbl, yv in labels:
        ax.text(x_lbl, yv, lbl, fontsize=fs_lbl, va="center", ha="left", bbox=bbox_kw)

    # Developer dots (no sideways, ever)
    vals = df_dev[PRICE_COL].to_numpy(dtype=float)
    xj = np.full(len(vals), x0, dtype=float)

    ax.scatter(
        xj, vals,
        s=dot_size,
        color=dev_dot_color,
        edgecolor="black",
        linewidth=dot_edgewidth,
        alpha=0.95,
        zorder=10
    )

    # Axes formatting
    ax.yaxis.set_major_formatter(FuncFormatter(fmt_currency))
    for t in ax.yaxis.get_ticklabels():
        t.set_fontsize(fs_tick)

    ax.grid(axis="y", alpha=0.18 if show_grid_y else 0.0)

    # No x axis
    ax.set_xticks([])
    ax.set_xlim(-1.0, 1.0)  # symmetric, centered


st.set_page_config(page_title="Developer Bid Feedback", layout="wide")
st.title("SAP Developer Bid Feedback")

uploaded = st.file_uploader("Upload financials.xlsx", type=["xlsx"])

if uploaded is not None:
    df_extracted = load_all_tables_from_workbook(uploaded)
    if df_extracted.empty:
        st.error("Could not find any table blocks with the required headers anywhere in the workbook.")
        st.stop()

    df_market_all = clean_market_df(df_extracted)
    df_dev_all    = clean_dev_df(df_extracted)

    # Term selector
    terms = sorted(df_market_all[TERM_COL].dropna().unique().tolist())
    terms = [int(t) if float(t).is_integer() else float(t) for t in terms]
    if not terms:
        st.error("No valid numeric terms found after cleaning.")
        st.stop()

    term_choice = st.selectbox("Select Term (years)", terms, index=0)

    # Developer selector within term
    df_dev_term = df_dev_all[term_equal(df_dev_all[TERM_COL], term_choice)].copy()
    devs = sorted(df_dev_term[DEV_COL].dropna().unique().tolist())
    devs = [d for d in devs if str(d).strip() != ""]
    if not devs:
        st.warning("No developers found for this term after cleaning.")
        st.stop()

    dev = st.selectbox("Select Developer", devs)

    df_dev = df_dev_term[df_dev_term[DEV_COL] == dev].copy()
    if df_dev.empty:
        st.warning("No developer bids found for that term after cleaning.")
        st.stop()

    # Tech selector
    tech_options = ["All Technologies"] + sorted(df_dev[TECH_COL].dropna().unique().tolist())
    tech_choice = st.selectbox("Select Technology", tech_options, index=0)
    tech_sel = None if tech_choice == "All Technologies" else tech_choice

    if tech_sel is not None:
        df_dev = df_dev[df_dev[TECH_COL] == tech_sel].copy()

    if df_dev.empty:
        st.warning("No developer bids for that term and tech after filtering.")
        st.stop()

    # Fonts list
    raw_fonts = sorted({f.name for f in fm.fontManager.ttflist})
    exclude_prefixes = ("cm", "CM", "TeX", "STIX", "ZWAdobeF", "ZW")
    filtered_fonts = [f for f in raw_fonts if not any(f.startswith(prefix) for prefix in exclude_prefixes)]
    font_options = ["Auto"] + filtered_fonts
    default_font_name  = "Calibri" if "Calibri" in filtered_fonts else "Auto"
    default_font_index = font_options.index(default_font_name) if default_font_name in font_options else 0

    with st.sidebar:
        st.subheader("Fonts")
        font_disp = st.selectbox("Font Family", font_options, index=default_font_index)
        plt.rcParams["font.family"] = "sans-serif" if font_disp == "Auto" else font_disp
        font_scale = st.slider("Font Scale", 0.5, 2.0, FONT_SCALE_DEFAULT, 0.05)

        show_grid_y = st.checkbox("Show Y Gridlines", False)

        st.subheader("Box Mode")
        box_mode = st.radio("Choose how the box should be drawn", ["P25-P75", "P25-P50", "Full"], index=0)

        st.subheader("Developer dots")
        dev_dot_color = st.color_picker("Developer dot color", DOT_COLOR_DEFAULT)

    # Market box data: filter to same term, and same tech only if selected
    term_is_sparse = int(float(term_choice)) in SPARSE_TERMS_ALWAYS_ALL

    df_market_term = df_market_all[term_equal(df_market_all[TERM_COL], term_choice)].copy()
    term_ok = (not term_is_sparse) and (len(df_market_term) >= MIN_N_FOR_TERM_BOX)

    df_market_for_box = df_market_term if term_ok else df_market_all.copy()

    if tech_sel is not None:
        df_market_for_box = df_market_for_box[df_market_for_box[TECH_COL] == tech_sel].copy()

    if df_market_for_box.empty:
        st.warning("No market data available for the selected term and tech.")
        st.stop()

    mkt_stats = compute_market_stats(df_market_for_box[PRICE_COL])
    if not np.isfinite(mkt_stats["p10"]) or not np.isfinite(mkt_stats["p90"]):
        st.warning("Market stats could not be computed from the available data.")
        st.stop()

    # Stable sort (Bid IDs internal)
    df_dev = df_dev.sort_values(by=[PROJECT_COL, BID_COL]).copy()
    dev_vals = df_dev[PRICE_COL].to_numpy(dtype=float)

    box_color = TECH_TO_BOX_COLOR.get(tech_choice, ALL_TECH_COLOR)
    ylim = compute_ylim(mkt_stats, dev_vals, extra_pad_abs=1.0)

    # Slightly taller bottom margin for footer label
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.subplots_adjust(bottom=0.14)

    draw_box_and_dev_overlay(
        ax,
        df_dev=df_dev,
        mkt_stats=mkt_stats,
        box_color=box_color,
        font_scale=font_scale,
        show_grid_y=show_grid_y,
        box_mode=box_mode,
        dev_dot_color=dev_dot_color
    )

    ax.set_ylim(*ylim)
    ax.set_ylabel(PRICE_COL, fontsize=11 * font_scale)
    ax.set_xlabel("")

    # No chart title. Add bottom-center caption like a legend.
    caption = f"{dev} | {tech_choice} | {term_choice} yr"
    fig.text(0.5, 0.08, caption, ha="center", va="center", fontsize=11 * font_scale)

    st.pyplot(fig, clear_figure=False)

    fname = f"{dev}_{tech_choice}_{term_choice}yr_bid_feedback_overlay.png".replace(" ", "_")
    st.download_button(
        "Download PNG",
        data=save_png_transparent(fig),
        file_name=fname,
        mime="image/png"
    )
