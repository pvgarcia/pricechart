import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib import colors as mcolors
from matplotlib import font_manager as fm
from io import BytesIO

plt.rcParams["font.family"] = "Calibri"

PRICE_COL  = "Price ($/MWh)"
DEV_COL    = "Developer"
ISO_COL    = "ISO/RTO"
HUB_COL    = "Settlement Hub"
TECH_COL   = "Technology"
TERM_COL   = "Term (years)"

REQUIRED_COLS = [PRICE_COL, DEV_COL, ISO_COL, HUB_COL, TECH_COL, TERM_COL]

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

# Visibility defaults
LABEL_XOFFSET_DEFAULT = 0.06
FONT_SCALE_DEFAULT    = 0.50
DOT_SIZE_DEFAULT      = 30
DOT_EDGE_DEFAULT      = 0.7
DOT_JITTER_DEFAULT    = 0.0  # default should be zero

# Sparse term logic
SPARSE_TERMS_ALWAYS_ALL = {4, 7, 15, 20}
ISO_MIN_N_FOR_TERM_BOX  = 5
HUB_MIN_N_FOR_TERM_BOX  = 5


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
    header_rows = find_header_rows(df_noheader, REQUIRED_COLS)
    if not header_rows:
        return pd.DataFrame()

    blocks = []
    header_rows = sorted(header_rows)
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

    df[ISO_COL]  = df[ISO_COL].astype("string").str.strip()
    df[HUB_COL]  = df[HUB_COL].astype("string").str.strip().str.upper()

    df[PRICE_COL] = pd.to_numeric(df[PRICE_COL], errors="coerce")
    df[TERM_COL]  = pd.to_numeric(df[TERM_COL], errors="coerce")

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

    df[TECH_COL] = df[TECH_COL].map(_norm_tech)
    df[HUB_COL]  = df[HUB_COL].replace(r".*\bALBERTA\b.*", "ALBERTA", regex=True)
    return df


def clean_market_df(df_extracted: pd.DataFrame) -> pd.DataFrame:
    df = df_extracted.copy()
    needed = [PRICE_COL, ISO_COL, HUB_COL, TECH_COL, TERM_COL]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        st.error(f"Missing required columns after extraction: {missing}")
        st.stop()

    df = normalize_common_cols(df)
    df = df.dropna(subset=[PRICE_COL, ISO_COL, HUB_COL, TERM_COL])
    df = df[df[ISO_COL].astype("string").str.len() > 0]
    df = df[df[HUB_COL].astype("string").str.len() > 0]
    return df


def clean_dev_df(df_extracted: pd.DataFrame) -> pd.DataFrame:
    df = df_extracted.copy()
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        st.error(f"Missing required columns after extraction: {missing}")
        st.stop()

    df = normalize_common_cols(df)

    df[DEV_COL] = normalize_dev_series(df[DEV_COL])
    df = df.dropna(subset=[DEV_COL])
    df = df[df[DEV_COL].str.len() > 0]

    df = df.dropna(subset=[PRICE_COL, ISO_COL, HUB_COL, TERM_COL])
    return df


@st.cache_data(show_spinner=False)
def precompute_iso_stats(df_market: pd.DataFrame) -> pd.DataFrame:
    qs = [0.10, 0.25, 0.50, 0.75, 0.90]
    return (
        df_market.groupby([ISO_COL])[PRICE_COL]
                 .quantile(qs)
                 .unstack(level=-1)
                 .rename(columns={0.10: "p10", 0.25: "p25", 0.50: "p50", 0.75: "p75", 0.90: "p90"})
                 .reset_index()
    )


@st.cache_data(show_spinner=False)
def precompute_iso_hub_stats(df_market: pd.DataFrame) -> pd.DataFrame:
    qs = [0.10, 0.25, 0.50, 0.75, 0.90]
    return (
        df_market.groupby([ISO_COL, HUB_COL])[PRICE_COL]
                 .quantile(qs)
                 .unstack(level=-1)
                 .rename(columns={0.10: "p10", 0.25: "p25", 0.50: "p50", 0.75: "p75", 0.90: "p90"})
                 .reset_index()
    )


def staggered_labels(stt, eps):
    pts = [("P90", stt["p90"]), ("P75", stt["p75"]), ("P50", stt["p50"]), ("P25", stt["p25"]), ("P10", stt["p10"])]
    placed, out = [], []
    for label, y in pts:
        y_adj = y
        for yy, _ in placed:
            if abs(y_adj - yy) < eps:
                y_adj = yy - eps * 0.6
        placed.append((y_adj, label))
        out.append((y_adj, label))
    return out


def is_skinny_box(row, threshold):
    try:
        return abs(float(row["p75"]) - float(row["p25"])) < float(threshold)
    except Exception:
        return False


def compute_ylim_from_boxes_and_dots(box_rows: list, dot_vals=None, extra_pad_abs=0.0, y_floor=None):
    lo = np.inf
    hi = -np.inf
    for r in box_rows:
        lo = min(lo, float(r["p10"]))
        hi = max(hi, float(r["p90"]))

    if dot_vals is not None and len(dot_vals):
        lo = min(lo, float(np.min(dot_vals)))
        hi = max(hi, float(np.max(dot_vals)))

    if lo == np.inf:
        lo, hi = 0.0, 1.0

    span = hi - lo if hi > lo else 1.0
    pad = 0.06 * span
    lo -= pad + extra_pad_abs
    hi += pad + extra_pad_abs

    if y_floor is not None:
        lo = float(y_floor)

    if not lo < hi:
        hi = lo + 1.0

    return lo, hi


def draw_combined_iso_chart(
    ax,
    *,
    boxes_by_iso: dict,
    df_dev: pd.DataFrame,
    box_width,
    hub_step,
    iso_gap,
    label_xoffset,
    edge_width,
    fill_alpha,
    cap_width_fr,
    show_grid_y,
    share_ylim,
    box_color,
    dot_color,
    font_scale,
    box_mode,
    dot_outlier_mode,
    dot_jitter,
    dot_size,
    dot_edgewidth
):
    rng = np.random.default_rng(12345)
    face = mcolors.to_rgba(box_color, fill_alpha)
    cap_half = box_width * cap_width_fr

    fs_tick = 10 * font_scale
    fs_lbl  = 9 * font_scale

    x_ticks = []
    x_labels = []
    separators = []
    x_cursor = 0.0
    label_once = False

    isos_sorted = sorted(boxes_by_iso.keys())

    for i, iso in enumerate(isos_sorted):
        df_dev_iso = df_dev[df_dev[ISO_COL] == iso].copy()
        hubs = sorted(df_dev_iso[HUB_COL].dropna().unique().tolist())
        if not hubs:
            continue

        xpos = {h: x_cursor + j * hub_step for j, h in enumerate(hubs)}
        group_xs = list(xpos.values())
        x_left = min(group_xs)
        x_right = max(group_xs)
        x_center = 0.5 * (x_left + x_right)

        box_row = boxes_by_iso[iso]
        stt = {k: float(box_row[k]) for k in ["p10", "p25", "p50", "p75", "p90"]}

        y25, y50, y75 = stt["p25"], stt["p50"], stt["p75"]
        y10, y90      = stt["p10"], stt["p90"]

        if box_mode == "P25-P75":
            y0, y1 = y25, y75
            labels = [("P25", y25), ("P75", y75)]
            whisk  = False
        elif box_mode == "P25-P50":
            y0, y1 = y25, y50
            labels = [("P25", y25), ("P50", y50)]
            whisk  = False
        elif box_mode == "P10-P50":
            y0, y1 = y10, y50
            labels = [("P10", y10), ("P50", y50)]
            whisk  = False
        else:
            y0, y1 = y25, y75
            labels = None
            whisk  = True

        ax.add_patch(
            plt.Rectangle(
                (x_center - box_width / 2, y0),
                box_width, y1 - y0,
                facecolor=face,
                edgecolor=box_color,
                linewidth=edge_width
            )
        )

        def label_pos(x):
            return x + box_width / 2 + max(0.005, label_xoffset), "left"

        bbox_kw = dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.75)

        if whisk:
            ax.vlines(x_center, y75, y90, color=box_color, linewidth=edge_width)
            ax.hlines(y90, x_center - cap_half, x_center + cap_half, color=box_color, linewidth=edge_width)

            ax.vlines(x_center, y25, y10, color=box_color, linewidth=edge_width)
            ax.hlines(y10, x_center - cap_half, x_center + cap_half, color=box_color, linewidth=edge_width)

            span = max(y90 - y10, 1.0)
            eps  = max(0.01 * span, 0.25)
            for y_adj, lbl in staggered_labels(stt, eps):
                xlbl, ha = label_pos(x_center)
                ax.text(xlbl, y_adj, lbl, fontsize=fs_lbl, va="center", ha=ha, bbox=bbox_kw)
        else:
            for lbl, yv in labels:
                xlbl, ha = label_pos(x_center)
                ax.text(xlbl, yv, lbl, fontsize=fs_lbl, va="center", ha=ha, bbox=bbox_kw)

        clip_lo, clip_hi = None, None
        if dot_outlier_mode == "Clip to P10-P90":
            clip_lo, clip_hi = stt["p10"], stt["p90"]
        elif dot_outlier_mode == "Clip to P25-P75":
            clip_lo, clip_hi = stt["p25"], stt["p75"]

        for h in hubs:
            vals = df_dev_iso.loc[df_dev_iso[HUB_COL] == h, PRICE_COL].to_numpy()
            if len(vals) == 0:
                continue

            plot_vals = vals.copy()
            if clip_lo is not None and clip_hi is not None:
                plot_vals = np.clip(plot_vals, clip_lo, clip_hi)

            x0 = xpos[h]
            if dot_jitter > 0:
                xj = x0 + rng.uniform(-dot_jitter, dot_jitter, size=len(plot_vals))
            else:
                xj = np.full(len(plot_vals), x0)

            ax.scatter(
                xj, plot_vals,
                s=dot_size,
                color=dot_color,
                edgecolor="black",
                linewidth=dot_edgewidth,
                alpha=0.95,
                label=f"{df_dev_iso[DEV_COL].iloc[0]} Projects" if not label_once else None
            )
            label_once = True

        for h in hubs:
            x_ticks.append(xpos[h])
            x_labels.append(f"{iso} {h}")

        x_cursor = x_right + iso_gap
        if i < len(isos_sorted) - 1:
            separators.append(x_right + iso_gap * 0.5)

    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels, rotation=25, ha="right", fontsize=fs_tick)

    ax.yaxis.set_major_formatter(FuncFormatter(fmt_currency))
    for lbl in ax.yaxis.get_ticklabels():
        lbl.set_fontsize(fs_tick)

    ax.grid(axis="y", alpha=0.2 if show_grid_y else 0.0)
    ax.set_ylim(*share_ylim)

    # KEY FIX: dynamic x padding based on hub_step (so squeeze actually works)
    if x_ticks:
        xmin, xmax = min(x_ticks), max(x_ticks)
        base_pad = hub_step * 0.6
        ax.set_xlim(xmin - base_pad, xmax + base_pad)

    for xs in separators:
        ax.axvline(xs, alpha=0.25, linewidth=1.0)

    if label_once:
        ax.legend(
            frameon=False,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.24),
            fontsize=fs_lbl
        )


st.set_page_config(page_title="ISO/RTO Price Distribution", layout="wide")
st.title("ISO/RTO Price Distribution")

uploaded = st.file_uploader("Upload financials.xlsx", type=["xlsx"])

if uploaded is not None:
    df_extracted = load_all_tables_from_workbook(uploaded)
    if df_extracted.empty:
        st.error("Could not find any table blocks with the required headers anywhere in the workbook.")
        st.stop()

    df_market_all = clean_market_df(df_extracted)
    df_dev_all    = clean_dev_df(df_extracted)

    terms = sorted(df_market_all[TERM_COL].dropna().unique().tolist())
    terms = [int(t) if float(t).is_integer() else float(t) for t in terms]
    if not terms:
        st.error("No valid numeric terms found after cleaning.")
        st.stop()

    term_choice = st.selectbox("Select Term (years)", terms, index=0)

    df_dev_term_all = df_dev_all[term_equal(df_dev_all[TERM_COL], term_choice)].copy()
    devs = sorted(df_dev_term_all[DEV_COL].dropna().unique().tolist())
    devs = [d for d in devs if str(d).strip() != ""]
    if not devs:
        st.warning("No developers found for this term after cleaning.")
        st.stop()

    dev = st.selectbox("Select Developer", devs)

    df_dev = df_dev_term_all[df_dev_term_all[DEV_COL] == dev].copy()
    if df_dev.empty:
        st.warning("No developer projects for that term after cleaning.")
        st.stop()

    tech_options = ["All Technologies"] + sorted(df_dev[TECH_COL].dropna().unique().tolist())
    tech_choice = st.selectbox("Select Technology", tech_options, index=0)
    tech_sel = None if tech_choice == "All Technologies" else tech_choice

    if tech_sel is not None:
        df_dev = df_dev[df_dev[TECH_COL] == tech_sel].copy()

    if df_dev.empty:
        st.warning("No developer projects for that term and tech after filtering.")
        st.stop()

    box_color = TECH_TO_BOX_COLOR.get(tech_choice, ALL_TECH_COLOR)

    raw_fonts = sorted({f.name for f in fm.fontManager.ttflist})
    exclude_prefixes = ("cm", "CM", "TeX", "STIX", "ZWAdobeF", "ZW")
    filtered_fonts = [f for f in raw_fonts if not any(f.startswith(prefix) for prefix in exclude_prefixes)]
    font_options = ["Auto"] + filtered_fonts
    default_font_name  = "Calibri" if "Calibri" in filtered_fonts else "Auto"
    default_font_index = font_options.index(default_font_name) if default_font_name in font_options else 0

    with st.sidebar:
        squeeze = st.slider("Squeeze chart width", 0.70, 1.00, 0.80)

        box_width = st.slider("Box Width", 0.2, 1.2, 0.80, 0.05)
        hub_step  = st.slider("Hub Spacing", 0.5, 1.5, 0.95, 0.05) * squeeze
        iso_gap   = st.slider("ISO group gap", 0.8, 3.0, 1.25, 0.05) * squeeze

        label_xoffset = st.slider("Label X Offset", -0.05, 0.15, LABEL_XOFFSET_DEFAULT, 0.005)
        edge_width = st.slider("Edge Width", 1.0, 3.0, 2.8, 0.1)
        fill_alpha = st.slider("Box Fill Alpha", 0.0, 0.5, 0.18, 0.01)
        show_grid_y = st.checkbox("Show Y Gridlines", False)

        st.subheader("Fonts")
        font_disp = st.selectbox("Font Family", font_options, index=default_font_index)
        plt.rcParams["font.family"] = "sans-serif" if font_disp == "Auto" else font_disp
        font_scale = st.slider("Font Scale", 0.5, 2.0, FONT_SCALE_DEFAULT, 0.05)

        st.subheader("Dot styling")
        dot_color = st.color_picker("Developer dot color", DOT_COLOR_DEFAULT)
        dot_outlier_mode = st.radio("Outlier handling (display only)", ["None", "Clip to P10-P90", "Clip to P25-P75"], index=1)
        dot_jitter = st.slider("Dot jitter (horizontal)", 0.0, 0.20, DOT_JITTER_DEFAULT, 0.01)
        dot_size = st.slider("Dot size", 10, 20, DOT_SIZE_DEFAULT, 10)
        dot_edge = st.slider("Dot edge width", 0.2, 1.5, DOT_EDGE_DEFAULT, 0.1)

        st.subheader("Market box fallback")
        use_all_market_points_box = st.checkbox(
            "If box is skinny, use ALL market points (ignore term + tech) for the box",
            value=False
        )
        skinny_box_threshold = st.number_input(
            "Skinny box threshold ($/MWh) for P75-P25",
            value=0.50,
            min_value=0.00,
            step=0.25
        )

        st.subheader("Y axis control")
        include_dev_ylim = st.checkbox("Include developer dots in Y limits", True)
        extra_pad = st.number_input("Extra padding ($/MWh)", value=1.00, min_value=0.00, step=0.25)
        force_ymin = st.checkbox("Force Y axis minimum", False)
        y_min_value = st.number_input("Y min", 0.0, step=1.0) if force_ymin else None

        st.subheader("Box Mode")
        box_mode = st.radio("Choose how the box should be drawn", ["P25-P75", "P25-P50", "P10-P50", "Full"], index=0)

        st.subheader("Box scope logic")
        use_hub_box_when_single = st.checkbox(
            "If developer has only one hub in an ISO, build box from that hub",
            value=True
        )

    isos = sorted(df_dev[ISO_COL].dropna().unique().tolist())
    term_is_sparse = int(float(term_choice)) in SPARSE_TERMS_ALWAYS_ALL

    boxes_by_iso = {}
    box_rows_for_ylim = []

    for iso in isos:
        df_dev_iso = df_dev[df_dev[ISO_COL] == iso].copy()
        if df_dev_iso.empty:
            continue

        hubs_dev = sorted(df_dev_iso[HUB_COL].dropna().unique().tolist())

        df_market_iso = df_market_all[df_market_all[ISO_COL] == iso].copy()
        if tech_sel is not None:
            df_market_iso = df_market_iso[df_market_iso[TECH_COL] == tech_sel].copy()

        df_market_iso_term = df_market_iso[term_equal(df_market_iso[TERM_COL], term_choice)].copy()

        iso_term_ok = (not term_is_sparse) and (len(df_market_iso_term) >= ISO_MIN_N_FOR_TERM_BOX)
        df_market_for_iso_stats = df_market_iso_term if iso_term_ok else df_market_iso

        iso_stats_df = precompute_iso_stats(df_market_for_iso_stats)
        box_row = iso_stats_df[iso_stats_df[ISO_COL] == iso]
        if box_row.empty:
            continue

        chosen_row = box_row.iloc[0]

        if iso_term_ok and is_skinny_box(chosen_row, skinny_box_threshold):
            iso_stats_all_terms = precompute_iso_stats(df_market_iso)
            box_row_all_terms = iso_stats_all_terms[iso_stats_all_terms[ISO_COL] == iso]
            if not box_row_all_terms.empty:
                chosen_row = box_row_all_terms.iloc[0]

        if use_all_market_points_box and is_skinny_box(chosen_row, skinny_box_threshold):
            df_market_iso_all = df_market_all[df_market_all[ISO_COL] == iso].copy()
            iso_stats_all_market = precompute_iso_stats(df_market_iso_all)
            box_row_all_market = iso_stats_all_market[iso_stats_all_market[ISO_COL] == iso]
            if not box_row_all_market.empty:
                chosen_row = box_row_all_market.iloc[0]

        if use_hub_box_when_single and len(hubs_dev) == 1:
            hub = hubs_dev[0]

            df_hub_term = df_market_iso_term[df_market_iso_term[HUB_COL] == hub].copy()
            hub_term_ok = iso_term_ok and (len(df_hub_term) >= HUB_MIN_N_FOR_TERM_BOX)

            if hub_term_ok:
                df_market_for_hub_stats = df_hub_term
            else:
                df_market_for_hub_stats = df_market_iso[df_market_iso[HUB_COL] == hub].copy()

            iso_hub_stats_df = precompute_iso_hub_stats(df_market_for_hub_stats)
            hub_row = iso_hub_stats_df[
                (iso_hub_stats_df[ISO_COL] == iso) &
                (iso_hub_stats_df[HUB_COL] == hub)
            ]
            if not hub_row.empty:
                chosen_row = hub_row.iloc[0]

                if use_all_market_points_box and is_skinny_box(chosen_row, skinny_box_threshold):
                    df_hub_all = df_market_all[
                        (df_market_all[ISO_COL] == iso) &
                        (df_market_all[HUB_COL] == hub)
                    ].copy()
                    hub_stats_all_market = precompute_iso_hub_stats(df_hub_all)
                    hub_row_all_market = hub_stats_all_market[
                        (hub_stats_all_market[ISO_COL] == iso) &
                        (hub_stats_all_market[HUB_COL] == hub)
                    ]
                    if not hub_row_all_market.empty:
                        chosen_row = hub_row_all_market.iloc[0]

        boxes_by_iso[iso] = chosen_row
        box_rows_for_ylim.append(chosen_row)

    if not boxes_by_iso:
        st.warning("No market stats available for the selected developer, term, and tech.")
        st.stop()

    dot_vals_all = df_dev[PRICE_COL].to_numpy() if include_dev_ylim else None
    ylim = compute_ylim_from_boxes_and_dots(
        box_rows_for_ylim,
        dot_vals=dot_vals_all,
        extra_pad_abs=extra_pad,
        y_floor=y_min_value
    )

    # Optional: make figure width responsive so whitespace shrinks further
    n_hubs = df_dev[HUB_COL].nunique()
    n_isos = len(boxes_by_iso)
    fig_width = max(8.0, min(18.0, (2.2 + 0.65 * (n_hubs + n_isos)) * squeeze))

    fig, ax = plt.subplots(figsize=(fig_width, 6))
    fig.subplots_adjust(bottom=0.25)

    draw_combined_iso_chart(
        ax,
        boxes_by_iso=boxes_by_iso,
        df_dev=df_dev,
        box_width=box_width,
        hub_step=hub_step,
        iso_gap=iso_gap,
        label_xoffset=label_xoffset,
        edge_width=edge_width,
        fill_alpha=fill_alpha,
        cap_width_fr=1/3,
        show_grid_y=show_grid_y,
        share_ylim=ylim,
        box_color=box_color,
        dot_color=dot_color,
        font_scale=font_scale,
        box_mode=box_mode,
        dot_outlier_mode=dot_outlier_mode,
        dot_jitter=dot_jitter,
        dot_size=dot_size,
        dot_edgewidth=dot_edge
    )

    ax.set_ylabel(PRICE_COL, fontsize=11 * font_scale)
    ax.set_xlabel("")  # removes the x-axis label text above the legend
    ax.set_title(dev, fontsize=13 * font_scale)

    st.pyplot(fig, clear_figure=False)

    fname = f"{dev}_{tech_choice}_{term_choice}yr_combined.png".replace(" ", "_")
    st.download_button(
        "Download PNG",
        data=save_png_transparent(fig),
        file_name=fname,
        mime="image/png"
    )