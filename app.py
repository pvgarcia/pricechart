# ------------------------------------------------------------
# ISO/RTO Price Distribution App – Updated
# - Calibri default font if installed
# - Auto-detect fonts but filter out TeX / CM / STIX / Adobe math fonts
# - Center single hub always ON (no sidebar toggle)
# - P25-P75 mode hides P50 label
# ------------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib import colors as mcolors
from matplotlib import font_manager as fm
from io import BytesIO

# ------------------------------------------------------------
# Global Default Font
# ------------------------------------------------------------

plt.rcParams["font.family"] = "Calibri"


# ------------------------------------------------------------
# Constants
# ------------------------------------------------------------

PRICE_COL  = "Price ($/MWh)"
DEV_COL    = "Developer"
ISO_COL    = "ISO/RTO"
HUB_COL    = "Settlement Hub"
TECH_COL   = "Technology"

TECH_COLORS_DEFAULT = {"Solar": "#FFC000", "Wind": "#00519B"}
MIXED_TECH_COLOR    = "#0EC477"
DOT_COLOR_DEFAULT   = "#5B6670"

LABEL_XOFFSET_DEFAULT = 0.03
FONT_SCALE_DEFAULT    = 0.80


# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------

def fmt_currency(x, pos):
    s = f"${abs(x):,.2f}"
    return f"({s})" if x < 0 else s


def clean_df(df):
    required = [PRICE_COL, DEV_COL, ISO_COL, HUB_COL, TECH_COL]
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(f"Missing required columns: {missing}")
        st.stop()

    df = df.copy()
    df[PRICE_COL] = pd.to_numeric(df[PRICE_COL], errors="coerce")
    df = df.dropna(subset=[PRICE_COL])

    df[DEV_COL]  = df[DEV_COL].astype(str).str.strip().str.title()
    df[ISO_COL]  = df[ISO_COL].astype(str).str.strip()
    df[HUB_COL]  = df[HUB_COL].astype(str).str.strip().str.upper()

    def _norm_tech(s):
        s = str(s).strip()
        low = s.lower()
        if ("solar" in low) and ("bess" in low or "storage" in low):
            return "Solar"
        if "solar" in low:
            return "Solar"
        if "wind" in low:
            return "Wind"
        return s.title()

    df[TECH_COL] = df[TECH_COL].map(_norm_tech)

    df[HUB_COL] = df[HUB_COL].replace(
        r".*\bALBERTA\b.*",
        "ALBERTA",
        regex=True
    )

    return df


@st.cache_data(show_spinner=False)
def precompute_hub_stats(df):
    qs = [0.10, 0.25, 0.50, 0.75, 0.90]

    per_tech = (
        df.groupby([ISO_COL, HUB_COL, TECH_COL])[PRICE_COL]
          .quantile(qs)
          .unstack(level=-1)
          .rename(columns={
              0.10: "p10", 0.25: "p25", 0.50: "p50",
              0.75: "p75", 0.90: "p90"
          })
          .reset_index()
    )

    all_tech = (
        df.groupby([ISO_COL, HUB_COL])[PRICE_COL]
          .quantile(qs)
          .unstack(level=-1)
          .rename(columns={
              0.10: "p10", 0.25: "p25", 0.50: "p50",
              0.75: "p75", 0.90: "p90"
          })
          .reset_index()
    )
    all_tech[TECH_COL] = "All Technologies"

    return pd.concat([per_tech, all_tech], ignore_index=True)


def staggered_labels(stt, eps):
    pts = [
        ("P90", stt["p90"]),
        ("P75", stt["p75"]),
        ("P50", stt["p50"]),
        ("P25", stt["p25"]),
        ("P10", stt["p10"]),
    ]
    placed, out = [], []
    for label, y in pts:
        y_adj = y
        for yy, _ in placed:
            if abs(y_adj - yy) < eps:
                y_adj = yy - eps * 0.6
        placed.append((y_adj, label))
        out.append((y_adj, label))
    return out


def compute_iso_ylim_for_scope(
    stats_df, iso, hubs, tech_or_none,
    dev_vals=None, extra_pad_abs=0.0, y_floor=None
):
    lo, hi = np.inf, -np.inf

    df_iso = stats_df[stats_df[ISO_COL] == iso]

    if tech_or_none is None:
        df_iso = df_iso[df_iso[TECH_COL] == "All Technologies"]
    else:
        df_iso = df_iso[df_iso[TECH_COL] == tech_or_none]

    df_iso = df_iso[df_iso[HUB_COL].isin(hubs)]

    if not df_iso.empty:
        lo = float(df_iso["p10"].min())
        hi = float(df_iso["p90"].max())

    if lo == np.inf:
        if dev_vals is not None and len(dev_vals):
            lo, hi = dev_vals.min(), dev_vals.max()
        else:
            lo, hi = 0, 1

    if dev_vals is not None and len(dev_vals):
        lo = min(lo, float(dev_vals.min()))
        hi = max(hi, float(dev_vals.max()))

    span = hi - lo if hi > lo else 1.0
    pad = 0.05 * span

    lo -= pad + extra_pad_abs
    hi += pad + extra_pad_abs

    if y_floor is not None:
        lo = float(y_floor)

    if not lo < hi:
        hi = lo + 1

    return lo, hi


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


# ------------------------------------------------------------
# Drawing function
# ------------------------------------------------------------

def draw_iso_panel(
    ax, *,
    df_all,
    stats_df,
    df_scope,
    iso,
    tech_or_none,
    dev,
    tech_label,
    box_width,
    hub_step,
    label_xoffset,
    edge_width,
    fill_alpha,
    cap_width_fr,
    show_grid_y,
    share_ylim,
    center_single,
    tech_colors,
    mixed_color,
    dot_color,
    legend_bottom,
    legend_frame,
    font_scale,
    box_mode,
    draw_dev_dots
):
    hubs = sorted(df_scope[HUB_COL].unique().tolist())
    if not hubs:
        ax.text(
            0.5, 0.5,
            f"No {tech_label} projects in {iso}",
            ha="center", va="center",
            transform=ax.transAxes
        )
        ax.axis("off")
        return

    # Center when only one hub
    if center_single and len(hubs) == 1:
        xpos = {hubs[0]: 0}
        ax.set_xlim(-hub_step, hub_step)
    else:
        xpos = {h: i * hub_step for i, h in enumerate(hubs)}

    # Select stats
    df_stats = stats_df[stats_df[ISO_COL] == iso]

    if tech_or_none is None:
        df_stats = df_stats[df_stats[TECH_COL] == "All Technologies"]
    else:
        df_stats = df_stats[df_stats[TECH_COL] == tech_or_none]

    df_stats = df_stats[df_stats[HUB_COL].isin(hubs)]

    hub_stats = {
        row[HUB_COL]: dict(
            p10=row["p10"], p25=row["p25"], p50=row["p50"],
            p75=row["p75"], p90=row["p90"]
        )
        for _, row in df_stats.iterrows()
    }

    box_color = mixed_color if tech_or_none is None else tech_colors.get(tech_or_none, "#333333")
    face = mcolors.to_rgba(box_color, fill_alpha)
    cap_half = box_width * cap_width_fr

    def label_pos(x):
        return x + box_width / 2 + max(0.005, label_xoffset), "left"

    fs_tick = 10 * font_scale
    fs_lbl  = 9 * font_scale

    for h in hubs:
        if h not in hub_stats:
            continue

        stt = hub_stats[h]
        x   = xpos[h]

        y25, y50, y75 = stt["p25"], stt["p50"], stt["p75"]
        y10, y90      = stt["p10"], stt["p90"]

        degenerate = (
            abs(y90 - y10) < 1e-9 and
            abs(y75 - y25) < 1e-9 and
            abs(y50 - y10) < 1e-9
        )

        if degenerate:
            ax.hlines(
                y50, x - box_width / 2, x + box_width / 2,
                color=box_color, linewidth=edge_width
            )
            xlbl, ha = label_pos(x)
            ax.text(xlbl, y50, "P50", fontsize=fs_lbl, va="center", ha=ha)
            continue

        # ------------ Box Modes ------------

        if box_mode == "P25-P75":
            y0, y1 = y25, y75
            labels = [("P25", y25), ("P75", y75)]  # Hide P50 in this mode
            whisk  = False

        elif box_mode == "P25-P50":
            y0, y1 = y25, y50
            labels = [("P25", y25), ("P50", y50)]
            whisk  = False

        elif box_mode == "P10-P50":
            y0, y1 = y10, y50
            labels = [("P10", y10), ("P50", y50)]
            whisk  = False

        else:  # Full (P10–P90)
            y0, y1 = y25, y75
            labels = None
            whisk  = True

        # Box
        ax.add_patch(
            plt.Rectangle(
                (x - box_width / 2, y0),
                box_width, y1 - y0,
                facecolor=face,
                edgecolor=box_color,
                linewidth=edge_width
            )
        )

        # Whiskers
        if whisk:
            ax.vlines(x, y75, y90, color=box_color, linewidth=edge_width)
            ax.hlines(y90, x - cap_half, x + cap_half,
                      color=box_color, linewidth=edge_width)

            ax.vlines(x, y25, y10, color=box_color, linewidth=edge_width)
            ax.hlines(y10, x - cap_half, x + cap_half,
                      color=box_color, linewidth=edge_width)

            span = max(y90 - y10, 1.0)
            eps  = max(0.01 * span, 0.25)
            for y_adj, lbl in staggered_labels(stt, eps):
                xlbl, ha = label_pos(x)
                ax.text(xlbl, y_adj, lbl, fontsize=fs_lbl, va="center", ha=ha)

        else:
            for lbl, yv in labels:
                xlbl, ha = label_pos(x)
                ax.text(xlbl, yv, lbl, fontsize=fs_lbl, va="center", ha=ha)

    # Draw developer dots
    if draw_dev_dots:
        label_once = False
        for h in hubs:
            vals = df_scope.loc[df_scope[HUB_COL] == h, PRICE_COL].to_numpy()
            if len(vals) == 0:
                continue

            x = xpos[h]
            ax.scatter(
                np.full(len(vals), x),
                vals,
                s=46,
                color=dot_color,
                edgecolor="black",
                linewidth=0.4,
                alpha=0.95,
                label=f"{dev} Projects" if not label_once else None
            )
            label_once = True

    ax.set_xticks([xpos[h] for h in hubs])
    ax.set_xticklabels(hubs, rotation=25, ha="right", fontsize=fs_tick)

    ax.yaxis.set_major_formatter(FuncFormatter(fmt_currency))
    for lbl in ax.yaxis.get_ticklabels():
        lbl.set_fontsize(fs_tick)

    ax.grid(axis="y", alpha=0.2 if show_grid_y else 0.0)
    ax.set_ylim(*share_ylim)
    ax.margins(x=0.15)

    if draw_dev_dots:
        ax.legend(
            frameon=legend_frame,
            loc=("lower center" if legend_bottom else "upper right"),
            bbox_to_anchor=(0.5, -0.22) if legend_bottom else None,
            fontsize=fs_lbl
        )


# ------------------------------------------------------------
# UI
# ------------------------------------------------------------

st.set_page_config(page_title="ISO/RTO Price Distribution", layout="wide")
st.title("ISO/RTO Price Distribution")

uploaded = st.file_uploader("Upload financials.xlsx", type=["xlsx"])

if uploaded is not None:

    df = clean_df(pd.read_excel(uploaded))

    # Filter hubs with <=3 rows
    hub_counts = df.groupby([ISO_COL, HUB_COL])[PRICE_COL].count()
    valid = hub_counts[hub_counts > 3].index

    df = (
        df.set_index([ISO_COL, HUB_COL])
          .loc[lambda x: x.index.isin(valid)]
          .reset_index()
    )

    if df.empty:
        st.error("All hubs were filtered out by the <=3 rows rule.")
        st.stop()

    df[ISO_COL] = df[ISO_COL].astype("category")
    df[HUB_COL] = df[HUB_COL].astype("category")
    df[TECH_COL] = df[TECH_COL].astype("category")
    df[DEV_COL] = df[DEV_COL].astype("category")

    stats_df = precompute_hub_stats(df)

    devs = sorted(df[DEV_COL].unique().tolist())
    dev  = st.selectbox("Select Developer", devs)

    df_dev = df[df[DEV_COL] == dev]

    techs = ["All Technologies"] + sorted(df_dev[TECH_COL].unique().tolist())
    tech_choice = st.selectbox("Select Technology", techs)

    all_techs  = (tech_choice == "All Technologies")
    tech_sel   = None if all_techs else tech_choice
    tech_label = tech_choice

    # ------------------------------------------------------------
    # Auto-detect fonts & filter out TeX/CM/STIX/Adobe math fonts
    # ------------------------------------------------------------

    raw_fonts = sorted({f.name for f in fm.fontManager.ttflist})

    exclude_prefixes = (
        "cm", "CM",          # Computer Modern fonts
        "TeX",               # TeX Gyre
        "STIX",              # STIX math fonts
        "ZWAdobeF",          # Adobe math
        "ZW",                # any ZW* Adobe fonts
    )

    filtered_fonts = [
        f for f in raw_fonts
        if not any(f.startswith(prefix) for prefix in exclude_prefixes)
    ]

    font_options = ["Auto"] + filtered_fonts

    default_font_name  = "Calibri" if "Calibri" in filtered_fonts else "Auto"
    default_font_index = font_options.index(default_font_name)

    # ------------------------------------------------------------
    # Sidebar
    # ------------------------------------------------------------

    with st.sidebar:

        box_width = st.slider("Box Width", 0.2, 1.2, 0.60, 0.05)
        hub_step  = st.slider("Hub Spacing", 0.5, 1.5, 0.90, 0.05)

        label_xoffset = st.slider(
            "Label X Offset",
            -0.05, 0.10,
            LABEL_XOFFSET_DEFAULT, 0.005
        )

        edge_width = st.slider("Edge Width", 1.0, 4.0, 2.2, 0.1)

        fill_alpha = st.slider("Box Fill Alpha", 0.0, 0.5, 0.12, 0.01)
        show_grid_y = st.checkbox("Show Y Gridlines", False)

        st.markdown("---")
        st.subheader("Fonts")

        font_disp = st.selectbox(
            "Font Family",
            font_options,
            index=default_font_index
        )

        if font_disp == "Auto":
            plt.rcParams["font.family"] = "sans-serif"
        else:
            plt.rcParams["font.family"] = font_disp

        font_scale = st.slider("Font Scale", 0.5, 2.0, FONT_SCALE_DEFAULT, 0.05)

        st.markdown("---")
        st.subheader("Colors")
        sol_color = st.color_picker("Solar color", TECH_COLORS_DEFAULT["Solar"])
        wind_color = st.color_picker("Wind color", TECH_COLORS_DEFAULT["Wind"])
        mixed_color = st.color_picker("All Technologies color", MIXED_TECH_COLOR)
        dot_color = st.color_picker("Developer dot color", DOT_COLOR_DEFAULT)
        tech_colors = {"Solar": sol_color, "Wind": wind_color}

        legend_bottom = st.checkbox("Legend at bottom center", True)
        legend_frame  = st.checkbox("Show legend border", False)

        st.markdown("---")
        st.subheader("Y axis control")
        include_dev_ylim = st.checkbox("Include developer dots in Y limits", True)
        extra_pad        = st.number_input("Extra padding ($/MWh)",
                                           value=1.00, min_value=0.00, step=0.25)
        force_ymin       = st.checkbox("Force Y axis minimum", False)
        y_min_value      = st.number_input("Y min", 0.0, step=1.0) if force_ymin else None

        st.markdown("---")
        st.subheader("Box Mode")
        box_mode = st.radio(
            "Choose how the box should be drawn",
            ["P25-P75", "P25-P50", "P10-P50", "Full"],
            index=0
        )

    # ------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------

    scope = df_dev if all_techs else df_dev[df_dev[TECH_COL] == tech_sel]
    isos  = sorted(scope[ISO_COL].unique().tolist())

    if not isos:
        st.info(f"No {tech_label} projects for this developer.")
        st.stop()

    center_single = True  # locked on

    for iso in isos:
        df_scope = scope[scope[ISO_COL] == iso]
        hubs = sorted(df_scope[HUB_COL].unique().tolist())
        if not hubs:
            continue

        dev_vals = df_scope[PRICE_COL].to_numpy() if include_dev_ylim else None

        ylim = compute_iso_ylim_for_scope(
            stats_df,
            iso,
            hubs,
            tech_sel,
            dev_vals=dev_vals,
            extra_pad_abs=extra_pad,
            y_floor=y_min_value
        )

        fig, ax = plt.subplots(figsize=(11, 6))

        draw_iso_panel(
            ax,
            df_all=df,
            stats_df=stats_df,
            df_scope=df_scope,
            iso=iso,
            tech_or_none=tech_sel,
            dev=dev,
            tech_label=tech_label,
            box_width=box_width,
            hub_step=hub_step,
            label_xoffset=label_xoffset,
            edge_width=edge_width,
            fill_alpha=fill_alpha,
            cap_width_fr=1/3,
            show_grid_y=show_grid_y,
            share_ylim=ylim,
            center_single=center_single,
            tech_colors=tech_colors,
            mixed_color=mixed_color,
            dot_color=dot_color,
            legend_bottom=legend_bottom,
            legend_frame=legend_frame,
            font_scale=font_scale,
            box_mode=box_mode,
            draw_dev_dots=True
        )

        ax.set_ylabel(PRICE_COL, fontsize=11 * font_scale)
        ax.set_title(f"{iso} {tech_label} Price Distribution",
                     fontsize=12 * font_scale)

        st.pyplot(fig, clear_figure=False)

        fname = f"{dev}_{tech_label}_{iso}.png".replace(" ", "_")

        st.download_button(
            "Download PNG",
            data=save_png_transparent(fig),
            file_name=fname,
            mime="image/png"
        )