# app.py
# ISO/RTO Price Distribution for a single developer.
# - Upload Excel (financials.xlsx)
# - Pick Developer (single), pick Technology (or All Technologies)
# - Per ISO charts for that developer
# - Box Mode: P25 to P50 only, P10 to P50 only, or Full (with whiskers)
# - P labels always outside; per hub nudges supported
# - Degenerate case: P50 line plus label
# - Transparent PNG download
# - Solar+Storage/BESS normalized to Solar
# - Y axis within each ISO is always shared across its hubs
# - When All Technologies is selected, developer dots are hidden
# - Font scale and font family controls

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib import colors as mcolors
from io import BytesIO

# ---------- Expected columns ----------
PRICE_COL  = "Price ($/MWh)"
DEV_COL    = "Developer"
ISO_COL    = "ISO/RTO"
HUB_COL    = "Settlement Hub"
TECH_COL   = "Technology"

# ---------- Defaults ----------
TECH_COLORS_DEFAULT = {"Solar": "#FFC000", "Wind": "#00519B"}
MIXED_TECH_COLOR    = "#0EC477"
DOT_COLOR_DEFAULT   = "#5B6670"

LABEL_XOFFSET_DEFAULT       = 0.03
CENTER_SINGLE_HUB_DEFAULT   = True
LEGEND_BOTTOM_CENTER_DEFAULT = True
LEGEND_FRAME_DEFAULT         = False
FONT_SCALE_DEFAULT           = 0.50

# ---------- Helpers ----------
def fmt_currency(x, pos):
    s = f"${abs(x):,.2f}"
    return f"({s})" if x < 0 else s

def pct_stats(vals: np.ndarray):
    p10, p25, p50, p75, p90 = np.percentile(vals, [10, 25, 50, 75, 90])
    return dict(p10=p10, p25=p25, p50=p50, p75=p75, p90=p90)

def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    required = [PRICE_COL, DEV_COL, ISO_COL, HUB_COL, TECH_COL]
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(f"Missing required columns: {missing}")
        st.stop()

    df = df.copy()
    df[PRICE_COL] = pd.to_numeric(df[PRICE_COL], errors="coerce")
    df = df.dropna(subset=[PRICE_COL])

    df[DEV_COL] = df[DEV_COL].astype(str).str.strip().str.title()
    df[ISO_COL] = df[ISO_COL].astype(str).str.strip()
    df[HUB_COL] = df[HUB_COL].astype(str).str.strip().str.upper()

    # Normalize Technology (treat Solar+Storage/BESS as Solar)
    def _norm_tech(s: str) -> str:
        s = str(s).strip()
        low = s.lower()
        if ("solar" in low) and (("bess" in low) or ("storage" in low)):
            return "Solar"
        if "solar" in low:
            return "Solar"
        if "wind" in low:
            return "Wind"
        return s.title()

    df[TECH_COL] = df[TECH_COL].map(_norm_tech)

    # Treat AESO / Alberta hubs as one
    df[HUB_COL] = df[HUB_COL].replace(
        to_replace=r".*\bALBERTA\b.*",
        value="ALBERTA",
        regex=True
    )
    return df

def staggered_labels(stt: dict, eps: float):
    items = [("P90", stt["p90"]),
             ("P50", stt["p50"]),
             ("P25", stt["p25"]),
             ("P10", stt["p10"])]
    out, placed = [], []
    for name, y in items:
        y_adj = y
        for y_prev, _ in placed:
            if abs(y_adj - y_prev) < eps:
                y_adj = y_prev - eps * 0.6
        placed.append((y_adj, name))
        out.append((y_adj, name))
    return out

def compute_iso_ylim_for_scope(
    df_all,
    iso,
    hubs,
    tech_or_none,
    dev_vals=None,
    extra_pad_abs=0.0,
    y_floor=None
):
    lo, hi = np.inf, -np.inf
    df_t = df_all[df_all[ISO_COL] == iso]
    if tech_or_none is not None:
        df_t = df_t[df_t[TECH_COL] == tech_or_none]

    for h in hubs:
        vals = df_t.loc[df_t[HUB_COL] == h, PRICE_COL].to_numpy()
        if len(vals) == 0:
            continue
        stt = pct_stats(vals)
        lo = min(lo, stt["p10"])
        hi = max(hi, stt["p90"])

    if lo == np.inf:
        if dev_vals is not None and len(dev_vals) > 0:
            lo, hi = float(np.min(dev_vals)), float(np.max(dev_vals))
        else:
            lo, hi = 0.0, 1.0

    if dev_vals is not None and len(dev_vals) > 0:
        lo = min(lo, float(np.min(dev_vals)))
        hi = max(hi, float(np.max(dev_vals)))

    pad_prop = 0.05 * (hi - lo if hi > lo else 1.0)
    lo -= pad_prop + max(0.0, float(extra_pad_abs))
    hi += pad_prop + max(0.0, float(extra_pad_abs))

    if y_floor is not None:
        lo = float(y_floor)

    if not (lo < hi):
        hi = lo + 1.0

    return (lo, hi)

def save_png_transparent(fig) -> BytesIO:
    orig_facecolors = []
    for ax in fig.get_axes():
        orig_facecolors.append(ax.get_facecolor())
        ax.set_facecolor("none")
    fig.patch.set_alpha(0.0)

    buf = BytesIO()
    fig.savefig(
        buf,
        format="png",
        dpi=220,
        bbox_inches="tight",
        transparent=True
    )
    buf.seek(0)

    for ax, fc in zip(fig.get_axes(), orig_facecolors):
        ax.set_facecolor(fc)
    fig.patch.set_alpha(1.0)
    return buf

# ---------- Drawing: per ISO panels (single developer) ----------
def draw_iso_panel(
    ax, *,
    df_all,
    df_panel_scope,
    iso,
    tech_or_none,
    dev_name,
    tech_label,
    box_width,
    hub_step,
    label_xoffset,
    edge_width,
    fill_alpha,
    cap_width_fr,
    show_grid_y,
    share_ylim=None,
    label_offsets_for_iso=None,
    center_single_hub=True,
    tech_colors=None,
    mixed_color=MIXED_TECH_COLOR,
    dot_color="#5B6670",
    legend_bottom_center=True,
    legend_frame=False,
    font_scale=1.0,
    box_mode="P25-P50",
    draw_dev_dots=True
):
    if tech_colors is None:
        tech_colors = TECH_COLORS_DEFAULT

    dev_hubs = sorted(df_panel_scope[HUB_COL].unique().tolist())
    if not dev_hubs:
        ax.text(
            0.5,
            0.5,
            f"No {tech_label} projects in {iso}",
            ha="center",
            va="center",
            transform=ax.transAxes
        )
        ax.axis("off")
        return

    if center_single_hub and len(dev_hubs) == 1:
        xpos = {dev_hubs[0]: 0.0}
        ax.set_xlim(-hub_step, hub_step)
    else:
        xpos = {h: i * hub_step for i, h in enumerate(dev_hubs)}

    df_boxes = df_all[df_all[ISO_COL] == iso]
    if tech_or_none is not None:
        df_boxes = df_boxes[df_boxes[TECH_COL] == tech_or_none]

    box_color = (
        mixed_color
        if tech_or_none is None
        else tech_colors.get(tech_or_none, "#333333")
    )
    face_rgba = mcolors.to_rgba(box_color, fill_alpha)

    hub_stats = {}
    for h in dev_hubs:
        vals = df_boxes.loc[df_boxes[HUB_COL] == h, PRICE_COL].to_numpy()
        if len(vals) > 0:
            hub_stats[h] = pct_stats(vals)

    TOL = 1e-9
    cap_half = min(box_width * cap_width_fr, hub_step * cap_width_fr)

    def label_pos(x):
        return x + box_width / 2 + max(0.005, label_xoffset), "left"

    fs_tick = 10 * font_scale
    fs_lbl  = 9 * font_scale
    lw_edge = edge_width

    for h in dev_hubs:
        if h not in hub_stats:
            continue
        stt = hub_stats[h]
        x = xpos[h]

        degenerate = (
            abs(stt["p90"] - stt["p10"]) < TOL
            and abs(stt["p75"] - stt["p25"]) < TOL
            and abs(stt["p50"] - stt["p10"]) < TOL
        )

        offsets_for_hub = label_offsets_for_iso.get(h, {}) if label_offsets_for_iso else {}

        if degenerate:
            ax.hlines(
                stt["p50"],
                x - box_width / 2,
                x + box_width / 2,
                color=box_color,
                linewidth=lw_edge,
                zorder=2,
            )
            y_off = offsets_for_hub.get("P50", 0.0)
            x_lbl, ha = label_pos(x)
            ax.text(
                x_lbl,
                stt["p50"] + y_off,
                "P50",
                va="center",
                ha=ha,
                fontsize=fs_lbl,
                zorder=3,
            )
        else:
            if box_mode == "P25-P50":
                y0, y1 = stt["p25"], stt["p50"]
                labels_to_show = [("P25", stt["p25"]), ("P50", stt["p50"])]
                draw_whiskers = False
            elif box_mode == "P10-P50":
                y0, y1 = stt["p10"], stt["p50"]
                labels_to_show = [("P10", stt["p10"]), ("P50", stt["p50"])]
                draw_whiskers = False
            else:
                y0, y1 = stt["p25"], stt["p50"]
                labels_to_show = None
                draw_whiskers = True

            ax.add_patch(
                plt.Rectangle(
                    (x - box_width / 2, y0),
                    box_width,
                    y1 - y0,
                    facecolor=face_rgba,
                    edgecolor=box_color,
                    linewidth=lw_edge,
                    zorder=1,
                )
            )

            if draw_whiskers:
                ax.vlines(
                    x,
                    stt["p50"],
                    stt["p90"],
                    color=box_color,
                    linewidth=lw_edge,
                    zorder=1,
                )
                ax.hlines(
                    stt["p90"],
                    x - cap_half,
                    x + cap_half,
                    color=box_color,
                    linewidth=lw_edge,
                    zorder=1,
                )
                ax.vlines(
                    x,
                    stt["p25"],
                    stt["p10"],
                    color=box_color,
                    linewidth=lw_edge,
                    zorder=1,
                )
                ax.hlines(
                    stt["p10"],
                    x - cap_half,
                    x + cap_half,
                    color=box_color,
                    linewidth=lw_edge,
                    zorder=1,
                )

                span = max(stt["p90"] - stt["p10"], 1.0)
                eps = max(0.01 * span, 0.25)
                for y_auto, label in staggered_labels(stt, eps):
                    y_off = offsets_for_hub.get(label, 0.0)
                    x_lbl, ha = label_pos(x)
                    ax.text(
                        x_lbl,
                        y_auto + y_off,
                        label,
                        va="center",
                        ha=ha,
                        fontsize=fs_lbl,
                        zorder=3,
                    )
            else:
                for label, yval in labels_to_show:
                    y_off = offsets_for_hub.get(label, 0.0)
                    x_lbl, ha = label_pos(x)
                    ax.text(
                        x_lbl,
                        yval + y_off,
                        label,
                        va="center",
                        ha=ha,
                        fontsize=fs_lbl,
                        zorder=3,
                    )

    dot_drawn = False
    if draw_dev_dots:
        for h in dev_hubs:
            vals = df_panel_scope.loc[df_panel_scope[HUB_COL] == h, PRICE_COL].to_numpy()
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
                zorder=5,
                label=f"{dev_name} Projects" if not dot_drawn else None,
            )
            dot_drawn = True

    ax.set_xticks([xpos[h] for h in dev_hubs])
    ax.set_xticklabels(dev_hubs, rotation=25, ha="right", fontsize=fs_tick)
    ax.yaxis.set_major_formatter(FuncFormatter(fmt_currency))
    for lbl in ax.yaxis.get_ticklabels():
        lbl.set_fontsize(fs_tick)
    ax.grid(axis="y", alpha=(0.2 if show_grid_y else 0.0))

    if share_ylim is not None:
        ax.set_ylim(*share_ylim)

    ax.margins(x=0.15)

    if draw_dev_dots and dot_drawn:
        if legend_bottom_center:
            ax.legend(
                frameon=legend_frame,
                loc="lower center",
                bbox_to_anchor=(0.5, -0.22),
                fontsize=fs_lbl,
            )
        else:
            ax.legend(
                frameon=legend_frame,
                loc="upper right",
                fontsize=fs_lbl,
            )

# ---------- UI ----------
st.set_page_config(page_title="ISO/RTO Price Distribution", layout="wide")
st.title("ISO/RTO Price Distribution")

uploaded = st.file_uploader("Upload financials.xlsx", type=["xlsx"])

if uploaded is not None:
    raw = pd.read_excel(uploaded)
    df = clean_df(raw)

    # Developer picker (single dev only)
    devs = sorted(df[DEV_COL].dropna().unique().tolist())
    if not devs:
        st.error("No developers found in data.")
        st.stop()

    dev_choice = st.selectbox("Select Developer", options=devs, index=0)
    dev = dev_choice
    df_dev_scope = df[df[DEV_COL] == dev]

    # Technology dropdown (per developer)
    tech_options = sorted(df_dev_scope[TECH_COL].unique().tolist())
    tech_options = ["All Technologies"] + tech_options
    tech_choice = st.selectbox("Select Technology", options=tech_options, index=0)
    all_techs = (tech_choice == "All Technologies")
    tech_selected = None if all_techs else tech_choice
    tech_label = "All Technologies" if all_techs else tech_choice

    # Sidebar
    with st.sidebar:
        st.header("Chart Settings")
        box_width     = st.slider("Box Width", 0.2, 1.2, 0.60, 0.05)
        hub_step      = st.slider("Hub Spacing", 0.5, 1.5, 0.90, 0.05)
        label_xoffset = st.slider("Label X Offset", -0.05, 0.10, LABEL_XOFFSET_DEFAULT, 0.005)
        edge_width    = st.slider("Edge Width", 1.0, 4.0, 2.2, 0.1)
        fill_alpha    = st.slider("Box Fill Alpha", 0.0, 0.5, 0.12, 0.01)
        show_grid_y   = st.checkbox("Show Y Gridlines", value=False)

        st.markdown("---")
        center_single_hub = st.checkbox(
            "Center layout when only one hub",
            value=CENTER_SINGLE_HUB_DEFAULT,
        )

        st.markdown("---")
        st.subheader("Fonts")
        font_family = st.selectbox(
            "Font family",
            [
                "Auto",
                "DejaVu Sans",
                "Arial",
                "Helvetica",
                "Calibri",
                "Verdana",
                "Times New Roman",
                "Georgia",
                "Courier New",
                "sans-serif",
                "serif",
                "monospace",
            ],
            index=0,
        )
        font_scale = st.slider(
            "Font scale",
            0.50,
            1.50,
            FONT_SCALE_DEFAULT,
            0.05,
        )
        if font_family != "Auto":
            plt.rcParams["font.family"] = font_family

        st.markdown("---")
        st.subheader("Colors")
        solar_color = st.color_picker(
            "Solar color",
            value=TECH_COLORS_DEFAULT["Solar"],
        )
        wind_color  = st.color_picker(
            "Wind color",
            value=TECH_COLORS_DEFAULT["Wind"],
        )
        mixed_color = st.color_picker(
            "All Technologies color",
            value=MIXED_TECH_COLOR,
        )
        dot_color   = st.color_picker(
            "Developer dot color",
            value=DOT_COLOR_DEFAULT,
        )
        tech_colors = {"Solar": solar_color, "Wind": wind_color}
        legend_bottom_center = st.checkbox(
            "Legend at bottom center",
            value=LEGEND_BOTTOM_CENTER_DEFAULT,
        )
        legend_frame = st.checkbox(
            "Show legend border",
            value=LEGEND_FRAME_DEFAULT,
        )

        st.markdown("---")
        st.subheader("Y axis control")
        include_dev_in_ylim = st.checkbox(
            "Include developer dots when setting Y limits",
            value=True,
        )
        extra_pad_abs = st.number_input(
            "Extra padding ($/MWh)",
            value=1.00,
            min_value=0.00,
            step=0.25,
        )
        force_ymin = st.checkbox(
            "Force Y axis minimum",
            value=False,
        )
        y_min_value = (
            st.number_input(
                "Y min (if forced)",
                value=0.00,
                step=1.0,
            )
            if force_ymin
            else None
        )

        st.markdown("---")
        st.subheader("Box Mode")
        box_mode = st.radio(
            "Choose how the box should be drawn",
            options=["P25-P50", "P10-P50", "Full"],
            index=0,
            help="P25 to P50 or P10 to P50 draw only a solid box with no whiskers. Full shows whiskers P10 to P90.",
        )

    # Scope for ISO list
    scope_for_isos = (
        df_dev_scope
        if all_techs
        else df_dev_scope[df_dev_scope[TECH_COL] == tech_selected]
    )
    dev_isos = sorted(scope_for_isos[ISO_COL].unique().tolist())
    if not dev_isos:
        st.info(f"No {tech_label} projects found for this selection.")
        st.stop()

    # Label offsets UI
    st.subheader("Optional: Label Offsets (per hub, $/MWh)")
    st.caption(
        "Use small values like plus or minus 0.10 to nudge P10 P25 P50 P90 up or down. Leave blank to skip."
    )
    label_offsets_all_iso = {}
    for iso in dev_isos:
        hubs_for_iso = sorted(
            scope_for_isos.loc[scope_for_isos[ISO_COL] == iso, HUB_COL].unique().tolist()
        )
        if not hubs_for_iso:
            continue
        with st.expander(f"{iso} - Set label offsets"):
            iso_offsets = {}
            for h in hubs_for_iso:
                cols = st.columns(4)
                p10 = cols[0].text_input(
                    f"{h} P10",
                    value="",
                    key=f"{iso}_{tech_label}_{h}_P10",
                )
                p25 = cols[1].text_input(
                    f"{h} P25",
                    value="",
                    key=f"{iso}_{tech_label}_{h}_P25",
                )
                p50 = cols[2].text_input(
                    f"{h} P50",
                    value="",
                    key=f"{iso}_{tech_label}_{h}_P50",
                )
                p90 = cols[3].text_input(
                    f"{h} P90",
                    value="",
                    key=f"{iso}_{tech_label}_{h}_P90",
                )
                inner = {}
                for name, val in [
                    ("P10", p10),
                    ("P25", p25),
                    ("P50", p50),
                    ("P90", p90),
                ]:
                    if val.strip():
                        try:
                            inner[name] = float(val.strip())
                        except Exception:
                            st.warning(
                                f"{iso}/{h} {name}: not a number, ignored."
                            )
                if inner:
                    iso_offsets[h] = inner
            label_offsets_all_iso[iso] = iso_offsets

    # --------- Render: per ISO panels for this developer ---------
    for iso in dev_isos:
        df_panel_scope = scope_for_isos[scope_for_isos[ISO_COL] == iso]
        hubs_this_panel = sorted(df_panel_scope[HUB_COL].unique().tolist())
        if not hubs_this_panel:
            continue

        draw_dev_dots = not all_techs
        dev_vals = (
            df_panel_scope[PRICE_COL].to_numpy()
            if (draw_dev_dots and include_dev_in_ylim)
            else None
        )

        ylim = compute_iso_ylim_for_scope(
            df,
            iso,
            hubs_this_panel,
            tech_selected,
            dev_vals=dev_vals,
            extra_pad_abs=extra_pad_abs,
            y_floor=y_min_value if force_ymin else None,
        )

        fig, ax = plt.subplots(1, 1, figsize=(11, 6))
        draw_iso_panel(
            ax,
            df_all=df,
            df_panel_scope=df_panel_scope,
            iso=iso,
            tech_or_none=tech_selected,
            dev_name=dev,
            tech_label=tech_label,
            box_width=box_width,
            hub_step=hub_step,
            label_xoffset=label_xoffset,
            edge_width=edge_width,
            fill_alpha=fill_alpha,
            cap_width_fr=1 / 3,
            show_grid_y=show_grid_y,
            share_ylim=ylim,
            label_offsets_for_iso=label_offsets_all_iso.get(iso, {}),
            center_single_hub=center_single_hub,
            tech_colors=tech_colors,
            mixed_color=mixed_color,
            dot_color=dot_color,
            legend_bottom_center=legend_bottom_center,
            legend_frame=legend_frame,
            font_scale=font_scale,
            box_mode=box_mode,
            draw_dev_dots=draw_dev_dots,
        )
        ax.set_xlabel(None)
        ax.set_ylabel(PRICE_COL, fontsize=11 * font_scale)
        ax.set_title(
            f"{iso} {tech_label} Price Distribution • {dev}",
            fontsize=12 * font_scale,
        )

        st.pyplot(fig, clear_figure=False)
        safe_dev = dev.replace(" ", "_")
        safe_tech = tech_label.replace(" ", "_")
        fname = (
            f"{safe_dev}_{safe_tech}_{iso}_settlement_hub_prices.png"
            .replace(" ", "_")
        )
        st.download_button(
            "Download PNG",
            data=save_png_transparent(fig),
            file_name=fname,
            mime="image/png",
        )
