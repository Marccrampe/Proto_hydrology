import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from hydro_model import (
    build_baseline_hydrograph_from_event,
    apply_nbs_to_hydrograph,
    get_event_intensity_mm_hr,
)

st.set_page_config(page_title="Houston NBS Flood Explorer", layout="wide")

# -----------------------------
# Data loading
# -----------------------------
events = pd.read_csv("events.csv")
watersheds = pd.read_csv("watershed.csv")
gauges = pd.read_csv("gauge.csv")
nbs_catalog = pd.read_csv("nbs_catalog.csv")

# -----------------------------
# Helpers
# -----------------------------
def build_synthetic_hydrograph(rainfall_mm, duration_hr, impervious_pct, intensity_mm_hr):
    t = np.linspace(0, duration_hr * 1.6, 500)
    peak_base = rainfall_mm * (0.75 + impervious_pct / 100.0) * (0.85 + intensity_mm_hr / 20.0)
    center = duration_hr * 0.58
    width = max(duration_hr / 4.5, 2.0)
    q = peak_base * np.exp(-((t - center) ** 2) / (2 * width ** 2))
    recession = np.exp(-0.028 * np.maximum(t - center, 0))
    q = q * recession
    return t, q

def aggregate_nbs_effects(selected_df, event_row):
    rainfall_mm = float(event_row["rainfall_mm"])
    total_runoff = 0.0
    total_peak = 0.0
    total_lag = 0.0

    details = []
    for _, row in selected_df.iterrows():
        coverage = float(row["coverage_pct"]) / 100.0
        runoff = float(row["max_effect_runoff"]) * coverage
        peak = float(row["max_effect_peak"]) * coverage
        lag = float(row["max_effect_lag_hr"]) * coverage

        # Extreme storms reduce GI effectiveness slightly
        if rainfall_mm >= 250 and row["family"] == "GI":
            runoff *= 0.72
            peak *= 0.75
            lag *= 0.85

        if rainfall_mm >= 250 and row["family"] == "Storage":
            runoff *= 0.90
            peak *= 0.95
            lag *= 0.95

        total_runoff += runoff
        total_peak += peak
        total_lag += lag

        details.append({
            "solution": row["name"],
            "family": row["family"],
            "coverage_pct": row["coverage_pct"],
            "runoff_red_pct": 100 * runoff,
            "peak_red_pct": 100 * peak,
            "lag_hr": lag
        })

    # Cap combined effects to avoid unrealistic totals
    total_runoff = min(total_runoff, 0.58)
    total_peak = min(total_peak, 0.72)
    total_lag = min(total_lag, 2.8)

    return total_runoff, total_peak, total_lag, pd.DataFrame(details)

def create_watershed_grid(n=90):
    x = np.linspace(0, 1, n)
    y = np.linspace(0, 1, n)
    X, Y = np.meshgrid(x, y)

    # Watershed mask: elongated urban basin
    mask = (((X - 0.50) / 0.42) ** 2 + ((Y - 0.52) / 0.34) ** 2) <= 1.0

    # Imperviousness proxy higher near central urban corridor
    urban_core = np.exp(-((X - 0.52) ** 2 / 0.03 + (Y - 0.55) ** 2 / 0.08))
    side_corridor = 0.7 * np.exp(-((X - 0.35) ** 2 / 0.02 + (Y - 0.48) ** 2 / 0.12))
    impervious = np.clip(0.15 + 0.7 * urban_core + 0.4 * side_corridor, 0, 1)

    # Bayou / drainage path
    channel = np.abs(Y - (0.68 - 0.55 * X + 0.03 * np.sin(8 * X))) < 0.03

    # Low-lying/flood-prone tendency near channel and lower-right outlet side
    flood_sus = (
        0.55 * np.exp(-((Y - (0.68 - 0.55 * X)) ** 2) / 0.0035)
        + 0.30 * np.exp(-((X - 0.78) ** 2 / 0.025 + (Y - 0.30) ** 2 / 0.03))
        + 0.25 * impervious
    )
    flood_sus = np.clip(flood_sus, 0, 1)
    flood_sus[~mask] = np.nan
    impervious[~mask] = np.nan

    return X, Y, mask, impervious, channel, flood_sus

def allocate_nbs_spatial(selected_df, mask, impervious, channel, X, Y):
    n = mask.shape[0]
    alloc = np.zeros((n, n), dtype=int)  # 0 no project, >0 category ids

    # Order categories for plotting
    category_map = {}
    category_names = []
    current_id = 1

    # Candidate areas
    valid_cells = np.argwhere(mask)

    for _, row in selected_df.iterrows():
        name = row["name"]
        family = row["family"]
        coverage = float(row["coverage_pct"]) / 100.0
        if coverage <= 0:
            continue

        category_map[name] = current_id
        category_names.append(name)
        cat_id = current_id
        current_id += 1

        target_count = max(1, int(np.sum(mask) * coverage * 0.45))

        if family == "GI":
            # Prioritize urban/impervious areas
            score = np.nan_to_num(impervious.copy(), nan=0.0)
            if "Green Roof" in name:
                score *= 1.1
            if "Infiltration Trench" in name:
                score = score * 1.2 + 0.2 * np.nan_to_num((1 - np.abs(Y - 0.5)), nan=0)
            if "Grassed Swale" in name or "Bioretention" in name:
                score = score * 1.0 + 0.15 * np.nan_to_num(channel, nan=0)
        else:
            # Storage near channels / low zones / outlet side
            channel_f = np.nan_to_num(channel.astype(float), nan=0.0)
            score = 0.75 * channel_f + 0.35 * np.nan_to_num((1 - np.abs(X - 0.75)), nan=0)

        score[~mask] = -999
        score[alloc > 0] *= 0.7  # avoid full overlap

        flat_idx = np.argsort(score.ravel())[::-1]
        chosen = 0
        for idx in flat_idx:
            r, c = np.unravel_index(idx, score.shape)
            if not mask[r, c]:
                continue
            # For storage, create more clustered patches
            if family == "Storage":
                rr0, rr1 = max(0, r - 1), min(n, r + 2)
                cc0, cc1 = max(0, c - 1), min(n, c + 2)
                patch = alloc[rr0:rr1, cc0:cc1]
                if np.all(patch == 0):
                    alloc[rr0:rr1, cc0:cc1] = cat_id
                    chosen += patch.size
                else:
                    continue
            else:
                if alloc[r, c] == 0:
                    alloc[r, c] = cat_id
                    chosen += 1

            if chosen >= target_count:
                break

    return alloc, category_names

def compute_flood_masks(flood_sus, alloc, selected_df, base_threshold):
    flood_before = np.nan_to_num(flood_sus.copy(), nan=0.0)

    reduction = np.zeros_like(flood_before)

    for idx, row in selected_df.iterrows():
        name = row["name"]
        coverage = float(row["coverage_pct"]) / 100.0
        family = row["family"]

        if family == "GI":
            local_red = 0.10 + 0.20 * coverage
        else:
            local_red = 0.14 + 0.22 * coverage

        cat_id = idx + 1  # aligned after selected_df reset_index(drop=True)
        reduction += (alloc == cat_id) * local_red

    # Mild diffuse benefit across watershed from connectivity / distributed retention
    reduction += 0.04 * (alloc > 0)

    reduction = np.clip(reduction, 0, 0.55)
    flood_after = flood_before * (1 - reduction)

    before_mask = flood_before >= base_threshold
    after_mask = flood_after >= base_threshold

    return before_mask, after_mask

# -----------------------------
# Title / layout
# -----------------------------
st.title("Houston Nature-Based Solutions Flood Explorer")
st.caption("V1.3 conceptual prototype: event-based scenario comparison, watershed attributes, spatial NBS allocation, and estimated flood extent.")

# -----------------------------
# Event / watershed selection
# -----------------------------
with st.sidebar:
    st.header("1) Event")
    selected_event_name = st.selectbox("Select event", events["name"].tolist())
    event_row = events.loc[events["name"] == selected_event_name].iloc[0]

    watershed_row = watersheds.loc[watersheds["watershed_id"] == event_row["watershed_id"]].iloc[0]
    gauge_row = gauges.loc[gauges["gauge_id"] == event_row["gauge_id"]].iloc[0]

    st.header("2) Solutions")
    st.caption("Choose one or several solutions and assign a coverage percentage.")

# -----------------------------
# Event and watershed cards
# -----------------------------
top1, top2 = st.columns([1.2, 1.0])

with top1:
    st.subheader("Selected Event")
    c1, c2, c3 = st.columns(3)
    c1.metric("Dates", f"{event_row['date_start']} → {event_row['date_end']}")
    c2.metric("Return period", str(event_row["return_period"]))
    c3.metric("Rainfall", f"{event_row['rainfall_mm']} mm")

    c4, c5, c6 = st.columns(3)
    c4.metric("Rain type", str(event_row["rain_type"]))
    c5.metric("Duration", f"{event_row['duration_hr']} h")
    intensity_val = get_event_intensity_mm_hr(event_row)
    c6.metric("Mean intensity", f"{round(intensity_val, 2)} mm/h")
with top2:
    st.subheader("Watershed Context")
    c1, c2, c3 = st.columns(3)
    c1.metric("Watershed", watershed_row["name"])
    c2.metric("Area", f"{watershed_row['area_km2']} km²")
    c3.metric("Imperviousness", f"{watershed_row['impervious_pct']}%")

    c4, c5, c6 = st.columns(3)
    c4.metric("Initial abstraction", f"{watershed_row['initial_abstraction_mm']} mm")
    c5.metric("Curve Number", f"{watershed_row['curve_number']}")
    c6.metric("Urban LULC", f"{watershed_row['lulc_urban_pct']}%")

    st.caption(
        f"Gauge: {gauge_row['name']} | Green LULC: {watershed_row['lulc_green_pct']}% | "
        f"Water/wet areas: {watershed_row['lulc_water_pct']}%"
    )

# -----------------------------
# NBS selector table
# -----------------------------
st.subheader("Nature-Based Solutions Selection")

editor_df = nbs_catalog.copy()
editor_df["use"] = False
editor_df["coverage_pct"] = 0

edited = st.data_editor(
    editor_df[["family", "name", "use", "coverage_pct", "notes"]],
    hide_index=True,
    use_container_width=True,
    column_config={
        "family": st.column_config.TextColumn("Family", disabled=True),
        "name": st.column_config.TextColumn("Solution", disabled=True),
        "use": st.column_config.CheckboxColumn("Use"),
        "coverage_pct": st.column_config.NumberColumn("Coverage (%)", min_value=0, max_value=50, step=5),
        "notes": st.column_config.TextColumn("Hydrologic role", disabled=True),
    },
    key="nbs_editor"
)

selected_df = nbs_catalog.copy()
selected_df["use"] = edited["use"]
selected_df["coverage_pct"] = edited["coverage_pct"]
selected_df = selected_df[(selected_df["use"] == True) & (selected_df["coverage_pct"] > 0)].reset_index(drop=True)

if selected_df.empty:
    st.warning("Select at least one solution and assign a coverage percentage to generate the scenario.")
    st.stop()


# -----------------------------
# Hydrograph computation
# -----------------------------

baseline = build_baseline_hydrograph_from_event(event_row, watershed_row)
scenario = apply_nbs_to_hydrograph(baseline, selected_df, event_row)

t = scenario["time_hr"]
t_mod = scenario["time_mod_hr"]
Q_base = scenario["q_base_m3s"]
Q_mod = scenario["q_mod_m3s"]

peak_base_val = scenario["peak_base_m3s"]
peak_mod_val = scenario["peak_mod_m3s"]
lag_base = scenario["t_peak_base_hr"]
lag_mod = scenario["t_peak_mod_hr"]
runoff_base = scenario["runoff_base_m3"]
runoff_mod = scenario["runoff_mod_m3"]

peak_reduction_pct = scenario["peak_reduction_pct"]
runoff_reduction_pct = scenario["runoff_reduction_pct"]
lag_increase_hr = scenario["lag_increase_hr"]
details_df = scenario["details_df"]

# -----------------------------
# Synthetic spatial layers
# -----------------------------
X, Y, mask, impervious, channel, flood_sus = create_watershed_grid(n=95)

selected_spatial = selected_df.copy()
alloc, category_names = allocate_nbs_spatial(selected_spatial, mask, impervious, channel, X, Y)

# Base threshold depends on event severity
rain_mm = float(event_row["rainfall_mm"])
if rain_mm >= 250:
    base_threshold = 0.28
elif rain_mm >= 140:
    base_threshold = 0.38
elif rain_mm >= 80:
    base_threshold = 0.48
else:
    base_threshold = 0.58

before_mask, after_mask = compute_flood_masks(flood_sus, alloc, selected_spatial, base_threshold)

flood_before_area = np.sum(before_mask)
flood_after_area = np.sum(after_mask)
flood_extent_reduction_pct = 100 * (1 - flood_after_area / max(flood_before_area, 1))

# -----------------------------
# Outputs layout
# -----------------------------
left, right = st.columns([1.1, 1.0])

with left:
    st.subheader("Hydrograph: Before vs After")
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(t, Q_base, "--", lw=2.2, label="Baseline")
    ax.plot(t_mod, Q_mod, lw=2.4, label="With selected NBS")
    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("Discharge (synthetic units)")
    ax.set_title("Outlet response")
    ax.grid(alpha=0.3)
    ax.legend()
    st.pyplot(fig, use_container_width=True)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Peak reduction", f"{peak_reduction_pct:.1f}%")
    m2.metric("Runoff reduction", f"{runoff_reduction_pct:.1f}%")
    m3.metric("Lag increase", f"{lag_increase_hr:.2f} h")
    m4.metric("Flood extent reduction", f"{flood_extent_reduction_pct:.1f}%")
    m5, m6 = st.columns(2)
    m5.metric("Runoff coeff. (base)", f"{scenario['effective_runoff_coeff_base']:.2f}")
    m6.metric("Runoff coeff. (with NBS)", f"{scenario['effective_runoff_coeff_mod']:.2f}")

    st.subheader("Performance by Selected Solution")
    if not details_df.empty:
        show_df = details_df.copy()
        show_df["runoff_red_pct"] = show_df["runoff_red_pct"].round(1)
        show_df["peak_red_pct"] = show_df["peak_red_pct"].round(1)
        show_df["lag_hr"] = show_df["lag_hr"].round(2)
        st.dataframe(
            show_df.rename(columns={
                "solution": "Solution",
                "family": "Family",
                "coverage_pct": "Coverage (%)",
                "runoff_red_pct": "Runoff red. (%)",
                "peak_red_pct": "Peak red. (%)",
                "lag_hr": "Lag (h)"
            }),
            use_container_width=True,
            hide_index=True
        )

with right:
    st.subheader("Watershed Map: NBS Spatial Allocation")
    # Base raster for map
    category_map = np.full(mask.shape, np.nan)
    category_map[mask] = 0
    category_map[alloc > 0] = alloc[alloc > 0]

    n_cat = int(np.nanmax(np.nan_to_num(category_map, nan=0)))
    colors = ["#d9d9d9", "#2ca25f", "#99d8c9", "#66c2a4", "#41ae76", "#238b45", "#006d2c", "#3182bd", "#6baed6", "#9ecae1"]
    cmap = ListedColormap(colors[: max(n_cat + 1, 2)])

    fig_map, axm = plt.subplots(figsize=(7.2, 6.1))
    axm.imshow(category_map, cmap=cmap, origin="lower")
    # overlay channel
    ch = np.where(channel & mask, 1.0, np.nan)
    axm.imshow(ch, cmap=ListedColormap(["#2166ac"]), origin="lower", alpha=0.8)
    axm.set_xticks([])
    axm.set_yticks([])
    axm.set_title("Categorized implementation map")
    st.pyplot(fig_map, use_container_width=True)

    legend_lines = ["0 = untreated / baseline urban area"]
    for i, name in enumerate(category_names, start=1):
        legend_lines.append(f"{i} = {name}")
    legend_lines.append("Blue line = bayou / main drainage corridor")
    st.caption(" | ".join(legend_lines))

# -----------------------------
# Flood extent maps
# -----------------------------
st.subheader("Estimated Flood Extent")

f1, f2 = st.columns(2)

with f1:
    fig_b, axb = plt.subplots(figsize=(6.8, 5.8))
    base_rgb = np.zeros((*before_mask.shape, 3))
    # urban gray
    base_rgb[..., 0] = np.where(mask, 0.82, 1.0)
    base_rgb[..., 1] = np.where(mask, 0.82, 1.0)
    base_rgb[..., 2] = np.where(mask, 0.82, 1.0)
    # roads / urban corridor feel
    roads = ((np.abs(Y - 0.55) < 0.012) | (np.abs(X - 0.52) < 0.010) | (np.abs(Y - (0.42 + 0.18*np.sin(7*X))) < 0.010)) & mask
    base_rgb[roads] = [0.68, 0.68, 0.68]
    # buildings-ish blocks
    blocks = (((np.sin(20*X) > 0.82) & (np.sin(18*Y) > 0.80)) & mask & (~roads))
    base_rgb[blocks] = [0.52, 0.52, 0.52]
    axb.imshow(base_rgb, origin="lower")
    flood_overlay = np.zeros((*before_mask.shape, 4))
    flood_overlay[..., 2] = 0.95
    flood_overlay[..., 1] = 0.45
    flood_overlay[..., 3] = np.where(before_mask, 0.62, 0.0)
    axb.imshow(flood_overlay, origin="lower")
    axb.set_title("Before NBS")
    axb.set_xticks([])
    axb.set_yticks([])
    st.pyplot(fig_b, use_container_width=True)

with f2:
    fig_a, axa = plt.subplots(figsize=(6.8, 5.8))
    axa.imshow(base_rgb, origin="lower")
    flood_overlay2 = np.zeros((*after_mask.shape, 4))
    flood_overlay2[..., 2] = 0.95
    flood_overlay2[..., 1] = 0.45
    flood_overlay2[..., 3] = np.where(after_mask, 0.62, 0.0)
    axa.imshow(flood_overlay2, origin="lower")
    axa.set_title("After NBS")
    axa.set_xticks([])
    axa.set_yticks([])
    st.pyplot(fig_a, use_container_width=True)

st.caption(
    "Flood maps are estimated scenario visualizations derived from a synthetic flood-susceptibility layer and literature-based NBS performance effects. "
    "They are intended for comparative interpretation, not as calibrated street-scale hydraulic simulations."
)