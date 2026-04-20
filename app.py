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

def create_watershed_grid(n=110):
    x = np.linspace(0, 1, n)
    y = np.linspace(0, 1, n)
    X, Y = np.meshgrid(x, y)

    # Watershed mask
    mask = (((X - 0.50) / 0.42) ** 2 + ((Y - 0.52) / 0.34) ** 2) <= 1.0

    # Urbanized core
    urban_core = np.exp(-((X - 0.52) ** 2 / 0.025 + (Y - 0.53) ** 2 / 0.06))
    side_urban = 0.7 * np.exp(-((X - 0.35) ** 2 / 0.02 + (Y - 0.48) ** 2 / 0.10))
    impervious = np.clip(0.18 + 0.75 * urban_core + 0.35 * side_urban, 0, 1)

    # Main outlet zone: lower-right part of basin
    outlet_zone = np.exp(-((X - 0.78) ** 2 / 0.020 + (Y - 0.22) ** 2 / 0.020))

    # Street network proxy
    vertical_streets = (np.abs((X * 100) % 12 - 6) < 0.7)
    horizontal_streets = (np.abs((Y * 100) % 11 - 5.5) < 0.7)
    diagonal_main = np.abs(Y - (0.62 - 0.48 * X)) < 0.015
    curved_corridor = np.abs(Y - (0.38 + 0.15 * np.sin(6 * X))) < 0.012

    roads = (vertical_streets | horizontal_streets | diagonal_main | curved_corridor) & mask

    # Buildings proxy = urban but not roads
    building_blocks = (
        (np.sin(22 * X) > 0.80) &
        (np.sin(20 * Y) > 0.78) &
        (~roads) &
        mask
    )

    # Low spots proxy: more likely near outlet + some local depressions
    low_spots = (
        0.55 * outlet_zone +
        0.25 * np.exp(-((X - 0.60) ** 2 / 0.015 + (Y - 0.35) ** 2 / 0.020)) +
        0.18 * np.exp(-((X - 0.30) ** 2 / 0.018 + (Y - 0.58) ** 2 / 0.025))
    )

    # Downstream tendency
    downstream = np.clip(1.0 - Y, 0, 1)

    # STREET-BASED urban flood susceptibility
    flood_sus = (
        0.55 * roads.astype(float) +
        0.28 * impervious +
        0.30 * low_spots +
        0.20 * downstream
    )

    # Buildings should flood less than streets/open spaces
    flood_sus[building_blocks] *= 0.35

    flood_sus = np.clip(flood_sus, 0, 1)

    flood_sus[~mask] = np.nan
    impervious[~mask] = np.nan

    return X, Y, mask, impervious, roads, building_blocks, flood_sus, outlet_zone


def allocate_nbs_spatial(selected_df, mask, impervious, roads, X, Y, outlet_zone):
    
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
            # Prioritize urban and road-adjacent cells
            score = np.nan_to_num(impervious.copy(), nan=0.0)

            if "Green Roof" in name:
                # More central dense urban fabric
                score = 1.15 * score + 0.10 * np.nan_to_num((1 - np.abs(X - 0.52)), nan=0)

            elif "Grassed Swale" in name:
                # Prefer corridors and roads
                score = 0.75 * score + 0.45 * roads.astype(float)

            elif "Bioretention" in name:
                score = 0.90 * score + 0.30 * roads.astype(float)

            elif "Infiltration Trench" in name:
                score = 1.10 * score + 0.35 * roads.astype(float)

            elif "Rain Barrel" in name or "Cistern" in name:
                score = 1.00 * score + 0.15 * np.nan_to_num((1 - np.abs(Y - 0.5)), nan=0)

        else:
            # Storage near outlet and low urban runoff concentration zones
            score = 0.65 * outlet_zone + 0.20 * roads.astype(float) + 0.15 * np.nan_to_num((1 - Y), nan=0)


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
X, Y, mask, impervious, roads, building_blocks, flood_sus, outlet_zone = create_watershed_grid(n=110)
selected_spatial = selected_df.copy()
alloc, category_names = allocate_nbs_spatial(selected_spatial, mask, impervious, roads, X, Y, outlet_zone)

# Base threshold depends on event severity
rain_mm = float(event_row["rainfall_mm"])
if rain_mm >= 250:
    base_threshold = 0.42
elif rain_mm >= 140:
    base_threshold = 0.52
elif rain_mm >= 80:
    base_threshold = 0.60
else:
    base_threshold = 0.68

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

        col_map = {
            "solution": "Solution",
            "family": "Family",
            "coverage_pct": "Coverage (%)",
            "runoff_reduction_pct": "Runoff red. (%)",
            "peak_reduction_pct": "Peak red. (%)",
            "lag_add_hr": "Lag (h)"
        }

        for col in ["runoff_reduction_pct", "peak_reduction_pct"]:
            if col in show_df.columns:
                show_df[col] = show_df[col].round(1)

        if "lag_add_hr" in show_df.columns:
            show_df["lag_add_hr"] = show_df["lag_add_hr"].round(2)

        keep_cols = [c for c in col_map.keys() if c in show_df.columns]

        st.dataframe(
            show_df[keep_cols].rename(columns=col_map),
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

    roads_img = np.where(roads & mask, 1.0, np.nan)
    axm.imshow(roads_img, cmap=ListedColormap(["#4d4d4d"]), origin="lower", alpha=0.35)

    axm.set_xticks([])
    axm.set_yticks([])
    axm.set_title("Categorized implementation map")
    st.pyplot(fig_map, use_container_width=True)

    legend_lines = ["0 = untreated / baseline urban area"]
    for i, name in enumerate(category_names, start=1):
        legend_lines.append(f"{i} = {name}")
    legend_lines.append("Dark gray = street network")
    st.caption(" | ".join(legend_lines))

# -----------------------------
# Flood extent maps
# -----------------------------
st.subheader("Estimated Flood Extent")

f1, f2 = st.columns(2)

with f1:
    fig_b, axb = plt.subplots(figsize=(6.8, 5.8))
    base_rgb = np.ones((*before_mask.shape, 3))

    # Background outside watershed
    base_rgb[:] = [0.96, 0.96, 0.96]

    # Urban land
    base_rgb[mask] = [0.82, 0.82, 0.82]

    # Streets
    base_rgb[roads] = [0.63, 0.63, 0.63]

    # Buildings
    base_rgb[building_blocks] = [0.48, 0.48, 0.48]

    axb.imshow(base_rgb, origin="lower")

    flood_overlay = np.zeros((*before_mask.shape, 4))
    flood_overlay[..., 2] = 0.95
    flood_overlay[..., 1] = 0.50
    flood_overlay[..., 3] = np.where(before_mask, 0.65, 0.0)

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
    flood_overlay2[..., 1] = 0.50
    flood_overlay2[..., 3] = np.where(after_mask, 0.65, 0.0)

    axa.imshow(flood_overlay2, origin="lower")
    axa.set_title("After NBS")
    axa.set_xticks([])
    axa.set_yticks([])
    st.pyplot(fig_a, use_container_width=True)

st.caption(
    "Flood maps are estimated scenario visualizations derived from a synthetic flood-susceptibility layer and literature-based NBS performance effects. "
    "They are intended for comparative interpretation, not as calibrated street-scale hydraulic simulations."
)
