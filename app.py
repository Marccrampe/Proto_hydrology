import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import folium
from folium.raster_layers import ImageOverlay
from streamlit_folium import st_folium
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
def rgba_from_intensity(mask, intensity, color=(40, 120, 255), max_alpha=120):
    h, w = mask.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[..., 0] = color[0]
    rgba[..., 1] = color[1]
    rgba[..., 2] = color[2]

    alpha = np.clip(intensity * max_alpha, 0, max_alpha).astype(np.uint8)
    alpha[~mask] = 0
    rgba[..., 3] = alpha
    return rgba


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
    vertical_streets = np.abs((X * 100) % 12 - 6) < 0.7
    horizontal_streets = np.abs((Y * 100) % 11 - 5.5) < 0.7
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

    # Low spots
    low_spots = (
        0.55 * outlet_zone +
        0.25 * np.exp(-((X - 0.60) ** 2 / 0.015 + (Y - 0.35) ** 2 / 0.020)) +
        0.18 * np.exp(-((X - 0.30) ** 2 / 0.018 + (Y - 0.58) ** 2 / 0.025))
    )

    # Downstream tendency
    downstream = np.clip(1.0 - Y, 0, 1)

    # Urban flood susceptibility
    flood_sus = (
        0.55 * roads.astype(float) +
        0.28 * impervious +
        0.30 * low_spots +
        0.20 * downstream
    )

    # Buildings flood less than streets/open spaces
    flood_sus[building_blocks] *= 0.35

    flood_sus = np.clip(flood_sus, 0, 1)
    flood_sus[~mask] = np.nan
    impervious[~mask] = np.nan

    return X, Y, mask, impervious, roads, building_blocks, flood_sus, outlet_zone


def allocate_nbs_spatial(selected_df, mask, impervious, roads, X, Y, outlet_zone):
    n = mask.shape[0]
    alloc = np.zeros((n, n), dtype=int)

    category_names = []
    current_id = 1

    for _, row in selected_df.iterrows():
        name = row["name"]
        family = row["family"]
        coverage = float(row["coverage_pct"]) / 100.0
        if coverage <= 0:
            continue

        category_names.append(name)
        cat_id = current_id
        current_id += 1

        target_count = max(1, int(np.sum(mask) * coverage * 0.45))

        if family == "GI":
            score = np.nan_to_num(impervious.copy(), nan=0.0)

            if "Green Roof" in name:
                score = 1.15 * score + 0.10 * np.nan_to_num((1 - np.abs(X - 0.52)), nan=0)

            elif "Grassed Swale" in name:
                score = 0.75 * score + 0.45 * roads.astype(float)

            elif "Bioretention" in name:
                score = 0.90 * score + 0.30 * roads.astype(float)

            elif "Infiltration Trench" in name:
                score = 1.10 * score + 0.35 * roads.astype(float)

            elif "Rain Barrel" in name or "Cistern" in name:
                score = 1.00 * score + 0.15 * np.nan_to_num((1 - np.abs(Y - 0.5)), nan=0)

        else:
            score = 0.65 * outlet_zone + 0.20 * roads.astype(float) + 0.15 * np.nan_to_num((1 - Y), nan=0)

        score[~mask] = -999
        score[alloc > 0] *= 0.7

        flat_idx = np.argsort(score.ravel())[::-1]
        chosen = 0

        for idx in flat_idx:
            r, c = np.unravel_index(idx, score.shape)
            if not mask[r, c]:
                continue

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
        coverage = float(row["coverage_pct"]) / 100.0
        family = row["family"]

        if family == "GI":
            local_red = 0.10 + 0.20 * coverage
        else:
            local_red = 0.14 + 0.22 * coverage

        cat_id = idx + 1
        reduction += (alloc == cat_id) * local_red

    reduction += 0.04 * (alloc > 0)
    reduction = np.clip(reduction, 0, 0.55)

    flood_after = flood_before * (1 - reduction)

    before_mask = flood_before >= base_threshold
    after_mask = flood_after >= base_threshold

    return before_mask, after_mask


# -----------------------------
# Title
# -----------------------------
st.title("Houston Nature-Based Solutions Flood Explorer")
st.caption(
    "Conceptual prototype: event-based hydrologic response, watershed attributes, "
    "spatial NBS allocation, and estimated urban flood extent."
)

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

# Dynamic map bounds centered on gauge
gauge_lat = float(gauge_row["lat"])
gauge_lon = float(gauge_row["lon"])
lat_pad = 0.018
lon_pad = 0.025

map_bounds = [
    [gauge_lat - lat_pad, gauge_lon - lon_pad],
    [gauge_lat + lat_pad, gauge_lon + lon_pad],
]
map_center = [gauge_lat, gauge_lon]

# -----------------------------
# Event and watershed cards
# -----------------------------
top1, top2 = st.columns([1.2, 1.0])

with top1:
    st.subheader("Selected Event")
    c1, c2, c3 = st.columns(3)
    c1.metric("Dates", f"{event_row.get('date_start', 'N/A')} → {event_row.get('date_end', 'N/A')}")
    c2.metric("Return period", str(event_row.get("return_period", "N/A")))
    c3.metric("Rainfall", f"{event_row.get('rainfall_mm', 'N/A')} mm")

    c4, c5, c6 = st.columns(3)
    c4.metric("Rain type", str(event_row.get("rain_type", "N/A")))
    c5.metric("Duration", f"{event_row.get('duration_hr', 'N/A')} h")
    intensity_val = get_event_intensity_mm_hr(event_row)
    c6.metric("Mean intensity", f"{round(intensity_val, 2)} mm/h")

with top2:
    st.subheader("Watershed Context")
    c1, c2, c3 = st.columns(3)
    c1.metric("Watershed", watershed_row.get("name", "N/A"))
    c2.metric("Area", f"{watershed_row.get('area_km2', 'N/A')} km²")
    c3.metric("Imperviousness", f"{watershed_row.get('impervious_pct', 'N/A')}%")

    initial_abstraction = watershed_row.get("initial_abstraction_mm", "N/A")
    curve_number = watershed_row.get("curve_number", "N/A")
    lulc_urban = watershed_row.get("lulc_urban_pct", "N/A")
    lulc_green = watershed_row.get("lulc_green_pct", "N/A")
    lulc_water = watershed_row.get("lulc_water_pct", "N/A")

    c4, c5, c6 = st.columns(3)
    c4.metric("Initial abstraction", f"{initial_abstraction} mm")
    c5.metric("Curve Number", f"{curve_number}")
    c6.metric("Urban LULC", f"{lulc_urban}%")

    st.caption(
        f"Gauge: {gauge_row.get('name', 'N/A')} | Green LULC: {lulc_green}% | "
        f"Water/wet areas: {lulc_water}%"
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
    key="nbs_editor",
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

peak_reduction_pct = scenario["peak_reduction_pct"]
runoff_reduction_pct = scenario["runoff_reduction_pct"]
lag_increase_hr = scenario["lag_increase_hr"]
details_df = scenario["details_df"]

# -----------------------------
# Synthetic spatial layers
# -----------------------------
X, Y, mask, impervious, roads, building_blocks, flood_sus, outlet_zone = create_watershed_grid(n=110)
alloc, category_names = allocate_nbs_spatial(selected_df.copy(), mask, impervious, roads, X, Y, outlet_zone)

rain_mm = float(event_row["rainfall_mm"])
if rain_mm >= 250:
    base_threshold = 0.42
elif rain_mm >= 140:
    base_threshold = 0.52
elif rain_mm >= 80:
    base_threshold = 0.60
else:
    base_threshold = 0.68

before_mask, after_mask = compute_flood_masks(flood_sus, alloc, selected_df.copy(), base_threshold)

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
    ax.set_ylabel("Discharge (m³/s)")
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
            "lag_add_hr": "Lag (h)",
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
            hide_index=True,
        )

with right:
    st.subheader("Watershed Map: NBS Spatial Allocation")
    category_map = np.full(mask.shape, np.nan)
    category_map[mask] = 0
    category_map[alloc > 0] = alloc[alloc > 0]

    n_cat = int(np.nanmax(np.nan_to_num(category_map, nan=0)))
    colors = [
        "#d9d9d9",
        "#2ca25f",
        "#99d8c9",
        "#66c2a4",
        "#41ae76",
        "#238b45",
        "#006d2c",
        "#3182bd",
        "#6baed6",
        "#9ecae1",
    ]
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
# Folium flood extent maps
# -----------------------------
st.subheader("Estimated Flood Extent on Basemap")

col_map1, col_map2 = st.columns(2)

before_rgba = rgba_from_intensity(before_mask, np.nan_to_num(flood_sus, nan=0.0))
after_rgba = rgba_from_intensity(after_mask, np.nan_to_num(flood_sus, nan=0.0))

with col_map1:
    st.markdown("**Before NBS**")
    m_before = folium.Map(location=map_center, zoom_start=14, tiles=None)

    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Imagery",
        name="Satellite",
        overlay=False,
        control=True,
    ).add_to(m_before)

    folium.CircleMarker(
        location=[gauge_lat, gauge_lon],
        radius=6,
        color="red",
        fill=True,
        fill_color="red",
        fill_opacity=0.9,
        tooltip=f"Outlet gauge: {gauge_row['name']}",
    ).add_to(m_before)

    ImageOverlay(
        image=before_rgba,
        bounds=map_bounds,
        opacity=0.55,
        interactive=True,
        cross_origin=False,
        zindex=10,
    ).add_to(m_before)

    folium.LayerControl().add_to(m_before)
    st_folium(m_before, width=650, height=500, key="before_map")

with col_map2:
    st.markdown("**After NBS**")
    m_after = folium.Map(location=map_center, zoom_start=14, tiles=None)

    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Imagery",
        name="Satellite",
        overlay=False,
        control=True,
    ).add_to(m_after)

    folium.CircleMarker(
        location=[gauge_lat, gauge_lon],
        radius=6,
        color="red",
        fill=True,
        fill_color="red",
        fill_opacity=0.9,
        tooltip=f"Outlet gauge: {gauge_row['name']}",
    ).add_to(m_after)

    ImageOverlay(
        image=after_rgba,
        bounds=map_bounds,
        opacity=0.55,
        interactive=True,
        cross_origin=False,
        zindex=10,
    ).add_to(m_after)

    folium.LayerControl().add_to(m_after)
    st_folium(m_after, width=650, height=500, key="after_map")

st.caption(
    "Flood maps are estimated scenario visualizations derived from a synthetic urban flood-susceptibility layer, "
    "anchored on the outlet gauge location and literature-based NBS performance effects. "
    "They are intended for comparative interpretation, not as calibrated street-scale hydraulic simulations."
)
