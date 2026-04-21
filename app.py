import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import folium
from folium.raster_layers import ImageOverlay
from streamlit_folium import st_folium
from matplotlib.colors import ListedColormap

import osmnx as ox
import geopandas as gpd
from shapely.geometry import box
from rasterio.features import rasterize
from rasterio.transform import from_bounds

from hydro_model import (
    build_baseline_hydrograph_from_event,
    apply_nbs_to_hydrograph,
    get_event_intensity_mm_hr,
)

st.set_page_config(page_title="Houston NBS Flood Explorer", layout="wide")


# ============================================================
# DATA
# ============================================================
@st.cache_data
def load_tabular_data():
    events = pd.read_csv("events.csv")
    watersheds = pd.read_csv("watershed.csv")
    gauges = pd.read_csv("gauge.csv")
    nbs_catalog = pd.read_csv("nbs_catalog.csv")
    return events, watersheds, gauges, nbs_catalog


events, watersheds, gauges, nbs_catalog = load_tabular_data()


# ============================================================
# HELPERS
# ============================================================
def rgba_from_intensity(mask, intensity, color=(30, 110, 255), max_alpha=255):
    h, w = mask.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[..., 0] = color[0]
    rgba[..., 1] = color[1]
    rgba[..., 2] = color[2]
    alpha = np.clip(intensity * max_alpha, 0, max_alpha).astype(np.uint8)
    alpha[~mask] = 0
    rgba[..., 3] = alpha
    return rgba


def smooth2d(arr, n_iter=2):
    out = arr.astype(float).copy()
    for _ in range(n_iter):
        out = (
            out
            + np.roll(out, 1, axis=0)
            + np.roll(out, -1, axis=0)
            + np.roll(out, 1, axis=1)
            + np.roll(out, -1, axis=1)
            + np.roll(np.roll(out, 1, axis=0), 1, axis=1)
            + np.roll(np.roll(out, 1, axis=0), -1, axis=1)
            + np.roll(np.roll(out, -1, axis=0), 1, axis=1)
            + np.roll(np.roll(out, -1, axis=0), -1, axis=1)
        ) / 9.0
    return out


@st.cache_resource(show_spinner=False)
def load_osm_layers(north, south, east, west):
    bbox_poly = box(west, south, east, north)

    # Roads
    roads = ox.features_from_bbox(
        north, south, east, west,
        tags={"highway": True}
    )
    roads = roads[roads.geometry.type.isin(["LineString", "MultiLineString"])].copy()
    roads = roads.to_crs(4326)
    roads = roads[roads.intersects(bbox_poly)]

    # Buildings
    buildings = ox.features_from_bbox(
        north, south, east, west,
        tags={"building": True}
    )
    buildings = buildings[buildings.geometry.type.isin(["Polygon", "MultiPolygon"])].copy()
    buildings = buildings.to_crs(4326)
    buildings = buildings[buildings.intersects(bbox_poly)]

    # Waterways
    water = ox.features_from_bbox(
        north, south, east, west,
        tags={"waterway": True}
    )
    water = water[water.geometry.type.isin(["LineString", "MultiLineString", "Polygon", "MultiPolygon"])].copy()
    water = water.to_crs(4326)
    water = water[water.intersects(bbox_poly)]

    return roads, buildings, water


def rasterize_gdf(gdf, bounds, out_shape, all_touched=True, default_value=1):
    west, south, east, north = bounds
    transform = from_bounds(west, south, east, north, out_shape[1], out_shape[0])

    if gdf is None or len(gdf) == 0:
        return np.zeros(out_shape, dtype=np.uint8)

    shapes = [(geom, default_value) for geom in gdf.geometry if geom is not None and not geom.is_empty]
    if not shapes:
        return np.zeros(out_shape, dtype=np.uint8)

    arr = rasterize(
        shapes,
        out_shape=out_shape,
        transform=transform,
        fill=0,
        all_touched=all_touched,
        dtype="uint8",
    )
    return arr


def derive_one_side_watershed_from_river(river_arr):
    """
    Build a one-sided watershed mask from a rasterized river.
    Keep the SOUTH side of the river (rows below the river line).
    """
    h, w = river_arr.shape
    mask = np.zeros((h, w), dtype=bool)

    river_cols = []
    river_rows = []

    for c in range(w):
        rows = np.where(river_arr[:, c] > 0)[0]
        if len(rows) > 0:
            river_cols.append(c)
            river_rows.append(np.mean(rows))

    if len(river_cols) < 3:
        # fallback: lower part of domain
        mask[h // 3 :, :] = True
        return mask

    river_cols = np.array(river_cols)
    river_rows = np.array(river_rows)

    full_cols = np.arange(w)
    interp_rows = np.interp(full_cols, river_cols, river_rows)

    for c in range(w):
        r0 = int(np.clip(interp_rows[c], 0, h - 1))
        mask[r0 + 1 :, c] = True  # south side only

    return mask


def build_real_spatial_layers(gauge_lat, gauge_lon, lat_pad, lon_pad, n=280):
    north = gauge_lat + lat_pad
    south = gauge_lat - lat_pad
    east = gauge_lon + lon_pad
    west = gauge_lon - lon_pad

    bounds = (west, south, east, north)

    # grids in geographic coordinates
    lon = np.linspace(west, east, n)
    lat = np.linspace(north, south, n)  # row 0 = north
    Lon, Lat = np.meshgrid(lon, lat)

    roads_gdf, buildings_gdf, water_gdf = load_osm_layers(north, south, east, west)

    roads = rasterize_gdf(roads_gdf, bounds, (n, n), all_touched=True, default_value=1).astype(bool)
    buildings = rasterize_gdf(buildings_gdf, bounds, (n, n), all_touched=True, default_value=1).astype(bool)
    river = rasterize_gdf(water_gdf, bounds, (n, n), all_touched=True, default_value=1).astype(bool)

    # one-sided mask
    mask = derive_one_side_watershed_from_river(river)

    # if river extraction weak, keep local half near outlet
    if mask.sum() < 0.12 * n * n:
        mask = np.zeros((n, n), dtype=bool)
        mask[n // 3 :, :] = True

    # outlet zone around gauge
    dlon = (Lon - gauge_lon) / max(lon_pad, 1e-6)
    dlat = (Lat - gauge_lat) / max(lat_pad, 1e-6)
    outlet_zone = np.exp(-(dlon**2 / 0.08 + dlat**2 / 0.08))
    outlet_zone[~mask] = 0.0

    # synthetic slope toward gauge / outlet
    dist = np.sqrt(dlon**2 + dlat**2)
    dist = dist / np.nanmax(dist)
    slope = 1.0 - dist
    slope = np.clip(slope, 0, 1) ** 1.3
    slope[~mask] = np.nan

    # impervious proxy from real layers
    roads_f = roads.astype(float)
    buildings_f = buildings.astype(float)
    river_f = river.astype(float)

    impervious = 0.45 * smooth2d(roads_f, 2) + 0.65 * smooth2d(buildings_f, 2)
    impervious = np.clip(impervious, 0, 1)
    impervious[~mask] = np.nan

    # low spots / accumulation
    low_spots = 0.55 * outlet_zone + 0.18 * smooth2d(river_f, 4) + 0.18 * np.nan_to_num(slope, nan=0.0)
    low_spots[~mask] = 0.0

    flow_accum = np.nan_to_num(slope, nan=0.0) * np.nan_to_num(impervious, nan=0.0)
    flow_accum = np.clip(flow_accum, 0, 1)

    flood_sus = (
        0.42 * roads_f
        + 0.18 * np.nan_to_num(impervious, nan=0.0)
        + 0.20 * low_spots
        + 0.26 * np.nan_to_num(slope, nan=0.0)
        + 0.14 * flow_accum
        + 0.40 * outlet_zone
    )

    # reduce flooding in buildings and in the river itself
    flood_sus[buildings] *= 0.30
    flood_sus[river] *= 0.08
    flood_sus = np.clip(flood_sus, 0, 1)
    flood_sus[~mask] = np.nan

    return {
        "Lon": Lon,
        "Lat": Lat,
        "mask": mask,
        "roads": roads,
        "buildings": buildings,
        "river": river,
        "impervious": impervious,
        "outlet_zone": outlet_zone,
        "slope": slope,
        "flood_sus": flood_sus,
        "bounds": bounds,
        "roads_gdf": roads_gdf,
        "buildings_gdf": buildings_gdf,
        "water_gdf": water_gdf,
    }


def create_synthetic_spatial_layers(n=220):
    x = np.linspace(0, 1, n)
    y = np.linspace(0, 1, n)
    X, Y = np.meshgrid(x, y)

    river_centerline = 0.45 + 0.03 * np.sin(8 * X)
    river = np.abs(Y - river_centerline) < 0.018

    mask = Y > river_centerline
    roads = (
        (np.abs((X * 100) % 12 - 6) < 0.5)
        | (np.abs((Y * 100) % 11 - 5.5) < 0.5)
    ) & mask

    buildings = (
        (np.sin(20 * X) > 0.82)
        & (np.sin(18 * Y) > 0.82)
        & (~roads)
        & (~river)
        & mask
    )

    outlet_zone = np.exp(-((X - 0.72) ** 2 / 0.03 + (Y - 0.65) ** 2 / 0.03))
    slope = 1 - np.sqrt((X - 0.72) ** 2 + (Y - 0.65) ** 2)
    slope = np.clip(slope, 0, 1)
    slope[~mask] = np.nan

    impervious = 0.55 * smooth2d(roads.astype(float), 2) + 0.65 * smooth2d(buildings.astype(float), 2)
    impervious = np.clip(impervious, 0, 1)
    impervious[~mask] = np.nan

    low_spots = 0.5 * outlet_zone + 0.2 * smooth2d(river.astype(float), 3)
    flow_accum = np.nan_to_num(slope, nan=0.0) * np.nan_to_num(impervious, nan=0.0)

    flood_sus = (
        0.42 * roads.astype(float)
        + 0.18 * np.nan_to_num(impervious, nan=0.0)
        + 0.20 * low_spots
        + 0.26 * np.nan_to_num(slope, nan=0.0)
        + 0.14 * flow_accum
        + 0.40 * outlet_zone
    )
    flood_sus[buildings] *= 0.30
    flood_sus[river] *= 0.08
    flood_sus = np.clip(flood_sus, 0, 1)
    flood_sus[~mask] = np.nan

    return {
        "Lon": X,
        "Lat": Y,
        "mask": mask,
        "roads": roads,
        "buildings": buildings,
        "river": river,
        "impervious": impervious,
        "outlet_zone": outlet_zone,
        "slope": slope,
        "flood_sus": flood_sus,
        "bounds": (0, 0, 1, 1),
    }


def allocate_nbs_spatial_real(selected_df, mask, roads, buildings, impervious, outlet_zone):
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

        target_count = max(1, int(np.sum(mask) * coverage * 0.18))

        if family == "GI":
            score = np.nan_to_num(impervious.copy(), nan=0.0)

            if "Green Roof" in name:
                score = 0.85 * score + 0.75 * buildings.astype(float)
            elif "Grassed Swale" in name:
                score = 0.55 * score + 0.75 * roads.astype(float)
            elif "Bioretention" in name:
                score = 0.60 * score + 0.55 * roads.astype(float)
            elif "Infiltration Trench" in name:
                score = 0.70 * score + 0.70 * roads.astype(float)
            elif "Rain Barrel" in name or "Cistern" in name:
                score = 0.45 * score + 0.85 * buildings.astype(float)
        else:
            score = 0.70 * outlet_zone + 0.20 * roads.astype(float)

        score[~mask] = -999
        score[alloc > 0] *= 0.7

        flat_idx = np.argsort(score.ravel())[::-1]
        chosen = 0

        for idx in flat_idx:
            r, c = np.unravel_index(idx, score.shape)
            if not mask[r, c]:
                continue

            if family == "Storage":
                rr0, rr1 = max(0, r - 2), min(n, r + 3)
                cc0, cc1 = max(0, c - 2), min(n, c + 3)
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


# ============================================================
# TITLE
# ============================================================
st.title("Houston Nature-Based Solutions Flood Explorer")
st.caption(
    "Conceptual prototype: event-based hydrologic response, watershed attributes, "
    "real street/building-based spatial layers, and estimated urban flood extent."
)

# ============================================================
# EVENT / WATERSHED SELECTION
# ============================================================
with st.sidebar:
    st.header("1) Event")
    selected_event_name = st.selectbox("Select event", events["name"].tolist())
    event_row = events.loc[events["name"] == selected_event_name].iloc[0]

    watershed_row = watersheds.loc[watersheds["watershed_id"] == event_row["watershed_id"]].iloc[0]
    gauge_row = gauges.loc[gauges["gauge_id"] == event_row["gauge_id"]].iloc[0]

    st.header("2) Solutions")
    st.caption("Choose one or several solutions and assign a coverage percentage.")

gauge_lat = float(gauge_row["lat"])
gauge_lon = float(gauge_row["lon"])

lat_pad = 0.010
lon_pad = 0.014

map_bounds = [
    [gauge_lat - lat_pad, gauge_lon - lon_pad],
    [gauge_lat + lat_pad, gauge_lon + lon_pad],
]
map_center = [gauge_lat, gauge_lon]

# ============================================================
# EVENT / WATERSHED INFO
# ============================================================
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

    c4, c5, c6 = st.columns(3)
    c4.metric("Initial abstraction", f"{watershed_row.get('initial_abstraction_mm', 'N/A')} mm")
    c5.metric("Curve Number", f"{watershed_row.get('curve_number', 'N/A')}")
    c6.metric("Urban LULC", f"{watershed_row.get('lulc_urban_pct', 'N/A')}%")

    st.caption(
        f"Gauge: {gauge_row.get('name', 'N/A')} | Green LULC: {watershed_row.get('lulc_green_pct', 'N/A')}% | "
        f"Water/wet areas: {watershed_row.get('lulc_water_pct', 'N/A')}%"
    )

# ============================================================
# NBS SELECTOR
# ============================================================
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

# ============================================================
# HYDROGRAPH
# ============================================================
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

# ============================================================
# SPATIAL LAYERS
# ============================================================
use_fallback = False
try:
    spatial = build_real_spatial_layers(
        gauge_lat=gauge_lat,
        gauge_lon=gauge_lon,
        lat_pad=lat_pad,
        lon_pad=lon_pad,
        n=280,
    )
except Exception:
    spatial = create_synthetic_spatial_layers(n=220)
    use_fallback = True

mask = spatial["mask"]
roads = spatial["roads"]
buildings = spatial["buildings"]
river = spatial["river"]
impervious = spatial["impervious"]
outlet_zone = spatial["outlet_zone"]
slope = spatial["slope"]
flood_sus = spatial["flood_sus"]

alloc, category_names = allocate_nbs_spatial_real(
    selected_df.copy(),
    mask=mask,
    roads=roads,
    buildings=buildings,
    impervious=impervious,
    outlet_zone=outlet_zone,
)

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

before_mask_clean = before_mask & (~river)
after_mask_clean = after_mask & (~river)

before_intensity = np.nan_to_num(flood_sus, nan=0.0).copy()
after_intensity = np.nan_to_num(flood_sus, nan=0.0).copy()
before_intensity[river] = 0.0
after_intensity[river] = 0.0

# ============================================================
# OUTPUTS
# ============================================================
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

    with st.expander("Debug: slope"):
        fig_s, ax_s = plt.subplots(figsize=(6, 5))
        ax_s.imshow(np.nan_to_num(slope, nan=0.0), cmap="terrain")
        ax_s.set_xticks([])
        ax_s.set_yticks([])
        ax_s.set_title("Slope toward outlet")
        st.pyplot(fig_s, use_container_width=True)

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
    axm.imshow(category_map, cmap=cmap, origin="upper")

    roads_img = np.where(roads & mask, 1.0, np.nan)
    axm.imshow(roads_img, cmap=ListedColormap(["#4d4d4d"]), origin="upper", alpha=0.25)

    river_img = np.where(river, 1.0, np.nan)
    axm.imshow(river_img, cmap=ListedColormap(["#6baed6"]), origin="upper", alpha=0.45)

    axm.set_xticks([])
    axm.set_yticks([])
    axm.set_title("Categorized implementation map")
    st.pyplot(fig_map, use_container_width=True)

    legend_lines = ["0 = untreated / baseline urban area"]
    for i, name in enumerate(category_names, start=1):
        legend_lines.append(f"{i} = {name}")
    legend_lines.append("Dark gray = street network")
    legend_lines.append("Light blue = river / bayou")
    st.caption(" | ".join(legend_lines))

# ============================================================
# FOLIUM MAPS
# ============================================================
st.subheader("Estimated Flood Extent on Basemap")

if use_fallback:
    st.warning("OSM layers could not be loaded. A synthetic fallback spatial grid is being used.")

col_map1, col_map2 = st.columns(2)

before_rgba = rgba_from_intensity(before_mask_clean, before_intensity)
after_rgba = rgba_from_intensity(after_mask_clean, after_intensity)

with col_map1:
    st.markdown("**Before NBS**")
    m_before = folium.Map(location=map_center, zoom_start=15, tiles=None)

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
        fill_opacity=0.95,
        tooltip=f"Outlet gauge: {gauge_row['name']}",
    ).add_to(m_before)

    ImageOverlay(
        image=before_rgba,
        bounds=map_bounds,
        opacity=1.0,
        interactive=True,
        cross_origin=False,
        zindex=10,
    ).add_to(m_before)

    folium.LayerControl().add_to(m_before)
    st_folium(m_before, width=700, height=520, key="before_map")

with col_map2:
    st.markdown("**After NBS**")
    m_after = folium.Map(location=map_center, zoom_start=15, tiles=None)

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
        fill_opacity=0.95,
        tooltip=f"Outlet gauge: {gauge_row['name']}",
    ).add_to(m_after)

    ImageOverlay(
        image=after_rgba,
        bounds=map_bounds,
        opacity=1.0,
        interactive=True,
        cross_origin=False,
        zindex=10,
    ).add_to(m_after)

    folium.LayerControl().add_to(m_after)
    st_folium(m_after, width=700, height=520, key="after_map")

st.caption(
    "Flood maps are scenario visualizations derived from real local street/building/waterway layers when available, "
    "combined with outlet-driven slope and literature-based NBS performance effects. "
    "They are intended for comparative interpretation, not as calibrated hydraulic simulations."
)
