import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import folium
from folium.raster_layers import ImageOverlay
from streamlit_folium import st_folium
from matplotlib.colors import ListedColormap

import osmnx as ox
from shapely.geometry import box
from rasterio.features import rasterize
from rasterio.transform import from_bounds

from hydro_model import (
    build_baseline_hydrograph_from_event,
    apply_nbs_to_hydrograph,
    get_event_intensity_mm_hr,
)

st.set_page_config(
    page_title="Houston Urban Flood Explorer",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ============================================================
# STYLE
# ============================================================
st.markdown(
    """
    <style>
    .block-container {
        padding-top: 2.0rem;
        padding-bottom: 2.0rem;
        max-width: 1500px;
    }

    .main-title {
        font-size: 2.8rem;
        font-weight: 850;
        line-height: 1.05;
        margin-bottom: 0.25rem;
    }

    .subtitle {
        color: #9ca3af;
        font-size: 1.02rem;
        margin-bottom: 2rem;
    }

    .section-title {
        font-size: 1.55rem;
        font-weight: 800;
        margin-top: 1.8rem;
        margin-bottom: 0.85rem;
    }

    .panel-title {
        font-size: 0.78rem;
        font-weight: 800;
        letter-spacing: 0.08em;
        color: #94a3b8;
        margin-top: 0.4rem;
        margin-bottom: 0.6rem;
        text-transform: uppercase;
    }

    .card {
        padding: 1.05rem 1.15rem;
        border-radius: 18px;
        border: 1px solid rgba(255,255,255,0.10);
        background: linear-gradient(135deg, rgba(30,41,59,0.96), rgba(15,23,42,0.96));
        min-height: 115px;
        box-shadow: 0 8px 24px rgba(0,0,0,0.18);
    }

    .card-label {
        color: #cbd5e1;
        font-size: 0.82rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-bottom: 0.35rem;
    }

    .card-value {
        color: white;
        font-size: 1.95rem;
        font-weight: 850;
        line-height: 1.1;
    }

    .card-note {
        color: #94a3b8;
        font-size: 0.82rem;
        margin-top: 0.35rem;
    }

    .blue-card {
        border-left: 6px solid #38bdf8;
    }

    .orange-card {
        border-left: 6px solid #fb923c;
    }

    .green-card {
        border-left: 6px solid #34d399;
    }

    .purple-card {
        border-left: 6px solid #a78bfa;
    }

    .compact-context {
        padding: 0.85rem 1rem;
        border-radius: 16px;
        background: rgba(30,41,59,0.72);
        border: 1px solid rgba(255,255,255,0.10);
        color: #d1d5db;
        font-size: 0.94rem;
    }

    .nbs-card {
        padding: 1rem 1.1rem;
        border-radius: 18px;
        border: 1px solid rgba(255,255,255,0.10);
        background: rgba(15,23,42,0.72);
        margin-bottom: 0.8rem;
    }

    .nbs-name {
        font-size: 1.05rem;
        font-weight: 800;
        color: #f8fafc;
        margin-bottom: 0.15rem;
    }

    .nbs-family {
        display: inline-block;
        padding: 0.12rem 0.55rem;
        border-radius: 999px;
        background: rgba(52,211,153,0.14);
        color: #86efac;
        font-size: 0.75rem;
        font-weight: 800;
        margin-bottom: 0.4rem;
    }

    .storage-family {
        background: rgba(56,189,248,0.14);
        color: #7dd3fc;
    }

    .small-muted {
        color: #94a3b8;
        font-size: 0.88rem;
    }

    .result-card {
        padding: 1rem 1.1rem;
        border-radius: 18px;
        background: rgba(15,23,42,0.85);
        border: 1px solid rgba(255,255,255,0.10);
        min-height: 110px;
    }

    .result-label {
        color: #cbd5e1;
        font-size: 0.82rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .result-value {
        font-size: 2.1rem;
        font-weight: 900;
        color: white;
        margin-top: 0.25rem;
    }

    div[data-testid="stMetric"] {
        background: rgba(15,23,42,0.50);
        padding: 1rem;
        border-radius: 16px;
        border: 1px solid rgba(255,255,255,0.08);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# DATA
# ============================================================
@st.cache_data
def load_tabular_data():
    return (
        pd.read_csv("events.csv"),
        pd.read_csv("watershed.csv"),
        pd.read_csv("gauge.csv"),
        pd.read_csv("nbs_catalog.csv"),
    )


events, watersheds, gauges, nbs_catalog = load_tabular_data()

# ============================================================
# HELPERS
# ============================================================
def safe_text(value, fallback="N/A"):
    if pd.isna(value):
        return fallback
    return str(value)


def short_text(value, max_chars=34):
    value = safe_text(value)
    if len(value) <= max_chars:
        return value
    return value[: max_chars - 3] + "..."


def card(label, value, note="", color_class="blue-card"):
    st.markdown(
        f"""
        <div class="card {color_class}">
            <div class="card-label">{label}</div>
            <div class="card-value">{value}</div>
            <div class="card-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def result_card(label, value, note="", accent="#38bdf8"):
    st.markdown(
        f"""
        <div class="result-card" style="border-left: 6px solid {accent};">
            <div class="result-label">{label}</div>
            <div class="result-value">{value}</div>
            <div class="card-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


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


def features_from_bbox_compat(north, south, east, west, tags):
    bbox = (west, south, east, north)
    try:
        return ox.features_from_bbox(bbox, tags=tags)
    except TypeError:
        return ox.features_from_bbox(north, south, east, west, tags=tags)


@st.cache_data(show_spinner=False)
def load_osm_layers(north, south, east, west):
    bbox_poly = box(west, south, east, north)

    roads = features_from_bbox_compat(north, south, east, west, {"highway": True})
    roads = roads[roads.geometry.type.isin(["LineString", "MultiLineString"])].copy()
    roads = roads.to_crs(4326)
    roads = roads[roads.intersects(bbox_poly)]

    buildings = features_from_bbox_compat(north, south, east, west, {"building": True})
    buildings = buildings[buildings.geometry.type.isin(["Polygon", "MultiPolygon"])].copy()
    buildings = buildings.to_crs(4326)
    buildings = buildings[buildings.intersects(bbox_poly)]

    water = features_from_bbox_compat(north, south, east, west, {"waterway": True})
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

    return rasterize(
        shapes,
        out_shape=out_shape,
        transform=transform,
        fill=0,
        all_touched=all_touched,
        dtype="uint8",
    )


def derive_one_side_watershed_from_river(river_arr):
    h, w = river_arr.shape
    mask = np.zeros((h, w), dtype=bool)

    river_cols, river_rows = [], []
    for c in range(w):
        rows = np.where(river_arr[:, c] > 0)[0]
        if len(rows) > 0:
            river_cols.append(c)
            river_rows.append(np.mean(rows))

    if len(river_cols) < 3:
        mask[h // 3 :, :] = True
        return mask

    river_cols = np.array(river_cols)
    river_rows = np.array(river_rows)
    interp_rows = np.interp(np.arange(w), river_cols, river_rows)

    for c in range(w):
        r0 = int(np.clip(interp_rows[c], 0, h - 1))
        mask[r0 + 1 :, c] = True

    return mask


def build_real_spatial_layers(gauge_lat, gauge_lon, lat_pad, lon_pad, n=280):
    north = gauge_lat + lat_pad
    south = gauge_lat - lat_pad
    east = gauge_lon + lon_pad
    west = gauge_lon - lon_pad
    bounds = (west, south, east, north)

    lon = np.linspace(west, east, n)
    lat = np.linspace(north, south, n)
    Lon, Lat = np.meshgrid(lon, lat)

    roads_gdf, buildings_gdf, water_gdf = load_osm_layers(north, south, east, west)

    roads = rasterize_gdf(roads_gdf, bounds, (n, n), all_touched=True).astype(bool)
    buildings = rasterize_gdf(buildings_gdf, bounds, (n, n), all_touched=True).astype(bool)
    river = rasterize_gdf(water_gdf, bounds, (n, n), all_touched=True).astype(bool)

    mask = derive_one_side_watershed_from_river(river)
    if mask.sum() < 0.12 * n * n:
        mask = np.zeros((n, n), dtype=bool)
        mask[n // 3 :, :] = True

    dlon = (Lon - gauge_lon) / max(lon_pad, 1e-6)
    dlat = (Lat - gauge_lat) / max(lat_pad, 1e-6)

    outlet_zone = np.exp(-(dlon**2 / 0.08 + dlat**2 / 0.08))
    outlet_zone[~mask] = 0.0

    dist = np.sqrt(dlon**2 + dlat**2)
    dist = dist / np.nanmax(dist)
    slope = np.clip(1.0 - dist, 0, 1) ** 1.3
    slope[~mask] = np.nan

    roads_f = roads.astype(float)
    buildings_f = buildings.astype(float)
    river_f = river.astype(float)

    impervious = 0.45 * smooth2d(roads_f, 2) + 0.65 * smooth2d(buildings_f, 2)
    impervious = np.clip(impervious, 0, 1)
    impervious[~mask] = np.nan

    low_spots = 0.55 * outlet_zone + 0.18 * smooth2d(river_f, 4) + 0.18 * np.nan_to_num(slope, nan=0.0)
    low_spots[~mask] = 0.0

    flow_accum = np.clip(
        np.nan_to_num(slope, nan=0.0) * np.nan_to_num(impervious, nan=0.0),
        0,
        1,
    )

    flood_sus = (
        0.42 * roads_f
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
        "mask": mask,
        "roads": roads,
        "buildings": buildings,
        "river": river,
        "impervious": impervious,
        "outlet_zone": outlet_zone,
        "slope": slope,
        "flood_sus": flood_sus,
        "roads_count": len(roads_gdf),
        "buildings_count": len(buildings_gdf),
        "water_count": len(water_gdf),
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
    slope = np.clip(1 - np.sqrt((X - 0.72) ** 2 + (Y - 0.65) ** 2), 0, 1)
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
        "mask": mask,
        "roads": roads,
        "buildings": buildings,
        "river": river,
        "impervious": impervious,
        "outlet_zone": outlet_zone,
        "slope": slope,
        "flood_sus": flood_sus,
        "roads_count": 0,
        "buildings_count": 0,
        "water_count": 0,
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
        local_red = (0.10 + 0.20 * coverage) if family == "GI" else (0.14 + 0.22 * coverage)
        reduction += (alloc == (idx + 1)) * local_red

    reduction += 0.04 * (alloc > 0)
    reduction = np.clip(reduction, 0, 0.55)

    flood_after = flood_before * (1 - reduction)
    return flood_before >= base_threshold, flood_after >= base_threshold


# ============================================================
# APP HEADER
# ============================================================
st.markdown(
    """
    <div class="main-title">Houston Urban Flood Explorer</div>
    <div class="subtitle">
        Event-based hydrologic response and nature-based mitigation scenarios for urban flood risk reduction.
    </div>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# 1. EVENT SELECTION
# ============================================================
st.markdown('<div class="section-title">1. Select Storm Event</div>', unsafe_allow_html=True)

event_col, help_col = st.columns([1.25, 1.0])

with event_col:
    selected_event_name = st.selectbox(
        "Choose a rainfall event",
        events["name"].tolist(),
        index=0,
        help="Each event contains rainfall depth, duration, return period, and watershed/gauge links.",
    )

event_row = events.loc[events["name"] == selected_event_name].iloc[0]
watershed_row = watersheds.loc[watersheds["watershed_id"] == event_row["watershed_id"]].iloc[0]
gauge_row = gauges.loc[gauges["gauge_id"] == event_row["gauge_id"]].iloc[0]

with help_col:
    st.markdown(
        """
        <div class="compact-context">
        <b>Workflow:</b> select a storm event, configure NBS coverage, then compare outlet hydrograph and
        estimated street-level flood extent before and after implementation.
        </div>
        """,
        unsafe_allow_html=True,
    )

gauge_lat = float(gauge_row["lat"])
gauge_lon = float(gauge_row["lon"])
lat_pad = 0.010
lon_pad = 0.014
map_bounds = [[gauge_lat - lat_pad, gauge_lon - lon_pad], [gauge_lat + lat_pad, gauge_lon + lon_pad]]
map_center = [gauge_lat, gauge_lon]

st.markdown("### Event summary")
e1, e2, e3, e4 = st.columns(4)

with e1:
    card("Return period", safe_text(event_row.get("return_period", "N/A")), "Event severity", "orange-card")

with e2:
    card(
        "Rainfall",
        f'{safe_text(event_row.get("rainfall_mm", "N/A"))} mm',
        f'{safe_text(event_row.get("date_start", "N/A"))} → {safe_text(event_row.get("date_end", "N/A"))}',
        "blue-card",
    )

with e3:
    card(
        "Duration",
        f'{safe_text(event_row.get("duration_hr", "N/A"))} h',
        f'Mean intensity: {round(get_event_intensity_mm_hr(event_row), 2)} mm/h',
        "purple-card",
    )

with e4:
    card("Rain type", short_text(event_row.get("rain_type", "N/A"), 20), "Storm classification", "green-card")

st.markdown("### Watershed context")
st.markdown(
    f"""
    <div class="compact-context">
    <b>{safe_text(watershed_row.get("name", "Watershed"))}</b> &nbsp; | &nbsp;
    Area: <b>{safe_text(watershed_row.get("area_km2", "N/A"))} km²</b> &nbsp; | &nbsp;
    Imperviousness: <b>{safe_text(watershed_row.get("impervious_pct", "N/A"))}%</b> &nbsp; | &nbsp;
    Curve Number: <b>{safe_text(watershed_row.get("curve_number", "N/A"))}</b> &nbsp; | &nbsp;
    Initial abstraction: <b>{safe_text(watershed_row.get("initial_abstraction_mm", "N/A"))} mm</b><br>
    Gauge: <b>{safe_text(gauge_row.get("name", "N/A"))}</b> &nbsp; | &nbsp;
    Urban LULC: <b>{safe_text(watershed_row.get("lulc_urban_pct", "N/A"))}%</b> &nbsp; | &nbsp;
    Green LULC: <b>{safe_text(watershed_row.get("lulc_green_pct", "N/A"))}%</b> &nbsp; | &nbsp;
    Water/wet areas: <b>{safe_text(watershed_row.get("lulc_water_pct", "N/A"))}%</b>
    </div>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# 2. NBS CONFIGURATION
# ============================================================
st.markdown('<div class="section-title">2. Configure Nature-Based Solutions</div>', unsafe_allow_html=True)
st.caption("Set coverage using sliders. Coverage represents the share of eligible urban area treated by each solution.")

selected_rows = []

for _, row in nbs_catalog.iterrows():
    family = safe_text(row.get("family", "NBS"))
    name = safe_text(row.get("name", "Solution"))
    notes = safe_text(row.get("notes", ""))
    max_coverage = 50

    default_coverage = 0
    if name == "Bioretention":
        default_coverage = 50
    elif name == "Green Roof":
        default_coverage = 20
    elif name == "Detention Pond":
        default_coverage = 50

    st.markdown('<div class="nbs-card">', unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1.45, 2.4, 2.1])

    with c1:
        family_class = "storage-family" if family.lower() == "storage" else ""
        st.markdown(
            f"""
            <div class="nbs-name">{name}</div>
            <span class="nbs-family {family_class}">{family}</span>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(f'<div class="small-muted">{notes}</div>', unsafe_allow_html=True)

    with c3:
        coverage = st.slider(
            f"{name} coverage",
            min_value=0,
            max_value=max_coverage,
            value=default_coverage,
            step=5,
            key=f"coverage_{name}",
        )

    st.markdown("</div>", unsafe_allow_html=True)

    if coverage > 0:
        tmp = row.copy()
        tmp["use"] = True
        tmp["coverage_pct"] = coverage
        selected_rows.append(tmp)

selected_df = pd.DataFrame(selected_rows)

if selected_df.empty:
    st.warning("Set at least one NBS coverage slider above 0% to generate the scenario.")
    st.stop()

# ============================================================
# HYDROGRAPH + SCENARIO
# ============================================================
baseline = build_baseline_hydrograph_from_event(event_row, watershed_row)
scenario = apply_nbs_to_hydrograph(baseline, selected_df, event_row)

t = scenario["time_hr"]
t_mod = scenario["time_mod_hr"]
Q_base = scenario["q_base_m3s"]
Q_mod = scenario["q_mod_m3s"]
details_df = scenario["details_df"]

# ============================================================
# SPATIAL LAYERS
# ============================================================
use_fallback = False
spatial_error = None

try:
    spatial = build_real_spatial_layers(
        gauge_lat=gauge_lat,
        gauge_lon=gauge_lon,
        lat_pad=lat_pad,
        lon_pad=lon_pad,
        n=280,
    )
except Exception as e:
    spatial_error = str(e)
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
    mask,
    roads,
    buildings,
    impervious,
    outlet_zone,
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
flood_extent_reduction_pct = 100 * (1 - np.sum(after_mask) / max(np.sum(before_mask), 1))

before_mask_clean = before_mask & (~river)
after_mask_clean = after_mask & (~river)

before_intensity = np.nan_to_num(flood_sus, nan=0.0).copy()
after_intensity = np.nan_to_num(flood_sus, nan=0.0).copy()
before_intensity[river] = 0.0
after_intensity[river] = 0.0

# ============================================================
# 3. RESULTS
# ============================================================
st.markdown('<div class="section-title">3. Scenario Results</div>', unsafe_allow_html=True)

k1, k2, k3, k4 = st.columns(4)
with k1:
    result_card("Peak reduction", f"{scenario['peak_reduction_pct']:.1f}%", "Lower outlet peak", "#fb923c")
with k2:
    result_card("Runoff reduction", f"{scenario['runoff_reduction_pct']:.1f}%", "Lower total volume", "#38bdf8")
with k3:
    result_card("Lag increase", f"{scenario['lag_increase_hr']:.2f} h", "Delayed response", "#a78bfa")
with k4:
    result_card("Flood extent reduction", f"{flood_extent_reduction_pct:.1f}%", "Estimated spatial impact", "#34d399")

left, right = st.columns([1.08, 1.0])

# ============================================================
# LEFT: HYDROGRAPH + PROGRESS TABLE
# ============================================================
with left:
    st.markdown('<div class="panel-title">🔵 Outlet hydrograph</div>', unsafe_allow_html=True)

    fig, ax = plt.subplots(figsize=(9.5, 4.7))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    t_arr = np.array(t)
    q_b = np.array(Q_base)
    t_marr = np.array(t_mod)
    q_m = np.array(Q_mod)

    q_m_interp = np.interp(t_arr, t_marr, q_m)

    ax.fill_between(
        t_arr,
        q_m_interp,
        q_b,
        where=q_b >= q_m_interp,
        color="#F4B183",
        alpha=0.22,
        label="_nolegend_",
    )

    ax.plot(t_arr, q_b, color="#2E86DE", lw=2.6, label="Baseline", zorder=3)
    ax.plot(t_marr, q_m, color="#4C8C2B", lw=2.6, label="With NBS", zorder=3)

    base_peak_idx = int(np.argmax(q_b))
    mod_peak_idx = int(np.argmax(q_m))

    ax.scatter(t_arr[base_peak_idx], q_b[base_peak_idx], color="#2E86DE", s=28, zorder=5)
    ax.text(
        t_arr[base_peak_idx] + 1,
        q_b[base_peak_idx],
        f"{q_b[base_peak_idx]:.0f}",
        color="#2E86DE",
        fontsize=11,
        fontweight="bold",
    )

    ax.scatter(t_marr[mod_peak_idx], q_m[mod_peak_idx], color="#4C8C2B", s=28, zorder=5)
    ax.text(
        t_marr[mod_peak_idx] + 1,
        q_m[mod_peak_idx],
        f"{q_m[mod_peak_idx]:.0f}",
        color="#4C8C2B",
        fontsize=11,
        fontweight="bold",
    )

    ax.set_xlabel("Time (hours)", fontsize=10, color="#6b7280")
    ax.set_ylabel("Discharge (m³/s)", fontsize=10, color="#6b7280")
    ax.tick_params(colors="#9ca3af", labelsize=9)
    ax.grid(axis="y", alpha=0.30, color="#e5e7eb")
    ax.legend(fontsize=10, framealpha=0, labelcolor="#444", loc="upper right")

    for spine in ax.spines.values():
        spine.set_edgecolor("#e5e7eb")

    plt.tight_layout(pad=0.5)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.markdown('<div class="panel-title">🟢 Performance by solution</div>', unsafe_allow_html=True)

    if not details_df.empty:
        show_df = details_df.copy()

        # Harmonize possible column names from hydro_model.py
        if "solution" not in show_df.columns and "name" in show_df.columns:
            show_df["solution"] = show_df["name"]

        if "runoff_reduction_pct" not in show_df.columns and "runoff_red_pct" in show_df.columns:
            show_df["runoff_reduction_pct"] = show_df["runoff_red_pct"]

        if "peak_reduction_pct" not in show_df.columns and "peak_red_pct" in show_df.columns:
            show_df["peak_reduction_pct"] = show_df["peak_red_pct"]

        if "lag_add_hr" not in show_df.columns and "lag_hr" in show_df.columns:
            show_df["lag_add_hr"] = show_df["lag_hr"]

        for col in ["runoff_reduction_pct", "peak_reduction_pct"]:
            if col in show_df.columns:
                show_df[col] = show_df[col].round(1)

        if "lag_add_hr" in show_df.columns:
            show_df["lag_add_hr"] = show_df["lag_add_hr"].round(2)

        if "coverage_pct" in show_df.columns:
            show_df["coverage_pct"] = show_df["coverage_pct"].astype(float).round(0)

        col_map = {
            "solution": "Solution",
            "family": "Type",
            "coverage_pct": "Coverage",
            "runoff_reduction_pct": "Runoff ↓",
            "peak_reduction_pct": "Peak ↓",
            "lag_add_hr": "Lag (h)",
        }

        keep_cols = [c for c in col_map.keys() if c in show_df.columns]

        table_df = show_df[keep_cols].rename(columns=col_map)

        st.dataframe(
            table_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Coverage": st.column_config.NumberColumn(
                    "Coverage",
                    format="%d%%",
                ),
                "Runoff ↓": st.column_config.ProgressColumn(
                    "Runoff ↓",
                    format="%.1f%%",
                    min_value=0,
                    max_value=50,
                ),
                "Peak ↓": st.column_config.ProgressColumn(
                    "Peak ↓",
                    format="%.1f%%",
                    min_value=0,
                    max_value=50,
                ),
                "Lag (h)": st.column_config.NumberColumn(
                    "Lag (h)",
                    format="%.2f",
                ),
            },
        )

# ============================================================
# RIGHT: MAPS + ASSUMPTIONS
# ============================================================
with right:
    st.subheader("NBS Spatial Allocation")
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
    fig_map.patch.set_facecolor("white")
    axm.set_facecolor("white")
    axm.imshow(category_map, cmap=cmap, origin="upper")
    axm.imshow(
        np.where(roads & mask, 1.0, np.nan),
        cmap=ListedColormap(["#4d4d4d"]),
        origin="upper",
        alpha=0.25,
    )
    axm.imshow(
        np.where(river, 1.0, np.nan),
        cmap=ListedColormap(["#6baed6"]),
        origin="upper",
        alpha=0.45,
    )
    axm.set_xticks([])
    axm.set_yticks([])
    axm.set_title("Categorized implementation map")
    st.pyplot(fig_map, use_container_width=True)
    plt.close(fig_map)

    legend_lines = ["0 = untreated / baseline urban area"]
    for i, name in enumerate(category_names, start=1):
        legend_lines.append(f"{i} = {name}")
    legend_lines.append("Dark gray = street network")
    legend_lines.append("Light blue = river / bayou")
    st.caption(" | ".join(legend_lines))

    with st.expander("Model assumptions and status"):
        st.markdown(
            """
            - This is a conceptual decision-support prototype, not a calibrated 2D hydraulic simulation.
            - Hydrologic effects are based on literature-derived NBS performance parameters.
            - Flood extent is estimated using local streets/buildings/waterways, an outlet-driven slope proxy,
              and relative flood susceptibility.
            - Results should be interpreted comparatively across scenarios.
            """
        )
        if use_fallback:
            st.warning("Fallback synthetic grid is active.")
            if spatial_error:
                st.code(spatial_error)
        else:
            st.success("OSM layers loaded successfully.")
            st.write(
                f"roads={spatial['roads_count']}, "
                f"buildings={spatial['buildings_count']}, "
                f"waterways={spatial['water_count']}"
            )

# ============================================================
# 4. FLOOD MAPS
# ============================================================
st.markdown('<div class="section-title">4. Estimated Flood Extent on Basemap</div>', unsafe_allow_html=True)

if use_fallback:
    st.warning("OSM layers could not be loaded. A synthetic fallback spatial grid is being used.")

st.caption("Blue overlay = estimated street-level flood susceptibility. Red marker = outlet gauge.")

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
    "They are intended for comparative interpretation, not calibrated hydraulic prediction."
)
