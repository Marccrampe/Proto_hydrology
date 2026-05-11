import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import folium
from folium.raster_layers import ImageOverlay
from streamlit_folium import st_folium

import osmnx as ox
from shapely.geometry import box
from rasterio.features import rasterize
from rasterio.transform import from_bounds
from matplotlib.colors import ListedColormap

from hydro_model import (
    build_baseline_hydrograph_from_event,
    apply_nbs_to_hydrograph,
    get_event_intensity_mm_hr,
)

# ── page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Houston Flood Explorer",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── global CSS ───────────────────────────────────────────────
st.markdown("""
<style>
/* reset streamlit chrome */
.block-container { padding: 0 !important; max-width: 100% !important; }
header[data-testid="stHeader"] { display: none; }
[data-testid="stSidebar"] { display: none; }

/* ---------- topbar ---------- */
.topbar {
    display: flex; align-items: center; justify-content: space-between;
    padding: 10px 24px;
    border-bottom: 1px solid var(--border);
    background: var(--bg-card);
    position: sticky; top: 0; z-index: 100;
}
.brand { font-size: 14px; font-weight: 600; display: flex; align-items: center; gap: 8px; }
.brand-dot { width: 8px; height: 8px; border-radius: 50%; background: #378ADD; }

.steps { display: flex; align-items: center; gap: 6px; }
.step {
    display: flex; align-items: center; gap: 5px;
    padding: 4px 12px; border-radius: 99px;
    font-size: 12px; font-weight: 500; color: #888;
    transition: all .15s;
}
.step.active { background: #EBF4FF; color: #185FA5; }
.step.done { color: #3B6D11; }
.step-sep { color: #ccc; font-size: 11px; }

/* ---------- layout ---------- */
.app-body {
    display: grid;
    grid-template-columns: 300px 1fr;
    min-height: calc(100vh - 48px);
}

/* ---------- sidebar ---------- */
.sidebar {
    background: #f8f9fa;
    border-right: 1px solid var(--border);
    padding: 16px 14px;
    display: flex; flex-direction: column; gap: 14px;
    overflow-y: auto;
    height: calc(100vh - 48px);
    position: sticky; top: 48px;
}
.sidebar-card {
    background: white;
    border: 1px solid #e5e7eb;
    border-radius: 12px;
    padding: 14px 14px 12px;
}
.sidebar-label {
    font-size: 10px; font-weight: 700; text-transform: uppercase;
    letter-spacing: .08em; color: #9ca3af; margin-bottom: 10px;
}
.event-badge {
    display: inline-flex; align-items: center; gap: 4px;
    background: #FAECE7; color: #993C1D;
    font-size: 10px; font-weight: 700;
    padding: 3px 8px; border-radius: 99px; margin-bottom: 8px;
}
.event-name { font-size: 13px; font-weight: 700; margin-bottom: 3px; color: #111; }
.event-meta { font-size: 12px; color: #6b7280; line-height: 1.65; }
.pill { display: inline-block; font-size: 10px; font-weight: 600;
    padding: 2px 8px; border-radius: 99px; margin: 2px 2px 0 0; }
.pill-blue { background: #EBF4FF; color: #185FA5; }
.pill-amber { background: #FAEEDA; color: #854F0B; }
.pill-gray { background: #f3f4f6; color: #6b7280; border: 1px solid #e5e7eb; }

.nbs-item { padding: 10px 0; border-bottom: 1px solid #f3f4f6; }
.nbs-item:last-child { border-bottom: none; padding-bottom: 2px; }
.nbs-header { display: flex; align-items: center; justify-content: space-between; margin-bottom: 6px; }
.nbs-name { font-size: 12px; font-weight: 600; color: #111; }
.nbs-tag { font-size: 10px; font-weight: 700; padding: 2px 7px; border-radius: 99px; }
.tag-gi { background: #EAF3DE; color: #3B6D11; }
.tag-storage { background: #EBF4FF; color: #185FA5; }

/* ---------- KPI bar ---------- */
.kpi-bar {
    display: grid; grid-template-columns: repeat(4, 1fr);
    gap: 0; border-bottom: 1px solid #e5e7eb;
    background: white;
}
.kpi-cell {
    padding: 14px 20px;
    border-right: 1px solid #e5e7eb;
}
.kpi-cell:last-child { border-right: none; }
.kpi-label { font-size: 10px; font-weight: 700; text-transform: uppercase;
    letter-spacing: .07em; color: #9ca3af; margin-bottom: 5px; }
.kpi-val { font-size: 22px; font-weight: 700; color: #111; line-height: 1.1; }
.kpi-delta { font-size: 11px; margin-top: 3px; }
.delta-good { color: #3B6D11; }
.delta-neutral { color: #6b7280; }

/* ---------- main content ---------- */
.main-content {
    display: flex; flex-direction: column;
    background: #f8f9fa;
    overflow-y: auto;
    height: calc(100vh - 48px);
}
.content-grid {
    display: grid; grid-template-columns: 1fr 1fr;
    gap: 14px;
    padding: 14px;
    flex: 1;
}
.panel {
    background: white;
    border: 1px solid #e5e7eb;
    border-radius: 12px;
    padding: 16px;
}
.panel-title {
    font-size: 12px; font-weight: 700; text-transform: uppercase;
    letter-spacing: .07em; color: #6b7280;
    margin-bottom: 12px; display: flex; align-items: center; gap: 6px;
}
.panel-title-dot { width: 6px; height: 6px; border-radius: 50%; background: #378ADD; }

/* ---------- map toggle ---------- */
.map-toggle-wrap { display: flex; justify-content: center; margin-bottom: 10px; }
.model-note {
    background: #f8f9fa; border-radius: 8px;
    border: 1px solid #e5e7eb;
    padding: 10px 12px; font-size: 11px; color: #9ca3af;
    line-height: 1.6; margin-top: 10px;
}

/* ---------- misc ---------- */
div[data-testid="stSelectbox"] > label { display: none; }
div[data-testid="stSlider"] > label { display: none; }
div[data-testid="stRadio"] > label { display: none; }
div[data-testid="stRadio"] [data-testid="stMarkdownContainer"] { display: none; }
</style>
""", unsafe_allow_html=True)

# ── CSS variables (light mode compat) ────────────────────────
st.markdown("""
<style>
:root {
    --border: #e5e7eb;
    --bg-card: #ffffff;
    --text-muted: #6b7280;
}
</style>
""", unsafe_allow_html=True)

# ── helpers ──────────────────────────────────────────────────
def safe(v, fb="N/A"):
    return fb if pd.isna(v) else str(v)

def short(v, n=30):
    v = safe(v)
    return v if len(v) <= n else v[:n-3] + "..."

def smooth2d(arr, n_iter=2):
    out = arr.astype(float).copy()
    for _ in range(n_iter):
        out = (out
            + np.roll(out,1,0)+np.roll(out,-1,0)
            + np.roll(out,1,1)+np.roll(out,-1,1)
            + np.roll(np.roll(out,1,0),1,1)
            + np.roll(np.roll(out,1,0),-1,1)
            + np.roll(np.roll(out,-1,0),1,1)
            + np.roll(np.roll(out,-1,0),-1,1)
        ) / 9.0
    return out

def rgba_from_intensity(mask, intensity, color=(55, 136, 221), max_alpha=210):
    h, w = mask.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[..., 0] = color[0]; rgba[..., 1] = color[1]; rgba[..., 2] = color[2]
    alpha = np.clip(intensity * max_alpha, 0, max_alpha).astype(np.uint8)
    alpha[~mask] = 0
    rgba[..., 3] = alpha
    return rgba

def features_from_bbox_compat(north, south, east, west, tags):
    bbox = (west, south, east, north)
    try:
        return ox.features_from_bbox(bbox, tags=tags)
    except TypeError:
        return ox.features_from_bbox(north, south, east, west, tags=tags)

@st.cache_data(show_spinner=False)
def load_osm_layers(north, south, east, west):
    bbox_poly = box(west, south, east, north)
    def fetch(tags, types):
        gdf = features_from_bbox_compat(north, south, east, west, tags)
        gdf = gdf[gdf.geometry.type.isin(types)].copy().to_crs(4326)
        return gdf[gdf.intersects(bbox_poly)]

    roads = fetch({"highway": True}, ["LineString","MultiLineString"])
    buildings = fetch({"building": True}, ["Polygon","MultiPolygon"])
    water = fetch({"waterway": True}, ["LineString","MultiLineString","Polygon","MultiPolygon"])
    return roads, buildings, water

def rasterize_gdf(gdf, bounds, out_shape):
    west, south, east, north = bounds
    transform = from_bounds(west, south, east, north, out_shape[1], out_shape[0])
    if gdf is None or len(gdf) == 0:
        return np.zeros(out_shape, dtype=np.uint8)
    shapes = [(g, 1) for g in gdf.geometry if g is not None and not g.is_empty]
    if not shapes:
        return np.zeros(out_shape, dtype=np.uint8)
    return rasterize(shapes, out_shape=out_shape, transform=transform,
                     fill=0, all_touched=True, dtype="uint8")

def derive_watershed_mask(river_arr):
    h, w = river_arr.shape
    mask = np.zeros((h, w), dtype=bool)
    cols, rows = [], []
    for c in range(w):
        rs = np.where(river_arr[:, c] > 0)[0]
        if len(rs) > 0:
            cols.append(c); rows.append(np.mean(rs))
    if len(cols) < 3:
        mask[h//3:, :] = True
        return mask
    interp = np.interp(np.arange(w), cols, rows)
    for c in range(w):
        mask[int(np.clip(interp[c], 0, h-1))+1:, c] = True
    return mask

@st.cache_data(show_spinner=False)
def build_spatial(gauge_lat, gauge_lon, lat_pad, lon_pad, n=280):
    north, south = gauge_lat+lat_pad, gauge_lat-lat_pad
    east, west   = gauge_lon+lon_pad, gauge_lon-lon_pad
    bounds = (west, south, east, north)
    lon = np.linspace(west, east, n); lat = np.linspace(north, south, n)
    Lon, Lat = np.meshgrid(lon, lat)

    roads_gdf, bldg_gdf, water_gdf = load_osm_layers(north, south, east, west)
    roads     = rasterize_gdf(roads_gdf, bounds, (n,n)).astype(bool)
    buildings = rasterize_gdf(bldg_gdf,  bounds, (n,n)).astype(bool)
    river     = rasterize_gdf(water_gdf, bounds, (n,n)).astype(bool)

    mask = derive_watershed_mask(river)
    if mask.sum() < 0.12*n*n:
        mask = np.zeros((n,n), bool); mask[n//3:, :] = True

    dlon = (Lon-gauge_lon)/max(lon_pad,1e-6)
    dlat = (Lat-gauge_lat)/max(lat_pad,1e-6)
    outlet_zone = np.exp(-(dlon**2/0.08 + dlat**2/0.08)); outlet_zone[~mask] = 0.
    dist = np.sqrt(dlon**2+dlat**2)/np.nanmax(np.sqrt(dlon**2+dlat**2))
    slope = np.clip(1.-dist,0,1)**1.3; slope[~mask] = np.nan

    impervious = np.clip(0.45*smooth2d(roads.astype(float),2) + 0.65*smooth2d(buildings.astype(float),2), 0, 1)
    impervious[~mask] = np.nan

    low_spots   = 0.55*outlet_zone + 0.18*smooth2d(river.astype(float),4) + 0.18*np.nan_to_num(slope,nan=0.)
    flow_accum  = np.clip(np.nan_to_num(slope,nan=0.)*np.nan_to_num(impervious,nan=0.), 0, 1)
    flood_sus   = (0.42*roads + 0.18*np.nan_to_num(impervious,nan=0.) + 0.20*low_spots
                   + 0.26*np.nan_to_num(slope,nan=0.) + 0.14*flow_accum + 0.40*outlet_zone)
    flood_sus[buildings] *= 0.30; flood_sus[river] *= 0.08
    flood_sus = np.clip(flood_sus, 0, 1); flood_sus[~mask] = np.nan

    return dict(mask=mask, roads=roads, buildings=buildings, river=river,
                impervious=impervious, outlet_zone=outlet_zone, slope=slope,
                flood_sus=flood_sus,
                roads_count=len(roads_gdf), buildings_count=len(bldg_gdf), water_count=len(water_gdf))

def build_synthetic(n=220):
    x = np.linspace(0,1,n); y = np.linspace(0,1,n)
    X, Y = np.meshgrid(x, y)
    river_cl = 0.45 + 0.03*np.sin(8*X)
    river = np.abs(Y-river_cl) < 0.018; mask = Y > river_cl
    roads = ((np.abs((X*100)%12-6)<0.5)|(np.abs((Y*100)%11-5.5)<0.5)) & mask
    buildings = (np.sin(20*X)>0.82)&(np.sin(18*Y)>0.82)&(~roads)&(~river)&mask
    outlet_zone = np.exp(-((X-0.72)**2/0.03+(Y-0.65)**2/0.03))
    slope = np.clip(1-np.sqrt((X-0.72)**2+(Y-0.65)**2),0,1); slope[~mask]=np.nan
    impervious = np.clip(0.55*smooth2d(roads.astype(float),2)+0.65*smooth2d(buildings.astype(float),2),0,1)
    impervious[~mask]=np.nan
    low_spots  = 0.5*outlet_zone+0.2*smooth2d(river.astype(float),3)
    flow_accum = np.nan_to_num(slope,nan=0.)*np.nan_to_num(impervious,nan=0.)
    flood_sus  = (0.42*roads+0.18*np.nan_to_num(impervious,nan=0.)+0.20*low_spots
                  +0.26*np.nan_to_num(slope,nan=0.)+0.14*flow_accum+0.40*outlet_zone)
    flood_sus[buildings]*=0.30; flood_sus[river]*=0.08
    flood_sus=np.clip(flood_sus,0,1); flood_sus[~mask]=np.nan
    return dict(mask=mask,roads=roads,buildings=buildings,river=river,
                impervious=impervious,outlet_zone=outlet_zone,slope=slope,flood_sus=flood_sus,
                roads_count=0,buildings_count=0,water_count=0)

def allocate_nbs(selected_df, mask, roads, buildings, impervious, outlet_zone):
    n = mask.shape[0]; alloc = np.zeros((n,n), dtype=int); cats = []
    for i, (_, row) in enumerate(selected_df.iterrows(), start=1):
        name=row["name"]; family=row["family"]; coverage=float(row["coverage_pct"])/100.
        if coverage <= 0: continue
        cats.append(name)
        target = max(1, int(np.sum(mask)*coverage*0.18))
        if family=="GI":
            score = np.nan_to_num(impervious.copy(), nan=0.)
            if "Green Roof" in name:   score = 0.85*score + 0.75*buildings
            elif "Swale" in name:      score = 0.55*score + 0.75*roads
            elif "Bioretention" in name: score = 0.60*score + 0.55*roads
            elif "Infiltration" in name: score = 0.70*score + 0.70*roads
            elif any(k in name for k in ["Rain Barrel","Cistern"]): score = 0.45*score + 0.85*buildings
        else:
            score = 0.70*outlet_zone + 0.20*roads
        score[~mask]=-999; score[alloc>0]*=0.7
        flat = np.argsort(score.ravel())[::-1]; chosen=0
        for idx in flat:
            r,c = np.unravel_index(idx, score.shape)
            if not mask[r,c]: continue
            if family=="Storage":
                r0,r1=max(0,r-2),min(n,r+3); c0,c1=max(0,c-2),min(n,c+3)
                patch=alloc[r0:r1,c0:c1]
                if np.all(patch==0): alloc[r0:r1,c0:c1]=i; chosen+=patch.size
            else:
                if alloc[r,c]==0: alloc[r,c]=i; chosen+=1
            if chosen>=target: break
    return alloc, cats

def compute_floods(flood_sus, alloc, selected_df, threshold):
    fb = np.nan_to_num(flood_sus.copy(), nan=0.)
    red = np.zeros_like(fb)
    for idx, (_, row) in enumerate(selected_df.iterrows()):
        cov=float(row["coverage_pct"])/100.; fam=row["family"]
        lr = (0.10+0.20*cov) if fam=="GI" else (0.14+0.22*cov)
        red += (alloc==(idx+1))*lr
    red += 0.04*(alloc>0); red = np.clip(red,0,0.55)
    fa = fb*(1-red)
    return fb>=threshold, fa>=threshold

# ── data ─────────────────────────────────────────────────────
@st.cache_data
def load_data():
    return (pd.read_csv("events.csv"), pd.read_csv("watershed.csv"),
            pd.read_csv("gauge.csv"),  pd.read_csv("nbs_catalog.csv"))

events, watersheds, gauges, nbs_catalog = load_data()

# ══════════════════════════════════════════════════════════════
# TOPBAR
# ══════════════════════════════════════════════════════════════
st.markdown("""
<div class="topbar">
  <div class="brand">
    <div class="brand-dot"></div>
    Houston flood explorer
  </div>
  <div class="steps">
    <span class="step done">✓ Event</span>
    <span class="step-sep">›</span>
    <span class="step active">NBS config</span>
    <span class="step-sep">›</span>
    <span class="step">Results</span>
    <span class="step-sep">›</span>
    <span class="step">Flood maps</span>
  </div>
  <div style="font-size:11px;color:#9ca3af">Conceptual prototype · Harris County</div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# APP BODY  —  sidebar + main in two columns
# ══════════════════════════════════════════════════════════════
sidebar_col, main_col = st.columns([300, 900], gap="small")

# ── SIDEBAR ──────────────────────────────────────────────────
with sidebar_col:
    # --- event selector ---
    with st.container():
        st.markdown('<div class="sidebar-label" style="margin-top:8px">Storm event</div>', unsafe_allow_html=True)
        selected_event = st.selectbox("event", events["name"].tolist(), index=0, label_visibility="collapsed")

    event_row     = events.loc[events["name"]==selected_event].iloc[0]
    watershed_row = watersheds.loc[watersheds["watershed_id"]==event_row["watershed_id"]].iloc[0]
    gauge_row     = gauges.loc[gauges["gauge_id"]==event_row["gauge_id"]].iloc[0]

    rain_mm   = float(event_row["rainfall_mm"])
    duration  = safe(event_row.get("duration_hr","N/A"))
    intensity = round(get_event_intensity_mm_hr(event_row), 2)
    rp        = safe(event_row.get("return_period","N/A"))
    rtype     = safe(event_row.get("rain_type","N/A"))
    d0        = safe(event_row.get("date_start",""))
    d1        = safe(event_row.get("date_end",""))

    st.markdown(f"""
    <div class="sidebar-card">
      <div class="event-badge">⚠ {rp} return period</div>
      <div class="event-name">{selected_event}</div>
      <div class="event-meta">
        {rain_mm:.0f} mm &nbsp;·&nbsp; {duration} h &nbsp;·&nbsp; {intensity} mm/h avg<br>
        {d0} → {d1}
      </div>
      <div style="margin-top:8px">
        <span class="pill pill-amber">{rtype}</span>
        <span class="pill pill-gray">{safe(watershed_row.get("name",""))}</span>
        <span class="pill pill-blue">{safe(watershed_row.get("impervious_pct","N/A"))}% imp.</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # --- NBS sliders ---
    st.markdown('<div class="sidebar-label" style="margin-top:4px">Nature-based solutions</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-card">', unsafe_allow_html=True)

    selected_rows = []
    for _, row in nbs_catalog.iterrows():
        name   = safe(row.get("name",""))
        family = safe(row.get("family","NBS"))
        tag_cls = "tag-gi" if family=="GI" else "tag-storage"

        default = 50 if name in ("Bioretention","Detention Pond") else (20 if name=="Green Roof" else 0)

        st.markdown(f"""
        <div class="nbs-item">
          <div class="nbs-header">
            <span class="nbs-name">{name}</span>
            <span class="nbs-tag {tag_cls}">{family}</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

        coverage = st.slider(name, 0, 50, default, 5, key=f"cov_{name}",
                             format="%d%%", label_visibility="collapsed")

        if coverage > 0:
            tmp = row.copy(); tmp["use"]=True; tmp["coverage_pct"]=coverage
            selected_rows.append(tmp)

    st.markdown('</div>', unsafe_allow_html=True)

selected_df = pd.DataFrame(selected_rows)

if selected_df.empty:
    st.warning("Set at least one NBS coverage above 0% to run the scenario.")
    st.stop()

# ── HYDROGRAPH COMPUTATION ───────────────────────────────────
baseline = build_baseline_hydrograph_from_event(event_row, watershed_row)
scenario = apply_nbs_to_hydrograph(baseline, selected_df, event_row)

t       = scenario["time_hr"]
t_mod   = scenario["time_mod_hr"]
Q_base  = scenario["q_base_m3s"]
Q_mod   = scenario["q_mod_m3s"]
details = scenario["details_df"]

peak_red  = scenario["peak_reduction_pct"]
run_red   = scenario["runoff_reduction_pct"]
lag_inc   = scenario["lag_increase_hr"]
Q_peak_b  = float(np.max(Q_base))
Q_peak_a  = float(np.max(Q_mod))

# ── SPATIAL ──────────────────────────────────────────────────
gauge_lat, gauge_lon = float(gauge_row["lat"]), float(gauge_row["lon"])
lat_pad, lon_pad = 0.010, 0.014
map_bounds = [[gauge_lat-lat_pad, gauge_lon-lon_pad], [gauge_lat+lat_pad, gauge_lon+lon_pad]]

use_fallback = False
try:
    spatial = build_spatial(gauge_lat, gauge_lon, lat_pad, lon_pad, n=280)
except Exception:
    spatial = build_synthetic(n=220)
    use_fallback = True

mask      = spatial["mask"]
roads     = spatial["roads"]
buildings = spatial["buildings"]
river     = spatial["river"]
impervious= spatial["impervious"]
outlet_zone=spatial["outlet_zone"]
flood_sus = spatial["flood_sus"]

alloc, cat_names = allocate_nbs(selected_df.copy(), mask, roads, buildings, impervious, outlet_zone)

threshold = 0.42 if rain_mm>=250 else (0.52 if rain_mm>=140 else (0.60 if rain_mm>=80 else 0.68))
before_mask, after_mask = compute_floods(flood_sus, alloc, selected_df.copy(), threshold)
extent_red = 100*(1 - np.sum(after_mask)/max(np.sum(before_mask),1))

before_clean = before_mask & (~river)
after_clean  = after_mask  & (~river)
before_int   = np.nan_to_num(flood_sus, nan=0.); before_int[river]=0.
after_int    = before_int.copy()

# ══════════════════════════════════════════════════════════════
# MAIN COLUMN
# ══════════════════════════════════════════════════════════════
with main_col:

    # ── KPI BAR ──────────────────────────────────────────────
    st.markdown(f"""
    <div class="kpi-bar">
      <div class="kpi-cell">
        <div class="kpi-label">Peak reduction</div>
        <div class="kpi-val">{peak_red:.1f}%</div>
        <div class="kpi-delta delta-good">↓ {Q_peak_b:.0f} → {Q_peak_a:.0f} m³/s</div>
      </div>
      <div class="kpi-cell">
        <div class="kpi-label">Runoff volume</div>
        <div class="kpi-val">{run_red:.1f}%</div>
        <div class="kpi-delta delta-good">↓ Total volume reduced</div>
      </div>
      <div class="kpi-cell">
        <div class="kpi-label">Lag time</div>
        <div class="kpi-val">+{lag_inc:.2f} h</div>
        <div class="kpi-delta delta-neutral">More response time</div>
      </div>
      <div class="kpi-cell">
        <div class="kpi-label">Flood extent</div>
        <div class="kpi-val">−{extent_red:.1f}%</div>
        <div class="kpi-delta delta-good">↓ Street-level area</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── PANELS ───────────────────────────────────────────────
    left, right = st.columns(2, gap="small")

    # ── LEFT: hydrograph + table ─────────────────────────────
    with left:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title"><div class="panel-title-dot"></div> Outlet hydrograph</div>', unsafe_allow_html=True)

        fig, ax = plt.subplots(figsize=(7, 3.8))
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")

        # shade peak reduction area
        t_arr  = np.array(t);    q_b = np.array(Q_base)
        t_marr = np.array(t_mod); q_m = np.array(Q_mod)
        ax.fill_between(t_arr, q_b, np.interp(t_arr, t_marr, q_m),
                        where=q_b > np.interp(t_arr, t_marr, q_m),
                        alpha=0.10, color="#D85A30", label="_nolegend_")

        ax.plot(t_arr,  q_b, lw=2.2, color="#378ADD", label="Baseline",    zorder=3)
        ax.plot(t_marr, q_m, lw=2.2, color="#639922", label="With NBS",    zorder=3)

        ax.set_xlabel("Time (hours)", fontsize=10, color="#6b7280")
        ax.set_ylabel("Discharge (m³/s)", fontsize=10, color="#6b7280")
        ax.tick_params(colors="#9ca3af", labelsize=9)
        for spine in ax.spines.values(): spine.set_edgecolor("#e5e7eb")
        ax.grid(axis="y", alpha=0.3, color="#e5e7eb")
        ax.legend(fontsize=10, framealpha=0, labelcolor="#444")

        # peak annotations
        pk_b_t = float(t_arr[np.argmax(q_b)])
        pk_a_t = float(t_marr[np.argmax(q_m)])
        ax.annotate(f"{Q_peak_b:.0f}", xy=(pk_b_t, Q_peak_b),
                    xytext=(pk_b_t+3, Q_peak_b*0.97),
                    fontsize=9, color="#185FA5", fontweight="bold")
        ax.annotate(f"{Q_peak_a:.0f}", xy=(pk_a_t, Q_peak_a),
                    xytext=(pk_a_t+3, Q_peak_a*0.97),
                    fontsize=9, color="#3B6D11", fontweight="bold")

        plt.tight_layout(pad=0.5)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        # performance table
        if not details.empty:
            st.markdown('<div class="panel-title" style="margin-top:14px"><div class="panel-title-dot" style="background:#639922"></div> Performance by solution</div>', unsafe_allow_html=True)
            col_map = {"solution":"Solution","family":"Type",
                       "coverage_pct":"Coverage","runoff_reduction_pct":"Runoff ↓","peak_reduction_pct":"Peak ↓","lag_add_hr":"Lag (h)"}
            show = details.copy()
            for c in ["runoff_reduction_pct","peak_reduction_pct"]:
                if c in show.columns: show[c] = show[c].round(1)
            if "lag_add_hr" in show.columns: show["lag_add_hr"] = show["lag_add_hr"].round(2)
            keep = [c for c in col_map if c in show.columns]
            st.dataframe(show[keep].rename(columns=col_map),
                         use_container_width=True, hide_index=True,
                         column_config={
                             "Runoff ↓": st.column_config.ProgressColumn(format="%.1f%%", min_value=0, max_value=50),
                             "Peak ↓":   st.column_config.ProgressColumn(format="%.1f%%", min_value=0, max_value=50),
                         })

        st.markdown('</div>', unsafe_allow_html=True)

    # ── RIGHT: flood map toggle ───────────────────────────────
    with right:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title"><div class="panel-title-dot" style="background:#D85A30"></div> Estimated flood extent</div>', unsafe_allow_html=True)

        map_view = st.radio("View", ["Before NBS", "After NBS"],
                            horizontal=True, label_visibility="collapsed")

        if use_fallback:
            st.caption("⚠ OSM layers unavailable — synthetic grid active.")

        show_before = map_view == "Before NBS"
        active_mask = before_clean if show_before else after_clean
        active_int  = before_int  if show_before else after_int
        active_rgba = rgba_from_intensity(active_mask, active_int)

        m = folium.Map(location=[gauge_lat, gauge_lon], zoom_start=15, tiles=None)
        folium.TileLayer(
            tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
            attr="Esri World Imagery", name="Satellite"
        ).add_to(m)
        ImageOverlay(image=active_rgba, bounds=map_bounds,
                     opacity=1.0, interactive=True, zindex=10).add_to(m)
        folium.CircleMarker(
            location=[gauge_lat, gauge_lon], radius=7,
            color="white", weight=2, fill=True, fill_color="#E24B4A", fill_opacity=1.0,
            tooltip=f"Outlet: {gauge_row['name']}"
        ).add_to(m)
        folium.LayerControl().add_to(m)

        st_folium(m, width=None, height=460, use_container_width=True, key="flood_map")

        if show_before:
            st.caption(f"Baseline flood area · threshold {threshold:.2f}")
        else:
            pct = f"−{extent_red:.1f}%"
            st.caption(f"After NBS &nbsp; · &nbsp; **{pct} flood area** vs baseline")

        # NBS allocation mini-map
        st.markdown('<div class="panel-title" style="margin-top:14px"><div class="panel-title-dot" style="background:#a78bfa"></div> NBS spatial allocation</div>', unsafe_allow_html=True)

        cat_map = np.full(mask.shape, np.nan)
        cat_map[mask] = 0
        cat_map[alloc>0] = alloc[alloc>0]
        n_cat = int(np.nanmax(np.nan_to_num(cat_map,nan=0)))
        colors = ["#e5e7eb","#2ca25f","#99d8c9","#66c2a4","#41ae76","#238b45","#006d2c","#3182bd","#6baed6","#9ecae1"]
        cmap   = ListedColormap(colors[:max(n_cat+1,2)])

        fig2, ax2 = plt.subplots(figsize=(7, 3.0))
        fig2.patch.set_facecolor("white"); ax2.set_facecolor("white")
        ax2.imshow(cat_map, cmap=cmap, origin="upper", aspect="auto")
        ax2.imshow(np.where(roads&mask,1.,np.nan), cmap=ListedColormap(["#555"]),
                   origin="upper", alpha=0.20, aspect="auto")
        ax2.imshow(np.where(river,1.,np.nan), cmap=ListedColormap(["#7bc8e8"]),
                   origin="upper", alpha=0.40, aspect="auto")
        ax2.set_xticks([]); ax2.set_yticks([])
        for sp in ax2.spines.values(): sp.set_visible(False)

        # compact legend
        patches = [mpatches.Patch(color=colors[0], label="Untreated")]
        for i,nm in enumerate(cat_names, start=1):
            if i < len(colors): patches.append(mpatches.Patch(color=colors[i], label=nm))
        ax2.legend(handles=patches, fontsize=8, loc="lower right",
                   framealpha=0.85, edgecolor="#e5e7eb")

        plt.tight_layout(pad=0.3)
        st.pyplot(fig2, use_container_width=True)
        plt.close(fig2)

        st.markdown("""
        <div class="model-note">
          Conceptual decision-support prototype. Hydrologic effects from literature-derived NBS parameters.
          Flood extent uses local OSM street/building/waterway layers + outlet-driven slope proxy.
          Interpret results comparatively across scenarios, not as calibrated hydraulic prediction.
        </div>
        """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

# ── OSM status (collapsed) ───────────────────────────────────
with st.expander("Data & model status", expanded=False):
    if use_fallback:
        st.warning("Fallback synthetic spatial grid active — OSM could not be loaded.")
    else:
        st.success(f"OSM layers loaded · roads={spatial['roads_count']}, "
                   f"buildings={spatial['buildings_count']}, waterways={spatial['water_count']}")
    st.caption(f"Watershed: {safe(watershed_row.get('name'))} · "
               f"CN={safe(watershed_row.get('curve_number'))} · "
               f"Ia={safe(watershed_row.get('initial_abstraction_mm'))} mm · "
               f"Area={safe(watershed_row.get('area_km2'))} km²")
