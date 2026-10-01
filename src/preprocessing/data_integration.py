import re
import os
from pathlib import Path
from zipfile import Path as ZipPath
import pandas as pd
import geopandas as gpd
import logging
from datetime import datetime
import xarray as xr

from collections import defaultdict
from src.preprocessing.count_record_grdc import count_records_2020
from tqdm import tqdm
from shapely import wkt
# Add project root to path for src imports
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.utils.utils import *
import regionmask
from collections import defaultdict



# Define Base Paths
# BASE_DIR assumes script is in src/preprocessing relative to project root
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"

GRDC_DIR = str(DATA_DIR / "grdc")
PRECIPITATION_NC_DIR = str(DATA_DIR / "precipitation")

TIGGE_GRIB_MERGE = str(DATA_DIR / "g" / "2025_01.grib")
TIGGE_GRIB = str(DATA_DIR / "tigge2.grib")
TIGGE_GRIB_DIR = str(DATA_DIR / "g")

RAIN_DIR = str(DATA_DIR / "precipitation_grid_last")
ERA5_DAILY_DIR = str(DATA_DIR / "era5_merged")

HYDRPO_ATLAS_GPKG = str(DATA_DIR / "BasinATLAS_v10_lev08.csv")
HYDROBASIN_DIR = str(DATA_DIR / "HydroBasin.csv")

OUTPUT_DIR = str(BASE_DIR / "output")
STEP1_DIR = os.path.join(OUTPUT_DIR, "step1")
STEP2_DIR = os.path.join(OUTPUT_DIR, "step2")
STEP2B_DIR = os.path.join(OUTPUT_DIR, "step2b")
STEP3_DIR = os.path.join(OUTPUT_DIR, "step3")
STEP3B_DIR = os.path.join(OUTPUT_DIR, "step3b")
STEP4_DIR = os.path.join(OUTPUT_DIR, "step4")
STEP4B_DIR = os.path.join(OUTPUT_DIR, "step4b")
STEP4C_DIR = os.path.join(OUTPUT_DIR, "step4c")
FINAL_DIR = os.path.join(OUTPUT_DIR, "final")

os.makedirs(STEP1_DIR, exist_ok=True)
os.makedirs(STEP2_DIR, exist_ok=True)
os.makedirs(STEP2B_DIR, exist_ok=True)
os.makedirs(STEP3_DIR, exist_ok=True)
os.makedirs(STEP3B_DIR, exist_ok=True)
os.makedirs(STEP4_DIR, exist_ok=True)
os.makedirs(STEP4B_DIR, exist_ok=True)
os.makedirs(STEP4C_DIR, exist_ok=True)
os.makedirs(FINAL_DIR, exist_ok=True)

STEP1_UPSTREAM_CSV = os.path.join(STEP1_DIR, "grdc_station_upstream.csv")
STEP1_STATIC_CSV = os.path.join(STEP1_DIR, "grdc_upstream_static_features.csv")
STEP2_RAIN_MAP = os.path.join(STEP2_DIR, "grdc_rain_station_mapping.csv")
STEP3_PRCP_NOAA_CSV = os.path.join(STEP3_DIR, "grdc_upstream_precipitation_daily.csv")
STEP3_DISCHARGE_CSV = os.path.join(STEP3_DIR, "grdc_upstream_discharge_daily.csv")
STEP4_ECMWF_CSV = os.path.join(STEP4_DIR, "grdc_upstream_ecmwf_daily.csv")


STEP2B_DISCHARGE_CSV = os.path.join(STEP2B_DIR,"grdc_discharge.csv")
STEP3B_precipitaion_csv = os.path.join(STEP3B_DIR, "precipitation_extraction.csv")
STEP4B_ECMWF_CSV = os.path.join(STEP4B_DIR, "grdc_upstream_ecmwf_daily.csv")
STEP4B_ECMWF_opt_CSV = os.path.join(STEP4B_DIR, "grdc_upstream_ecmwf_daily_optimized.csv")
STEP4B_ECMWF_opt2_CSV = os.path.join(STEP4B_DIR, "grdc_upstream_ecmwf_daily_optimized2.csv")

STEP4C_ECMWF_CSV = os.path.join(STEP4C_DIR, "grdc_upstream_ecmwf_daily.csv")


FINAL_DISCHARGE_CSV = os.path.join(FINAL_DIR, "final_discharge_precip_ecmwf.csv")
FINAL_HINDCAST_CSV = os.path.join(FINAL_DIR, "hindcast__precip_ecmwf.csv")
FINAL_FORECAST_CSV = os.path.join(FINAL_DIR, "forecast_Precip_ecmwf.csv")

DIST_THRESH = 500
AREA_TOL = 0.2

PHYSICAL_LIMITS = {
    "t2m": (-90.0, 60.0),        
    "sp":  (30.0, 120.0),       
}


def build_grdc_upstream_mapping(
    grdc_dir,
    output_csv=STEP1_UPSTREAM_CSV,
    static_out_csv=STEP1_STATIC_CSV,
    hydoAtlas= HYDRPO_ATLAS_GPKG,
    hydroBasin = AGGREGATE_BASIN,
):
    # Loading data from GRDC files
    records = []
    warnings = [] 
    
    for fname in tqdm(
        os.listdir(grdc_dir),
        desc="STEP 1.1 – Reading GRDC headers",
        unit="file"
    ):
        fp = os.path.join(grdc_dir, fname)

        if not os.path.isfile(fp):
            continue

        try:
            header_lines, _, _ = read_file(fp)
            meta = parse_grdc_header(header_lines)

            if meta["lat"] is None or meta["lon"] is None:
                raise ValueError("Missing lat/lon")

            records.append(meta)

        except Exception as e:
            msg = f"[WARNING] Skip GRDC file {fname}: {e}"
            logging.warning(msg)
            warnings.append(msg)

    grdc_df = pd.DataFrame(records)
    grdc_df = grdc_df[["grdc_no", "lat", "lon", "catchment"]]
    
    if warnings:
        logging.warning(f"Skip GRDC file {fname}: {e}")
        logging.warning("WARNING LIST:")
        for w in warnings[:10]:
            logging.warning(w)
        if len(warnings) > 10:
            logging.warning(f"... ({len(warnings) - 10} more warnings)")
    
    # Aggregate HydroBasin to csv
    hydrobasin_df = aggregate_basin(
        input_dir= HYDROBASIN_DIR,
        output_csv=AGGREGATE_BASIN
    )
    
    # Load HydroATLAS and HydroBasins
    hydro_atlas_gdf = load_hydroatlas(hydoAtlas)
    hydro_basin_gdf = load_hydrobasins(hydroBasin) 
    basin_area_df = hydro_basin_gdf[["hybas_id", "sub_area","geometry"]].copy()
    hybas_geom = dict(
        zip(
            hydro_basin_gdf.hybas_id,
            hydro_basin_gdf.geometry
        )
    )

    # Assign GRDC to HydroATLAS basins
    grdc_gdf = gpd.GeoDataFrame(
        grdc_df,
        geometry=gpd.points_from_xy(grdc_df.lon, grdc_df.lat),
        crs="EPSG:4326"
    )
    
    joined = gpd.sjoin(
        grdc_gdf,
        hydro_atlas_gdf[["hybas_id", "up_area", "geometry"]],
        how="left",
        predicate="within"
    )
    
    results = []
    n_no_basin = 0
    n_ambiguous = 0

    for grdc_no, grp in joined.groupby("grdc_no"):

        if grp.hybas_id.isna().all():
            n_no_basin += 1
            continue

        candidates = grp.dropna(subset=["hybas_id"])

        if len(candidates) == 1:
            chosen = candidates.iloc[0]
        else:
            n_ambiguous += 1
            chosen = candidates.sort_values("up_area").iloc[0]

        results.append({
            "grdc_no": grdc_no,
            "lat": chosen.lat,
            "lon": chosen.lon,
            "catchment": chosen.catchment,
            "local_hybas_id": int(chosen.hybas_id),
            "local_up_area": chosen.up_area,
            
        })

    local_df = pd.DataFrame(results)
    
    total_assigned = len(local_df)
    total_grdc = joined["grdc_no"].nunique()

    logging.info(f"Total assigned stations to HydroATLAS basin: {total_assigned}")
    logging.info(f"Total unassigned stations: {n_no_basin}")
    logging.info(f"Ambiguous (border) stations: {n_ambiguous}")

    # Validate upstream area
    validated_df = validate_catchmentarea_uparea(
        local_df=local_df,
        area_tol= AREA_TOL
    )
    n_multi = (
        validated_df
        .groupby("local_hybas_id")["grdc_no"]
        .nunique()
        .gt(1)
        .sum()
    )
    logging.info(f"Local basins with >1 GRDC: {n_multi} ")
    # Trace upstream basins
    rev_graph = build_reverse_graph(hydro_basin_gdf)
    upstream_records = []

    for _, row in validated_df.iterrows():
        grdc_no = row.grdc_no
        local_id = int(row.local_hybas_id)

        upstream_ids = trace_upstream_basins(
            start_hybas_id=local_id,
            rev_graph=rev_graph
        )

        for hid in upstream_ids:
            upstream_records.append({
                "grdc_no": grdc_no,
                "hybas_id": hid,
                "is_local": 1 if hid == local_id else 0
            })

    upstream_map_df = pd.DataFrame(upstream_records)
    upstream_map_df = upstream_map_df.merge(
        basin_area_df,
        on="hybas_id",
        how="left"
    )
    upstream_map_df["geometry_wkt"] = upstream_map_df.geometry.apply(
        lambda g: g.wkt if g is not None else None
    )
    upstream_map_df = upstream_map_df.drop(columns="geometry")

    area_check_df = validate_upstream_totalarea(
        upstream_map_df=upstream_map_df,
        validated_df=validated_df,
        area_tol=0.1
    )
    
    upstream_map_df.to_csv(output_csv, index=False)

    # Extract static upstream 
    static_df = aggregate_static_atlas_from_upstream_gdf(
        upstream_map_df=upstream_map_df,
        hydro_atlas_gdf=hydro_atlas_gdf,
        out_csv=static_out_csv
    )
    
    return upstream_map_df

def build_reverse_graph(hydrobasins_gdf):
    """
    Build reverse river graph from HydroBASINS NEXT_DOWN.
    Return: dict[downstream_id] -> list[upstream_ids]
    """
    rev = defaultdict(list)

    for _, row in hydrobasins_gdf.iterrows():
        down = row.next_down
        up = row.hybas_id

        if pd.notna(down) and down > 0:
            rev[int(down)].append(int(up))

    return rev

def trace_upstream_basins(
    start_hybas_id: int,
    rev_graph: dict
):
    """
    Trace all upstream basins from a starting HYBAS_ID
    using DFS (loop-safe).
    """
    visited = set()
    stack = [int(start_hybas_id)]

    while stack:
        cur = stack.pop()
        if cur in visited:
            continue

        visited.add(cur)
        stack.extend(rev_graph.get(cur, []))

    return visited

def validate_upstream_totalarea(
    upstream_map_df: pd.DataFrame,
    validated_df: pd.DataFrame,
    area_tol: float = 0.2
):
    """
    Validate conservation of area:
    Sum of SUB_AREA (upstream basins) ≈ local UP_AREA

    Parameters
    ----------
    upstream_map_df : DataFrame
        Columns: grdc_no, hybas_id, sub_area
    validated_df : DataFrame
        Columns: grdc_no, local_hybas_id, local_up_area
    area_tol : float
        Relative tolerance (default 20%)
    """

    # sum SUB_AREA per GRDC
    sum_sub = (
        upstream_map_df
        .groupby("grdc_no")["sub_area"]
        .sum()
        .reset_index(name="sum_sub_area")
    )

    # attach local UP_AREA
    check_df = sum_sub.merge(
        validated_df[["grdc_no", "local_hybas_id", "local_up_area"]],
        on="grdc_no",
        how="left"
    )

    n_warn = 0

    for _, row in check_df.iterrows():
        sum_area = row.sum_sub_area
        up_area = row.local_up_area

        if pd.isna(sum_area) or pd.isna(up_area) or up_area <= 0:
            continue

        rel_diff = abs(sum_area - up_area) / up_area

        if rel_diff > area_tol:
            n_warn += 1
            logging.warning(
                f"Area mismatch for local basin {int(row.local_hybas_id)} "
                f"(GRDC {row.grdc_no}): "
                f"ΣSUB_AREA={sum_area:.2f}, UP_AREA={up_area:.2f}, "
                f"diff={rel_diff*100:.1f}%"
            )


    if n_warn > 0:
        logging.warning(f"({n_warn} warnings)")

    return check_df

def validate_catchmentarea_uparea(
    local_df: pd.DataFrame,
    area_tol: float = 0.2
):
    """
    Validate GRDC stations using upstream area consistency.

    Rules
    -----
    1. Auto-pass if catchment is missing or equals -999
    2. Reject if local_up_area is missing or <= 0
    3. Reject if relative difference > area_tol
    """

    valid_rows = []

    n_auto_pass = 0
    n_area_reject = 0
    n_missing_uparea = 0

    for _, row in local_df.iterrows():
        catchment = row.catchment
        up_area = row.local_up_area

        # Auto-pass: GRDC has no catchment info
        if pd.isna(catchment) or catchment in [-999, -999.0, -999.00]:
            n_auto_pass += 1
            valid_rows.append(row)
            continue

        # Reject: missing or invalid UP_AREA
        if pd.isna(up_area) or up_area <= 0:
            n_missing_uparea += 1
            continue

        # Reject: area mismatch
        rel_diff = abs(up_area - catchment) / catchment
        if rel_diff > area_tol:
            n_area_reject += 1
            continue

        # Valid
        valid_rows.append(row)

    validated_df = pd.DataFrame(valid_rows)

    logging.info(f"Valid stations after validation: {len(validated_df)}")
    logging.info(f"Auto-pass (no catchment)       : {n_auto_pass}")
    logging.info(f"Rejected (area mismatch)       : {n_area_reject}")
    logging.info(f"Rejected (missing UP_AREA)     : {n_missing_uparea}")

    return validated_df







def assign_rain_stations_to_basins(
    step1_upstream_csv =STEP1_UPSTREAM_CSV,
    rain_dir="precipitation",
    out_csv=STEP2_RAIN_MAP,
):
    # Load GRDC station upstream mapping
    step1 = pd.read_csv(
        step1_upstream_csv,
        dtype={"grdc_no": str}
    )
    needed_hybas  = set(step1.hybas_id.astype(int))
    
    # Load HydroATLAS geometry filtered by set of id in step 1
    hydro_gdf = load_hydroatlas_filtered(
        HYDRPO_ATLAS_GPKG,
        needed_hybas
    )

    logging.info(f"Total unique assigned hydro basins: {len(hydro_gdf)}")

    rain_meta = load_rain_station_metadata(rain_dir)
    rain_df = pd.DataFrame(rain_meta)

    rain_gdf = gpd.GeoDataFrame(
        rain_df,
        geometry=gpd.points_from_xy(rain_df.lon, rain_df.lat),
        crs="EPSG:4326"
    )
    logging.info(f"Total rain stations loaded: {len(rain_gdf)}")
    
    # Spatial join rain stations to HydroATLAS basins
    rain_in_basin = gpd.sjoin(
        rain_gdf,
        hydro_gdf[["hybas_id", "geometry"]],
        how="inner",
        predicate="within"
    )
    logging.info(f"Rain stations inside upstream basins: {len(rain_in_basin)}")

    records = []

    total_hybas = set(step1["hybas_id"].astype(int))
    skipped_hybas = set()
    for _, r in step1.iterrows():
        grdc_no = r.grdc_no
        hybas_id = int(r.hybas_id)
        is_local = int(r.is_local)
        sub_area = r.sub_area

        sub = rain_in_basin[rain_in_basin.hybas_id == hybas_id]

        if sub.empty:
            skipped_hybas.add(hybas_id)
            # logging.warning(
            #     "HYBAS %s (GRDC %s) has NO rain station → skipped",
            #     hybas_id, grdc_no
            # )
            continue

        for _, rr in sub.iterrows():
            records.append({
                "grdc_no": grdc_no,
                "hybas_id": hybas_id,
                "is_local": is_local,
                "sub_area": sub_area,
                "rain_station_id": rr.pre_no,
            })
            
    if skipped_hybas:
        logging.warning(
            "Skipped %d / %d HYBAS basins due to no rain station (%.1f%%)",
            len(skipped_hybas),
            len(total_hybas),
            100.0 * len(skipped_hybas) / len(total_hybas)
        )
        
    out_df = pd.DataFrame(records)
    out_df.to_csv(out_csv, index=False)
    basin_rain_cnt = (
        out_df
        .groupby("hybas_id")["rain_station_id"]
        .nunique()
    )      
    # Basin có > 2 trạm mưa
    basins_gt_2 = basin_rain_cnt[basin_rain_cnt >= 2]

    logging.info(f"Basins with >= 2 rain stations: {len(basins_gt_2)}")
    logging.info(f"Total mappings from rain station to GRDC station: {len(out_df)}")

    total_grdc = step1["grdc_no"].nunique()
    assigned_grdc = out_df["grdc_no"].nunique()

    logging.info(
        "GRDC stations assigned with rain data: %d / %d (%.1f%%)",
        assigned_grdc,
        total_grdc,
        100.0 * assigned_grdc / total_grdc if total_grdc > 0 else 0.0
    )
    return out_df

def aggregate_static_atlas_from_upstream_gdf(
    upstream_map_df: pd.DataFrame,
    hydro_atlas_gdf: gpd.GeoDataFrame,
    out_csv: str,
):
    """
    Aggregate HydroATLAS static attributes over upstream basins
    using SUB_AREA as weights.
    """

    static_cols = [
        "hybas_id",
        "ele_mt_uav",
        "slp_dg_uav",
        "cly_pc_uav",
        "snd_pc_uav",
        "slt_pc_uav",
        "for_pc_use",
        "urb_pc_use",
        "crp_pc_use",
        "ari_ix_uav",     
        "pre_mm_uyr",    
    ]

    hydro_static = hydro_atlas_gdf[static_cols].copy()

    df = upstream_map_df.merge(
        hydro_static,
        on="hybas_id",
        how="left"
    )

    records = []

    for grdc_no, grp in tqdm(
        df.groupby("grdc_no"),
        desc="STEP 1B – Aggregate HydroATLAS static"
    ):
        total_area = grp.sub_area.sum()

        if pd.isna(total_area) or total_area <= 0:
            continue

        rec = {
            "grdc_no": grdc_no,
            "area": total_area,   # km²
        }

        # --- intensive variables
        rec["mean_elevation"] = (grp.ele_mt_uav * grp.sub_area).sum() / total_area
        rec["mean_slope"]     = (grp.slp_dg_uav * grp.sub_area).sum() / total_area
        
        # --- fractional variables (%)
        rec["clay_fraction"]  = (grp.cly_pc_uav * grp.sub_area).sum() / total_area
        rec["sand_fraction"]  = (grp.snd_pc_uav * grp.sub_area).sum() / total_area
        rec["silt_fraction"]  = (grp.slt_pc_uav * grp.sub_area).sum() / total_area

        rec["forest_cover"]   = (grp.for_pc_use * grp.sub_area).sum() / total_area
        rec["urban_cover"]    = (grp.urb_pc_use * grp.sub_area).sum() / total_area
        rec["cropland_cover"] = (grp.crp_pc_use * grp.sub_area).sum() / total_area
        rec["aridity_index"]  = (grp.ari_ix_uav * grp.sub_area).sum() / total_area
        rec["mean_precip"]    = (grp.pre_mm_uyr * grp.sub_area).sum() / total_area

        records.append(rec)

    out_df = pd.DataFrame(records)

    # Làm đẹp khi output
    out_df = out_df.round({
        "area": 1,
        "mean_elevation": 1,
        "mean_slope": 2,
        "clay_fraction": 2,
        "sand_fraction": 2,
        "silt_fraction": 2,
        "forest_cover": 2,
        "urban_cover": 2,
        "cropland_cover": 2,
        "aridity_index": 3,
        "mean_precip": 1,
    })
    out_df.to_csv(out_csv, index=False)
    return out_df


def build_topology(step2_csv):
    """
    Return:
      grdc_to_hybas: dict[grdc_no] -> set(hybas_id)
      hybas_to_rain: dict[hybas_id] -> set(rain_station_id)
      hybas_area   : dict[hybas_id] -> sub_area
    """
    df = pd.read_csv(step2_csv)

    grdc_to_hybas = {}
    hybas_to_rain = {}
    hybas_area = {}

    for _, r in df.iterrows():
        g = str(r.grdc_no)
        h = int(r.hybas_id)
        s = str(r.rain_station_id)

        grdc_to_hybas.setdefault(g, set()).add(h)
        hybas_to_rain.setdefault(h, set()).add(s)

        # sub_area giống nhau cho mọi dòng cùng hybas
        if h not in hybas_area:
            hybas_area[h] = r.sub_area

    return grdc_to_hybas, hybas_to_rain, hybas_area

def compute_hybas_daily_precip(
    hybas_to_rain,
    station_daily
):
    """
    Return:
      hybas_daily[hybas_id][date] = mean precipitation
    """

    hybas_daily = {}

    for h, stations in tqdm(
        hybas_to_rain.items(),
        total=len(hybas_to_rain),
        desc="Computing HYBAS daily precipitation"
    ):
        daily_vals = {}

        for s in stations:
            for d, v in station_daily.get(s, {}).items():
                daily_vals.setdefault(d, []).append(v)

        hybas_daily[h] = {
            d: sum(vals) / len(vals)
            for d, vals in daily_vals.items()
        }

    return hybas_daily

def compute_grdc_daily_precip(
    grdc_to_hybas,
    hybas_daily,
    hybas_area
):
    """
    Area-weighted upstream precipitation per GRDC.

    Return:
      DataFrame(grdc_no, date, prcp_upstream_mm)
    """
    records = []

    for grdc_no, hybas_list in tqdm(
        grdc_to_hybas.items(),
        desc="Computing GRDC upstream precipitation",
        total=len(grdc_to_hybas)
    ):
        daily_num = {}
        daily_den = {}

        for h in hybas_list:
            area = hybas_area.get(h)
            if area is None or area <= 0:
                continue

            for d, v in hybas_daily.get(h, {}).items():
                daily_num[d] = daily_num.get(d, 0.0) + v * area
                daily_den[d] = daily_den.get(d, 0.0) + area

        for d in daily_num:
            if daily_den[d] > 0:
                records.append({
                    "grdc_no": grdc_no,
                    "date": d,
                    "prcp_upstream_mm": daily_num[d] / daily_den[d]
                })

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    return df
def process_discharge_and_precipitation(
    start_date,
    end_date,
    out_discharge_csv,
    out_precip_csv,
    rain_root_dir="precipitation",
    grdc_dir=GRDC_DIR,
    step2_map_csv=STEP2_RAIN_MAP,
):
    # Build topology from STEP 2
    grdc_to_hybas, hybas_to_rain, hybas_area = build_topology(step2_map_csv)
    station_ids = {s for v in hybas_to_rain.values() for s in v}

    # Load rain station daily data
    station_daily = load_rain_station_daily(
        station_ids,
        rain_dir=rain_root_dir,
        start_date="2000-01-01",
        end_date="2025-12-31"
    )

    # Average precipitation per HYBAS
    hybas_daily = compute_hybas_daily_precip(
        hybas_to_rain,
        station_daily
    )
    
    # Area-weighted upstream precipitation per GRDC
    prcp_df = compute_grdc_daily_precip(
        grdc_to_hybas,
        hybas_daily,
        hybas_area
    )
    
    # Load GRDC discharge data
    discharge_df = load_grdc_discharge_df(
        grdc_dir=grdc_dir,
        valid_grdc_ids=set(grdc_to_hybas.keys()),
        start_date=start_date,
        end_date=end_date
    )

    # Merge precipitation with discharge
    final_df, missing_stats = merge_with_discharge(
        prcp_df,
        discharge_df
    )
    final_df = final_df.rename(columns={"discharge_cms": "q_obs"})
    final_df = final_df.rename(columns={"prcp_upstream_mm": "prcp_NOAA"})
    logging.info(f"Final columns: {final_df.columns.tolist()}")


    # Save discharge
    final_df[["date", "grdc_no", "q_obs"]].to_csv(
        out_discharge_csv, index=False
    )

    # Save NOAA precipitation
    final_df[["date", "grdc_no", "prcp_NOAA"]].to_csv(
        out_precip_csv, index=False
    )

    return final_df, missing_stats

def extract_discharge_data(
    start_date,
    end_date,
    out_csv = STEP2B_DISCHARGE_CSV,
    grdc_dir = GRDC_DIR,
    step1_csv=STEP1_UPSTREAM_CSV,

):
    step1 = pd.read_csv(step1_csv, dtype={"grdc_no": str})
    valid_grdc_ids = set(step1["grdc_no"].unique())
    
    discharge_df = load_grdc_discharge_df(
        grdc_dir=grdc_dir,
        valid_grdc_ids=valid_grdc_ids,
        start_date=start_date,
        end_date=end_date
    )
    discharge_df = discharge_df.rename(
        columns={"discharge_cms": "q_obs"}
    )

    discharge_df = discharge_df[
            ["date", "grdc_no", "q_obs"]
        ].copy()
    
    discharge_df.to_csv(out_csv, index=False)
    
    return discharge_df

# def get_required_dates_from_discharge(discharge_csv):
#     df = pd.read_csv(discharge_csv, parse_dates=["date"])
#     dates = sorted(df["date"].unique())
#     return dates

def extract_precipitation_data(
    precipitation_dir = PRECIPITATION_NC,
    discharge_csv = STEP2B_DISCHARGE_CSV,
    step1_upstream = STEP1_UPSTREAM_CSV,
    out_csv = STEP3B_precipitaion_csv,
    precip_var = "precip",
):
    discharge_df = pd.read_csv(discharge_csv, parse_dates=["date"])
    dates = sorted(discharge_df["date"].unique())
    t_start = min(dates)
    t_end   = max(dates)

    logging.info(
        "STEP3B: Extract precipitation for %d unique dates",
        len(dates)
    )
    
    step1 = pd.read_csv(step1_upstream)
    step1["geometry"] = step1["geometry_wkt"].apply(wkt.loads)
    gdf = gpd.GeoDataFrame(step1, geometry="geometry", crs="EPSG:4326")
    lon_min, lat_min, lon_max, lat_max = gdf.total_bounds
    
    logging.info(
        "STEP3B: Loaded %d upstream basin geometries",
        len(gdf)
    )

    lat_slice = slice(lat_max, lat_min)
    lon_min_ = lon_min % 360
    lon_max_ = lon_max % 360

    nc_files = sorted(
        os.path.join(precipitation_dir, f)
        for f in os.listdir(precipitation_dir)
        if f.endswith(".nc")
    )

    ds = xr.open_mfdataset(
        nc_files,
        combine="by_coords"
    )

    ds = ds.sel(
        time=slice(t_start, t_end),
        lat=lat_slice,
        lon=slice(lon_min_, lon_max_),
    )

    ds = ds.chunk({"time": 30, "lat": 200, "lon": 200})

    regions = regionmask.Regions(
        outlines=list(gdf.geometry),
        numbers=gdf.hybas_id.values,  
    )

    basin_mask = regions.mask(ds)
        
    basin_precip = (
        ds[precip_var]
        .groupby(basin_mask)
        .mean(dim="stacked_lat_lon", skipna=True)
    )

    basin_precip = basin_precip.rename({"mask": "hybas_id"})
    basin_precip = basin_precip.compute()

    df_basin = (
        basin_precip
        .to_dataframe(name="precip")
        .reset_index()
        .rename(columns={"time": "date"})
    )

    df_basin = df_basin.merge(
        gdf[["hybas_id", "grdc_no", "sub_area"]],
        on="hybas_id",
        how="left"
    )

    df_grdc = (
        df_basin
        .dropna(subset=["precip"])
        .groupby(["grdc_no", "date"])
        .apply(
            lambda g: (g.precip * g.sub_area).sum() / g.sub_area.sum()
        )
        .reset_index(name="prep_NOAA")
    )
    
    discharge_keys = discharge_df[["grdc_no", "date"]].drop_duplicates()

    df_grdc = df_grdc.merge(
        discharge_keys,
        on=["grdc_no", "date"],
        how="inner"
    )

    df_grdc["prep_NOAA"] = df_grdc["prep_NOAA"].round(2)
    df_grdc.to_csv(out_csv, index=False)

    logging.info(
        "STEP3B:Total records of precipitation:  %d ",
        len(df_grdc)
    )
    
    return df_grdc






def merge_with_discharge(
    prcp_df,
    discharge_df
):
    """
    Merge upstream precipitation with GRDC discharge.

    - Keep ALL discharge records
    - Count days with discharge but missing upstream precipitation
    - DROP those days from output dataframe

    Return:
      df_clean, missing_stats
    """

    # --------------------------------------------------
    # Merge: discharge is the reference
    # --------------------------------------------------
    df = discharge_df.merge(
        prcp_df,
        on=["grdc_no", "date"],
        how="left"
    )

    # --------------------------------------------------
    # Identify days with Q but missing P
    # --------------------------------------------------
    missing_mask = (
        df["discharge_cms"].notna() &
        df["prcp_upstream_mm"].isna()
    )

    # --------------------------------------------------
    # Count missing precipitation days per GRDC
    # --------------------------------------------------
    missing_stats = (
        df[missing_mask]
        .groupby("grdc_no")
        .size()
        .rename("n_missing_precip")
        .reset_index()
    )

    # --------------------------------------------------
    # DROP days without upstream precipitation
    # --------------------------------------------------
    df_clean = df[~missing_mask].copy()

    return df_clean, missing_stats

def validate_discharge_precipitation_output(
    in_csv,
    out_valid_csv
):
    """
    Validate STEP 3 output.

    Rules:
    - Drop rows with missing discharge
    - Drop rows with zero upstream precipitation

    Output:
    - Save validated CSV
    - Print statistics
    """
    logging.info("[VALIDATE] Loading STEP 3 output...")
    df = pd.read_csv(in_csv)

    n_total = len(df)

    # --- invalid conditions
    mask_no_discharge = df["discharge_cms"].isna()
    mask_zero_prcp = df["prcp_upstream_mm"] == 0

    n_no_discharge = mask_no_discharge.sum()
    n_zero_prcp = mask_zero_prcp.sum()

    # --- combine
    invalid_mask = mask_no_discharge | mask_zero_prcp
    n_invalid = invalid_mask.sum()

    df_valid = df[~invalid_mask].copy()

    n_valid = len(df_valid)

    # --- stats
    logging.info("VALIDATE STATS:")
    logging.info(f"Total records          : {n_total}")
    logging.info(f"Removed (no discharge) : {n_no_discharge}")
    logging.info(f"Removed (zero precip)  : {n_zero_prcp}")
    logging.info(f"Total removed          : {n_invalid}")
    logging.info(f"Remaining valid        : {n_valid}")
    logging.info(f"Retention rate (%%)     : {n_valid / n_total * 100:.2f}")

    # --- save
    df_valid.to_csv(out_valid_csv, index=False)

    logging.info(f"VALIDATE DONE. Valid data saved → {out_valid_csv}")

    return df_valid





def extract_ecmwf_data_optimized(
    out_csv=STEP4B_ECMWF_CSV,
    tigge=TIGGE_GRIB,
    precipitation=STEP3B_precipitaion_csv,
    step1_upstream_csv=STEP1_UPSTREAM_CSV,
):
    # ============================================================
    # 1. LOAD REQUIRED DATES (from STEP3B)
    # ============================================================
    step3 = pd.read_csv(precipitation, parse_dates=["date"])
    dates = sorted(step3["date"].unique())

    # # # giữ nguyên logic debug hiện tại
    t_start = pd.Timestamp("2025-01-01")
    t_end   = pd.Timestamp("2025-01-31")
    # t_start = dates[0]
    # t_end   = dates[-1]

    logging.info(
        "STEP4B: Extract ECMWF for %d days (%s → %s)",
        len(dates), t_start.date(), t_end.date()
    )

    # ============================================================
    # 2. LOAD STEP1 + GEOMETRY (ONCE)
    # ============================================================
    step1 = pd.read_csv(step1_upstream_csv)

    

    step1["geometry"] = step1["geometry_wkt"].apply(wkt.loads)
    step1["geometry_shift"] = step1.geometry.apply(shift_geom_lon_360)

    gdf = gpd.GeoDataFrame(step1, geometry="geometry", crs="EPSG:4326")

    lon_min, lat_min, lon_max, lat_max = gdf.total_bounds
    bbox = (lon_min, lon_max, lat_min, lat_max)

    logging.info(
        "STEP4B: BBOX lon=[%.2f, %.2f], lat=[%.2f, %.2f]",
        lon_min, lon_max, lat_min, lat_max
    )

    # ============================================================
    # 3. OPEN TIGGE DATA (ONCE)
    # ============================================================
    ds_daily = open_tigge_daily(tigge, bbox, t_start, t_end)

    ds_acc   = open_tigge_acc(tigge, bbox, t_start, t_end)
    ds_daily = ds_daily.chunk({"time": 30})
    ds_acc   = ds_acc.chunk({"time": 30})

    # ============================================================
    # 4. BUILD REGION MASK (2D ONLY – OPTIMIZED)
    # ============================================================
    regions = regionmask.Regions(
        outlines=step1.geometry_shift,
        numbers=step1.hybas_id.values,
        overlap=True,
    )

    mask_daily_3d = regions.mask_3D(ds_daily.isel(time=0))
    mask_acc_3d   = regions.mask_3D(ds_acc.isel(time=0))

    # ============================================================
    # 5. HYBAS-LEVEL TIME SERIES (VECTORIZED)
    # ============================================================
    logging.info("STEP4B: Computing HYBAS time series (streaming)")
    daily_surface_vars = tuple(
        v for v in ("sp",) if v in ds_daily.data_vars
    )
    daily_hagl_vars = tuple(
        v for v in ("t2m",) if v in ds_daily.data_vars
    )
    acc_vars   = tuple(v for v in ("ssr", "str", "tp") if v in ds_acc.data_vars)

    hybas_daily_list = []
    hybas_acc_list   = []

    time_blocks = pd.date_range(
        ds_daily.time.min().values,
        ds_daily.time.max().values,
        freq="30D"
    )

    if len(time_blocks) < 2:
        time_blocks = [
            ds_daily.time.min().values,
            ds_daily.time.max().values,
        ]

    for t0, t1 in zip(time_blocks[:-1], time_blocks[1:]):
        logging.info(
            "  HYBAS block %s → %s",
            str(t0)[:10],
            str(t1)[:10],
        )
        ds_d = ds_daily.sel(time=slice(t0, t1))
        ds_a = ds_acc.sel(time=slice(t0, t1))

        # --- DAILY ---
        num_d = (
            ds_d[list(daily_surface_vars)]
            .expand_dims(region=mask_daily_3d.region)
            * mask_daily_3d
        ).sum(dim=("lat", "lon"), skipna=True)

        
        den_d = mask_daily_3d.sum(dim=("lat", "lon"))
        hybas_d = num_d / den_d

        den_d = mask_daily_3d.sum(dim=("lat", "lon"))
        hybas_d = num_d / den_d

        hybas_d = hybas_d.assign_coords(
            hybas_id=("region", mask_daily_3d.region.values)
        ).swap_dims({"region": "hybas_id"})
        
        # --- ACC ---
        hybas_a = (
            ds_a[list(acc_vars)]
            .expand_dims(region=mask_acc_3d.region)
            * mask_acc_3d
        ).sum(dim=("lat", "lon"), skipna=True)

        hybas_a = hybas_a.assign_coords(
            hybas_id=("region", mask_acc_3d.region.values)
        ).swap_dims({"region": "hybas_id"})

        hybas_daily_list.append(hybas_d)
        hybas_acc_list.append(hybas_a)

    # concat theo time
    hybas_daily = xr.concat(hybas_daily_list, dim="time")
    hybas_acc   = xr.concat(hybas_acc_list, dim="time")

    hybas_ds = xr.merge(
        [hybas_d, hybas_a],
        compat="override",
        join="outer"   # ✅ explicit
    )

    if "t2m" in ds_daily:
        logging.info("STEP4B: Aggregating t2m to HYBAS")

        num_t2m = (
            ds_daily["t2m"]
            .expand_dims(region=mask_daily_3d.region)
            * mask_daily_3d
        ).sum(dim=("lat", "lon"), skipna=True)

        den_t2m = mask_daily_3d.sum(dim=("lat", "lon"))

        hybas_t2m = (num_t2m / den_t2m).rename("t2m")

        hybas_t2m = (
            hybas_t2m
            .assign_coords(hybas_id=("region", mask_daily_3d.region.values))
            .swap_dims({"region": "hybas_id"})
        )

        hybas_ds = xr.merge([hybas_ds, hybas_t2m], compat="override")

    # ============================================================
    # 6. AGGREGATE HYBAS → GRDC (AREA-WEIGHTED – GIỮ NGUYÊN LOGIC)
    # ============================================================
    logging.info("STEP4B: Aggregating HYBAS → GRDC")

    frames = []

    for grdc_no, grp in tqdm(
        step1.groupby("grdc_no"),
        desc="STEP4B | GRDC aggregation",
        unit="station"
    ):
        grp = grp.copy()
        grp["hybas_id"] = grp["hybas_id"].astype(int)
        grp["sub_area"] = grp["sub_area"].astype(float)

        grp = grp[
            (grp["sub_area"] > 0) &
            (grp["hybas_id"].isin(hybas_ds.hybas_id.values))
        ]

        if grp.empty:
            continue

        hy_ids = grp["hybas_id"].tolist()
        areas  = grp["sub_area"].values

        w = xr.DataArray(
            areas,
            coords={"hybas_id": hy_ids},
            dims=("hybas_id",)
        )
        w = w / w.sum()   # normalize ONCE

        ds_grdc = xr.Dataset()

        for v in hybas_ds.data_vars:
            sub = hybas_ds[v].sel(hybas_id=hy_ids)
            ts = (sub * w).sum(dim="hybas_id", skipna=True)

            if "step" in ts.dims:
                ts = ts.mean(dim="step", skipna=True)

            ts_daily = (
                ts.groupby("time.date")
                .mean(skipna=True)
                .rename({"date": "time"})
            )

            ts_daily = ts_daily.assign_coords(
                time=pd.to_datetime(ts_daily["time"].values)
            )

            ds_grdc[v] = ts_daily

        if not ds_grdc.data_vars:
            continue

        df_grdc = ds_grdc.to_dataframe().reset_index()
        df_grdc["grdc_no"] = grdc_no
        frames.append(df_grdc)

    df = pd.concat(frames, ignore_index=True)

    # ============================================================
    # 7. FINAL CLEAN & SAVE (GIỮ NGUYÊN)
    # ============================================================
    df = df.rename(columns={"time": "date"})
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()

    KEEP_COLS = ["date", "grdc_no", "t2m", "sp", "ssr", "str", "tp"]
    df = df[[c for c in KEEP_COLS if c in df.columns]]

    dup = df.duplicated(subset=["grdc_no", "date"])
    if dup.any():
        raise RuntimeError(
            f"Duplicate (grdc_no, date) detected: {dup.sum()} rows"
        )

    df = df.sort_values(["grdc_no", "date"])
    if "tp" in df.columns:
        df["tp"] = df["tp"] * 1000

    # ssr, str: J/m2 → MJ/m2
    for col in ["ssr", "str"]:
        if col in df.columns:
            df[col] = df[col] / 1e6


    # 2) Làm tròn số (khuyên dùng)
    round_map = {
        "sp": 1,     # Pa → 0.1 Pa
        "t2m": 2,    # °C
        "tp": 2,     # mm
        "ssr": 3,    # MJ/m2
        "str": 3,
    }

    for col, nd in round_map.items():
        if col in df.columns:
            df[col] = df[col].round(nd)
    df.to_csv(out_csv, index=False)
    logging.info("STEP4B: ECMWF extraction done, saved to %s", out_csv)

    return df


def extract_ecmwf_data_from_dir(
    grib_root_dir,
    output_root_dir,
    precipitation_csv,
    step1_upstream_csv,
):
    """
    Wrapper chạy extract_ecmwf_data_optimized
    cho TỪNG FILE GRIB trong thư mục.
    KHÔNG sửa logic bên trong extract_ecmwf_data_optimized.
    """

    os.makedirs(output_root_dir, exist_ok=True)

    # ------------------------------------------------------------
    # 1. LẤY DANH SÁCH FILE GRIB (GIỮ NGUYÊN CẤU TRÚC THƯ MỤC)
    # ------------------------------------------------------------
    grib_files = []
    for root, _, files in os.walk(grib_root_dir):
        for f in files:
            if f.lower().endswith(".grib"):
                grib_files.append(os.path.join(root, f))

    grib_files.sort()

    if not grib_files:
        raise RuntimeError(f"No GRIB files found in {grib_root_dir}")

    logging.info("STEP4B: Found %d GRIB files", len(grib_files))

    # ------------------------------------------------------------
    # 2. CHẠY TỪNG FILE – KHÔNG ĐỤNG LOGIC
    # ------------------------------------------------------------
    for grib_file in grib_files:
        # mapping output 1–1
        rel_path = os.path.relpath(grib_file, grib_root_dir)
        out_csv = os.path.join(
            output_root_dir,
            os.path.splitext(rel_path)[0] + ".csv"
        )
        os.makedirs(os.path.dirname(out_csv), exist_ok=True)

        logging.info("==============================================")
        logging.info("STEP4B: Processing %s", grib_file)
        logging.info("OUTPUT → %s", out_csv)

        try:
            extract_ecmwf_data_optimized(
                out_csv=out_csv,
                tigge=grib_file,
                precipitation=precipitation_csv,
                step1_upstream_csv=step1_upstream_csv,
            )

        except Exception:
            logging.exception("FAILED processing %s", grib_file)
            continue

from shapely.ops import transform

def shift_geom_lon_360(geom):
    def _shift(x, y, z=None):
        x = x % 360
        return (x, y) if z is None else (x, y, z)
    return transform(_shift, geom)


def open_tigge_daily(grib_path, bbox, t_start, t_end):
    logging.info("STEP4B: Open TIGGE DAILY vars [sp + t2m]")

    ds_surface = xr.open_dataset(
        grib_path,
        engine="cfgrib",
        backend_kwargs={
            "indexpath": "",
            "filter_by_keys": {"typeOfLevel": "surface", "shortName": ["sp"]},
        },
    )

    ds_t2m = xr.open_dataset(
        grib_path,
        engine="cfgrib",
        backend_kwargs={
            "indexpath": "",
            "filter_by_keys": {
                "typeOfLevel": "heightAboveGround",
                "level": 2,
                "shortName": ["2t"],
            },
        },
    )   
    ds = xr.merge([ds_surface, ds_t2m], compat="override")

    if "step" in ds.dims:
        ds = ds.isel(step=-1)
        if "valid_time" in ds.coords:
            ds = ds.assign_coords(time=ds["valid_time"])

    ds = ds.rename({"longitude": "lon", "latitude": "lat"})

    lon_min, lon_max, lat_min, lat_max = bbox
    if ds.lon.max() > 180:
        lon_min %= 360
        lon_max %= 360

    ds = ds.sel(
        lon=slice(lon_min, lon_max),
        lat=slice(lat_max, lat_min),
        time=slice(t_start, t_end),
    )

    logging.info("STEP4B: ds_daily vars = %s", list(ds.data_vars))
    return ds



def open_tigge_acc(grib_path, bbox, t_start, t_end):
    logging.info("STEP4B: Open TIGGE ACC vars [ssr, str, tp]")

    ds = xr.open_dataset(
        grib_path,
        engine="cfgrib",
        backend_kwargs={
            "indexpath": "",
            "filter_by_keys": {"shortName": ["ssr", "str", "tp"]},
        },
    )

    ds = ds.rename({"longitude": "lon", "latitude": "lat"})

    lon_min, lon_max, lat_min, lat_max = bbox

    if ds.lon.max() > 180:
        lon_min = lon_min % 360
        lon_max = lon_max % 360

    ds = ds.sel(
        lon=slice(lon_min, lon_max),
        lat=slice(lat_max, lat_min),
        time=slice(t_start, t_end),
    )

    if ds.lon.size == 0 or ds.lat.size == 0:
        raise ValueError("EMPTY GRID AFTER TIGGE ACC SLICE")

    return ds

def compute_hybas_timeseries(
    ds_daily,
    ds_acc,
    mask_daily_3d,
    mask_acc_3d,
    daily_vars,
    acc_vars,
):
    out = {}   # hybas_id -> {var -> DataArray(time)}

    region_ids = mask_daily_3d.region.values

    for i, hid in enumerate(region_ids):
        vals = {}

        m_daily = mask_daily_3d.isel(region=i)
        m_acc   = mask_acc_3d.isel(region=i)

        if ds_daily is not None:
            for v in daily_vars:
                if v in ds_daily:
                    sub = ds_daily[v].where(m_daily)
                    if sub.count() > 0:
                        vals[v] = sub.mean(
                            dim=("lat", "lon"), skipna=True
                        )

        if ds_acc is not None:
            for v in acc_vars:
                if v in ds_acc:
                    sub = ds_acc[v].where(m_acc)
                    if sub.count() > 0:
                        vals[v] = sub.sum(
                            dim=("lat", "lon"), skipna=True
                        )

        if vals:
            out[int(hid)] = vals

    return out

def select_day_hours(ds, date):
    t = pd.to_datetime(ds.time.values)
    mask = (t.date == pd.Timestamp(date).date())
    if not mask.any():
        return None
    return ds.isel(time=mask)

def compute_hybas_daily_from_mask(
    ds_daily,
    ds_acc,
    mask_daily_3d,
    mask_acc_3d,
    daily_vars,
    acc_vars,
):
    out = {}

    region_ids = mask_daily_3d.region.values
    non_empty_regions = 0

    for i, hid in enumerate(region_ids):
        vals = {}

        if ds_daily is not None:
            for v in daily_vars:
                if v not in ds_daily:
                    continue

                m = mask_daily_3d.isel(region=i)
                sub = ds_daily[v].where(m)

                cnt = int(sub.count())


                if cnt > 0:
                    vals[v] = float(sub.mean().item())

        if ds_acc is not None:
            for v in acc_vars:
                if v not in ds_acc:
                    continue

                m = mask_acc_3d.isel(region=i)
                sub = ds_acc[v].where(m)

                cnt = int(sub.count())


                if cnt > 0:
                    vals[v] = float(sub.sum().item())

        if vals:
            out[int(hid)] = vals
            non_empty_regions += 1

    return out


def extract_era5_data_daily(
    era5_dir=ERA5_DAILY_DIR,
    step1_upstream_csv=STEP1_UPSTREAM_CSV,
    out_csv=STEP4C_ECMWF_CSV,
    precipitation_csv=STEP3B_precipitaion_csv,
):

    # ============================================================
    # 1. LOAD DATES & SPLIT BY YEAR
    # ============================================================
    prcp = pd.read_csv(precipitation_csv, parse_dates=["date"])
    dates = sorted(prcp["date"].dt.normalize().unique())
    year_batches = split_dates_by_year(dates)

    logging.info(
        "STEP4C: ERA5 DAILY %d dates (%s → %s)",
        len(dates), dates[0].date(), dates[-1].date()
    )

    # ============================================================
    # 2. LOAD STEP1 + GEOMETRY (ONCE)
    # ============================================================
    step1 = pd.read_csv(step1_upstream_csv)
    step1["geometry"] = step1["geometry_wkt"].apply(wkt.loads)
    gdf = gpd.GeoDataFrame(step1, geometry="geometry", crs="EPSG:4326")

    lon_min, lat_min, lon_max, lat_max = gdf.total_bounds
    logging.info(
        "STEP4C BBOX: lon=[%.2f, %.2f], lat=[%.2f, %.2f]",
        lon_min, lon_max, lat_min, lat_max
    )

    regions = regionmask.Regions(
        outlines=list(gdf.geometry),
        numbers=gdf.hybas_id.values,
    )

    hybas_area = dict(zip(step1.hybas_id, step1.sub_area))
    grdc_to_hybas = (
        step1.groupby("grdc_no")["hybas_id"]
        .apply(lambda x: list(set(map(int, x))))
        .to_dict()
    )

    VAR_ORDER = ["t2m", "tp", "sp", "ssr", "str"]
    nc_files = sorted(f for f in os.listdir(era5_dir) if f.endswith(".nc"))

    # ============================================================
    # 3. PROCESS EACH YEAR (BATCH)
    # ============================================================
    for year, year_dates in year_batches.items():

        logging.info("STEP4C: START YEAR %d", year)

        out_csv_year = out_csv.replace(".csv", f"_{year}.csv")
        if os.path.exists(out_csv_year):
            logging.info("STEP4C: %s exists → skip", out_csv_year)
            continue

        year_dates_set = set(pd.to_datetime(year_dates).date)

        # --------------------------------------------------------
        # year_data[day][hybas_id][var] = value
        # --------------------------------------------------------
        year_data = {}

        # ========================================================
        # 4. LOOP ERA5 FILES (VAR)
        # ========================================================
        for fname in nc_files:
            fpath = os.path.join(era5_dir, fname)

            with xr.open_dataset(fpath) as ds:
                var = list(ds.data_vars)[0]
                if var not in VAR_ORDER:
                    continue

                da = ds[var]
                logging.info("STEP4C: %s (%s)", fname, var)

                if "valid_time" in da.coords:
                    da = da.rename({"valid_time": "time"})

                da["time"] = pd.to_datetime(da.time.values).normalize()

                if da.lon.max() > 180:
                    da = da.assign_coords(
                        lon=((da.lon + 180) % 360) - 180
                    ).sortby("lon")

                da = da.sel(
                    lon=slice(lon_min, lon_max),
                    lat=slice(lat_max, lat_min),
                )

                if da.size == 0:
                    logging.warning("STEP4C: %s empty after bbox → skip", var)
                    continue

                mask = regions.mask(da)

                # ====================================================
                # 5. LOOP TIME → HYBAS (CALCULATE ONCE)
                # ====================================================
                for t in tqdm(da.time.values, desc=f"{var}-{year}"):
                    day = pd.to_datetime(t).date()
                    if day not in year_dates_set:
                        continue

                    da_day = da.sel(time=t)
                    means = da_day.groupby(mask).mean()

                    for hid in means.mask.values:
                        if hid == -1 or np.isnan(hid):
                            continue

                        val = float(means.sel(mask=hid).values)

                        year_data \
                            .setdefault(day, {}) \
                            .setdefault(int(hid), {})[var] = val

        # ============================================================
        # 6. AGGREGATE TO GRDC (NO RE-COMPUTE HYBAS)
        # ============================================================
        records = []

        for grdc_no, hybas_list in grdc_to_hybas.items():
            for day, hybas_map in year_data.items():

                row = {
                    "grdc_no": grdc_no,
                    "date": day
                }

                for var in VAR_ORDER:
                    num = 0.0
                    den = 0.0

                    for h in hybas_list:
                        area = hybas_area.get(h)
                        if area and h in hybas_map and var in hybas_map[h]:
                            num += hybas_map[h][var] * area
                            den += area

                    if den > 0:
                        row[var] = num / den

                records.append(row)

        if not records:
            logging.warning("STEP4C %d: No records", year)
            continue

        # ============================================================
        # 7. POST-PROCESS & SAVE
        # ============================================================
        df = pd.DataFrame(records)

        for c in ["grdc_no", "date"] + VAR_ORDER:
            if c not in df.columns:
                df[c] = np.nan

        df = df[["grdc_no", "date"] + VAR_ORDER]

        df["t2m"] = df["t2m"] - 273.15
        df["tp"]  = df["tp"] * 1000
        df["sp"]  = df["sp"] / 1000
        df["ssr"] = df["ssr"] / 1e6
        df["str"] = df["str"] / 1e6

        df.loc[df["tp"].abs() < 0.001, "tp"] = 0.0

        df = df.round({
            "t2m": 2,
            "tp": 3,
            "sp": 2,
            "ssr": 3,
            "str": 3,
        })

        df = df.sort_values(["grdc_no", "date"])
        df["date"] = df["date"].astype(str)

        df.to_csv(out_csv_year, index=False)
        logging.info("STEP4C: Saved %s", out_csv_year)

    logging.info("STEP4C: ALL YEARS COMPLETED")




def split_dates_by_year(dates):
    by_year = {}
    for d in dates:
        y = pd.Timestamp(d).year
        by_year.setdefault(y, []).append(pd.Timestamp(d))
    return by_year



def apply_physical_limits(df, name, limits_dict):
    """
    Filter dataframe by physical limits.
    Log number of removed values per variable.
    """
    df = df.copy()

    logging.info("STEP5: Apply physical limits for %s", name)

    for var, (vmin, vmax) in limits_dict.items():
        if var not in df.columns:
            continue

        before = df[var].notna().sum()

        invalid_mask = (df[var] < vmin) | (df[var] > vmax)
        n_invalid = invalid_mask.sum()

        if n_invalid > 0:
            logging.warning(
                "%s | %s: removed %d / %d values (%.2f%%) خارج [%s, %s]",
                name,
                var,
                n_invalid,
                before,
                100.0 * n_invalid / before if before > 0 else 0.0,
                vmin,
                vmax,
            )

            df.loc[invalid_mask, var] = np.nan

    return df
def rebuild_final_dataset(
    discharge_csv,
    precip_csv,
    era5_csv,
    ecmwf_dir,
    static_csv,
    upstream_csv,
    out_dir,
):

    logging.info("STEP5: rebuild alignment from scratch")

    os.makedirs(out_dir, exist_ok=True)

    # =========================================================
    # 1. LOAD TIME-SERIES FILES
    # =========================================================
    df_q  = pd.read_csv(discharge_csv, parse_dates=["date"])
    df_p  = pd.read_csv(precip_csv,    parse_dates=["date"])
    # df_e5 = pd.read_csv(era5_csv,       parse_dates=["date"])
    df_e5 =  load_step4c_dir(era5_csv)
    df_e5 = enforce_units_era5(df_e5)
    
    # df_ec = pd.read_csv(ecmwf_csv,      parse_dates=["date"])
    df_ec = load_step4b_dir(ecmwf_dir)
    df_ec = enforce_units_tigge(df_ec)


    df_st = pd.read_csv(static_csv)
    df_up = pd.read_csv(upstream_csv)

    
    df_e5 = apply_physical_limits(df_e5, "ERA5", PHYSICAL_LIMITS)
    df_ec = apply_physical_limits(df_ec, "TIGGE", PHYSICAL_LIMITS)
    df_q  = apply_physical_limits(df_q,  "DISCHARGE", PHYSICAL_LIMITS)
    df_p  = apply_physical_limits(df_p,  "PRECIP", PHYSICAL_LIMITS)

    # =========================================================
    # 2. BUILD COMMON (grdc_no, date) SET
    # =========================================================
    def normalize(df):
        df = df.copy()
        df["grdc_no"] = df["grdc_no"].astype(int)
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()
        return df

    df_q  = normalize(df_q)
    df_p  = normalize(df_p)
    df_e5 = normalize(df_e5)
    df_ec = normalize(df_ec)

    # =========================================================
    # 3. BUILD COMMON (grdc_no, date) VIA MERGE (CHUẨN)
    # =========================================================
    common = (
        df_q[["grdc_no", "date"]]
        .merge(df_p[["grdc_no", "date"]],  on=["grdc_no", "date"])
        .merge(df_e5[["grdc_no", "date"]], on=["grdc_no", "date"])
        .merge(df_ec[["grdc_no", "date"]], on=["grdc_no", "date"])
        .drop_duplicates()
    )

    if common.empty:
        raise RuntimeError("STEP5: NO common (grdc_no, date) found")

    logging.info("STEP5: common (grdc_no, date) = %d", len(common))

    # =========================================================
    # 4. FILTER ALL TIME-SERIES FILES (AN TOÀN)
    # =========================================================
    df_q  = df_q.merge(common, on=["grdc_no", "date"])
    df_p  = df_p.merge(common, on=["grdc_no", "date"])
    df_e5 = df_e5.merge(common, on=["grdc_no", "date"])
    df_ec = df_ec.merge(common, on=["grdc_no", "date"])
    df_ec = (
        df_ec
        .merge(common, on=["grdc_no", "date"], how="inner")
        .sort_values(["grdc_no", "date"])
        .drop_duplicates(subset=["grdc_no", "date"], keep="first")
        .reset_index(drop=True)
    )

   
    logging.info("STEP5: common (grdc_no, date) = %d", len(common))

    # =========================================================
    # 3. FILTER ALL TIME-SERIES FILES
    # =========================================================
  
    for df in (df_q, df_p, df_e5, df_ec):
        df.sort_values(["grdc_no", "date"], inplace=True)
        df.reset_index(drop=True, inplace=True)

    # =========================================================
    # 4. FILTER STATIC & UPSTREAM BY USED GRDC_NO
    # =========================================================
    # used_grdc = {g for g, _ in common}
    used_grdc = set(common["grdc_no"].unique())


    df_st = (
        df_st[df_st["grdc_no"].isin(used_grdc)]
        .drop_duplicates(subset=["grdc_no"])
        .reset_index(drop=True)
    )

    df_up = (
        df_up[df_up["grdc_no"].isin(used_grdc)]
        .reset_index(drop=True)
    )

    logging.info("STEP5: used GRDC stations = %d", len(used_grdc))

    # =========================================================
    # 4.5 COMPUTE TOTAL UPSTREAM AREA PER GRDC (m²)
    # =========================================================
    if "sub_area" not in df_up.columns:
        raise RuntimeError("STEP5: sub_area column missing in upstream file")

    df_up["sub_area_m2"] = df_up["sub_area"] * 1e6
    
    area_map = (
        df_up.groupby("grdc_no")["sub_area_m2"]
        .sum()
        .to_dict()
    )

    # =========================================================
    # 4.6 NORMALIZE DISCHARGE: m³/s → m/s
    # =========================================================
    # Giả định cột discharge tên là 'discharge'
    if "q_obs" not in df_q.columns:
        raise RuntimeError("STEP5: q_obs column not found in discharge file")

    df_q["area_m2"] = df_q["grdc_no"].map(area_map)

    if df_q["area_m2"].isna().any():
        miss = df_q[df_q["area_m2"].isna()]["grdc_no"].unique()
        raise RuntimeError(
            f"STEP5: missing upstream area for GRDC stations: {miss}"
        )
    df_q_raw = df_q.copy()

    # q_mm_day = Q / A * 1000 * 86400
    df_q["q_obs"] = (
        df_q["q_obs"] / df_q["area_m2"] * 1000.0 * 86400.0
    )

    df_q.drop(columns=["area_m2"], inplace=True)

    logging.info(
        "STEP5: q_obs range = %.3f → %.3f mm/day",
        df_q["q_obs"].min(),
        df_q["q_obs"].max()
    )
    df_q["q_obs"] = df_q["q_obs"].round(3)

    # =========================================================
    # 5. EXPORT (FORMAT GIỮ NGUYÊN)
    # =========================================================
    df_q_raw.to_csv(f"{out_dir}/step5_discharge_raw.csv", index=False)

    df_q.to_csv(f"{out_dir}/step5_discharge.csv", index=False)
    df_p.to_csv(f"{out_dir}/step5_precip.csv",    index=False)
    df_e5.to_csv(f"{out_dir}/step5_era5.csv",     index=False)
    df_ec.to_csv(f"{out_dir}/step5_ecmwf.csv",    index=False)

    df_st.to_csv(f"{out_dir}/step5_static.csv",   index=False)
    df_up.to_csv(f"{out_dir}/step5_upstream.csv", index=False)

    logging.info("STEP5: DONE")


def load_step4b_dir(step4b_dir):
    """
    Load all CSV files under step4b directory (recursive),
    merge into a single DataFrame.
    """

    csv_files = []

    for root, _, files in os.walk(step4b_dir):
        for f in files:
            if f.endswith(".csv"):
                csv_files.append(os.path.join(root, f))

    if not csv_files:
        raise RuntimeError(f"STEP5: no CSV files found in {step4b_dir}")

    logging.info(
        "STEP5: loading %d step4b CSV files", len(csv_files)
    )

    dfs = [
        pd.read_csv(f, parse_dates=["date"])
        for f in sorted(csv_files)
    ]

    df = pd.concat(dfs, ignore_index=True)

    logging.info(
        "STEP5: ECMWF merged from %d files → %d records, %d grdc, %d dates",
        len(csv_files),
        len(df),
        df["grdc_no"].nunique(),
        df["date"].nunique(),
    )
    logging.info(
        "STEP5 TOP 5 (grdc_no, date):\n%s",
        df[["grdc_no", "date"]].head(5).to_string(index=False)
    )

    return df

def load_step4c_dir(step4c_dir):
    csv_files = []
    for root, _, files in os.walk(step4c_dir):
        for f in files:
            if f.endswith(".csv"):
                csv_files.append(os.path.join(root, f))

    if not csv_files:
        raise RuntimeError(f"No CSV files found in {step4c_dir}")

    logging.info(
        "STEP5: loading %d step4c CSV files", len(csv_files)
    )

    dfs = [pd.read_csv(f) for f in sorted(csv_files)]
    df = pd.concat(dfs, ignore_index=True)

    logging.info(
        "STEP5: step4c merged → %d rows, %d grdc",
        len(df),
        df["grdc_no"].nunique()
    )

    return df


    
