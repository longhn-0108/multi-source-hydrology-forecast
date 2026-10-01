import re
import os
import pandas as pd
import glob
import geopandas as gpd
from shapely import wkt
from tqdm import tqdm
import logging
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

SEPARATOR_LINE = "#************************************************************"


# GRDC utils
def read_file(fp):
    """
    Read a GRDC station file and split it into logical sections.

    A standard GRDC text file is structured into multiple sections
    separated by a fixed delimiter line. This function reads the file
    and splits it into three main parts:
      1) Header section      : metadata and station information
      2) Variable definition : description of recorded variables
      3) Data section        : time-series discharge records

    Parameters
    ----------
    fp : Path to the GRDC station text file.

    Returns
    -------
    header_lines : list[str]
        Lines belonging to the header section.
    var_lines : list[str]
        Lines describing variable definitions.
    data_lines : list[str]
        Lines containing the actual observation data.

    Raises
    ------
    ValueError
        If the file does not contain at least three sections, which
        indicates that the file does not follow the GRDC format.
    """

    with open(fp, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    parts = []
    cur = []

    for line in lines:
        if line.strip() == SEPARATOR_LINE:
            parts.append(cur)
            cur = []
        else:
            cur.append(line)
    parts.append(cur)

    if len(parts) < 3:
        raise ValueError(f"File {fp} không đúng định dạng GRDC (thiếu 3 sectors).")

    return parts[0], parts[1], parts[2]


def parse_grdc_header(header_lines):
    """
    Parse metadata information from the header section of a GRDC file.

    Parameters
    ----------
    header_lines : list[str]
        Lines from the header section of a GRDC station file.

    Returns
    -------
    meta : dict
        Dictionary containing extracted metadata:
        {
            "grdc_no"   : str,
            "lat"       : float,
            "lon"       : float,
            "catchment" : float
        }

    Raises
    ------
    ValueError
        If the GRDC station identifier cannot be found in the header.
    """
    meta = {
        "grdc_no": None,
        "lat": None,
        "lon": None,
        "catchment": None
    }

    for line in header_lines:
        if meta["grdc_no"] is None:
            m = re.search(r"GRDC-No\.\:\s*(\d+)", line)
            if m:
                meta["grdc_no"] = m.group(1)

        if meta["lat"] is None:
            m = re.search(r"Latitude\s*\(DD\)\:\s*([-\d\.]+)", line)
            if m:
                meta["lat"] = float(m.group(1))

        if meta["lon"] is None:
            m = re.search(r"Longitude\s*\(DD\)\:\s*([-\d\.]+)", line)
            if m:
                meta["lon"] = float(m.group(1))

        if meta["catchment"] is None:
            m = re.search(r"Catchment area\s*\(km\)\:\s*([-\d\.]+)", line)
            if m:
                meta["catchment"] = float(m.group(1))

    if meta["grdc_no"] is None:
        raise ValueError("Không tìm thấy GRDC-No trong header")

    return meta
def parse_grdc_data_lines(
    data_lines,
    start_date,
    end_date,
    grdc_no=None,
):
    """
    Parse GRDC data section into list of (date, discharge).

    Only keep:
    - valid YYYY-MM-DD
    - discharge != -999
    - date in [start_date, end_date]
    """

    records = []

    DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

    # ---- counters ----
    n_bad_format = 0
    n_missing = 0
    n_parse_error = 0

    first_data_line = True   # 👈 đánh dấu dòng đầu

    for line in data_lines:
        line = line.strip()

        if not line or line.startswith("#"):
            continue

        # 👇 BỎ QUA dòng data đầu tiên (header)
        if first_data_line:
            first_data_line = False
            continue

        parts = [p.strip() for p in line.split(";")]
        if len(parts) < 2:
            n_bad_format += 1
            continue

        if not DATE_RE.match(parts[0]):
            n_bad_format += 1
            continue

        try:
            date = pd.to_datetime(parts[0])

            if not (start_date <= date <= end_date):
                continue

            q = float(parts[-1])
            if q == -999.0:
                n_missing += 1
                continue

            records.append((date, q))

        except Exception:
            n_parse_error += 1
            continue

    # ---- logging ----
    tag = f"GRDC {grdc_no}" if grdc_no else "GRDC"
    total_invalid = n_bad_format + n_missing + n_parse_error

    if total_invalid > 0:
        logging.warning(
            "%s invalid stats (excluding header): bad_format=%d, missing=%d, parse_error=%d",
            tag,
            n_bad_format,
            n_missing,
            n_parse_error,
        )

    return records


def load_grdc_discharge_df(
    grdc_dir,
    valid_grdc_ids,
    start_date,
    end_date
):
    """
    Load GRDC daily discharge into DataFrame.

    Return columns:
      grdc_no, date, discharge_cms
    """

    records = []

    valid_grdc_ids = set(map(str, valid_grdc_ids))
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    grdc_files = {
        fname.split("_")[0]: fname
        for fname in os.listdir(grdc_dir)
        if fname.endswith("_Q_Day.Cmd.txt")
    }

    logging.info(
        "[GRDC] Found %d discharge files in folder",
        len(grdc_files)
    )

    for grdc_no in tqdm(
        sorted(valid_grdc_ids),
        total=len(valid_grdc_ids),
        desc="Loading GRDC discharge"
    ):
        fname = grdc_files.get(grdc_no)

        if fname is None:
            logging.warning(
                "[GRDC] Missing discharge file for station %s",
                grdc_no
            )
            continue

        fp = os.path.join(grdc_dir, fname)

        try:
            _, _, data_lines = read_file(fp)
        except Exception as e:
            logging.warning(
                "[GRDC] Failed to read file %s: %s",
                fname, e
            )
            continue

        parsed = parse_grdc_data_lines(
            data_lines,
            start_date,
            end_date
        )

        if not parsed:
            logging.warning(
                "[GRDC] Station %s has no valid discharge data in range",
                grdc_no
            )
            continue

        for date, q in parsed:
            records.append({
                "grdc_no": grdc_no,
                "date": date,
                "discharge_cms": q
            })

    df = pd.DataFrame(
        records,
        columns=["grdc_no", "date", "discharge_cms"]
    )

    logging.info(
        "[GRDC] Loaded %d discharge records from %d stations",
        len(df),
        df["grdc_no"].nunique() if not df.empty else 0
    )

    return df


# Precipitation utils



def load_rain_station_metadata(rain_dir):
    """
    Load precipitation station metadata.

    Directory structure:
    precipitation/
        AR00000011.csv
        AR000087007.csv
        ...

    Each CSV contains:
    DATE,LATITUDE,LONGITUDE,PRCP

    Output:
    List of dicts:
        [
            { "pre_no": station_id, "lat": lat, "lon": lon },
            ...
        ]
    """
    precipitation_station_meta = []

    for fname in os.listdir(rain_dir):
        if not fname.lower().endswith(".csv"):
            continue

        pre_no = os.path.splitext(fname)[0]
        fpath = os.path.join(rain_dir, fname)

        try:
            # Chỉ cần đọc 1 dòng đầu
            df = pd.read_csv(
                fpath,
                usecols=["LATITUDE", "LONGITUDE"],
                nrows=1
            )
        except Exception as e:
            logging.warning(f"[WARN] Skip {fname}: {e}")
            continue

        if df.empty:
            continue

        lat = float(df.iloc[0]["LATITUDE"])
        lon = float(df.iloc[0]["LONGITUDE"])

        precipitation_station_meta.append({
            "pre_no": pre_no,
            "lat": lat,
            "lon": lon
        })

    return precipitation_station_meta


def load_rain_station_daily(
    station_ids,
    rain_dir,
    start_date,
    end_date
):
    """
    Return:
      station_daily[station_id][date] = prcp
    """

    station_daily = {}

    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    total_stations = len(station_ids)
    missing_files = 0
    empty_in_range = 0
    no_valid_prcp = 0
    total_records = 0

    logging.info(
        "Loading rain station daily data | stations=%d | range=%s → %s",
        total_stations, start_date.date(), end_date.date()
    )

    for sid in station_ids:
        fp = os.path.join(rain_dir, f"{sid}.csv")
        station_daily[sid] = {}

        if not os.path.exists(fp):
            logging.warning("Rain station file missing: %s", fp)
            missing_files += 1
            continue

        prep_col = "PRCP"
        df = pd.read_csv(
            fp,
            usecols=["DATE", prep_col]
        )

        if df.empty:
            logging.warning("Rain station file empty: %s", fp)
            empty_in_range += 1
            continue

        df["DATE"] = pd.to_datetime(df["DATE"])
        df = df[(df.DATE >= start_date) & (df.DATE <= end_date)]

        if df.empty:
            logging.warning(
                "Rain station %s has no data in range %s → %s",
                sid, start_date.date(), end_date.date()
            )
            empty_in_range += 1
            continue

        valid_cnt = 0
        for _, r in df.iterrows():
            if pd.notna(r[prep_col]):
                station_daily[sid][r.DATE] = r[prep_col]
                valid_cnt += 1

        if valid_cnt == 0:
            logging.warning(
                "Rain station %s has data but all PRCP are NaN (%d rows)",
                sid, len(df)
            )
            no_valid_prcp += 1
        else:
            total_records += valid_cnt

    logging.info(
        "Rain daily load summary | total=%d | missing_files=%d | "
        "no_data_in_range=%d | no_valid_prcp=%d | total_records=%d",
        total_stations,
        missing_files,
        empty_in_range,
        no_valid_prcp,
        total_records
    )

    return station_daily

# HydroAtlas utils

HYDRO_STATIC_COLS = [
    "ele_mt_uav", "slp_dg_uav",
    "cly_pc_uav", "snd_pc_uav", "slt_pc_uav",
    "crp_pc_use", "for_pc_use", "urb_pc_use",
    "ari_ix_uav", "pre_mm_uyr"
]

def load_hydroatlas(hydro_csv):
    cols = [
        "HYBAS_ID", "NEXT_DOWN", "UP_AREA", "SUB_AREA",
        "MAIN_BAS", "geometry_wkt"
    ] + HYDRO_STATIC_COLS

    hydro = pd.read_csv(hydro_csv, usecols=cols).rename(columns={
        "HYBAS_ID": "hybas_id",
        "NEXT_DOWN": "next_down",
        "UP_AREA": "up_area",
        "SUB_AREA": "sub_area"
    })

    hydro["geometry"] = hydro.geometry_wkt.apply(wkt.loads)
    hydro_gdf = gpd.GeoDataFrame(hydro, geometry="geometry", crs="EPSG:4326")

    return hydro_gdf

def load_hydroatlas_filtered(
    hydro_csv,
    hybas_ids: set
):
    """
    Load HydroATLAS only for required HYBAS_IDs
    (used in STEP 2 & STEP 3)
    """
    cols=["HYBAS_ID", "geometry_wkt"]


    hydro = pd.read_csv(
        hydro_csv,
        usecols=cols
    ).rename(columns={
        "HYBAS_ID": "hybas_id",
        "NEXT_DOWN": "next_down",
        "UP_AREA": "up_area",
        "SUB_AREA": "sub_area"
    })

    hydro = hydro[hydro.hybas_id.isin(hybas_ids)]

    hydro["geometry"] = hydro.geometry_wkt.apply(wkt.loads)

    hydro_gdf = gpd.GeoDataFrame(
        hydro,
        geometry="geometry",
        crs="EPSG:4326"
    )

    return hydro_gdf
def aggregate_basin(input_dir: str, output_csv: str):
    """
    Aggregate all HydroBASINS CSV files into ONE cleaned CSV
    (for STEP 1 upstream tracing)
    """

    csv_files = glob.glob(os.path.join(input_dir, "*.csv"))
    if not csv_files:
        raise ValueError(f"No CSV files found in {input_dir}")

    usecols = [
        "HYBAS_ID",
        "NEXT_DOWN",
        "NEXT_SINK",
        "MAIN_BAS",
        "SUB_AREA",
        "UP_AREA",
        "geometry_wkt",
    ]

    dfs = []

    for f in tqdm(
        sorted(csv_files),
        desc="STEP 1.2 – Aggregate HydroBASINS",
        unit="file"
    ):
        df = pd.read_csv(
            f,
            usecols=usecols
        )
        dfs.append(df)

    merged = pd.concat(dfs, ignore_index=True)

    # rename cho thống nhất pipeline
    merged = merged.rename(columns={
        "HYBAS_ID": "hybas_id",
        "NEXT_DOWN": "next_down",
        "NEXT_SINK": "next_sink",
        "MAIN_BAS": "main_bas",
        "SUB_AREA": "sub_area",
        "UP_AREA": "up_area",
    })

    merged = merged.drop_duplicates(subset="hybas_id")

    merged.to_csv(output_csv, index=False)

    return merged
    
def load_hydroatlas(hydro_atlas_csv):
    cols = [
        "HYBAS_ID",
        "NEXT_DOWN",
        "MAIN_BAS",
        "UP_AREA",
        "geometry_wkt",
    ] + HYDRO_STATIC_COLS

    df = pd.read_csv(
        hydro_atlas_csv,
        usecols=cols
    ).rename(columns={
        "HYBAS_ID": "hybas_id",
        "NEXT_DOWN": "next_down",
        "MAIN_BAS": "main_bas",
        "UP_AREA": "up_area",
    })

    df["geometry"] = df.geometry_wkt.apply(wkt.loads)

    gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")
    logging.info(f"Loaded {len(gdf)} basins from HydroATLAS")
    return gdf

def load_hydrobasins(hydro_basin_csv):
    cols = [
        "hybas_id",
        "next_down",
        "next_sink",
        "main_bas",
        "sub_area",
        "up_area",
        "geometry_wkt",
    ]

    df = pd.read_csv(
        hydro_basin_csv,
        usecols=cols
    )

    df["geometry"] = df.geometry_wkt.apply(wkt.loads)

    gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")
    logging.info(f"Loaded {len(gdf)} basins from HydroBASINS")
    return gdf

    
def setup_logging(
    output_dir="output",
    log_level=logging.INFO,
):
    """
    Setup logging:
    - Log to file: output/run_YYYYMMDD_HHMMSS.log
    - Log to console
    """

    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(
        output_dir, f"run_{timestamp}.log"
    )

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler()  # vẫn in ra terminal
        ],
    )

    logging.info("==============================================")
    logging.info("🚀 START NEW PIPELINE RUN")
    logging.info(f"Log file: {log_path}")
    logging.info("==============================================")

    return 


#Plot






def enforce_units_tigge(df):
    df = df.copy()
    df["t2m"] = (df["t2m"] - 273.15).round(2)   # K → °C
    df["sp"]  = (df["sp"] / 1000).round(2)      # Pa → kPa
    df["tp"]  = (np.log1p(df["tp"])).round(3)       # m → mm
   
    return df

def enforce_units_era5(df):
    df = df.copy()
    df["ssr"]  = (df["ssr"] * 10).round(3)    # Pa → kPa
    df["str"]  = (df["str"] * 10).round(3)    # Pa → kPa
    df["tp"]  = (df["tp"] * 10).round(3)    

    return df
