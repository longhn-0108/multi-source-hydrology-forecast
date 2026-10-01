import sys
from pathlib import Path
import logging
from datetime import datetime

# Add src to path to allow imports
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from src.preprocessing import data_integration as di
from src.utils.utils import setup_logging

def run_station_workflow(
    start_date="2019-01-01",
    end_date="2024-12-31",
    output_base_dir=None
):
    """
    Luồng 1: NOAA Station Data
    Step 1 -> Step 2 -> Step 3 -> Step 4 (ECMWF/ERA5) -> Step 5
    """
    logging.info("STARTING WORKFLOW 1: STATION DATA")
    
    # Defaults from data_integration if not passed
    # Using the constants defined in data_integration module
    
    # STEP 1: Build Upstream Mapping
    logging.info("--- Step 1: Upstream Mapping ---")
    di.build_grdc_upstream_mapping(
        grdc_dir=di.GRDC_DIR,
        output_csv=di.STEP1_UPSTREAM_CSV,
        static_out_csv=di.STEP1_STATIC_CSV,
        hydoAtlas=di.HYDRPO_ATLAS_GPKG,
        hydroBasin=di.AGGREGATE_BASIN
    )
    
    # STEP 2: Assign Rain Stations (Station Only)
    logging.info("--- Step 2: Assign Rain Stations ---")
    di.assign_rain_stations_to_basins(
        step1_upstream_csv=di.STEP1_UPSTREAM_CSV,
        rain_dir=di.RAIN_DIR,
        out_csv=di.STEP2_RAIN_MAP
    )
    
    # STEP 3: Process Discharge and Precipitation (Station)
    logging.info("--- Step 3: Process Discharge & Precip (Station) ---")
    di.process_discharge_and_precipitation(
        start_date=start_date,
        end_date=end_date,
        out_discharge_csv=di.STEP3_DISCHARGE_CSV,
        out_precip_csv=di.STEP3_PRCP_NOAA_CSV,
        rain_root_dir=di.RAIN_DIR,
        grdc_dir=di.GRDC_DIR,
        step2_map_csv=di.STEP2_RAIN_MAP
    )
    
    # Validate Output
    di.validate_discharge_precipitation_output(
        in_csv=di.STEP3_DISCHARGE_CSV, # Check if this is the right raw input, usually step3 returns raw
        out_valid_csv=di.STEP3_DISCHARGE_CSV.replace(".csv", "_valid.csv") # Or overwrite
    )
    
    # STEP 4: ECMWF / ERA5
    # Note: Using optimized extraction or daily extraction as per original main
    logging.info("--- Step 4: Extract ECMWF/ERA5 Data ---")
    
    # 4b: ECMWF (TIGGE)
    # Using extract_ecmwf_data_optimized
    di.extract_ecmwf_data_optimized(
        out_csv=di.STEP4B_ECMWF_opt2_CSV,
        tigge=di.TIGGE_GRIB_MERGE,
        step1_upstream_csv=di.STEP1_UPSTREAM_CSV,
        precipitation=di.STEP3_PRCP_NOAA_CSV
    )
    
    # 4c: ERA5
    di.extract_era5_data_daily(
        era5_dir=di.ERA5_DAILY_DIR,
        precipitation_csv=di.STEP3_PRCP_NOAA_CSV,
        step1_upstream_csv=di.STEP1_UPSTREAM_CSV,
        out_csv=di.STEP4C_ECMWF_CSV
    )
    
    # STEP 5: Rebuild Final Dataset
    logging.info("--- Step 5: Rebuild Final Dataset ---")
    out_dir_final = di.FINAL_DIR if output_base_dir is None else str(Path(output_base_dir) / "final_station")
    
    di.rebuild_final_dataset(
        discharge_csv=di.STEP3_DISCHARGE_CSV,
        precip_csv=di.STEP3_PRCP_NOAA_CSV,
        era5_csv=di.STEP4C_DIR, # Using dir for era5 as per step5_rebuild implementation in di
        ecmwf_dir=str(Path(di.OUTPUT_DIR) / "step4b"), # Should point to where Step 4b output is. 
        # Note: extract_ecmwf_data_optimized outputs to a single CSV (di.STEP4B_ECMWF_opt2_CSV).
        # rebuild_final_dataset expects a dir for `ecmwf_dir` via `load_step4b_dir`.
        # If we use `df_ec = load_step4b_dir(ecmwf_dir)`, we need to ensure the CSVs are in there.
        # The single file `di.STEP4B_ECMWF_opt2_CSV` is in `di.STEP4B_DIR`.
        static_csv=di.STEP1_STATIC_CSV,
        upstream_csv=di.STEP1_UPSTREAM_CSV,
        out_dir=out_dir_final
    )
    logging.info("Workflow 1 Completed.")


def run_grid_workflow(
    start_date="2000-01-01",
    end_date="2025-12-31",
    output_base_dir=None
):
    """
    Luồng 2: NOAA Grid Data
    Step 1 -> Step 2b -> Step 3b -> Step 4 -> Step 5
    """
    logging.info("STARTING WORKFLOW 2: GRID DATA")
    
    # STEP 1: Upstream Mapping (Shared)
    logging.info("--- Step 1: Upstream Mapping ---")
    di.build_grdc_upstream_mapping(
        grdc_dir=di.GRDC_DIR,
        output_csv=di.STEP1_UPSTREAM_CSV,
        static_out_csv=di.STEP1_STATIC_CSV,
        hydoAtlas=di.HYDRPO_ATLAS_GPKG,
        hydroBasin=di.AGGREGATE_BASIN
    )
    
    # STEP 2b: Extract Discharge (Standalone for Grid workflow)
    logging.info("--- Step 2b: Extract Discharge ---")
    di.extract_discharge_data(
        start_date=start_date,
        end_date=end_date,
        out_csv=di.STEP2B_DISCHARGE_CSV,
        grdc_dir=di.GRDC_DIR,
        step1_csv=di.STEP1_UPSTREAM_CSV
    )
    
    # STEP 3b: Extract Precipitation (Grid)
    logging.info("--- Step 3b: Extract Precipitation (Grid) ---")
    di.extract_precipitation_data(
        precipitation_dir=di.PRECIPITATION_NC_DIR,
        discharge_csv=di.STEP2B_DISCHARGE_CSV,
        step1_upstream=di.STEP1_UPSTREAM_CSV,
        out_csv=di.STEP3B_precipitaion_csv,
        precip_var="precip"
    )
    
    # STEP 4: ECMWF / ERA5
    logging.info("--- Step 4: Extract ECMWF/ERA5 Data ---")
    
    # 4b: ECMWF
    di.extract_ecmwf_data_optimized(
        out_csv=di.STEP4B_ECMWF_opt2_CSV,
        tigge=di.TIGGE_GRIB_MERGE,
        step1_upstream_csv=di.STEP1_UPSTREAM_CSV,
        precipitation=di.STEP3B_precipitaion_csv
    )
    
    # 4c: ERA5
    di.extract_era5_data_daily(
        era5_dir=di.ERA5_DAILY_DIR,
        precipitation_csv=di.STEP3B_precipitaion_csv,
        step1_upstream_csv=di.STEP1_UPSTREAM_CSV,
        out_csv=di.STEP4C_ECMWF_CSV
    )
    
    # STEP 5: Rebuild
    logging.info("--- Step 5: Rebuild Final Dataset ---")
    out_dir_final = di.FINAL_DIR if output_base_dir is None else str(Path(output_base_dir) / "final_grid")
    
    di.rebuild_final_dataset(
        discharge_csv=di.STEP2B_DISCHARGE_CSV,
        precip_csv=di.STEP3B_precipitaion_csv,
        era5_csv=di.STEP4C_DIR,
        ecmwf_dir=str(Path(di.OUTPUT_DIR) / "step4b"),
        static_csv=di.STEP1_STATIC_CSV,
        upstream_csv=di.STEP1_UPSTREAM_CSV,
        out_dir=out_dir_final
    )
    logging.info("Workflow 2 Completed.")

if __name__ == "__main__":
    import fire
    
    setup_logging(output_dir=di.OUTPUT_DIR)
    fire.Fire({
        'station': run_station_workflow,
        'grid': run_grid_workflow
    })
