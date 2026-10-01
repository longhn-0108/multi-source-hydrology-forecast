# Multi-source Hydrology Forecast

A multi-source hydrological data integration and **one-day-ahead streamflow forecasting** pipeline for data-scarce and ungauged basins.

The project integrates river discharge observations, precipitation, meteorological reanalysis/forecast data, and static basin attributes into a unified station–time dataset. A Handoff Forecast LSTM is then used to evaluate streamflow prediction from multi-source inputs.

## Highlights

- **Multi-source integration:** GRDC, NOAA precipitation, ERA5, TIGGE/ECMWF, HydroBASINS, and HydroATLAS.
- **Hydrological topology:** maps GRDC stations to local basins and reconstructs contributing upstream basins.
- **Two data workflows:** station-based NOAA precipitation and gridded NOAA precipitation.
- **Spatial aggregation:** basin-level precipitation and meteorological features are aggregated over upstream drainage areas.
- **Data harmonization:** temporal alignment, unit normalization, and physical plausibility checks.
- **Forecasting model:** Handoff Forecast LSTM with 365-day historical context and 1-day forecast horizon.
- **Evaluation:** RMSE, Bias, Pearson correlation, and Nash–Sutcliffe Efficiency (NSE).

## Research Context

The accompanying research report, **"Hydrological Prediction in Ungauged Basins"**, describes the construction of the multi-source dataset and its use for streamflow prediction in data-scarce and ungauged basins.

The reported baseline one-day-ahead result is:

| Model | RMSE ↓ | Bias ↓ | r ↑ | NSE ↑ |
|---|---:|---:|---:|---:|
| Caravan baseline | 1.839 | -0.916 | 0.044 | -1.093 |
| **Ours — Handoff Forecast LSTM** | **2.094** | **-0.001** | **0.806** | **0.634** |

The report also includes an ablation study covering meteorological precipitation (`tp`), NOAA precipitation, and static basin attributes. See [`results/results.md`](results/results.md) for the reported values and interpretation.

## Pipeline

```text
GRDC discharge ───────────────┐
NOAA precipitation ───────────┤
ERA5 reanalysis ──────────────┤
TIGGE / ECMWF forecast ───────┤──> Spatial & temporal integration
HydroBASINS topology ─────────┤            │
HydroATLAS static attributes ─┘            ▼
                                  Unified station-time dataset
                                             │
                                             ▼
                                  Handoff Forecast LSTM
                                             │
                                             ▼
                              1-day-ahead streamflow forecast
                                             │
                                             ▼
                              RMSE / Bias / r / NSE
```

## Workflows

### 1. Station-based workflow

Processes GRDC discharge together with station-based NOAA precipitation and meteorological inputs.

```bash
bash scripts/run_station_workflow.sh
```

or:

```bash
python src/run_pipeline.py station --start_date=2019-01-01 --end_date=2024-12-31
```

### 2. Grid-based workflow

Processes GRDC discharge with gridded NOAA precipitation.

```bash
bash scripts/run_grid_workflow.sh
```

or:

```bash
python src/run_pipeline.py grid --start_date=2000-01-01 --end_date=2025-12-31
```

Pipeline outputs are written under `output/` and organized by processing step.

## Data Setup

The raw datasets are **not included in this repository** because of their size and external distribution/licensing constraints.

Place the required files under `data/` following the structure below:

```text
data/
├── grdc/
├── precipitation/
├── precipitation_grid_last/
├── era5_merged/
├── g/
│   └── 2025_01.grib
├── BasinATLAS_v10_lev08.csv
└── HydroBasin.csv/
```

See [`data/README.md`](data/README.md) for the expected inputs and naming conventions.

## Environment

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

For the forecasting notebook, set the processed-data directory before opening the notebook:

```bash
export HYDRO_DATA_DIR=/path/to/processed/hydro/data
```

On Windows PowerShell:

```powershell
$env:HYDRO_DATA_DIR="C:\\path\\to\\processed\\hydro\\data"
```

Then open:

```text
notebooks/training_lstm.ipynb
```

## Repository Structure

```text
multi-source-hydrology-forecast/
├── README.md
├── LICENSE
├── .gitignore
├── requirements.txt
├── pyproject.toml
├── configs/
├── data/
│   └── README.md
├── docs/
│   ├── methodology.md
│   └── research_report.pdf
├── notebooks/
│   ├── training_lstm.ipynb
│   └── data_exploration_hydrobasins.ipynb
├── results/
│   └── results.md
├── scripts/
│   ├── run_station_workflow.sh
│   └── run_grid_workflow.sh
├── src/
│   ├── run_pipeline.py
│   ├── preprocessing/
│   └── utils/
├── tests/
└── outputs/
```

## Reproducibility Notes

- The repository intentionally excludes raw GRDC/NOAA/ERA5/TIGGE/HydroBASINS/HydroATLAS data and trained model checkpoints.
- The forecasting notebook uses a fixed random seed of `42`.
- The model uses a 365-day historical sequence and a 1-day forecast horizon.
- Training uses basin-level separation to reduce temporal leakage across train/validation/test sets.
- The numerical results shown in this README and `results/results.md` are the **reported results from the project research report**, not newly reproduced results in this cleaned repository.

## Limitations

The complete pipeline depends on large external geospatial and meteorological datasets. Consequently, a fresh clone cannot execute the full workflow until those datasets are downloaded and placed in the expected locations.

The included research report also discusses practical constraints related to data volume, spatial/temporal coverage, and the computational cost of processing GRIB/NetCDF products.

## License

MIT License. See [`LICENSE`](LICENSE).
