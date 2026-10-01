# Methodology

## 1. Multi-source data integration

The project constructs a unified station–time dataset from complementary hydrological, meteorological, precipitation, and basin-attribute sources.

### Step 1 — Upstream network identification

GRDC stations are mapped to HydroATLAS basins using station coordinates. Catchment-area consistency is checked, and the upstream HydroBASINS network is traced recursively. Static basin attributes are aggregated over the contributing upstream network.

### Step 2 — Discharge extraction

Daily GRDC discharge observations are parsed and aligned to the requested study period.

### Step 3 — Precipitation aggregation

NOAA precipitation is aggregated over the upstream basin network using area-weighted spatial aggregation. The repository supports both station-based and gridded precipitation workflows.

### Step 4 — Meteorological inputs

ERA5 reanalysis and TIGGE/ECMWF forecast variables are extracted and aggregated at basin level. The implementation is designed to reuse basin-level computations across stations sharing upstream basins.

### Step 5 — Dataset reconstruction

The final tables are aligned using `(grdc_no, date)` and selected variables are normalized/checked against physical limits before modeling.

## 2. Forecasting model

The research notebook implements a Handoff Forecast LSTM. The model uses:

- 365 days of historical dynamic inputs;
- 1 forecast day;
- ERA5/hindcast meteorological inputs;
- TIGGE/ECMWF forecast inputs;
- NOAA precipitation;
- static basin attributes.

The notebook applies `log1p` to selected non-negative variables, normalizes dynamic/static inputs using training-basin statistics, and trains with MSE loss and Adam optimization.

## 3. Evaluation

Reported metrics are RMSE, Bias, Pearson correlation `r`, and NSE. The accompanying report describes an 80/10/10 basin-level train/validation/test protocol. The current notebook implements a greedy basin split based on available sequence windows; therefore, the exact split logic should be treated as an implementation detail when reproducing the reported experiment.

## 4. Important reproducibility boundary

The cleaned repository does not contain the original large raw datasets or trained checkpoints. The reported numbers are retained as research-record results from the supplied report rather than claimed as independently reproduced results by this cleanup.
