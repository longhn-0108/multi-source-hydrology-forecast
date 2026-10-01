# Reported Results

The values below are transcribed from the supplied research report, **Hydrological Prediction in Ungauged Basins**.

## Baseline one-day-ahead forecasting

| Setting | RMSE ↓ | Bias ↓ | Pearson r ↑ | NSE ↑ |
|---|---:|---:|---:|---:|
| Caravan | 1.839 | -0.916 | 0.044 | -1.093 |
| **Ours — Handoff Forecast LSTM** | **2.094** | **-0.001** | **0.806** | **0.634** |

The report describes the proposed model as having stronger temporal agreement and positive NSE on the constructed test set, while the Caravan comparison has near-zero correlation and negative NSE.

## Ablation study

| Setting | RMSE ↓ | Bias ↓ | Pearson r ↑ | NSE ↑ |
|---|---:|---:|---:|---:|
| All inputs | 2.094 | -0.001 | 0.806 | 0.634 |
| w/o `tp` | 2.025 | -0.173 | 0.830 | 0.658 |
| w/o `precip_NOAA` | 2.899 | -0.421 | 0.775 | 0.597 |
| w/o static attributes | 2.853 | 0.925 | 0.784 | 0.320 |

According to the report, removing NOAA precipitation increases error and reduces correlation/NSE, while removing static attributes causes a larger degradation in RMSE and NSE. Removing `tp` is reported to slightly improve RMSE, correlation, and NSE in this experimental setting.

## Scope of these numbers

These are **reported research results from the supplied project report**. They are included here for portfolio transparency and are not presented as newly reproduced results from this cleaned GitHub repository.
