# ISA 444 Final Project: Hotel Demand Forecasting

## Overview
This project forecasts daily normalized room demand (occupancy rate) for 
19 hotel properties across a 28-day horizon. The goal is to compare 
traditional, machine learning, neural, and foundation model approaches 
and explain which methods perform best and why.

## Forecasting Task
- **Target:** Daily normalized room demand (0 = empty, 1 = fully booked)
- **Series:** 19 hotel properties
- **Horizon:** 28 days (h = 28)
- **Evaluation:** 5-fold time-series cross-validation, non-overlapping

## Models Compared


**| Family | Model |**

| Baseline | Naive, SeasonalNaive, RandomWalkWithDrift |

| Statistical | AutoETS, AutoARIMA |

| Machine Learning | LightGBM (via MLForecast) |

| Neural | AutoNBEATS, AutoNHITS (via NeuralForecast) |

| Foundation | Chronos (via TimeCopilot) |

## Evaluation Metrics
- ME (Bias)
- MAE
- RMSE
- MAPE (used cautiously — see findings)
- MASE (seasonality = 7)

## Key Findings
LightGBM was the clear winner, taking 75 out of 90 possible wins on MAE, 
MAPE, MASE, and RMSE. This makes sense given the dataset and the on-the-books 
(OTB) which give LightGBM access to rooms already reserved before the 
forecast period starts, which is information the other models don't have.

AutoNBEATS was more competitive on the bias metric, winning 18 series. 
It was better at staying centered around true demand even when 
its point forecasts weren't the most accurate overall. SeasonalNaive held 
its own on some of the series too, which makes sense for hotels with very 
predictable weekly patterns.

MAPE is not very reliable for this dataset. Because some hotels had zero 
occupancy on certain days, MAPE becomes undefined or inflated whenever 
the actual value hits zero. MAE and RMSE are the better metrics to rely 
on. Since demand is already scaled between 0 and 1, an MAE of 0.05 
means forecasts were off by about 5 percentage points of occupancy on 
average.

**Results by Metric**

**MAE:** LightGBM was the clear winner. It had an average MAE of 0.037 compared 
to the next best Chronos at 0.101. It won 75 out of 90 series. The otb gave it a huge edge.

**RMSE:** LightGBM averaged 0.050 while everyone else was above 0.12. It won 76 series. 
The gap here actually shows that LightGBM wasn't just better on average, but it also 
avoided the big misses that RMSE penalizes.

**MAPE:** LightGBM again at 0.108 average vs Chronos at 0.214. It is important 
to note that MAPE isn't super reliable here because some hotels hit 
zero occupancy on certain days, which makes MAPE break down. So take 
these numbers with a grain of salt and lean more on MAE and RMSE.

**MASE:** LightGBM averaged 0.357. A MASE under 1 means it actually beat 
the Seasonal Naive benchmark, which none of the other models managed to do consistently.

**Bias(ME):** LightGBM wasn't the winner here. AutoNBEATS won 18 
series and AutoETS took 15. Most models slightly under predicted demand on average, 
which makes sense given the gaps of 0 pulling averages down.

**AutoNHITS:** Showed inf across all metrics, which means something went 
wrong during that model's cross-validation. It should be excluded from any metric 
comparisons.

## Winning Model Forecast
Based on the cross-validation results, LightGBM was selected as the 
winning model and used to generate the final 28-day occupancy forecasts 
for all 19 hotel properties. These forecasts are saved in champion_lgbm_forecasts.csv

## Important:
I had colab assist me with the last code box.
