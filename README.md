# Time Series Forecasting with Echo State Networks
## ML Pipeline Engineering — Quantlase Lab Internship

End-to-end time series forecasting pipeline using Reservoir Computing 
(Echo State Networks), applied to S&P 500 closing price prediction.

Built as part of an ML engineering internship at Quantlase Lab,
Abu Dhabi — focused on refactoring and productionising ML code.

---

## What was built

- Refactored an unstructured ML script into a clean, modular, 
  class-based Python package (`esn.py`) with separated 
  preprocessing, training, and evaluation components
- Built a complete forecasting pipeline: data ingestion from 
  CSV → normalisation → reservoir state computation → 
  readout training → RMSE evaluation → visualisation
- Applied the pipeline to SPY (S&P 500 ETF) historical close 
  price data

---

## Tech Stack

Python · NumPy · Pandas · Matplotlib · Scikit-learn

---

## Structure

esn/
├── esn.py                  # ESN class — reservoir + readout
├── RC_StockPriceSPY.py     # forecasting pipeline
└── SPY.csv                 # S&P 500 historical data

---

## Note

This project was completed under internship supervision at 
Quantlase Lab and is for educational and demonstrative purposes 
only. Not intended for financial use.
