# Physical Reservoir Computing — SLM/CCD Pipeline
## Quantlase Lab ML Internship, Abu Dhabi

Implementation of physical reservoir computing using a 
Spatial Light Modulator (SLM) and CCD camera as the 
optical reservoir, alongside a software Echo State 
Network (ESN) reference implementation for comparison.

---

## What this project covers

**Physical RC Pipeline (imaging/)**
Uses a 1600×1152 SLM divided into 32-pixel macropixels 
(1,800 reservoir nodes) as a physical optical reservoir. 
CCD camera output is captured and block-averaged across 
the macropixel grid to extract reservoir states from 
real light interference patterns.

- `codeclass.py` — CCDProcessor class: maps SLM 
  macropixels to CCD coordinates, performs block 
  averaging to extract optical reservoir states
- `parameter_settings.ini` — hardware configuration 
  (CCD image map boundaries, macropixel geometry)

**Software ESN Reference (esn/)**
Computational ESN implementation for S&P 500 time 
series forecasting — used as a software baseline 
for the physical reservoir approach.

- `esn.py` — ESN class: InputLayer, ReservoirLayer 
  (leaky integrator, spectral radius normalisation), 
  OutputLayer (Ridge Regression readout)
- `RC_StockPriceSPY.py` — forecasting pipeline: 
  data ingestion, training, evaluation (MSE per 
  dimension), autonomous prediction mode

---

## Tech Stack

Python · NumPy · SciPy · OpenCV · Pandas · 
Matplotlib · configparser

---

## Note on ESN implementation

The software ESN builds on reservoir computing 
principles from published literature and an open-source 
reference implementation. The physical RC pipeline 
(CCD/SLM integration) was developed independently 
as part of the internship.
