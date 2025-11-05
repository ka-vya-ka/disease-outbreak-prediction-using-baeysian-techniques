# Bayesian Parameter Estimation for Disease Outbreak Prediction

## Overview
This project implements an advanced Bayesian SIR model using the MCMC algorithm in PyMC.
It estimates infection (β) and recovery (γ) rates from COVID-19 data and forecasts infection trends.

## Installation
To install dependencies, run:
    pip install -r requirements.txt

## Project Structure
bayesian_sir_project/
├── README.md
├── report.pdf
├── requirements.txt
├── data/
│   └── nation_level_daily.csv
├── src/
│   ├── main_sir_pyro.py
│   ├── utils.py
│   └── forecast.py
└── results/

## Training & Forecasting
To train and forecast, run:
    python src/main_sir_pyro.py

## Outputs
- results/trace_plots.png — Posterior traces of β and γ
- results/forecast_plot.png — Forecast vs. Actual infections
- results/metrics.txt — RMSE and posterior summary
