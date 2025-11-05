
import pandas as pd
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from utils import SIR_ODE, plot_results

# Load dataset and posterior results
df = pd.read_csv("bayesian_sir_project/data/nation_level_daily.csv")
df['Date'] = pd.to_datetime(df['Date'] + ' 2020', format='%d %B %Y', errors='coerce', dayfirst=True)
N = 1.4e9

train_size = int(0.8 * len(df))
train_df = df.iloc[:train_size]

with open("bayesian_sir_project/results/metrics.txt") as f:
    lines = f.readlines()
beta_post = float(lines[1].split(":")[1])
gamma_post = float(lines[2].split(":")[1])

S0, I0, R0 = train_df.loc[0, ["S","I","R"]].astype(float).to_numpy()
t_future = np.arange(len(df) + 30)
forecast = odeint(SIR_ODE, [S0, I0, R0], t_future, args=(beta_post, gamma_post, N))
I_forecast = forecast[:, 1]

plot_results(df["Date"], df["I"], I_forecast[:len(df)], train_size, save_path="bayesian_sir_project/results/forecast_extended.png")
