
import pandas as pd
import numpy as np
import pymc as pm
from scipy.integrate import odeint
import pytensor.tensor as pt
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from utils import SIR_ODE, plot_results

# --- Load Dataset ---
df = pd.read_csv("bayesian_sir_project/data/nation_level_daily.csv")
df['Date'] = pd.to_datetime(df['Date'] + ' 2020', format='%d %B %Y', errors='coerce', dayfirst=True)
df = df.sort_values('Date').reset_index(drop=True)

N = 1.4e9
df['Active'] = df['Daily Confirmed'] - df['Daily Recovered'] - df['Daily Deceased']
df['Removed'] = df['Daily Recovered'] + df['Daily Deceased']
df['Susceptible'] = N - df['Active'] - df['Removed']
df['S'], df['I'], df['R'] = df['Susceptible'], df['Active'], df['Removed']

# --- Train-Test Split ---
train_size = int(0.8 * len(df))
train_df, test_df = df.iloc[:train_size], df.iloc[train_size:]

# --- Prepare Training Data ---
t = np.arange(len(train_df), dtype=float)
I_obs = train_df["I"].astype(float).to_numpy()
S0, I0, R0 = train_df.loc[0, ["S", "I", "R"]].astype(float).to_numpy()

# --- Define Bayesian Model ---
class SIR_Solver(pm.CustomOp):
    def __init__(self, y0, t, N):
        self.y0, self.t, self.N = y0, t, N
        self.itypes, self.otypes = [pt.dscalar, pt.dscalar], [pt.dmatrix]
    def perform(self, node, inputs, outputs):
        beta, gamma = inputs
        sol = odeint(SIR_ODE, self.y0, self.t, args=(beta, gamma, self.N))
        outputs[0][0] = np.asarray(sol, dtype=np.float64)

sir_solver = SIR_Solver(y0=[S0, I0, R0], t=t, N=N)

with pm.Model() as model:
    beta = pm.TruncatedNormal("beta", mu=0.4, sigma=0.2, lower=0)
    gamma = pm.TruncatedNormal("gamma", mu=0.1, sigma=0.05, lower=0)
    sigma = pm.HalfNormal("sigma", sigma=1e6)

    sol = sir_solver(beta, gamma)
    I_hat = sol[:, 1]
    I_obs_ = pm.Normal("I_obs_", mu=I_hat, sigma=sigma, observed=I_obs)

    trace = pm.sample(1000, tune=1000, target_accept=0.9, random_seed=42, cores=2)

pm.plot_trace(trace, var_names=["beta", "gamma"])
plt.savefig("bayesian_sir_project/results/trace_plots.png")

# --- Forecasting ---
beta_post = trace.posterior["beta"].mean().item()
gamma_post = trace.posterior["gamma"].mean().item()
print(f"Posterior mean β = {beta_post:.4f}, γ = {gamma_post:.4f}, R₀ = {beta_post/gamma_post:.2f}")

t_future = np.arange(len(df))
forecast = odeint(SIR_ODE, [S0, I0, R0], t_future, args=(beta_post, gamma_post, N))
I_forecast = forecast[:, 1]

# --- Evaluation ---
I_pred_test = I_forecast[len(train_df):len(df)]
rmse = np.sqrt(mean_squared_error(test_df["I"], I_pred_test))
with open("bayesian_sir_project/results/metrics.txt", "w") as f:
    f.write(f"RMSE on Test Data: {rmse:.2f}\nPosterior β: {beta_post:.4f}\nPosterior γ: {gamma_post:.4f}\nR0: {beta_post/gamma_post:.2f}")

plot_results(df["Date"], df["I"], I_forecast, train_size, save_path="bayesian_sir_project/results/forecast_plot.png")
