
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# --- SIR Model Differential Equations ---
def SIR_ODE(y, t, beta, gamma, N):
    S, I, R = y
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return [dSdt, dIdt, dRdt]

# --- Plotting Function for Comparison ---
def plot_results(dates, I_obs, I_pred, train_size, save_path=None):
    plt.figure(figsize=(10,6))
    plt.plot(dates, I_obs, 'r', label='Observed Infected')
    plt.plot(dates, I_pred, 'k--', label='Predicted Infected')
    plt.axvline(dates[train_size], color='gray', linestyle='--', label='Train/Test Split')
    plt.legend()
    plt.xlabel('Date')
    plt.ylabel('Infected Population')
    plt.title('Bayesian SIR Forecast (PyMC)')
    plt.grid(True, alpha=0.3)
    if save_path:
        plt.savefig(save_path)
    plt.show()
