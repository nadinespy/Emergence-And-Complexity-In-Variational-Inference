import numpy as np
from matplotlib import pyplot as plt
from numba import njit
import os

# Create 'img' directory if it doesn't exist
os.makedirs('../img', exist_ok=True)

# Configure matplotlib for LaTeX typesetting and aesthetics
plt.rc('text', usetex=True)
plt.rc('font', size=15, family='serif', serif=['latin modern roman'])
plt.rc('legend', fontsize=15)
plt.rc('text.latex', preamble=r'\usepackage{amsmath,bm}')

# Get 3 distinct colors from the inferno colormap
cmap = plt.colormaps.get_cmap('inferno')
colors = [cmap(0.75), cmap(0.5), cmap(0.25)]

# --- Helper functions ---

def autocorrelation(x, max_lag):
    """Compute the autocorrelation of a signal up to max_lag."""
    x = x - np.mean(x)
    var = np.dot(x, x) / len(x)  # Variance = autocovariance at lag 0
    T = len(x)
    acorr = np.array([
        np.dot(x[:T - lag], x[lag:]) / (T - lag) / var
        for lag in range(max_lag + 1)
    ])
    return acorr

@njit
def simulate_linear_sde(A2, gamma, T):
    """Simulate a linear stochastic differential equation."""
    x = np.zeros((2, T))
    x[:, 0] = np.random.randn(2)
    for t in range(T - 1):
        noise = 0.01 * np.random.randn(2)
        x[:, t + 1] = x[:, t] - gamma * A2 @ (x[:, t] - noise)
    return x

# --- Model parameters ---

# Time lag for computing cross-covariances
time_lag = 1

# Identity matrix
I = np.eye(2)

# Simulation time steps
T = 10_000_000

# Dynamical system parameters
gamma = 0.01  # Learning rate
rho = 0.99    # Correlation between variables
weight = 1.0  # Weighting of off-diagonal coupling

# Reference covariance matrix
S = np.array([[1, np.sqrt(1*1)*rho],
              [np.sqrt(1*1)*rho, 1]])

# Precision (inverse covariance) matrix
A = np.linalg.inv(S)

# Modified precision matrix with a different coupling weight
A2 = A.copy()
A2[0,1] *= weight
A2[1,0] *= weight

print("Modified precision matrix A2:")
print(A2)

# Effective noise covariance matrix
K = np.linalg.inv(A2 @ I @ A2.T) / gamma**2

# Steady-state covariance matrices at lag 0 and lag 1
Sx0 = np.linalg.inv(K - (I - gamma * A2) @ K @ (I - gamma * A2))
Sx1 = (I - gamma * A2)**time_lag @ Sx0

print("Covariance matrix Sx0 (lag 0):")
print(Sx0)
print("Covariance matrix Sx1 (lag 1):")
print(Sx1)

# Full 4x4 joint covariance matrix of (x(t), x(t+1))
Sx = np.block([[Sx0, Sx1],
               [Sx1.T, Sx0]])

print("Full joint covariance matrix Sx:")
print(Sx)

# Submatrices for each variable
S_part1 = np.block([[Sx0[0,0], Sx1[0,0]],
                    [Sx1[0,0], Sx0[0,0]]])
S_part2 = np.block([[Sx0[1,1], Sx1[1,1]],
                    [Sx1[1,1], Sx0[1,1]]])

print("Submatrix S_part1 (variable x1):")
print(S_part1)

# --- Mutual information computation ---

# Mutual information of (x1, x2) jointly
MI_all = -0.5 * np.log(np.linalg.det(Sx)) + 2 * 0.5 * np.log(np.linalg.det(Sx0))

# Mutual information of x1 separately
MI_part1 = -0.5 * np.log(np.linalg.det(S_part1)) + 2 * 0.5 * np.log(S_part1[0,0])

# Mutual information of x2 separately
MI_part2 = -0.5 * np.log(np.linalg.det(S_part2)) + 2 * 0.5 * np.log(S_part2[0,0])

# Integrated information (synergy)
phi = MI_all - MI_part1 - MI_part2

print(f"MI_all={MI_all:.4f}, MI_part1={MI_part1:.4f}, MI_part2={MI_part2:.4f}, phi={phi:.4f}")

# --- Simulations ---

# Simulate the linear SDE
x = simulate_linear_sde(A2, gamma, T)


# --- Autocorrelation plots ---

max_lag = 500

# Compute autocorrelations
acf_x1 = autocorrelation(x[0], max_lag)
acf_x2 = autocorrelation(x[1], max_lag)
acf_sum = autocorrelation(x[0] + x[1], max_lag)

lags = np.arange(max_lag + 1)

plt.figure()
plt.plot(lags, acf_x1, color=colors[0], label=r'$x_1$', linewidth=2)
plt.plot(lags, acf_x2, color=colors[1], linestyle='--', label=r'$x_2$', linewidth=2)
plt.plot(lags, acf_sum, color=colors[2], label=r'$x_1 + x_2$', linewidth=2)

plt.xlabel(r'$\tau$')
plt.ylabel(r'$\mathrm{Corr}(\tau)$')
plt.legend()
plt.savefig('../img/Corr.pdf', bbox_inches='tight')

# --- Detailed time series plots ---

fig, axes = plt.subplots(3, 1, sharex=True)
steps = 1000
time = np.arange(steps)

# Extract recent samples
x1 = x[0, -steps:]
x2 = x[1, -steps:]
xsum = x1 + x2

# Plot x1
v1 = np.max(np.abs(x1))
axes[0].plot(time, x1, color=colors[0], linewidth=1.5)
axes[0].set_ylabel(r'$x_1$')
axes[0].set_xlim(0, steps-1)
axes[0].set_ylim(-v1, v1)

# Plot x2
v2 = np.max(np.abs(x2))
axes[1].plot(time, x2, color=colors[1], linewidth=1.5)
axes[1].set_ylabel(r'$x_2$')
axes[1].set_xlim(0, steps-1)
axes[1].set_ylim(-v2, v2)

# Plot x1+x2
vsum = np.max(np.abs(xsum))
axes[2].plot(time, xsum, color=colors[2], linewidth=1.5)
axes[2].set_ylabel(r'$x_1 + x_2$')
axes[2].set_xlabel('time')
axes[2].set_xlim(0, steps-1)
axes[2].set_ylim(-vsum, vsum)

plt.tight_layout()

plt.savefig('../img/trajectories.pdf', bbox_inches='tight')
plt.show()

