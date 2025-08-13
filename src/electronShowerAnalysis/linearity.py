import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os

# ---------- data ----------
beam_energy = np.array([300, 400, 500, 1000, 4000])
reco_energy = np.array([8170, 11484, 14787, 30490, 117908])

# std and count
stds = np.array([3527, 4135, 4687, 7515, 13170])
counts = np.array([50, 50, 50, 50, 50])
errors = stds / np.sqrt(counts)  
ratios = stds / reco_energy
ratio_err = errors / reco_energy

# ---------- function for fitting ----------
def resolution_model(E, a, b, c):
    return np.sqrt((a**2)/(E**2) + (b**2)/E + c**2)

# --------------------
p0 = [20, 5, 0.01]  
popt, pcov = curve_fit(resolution_model, beam_energy, ratios, sigma=ratio_err, absolute_sigma=True, p0=p0)
a, b, c = popt

# ---------- curve for fitting ----------
x_line = np.linspace(300, 4000, 400)
y_line = resolution_model(x_line, *popt)

# ---------- figures ----------
plt.figure(figsize=(10, 6))


plt.errorbar(beam_energy, ratios, yerr=ratio_err, fmt='o', color='black', ecolor='black', elinewidth=2, capsize=4)


plt.plot(x_line, y_line, color='blue', linewidth=2)

# style
plt.xlabel('Beam Energy (MeV)', fontsize=24)
plt.ylabel(r'$\sigma(E)/E$', fontsize=24)
plt.title('Energy Resolution Fit', fontsize=24)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.xlim(0, 4500)
plt.ylim(0, 0.5)
plt.tight_layout()

# ---------- save ----------
output_folder =
os.makedirs(output_folder, exist_ok=True)
plt.savefig(os.path.join(output_folder, "resolution_model_fit.png"), dpi=300)
plt.show()
