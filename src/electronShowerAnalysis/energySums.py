import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# ================== CONFIG ==================
PARAMS_CSV   = r"C:\TOT_parameters.csv"
REAL_DATA_CSV= r"C:\real data.csv"
PEDESTAL_CSV = r"C:\pedestals.csv"
OUT_DIR      = r"C:\Charge_Injection_energy"

#ADC_MAX_CODE = 1023
SAT_CUTOFF   = 1020  

# ================== HELPERS ==================
def gaussian(x, A, mu, sigma):
    return A * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

def vectorized_tot_to_adc(tot, p0, p1, a, b, c, d, fallback):
    # Solve: -p0*Q^2 + [(tot - p1) + p0*b]Q + [a - (tot - p1)*b] = 0
    A = -p0
    B = (tot - p1) + p0 * b
    C = a - (tot - p1) * b
    disc = B*B - 4*A*C

    # default to fallback
    out = fallback.copy()

    valid = np.isfinite(disc) & (disc >= 0) & np.isfinite(A) & (A != 0)
    if not np.any(valid):
        return out

    sqrt_disc = np.zeros_like(disc)
    sqrt_disc[valid] = np.sqrt(disc[valid])

    Q1 = np.full_like(tot, np.nan, dtype=float)
    Q2 = np.full_like(tot, np.nan, dtype=float)
    Q1[valid] = (-B[valid] + sqrt_disc[valid]) / (2*A[valid])
    Q2[valid] = (-B[valid] - sqrt_disc[valid]) / (2*A[valid])

    # choose physical root: Q > b and Q > 0 (prefer larger)
    cand1_ok = np.isfinite(Q1) & (Q1 > b) & (Q1 > 0)
    cand2_ok = np.isfinite(Q2) & (Q2 > b) & (Q2 > 0)

    Q = np.where(cand1_ok & cand2_ok, np.maximum(Q1, Q2),
         np.where(cand1_ok, Q1,
         np.where(cand2_ok, Q2, np.nan)))

    adc_equiv = c * Q + d
    ok = np.isfinite(adc_equiv) & (adc_equiv > 0)
    out[ok] = adc_equiv[ok]
    return out

# ================== LOAD ==================
os.makedirs(OUT_DIR, exist_ok=True)

# read only needed columns to save memory
params_cols = ["layer","strip","end",
               "TOT_threshold_TOTunits","tot_asym_slope","tot_asym_intercept",
               "tot_param_a","tot_param_b","adc_fit_slope","adc_fit_intercept"]
df_params = pd.read_csv(PARAMS_CSV, usecols=params_cols)

ped_cols = ["layer","strip","end","new_range"]
df_ped = pd.read_csv(PEDESTAL_CSV, usecols=ped_cols)

real_cols = ["pf_event","layer","strip",
             "adc_sum_end0","adc_max_end0","tot_end0",
             "adc_sum_end1","adc_max_end1","tot_end1"]
df_real = pd.read_csv(REAL_DATA_CSV, usecols=real_cols)

# enforce dtypes
for c in ["layer","strip","pf_event"]:
    df_real[c] = pd.to_numeric(df_real[c], errors="coerce").astype("Int64")

# ================== MERGE LOOKUPS (vectorized) ==================
# Build per-end param tables for fast merge
p0 = df_params.rename(columns={
    "TOT_threshold_TOTunits":"TOT_thr_TOT_0",
    "tot_asym_slope":"p0_0","tot_asym_intercept":"p1_0",
    "tot_param_a":"a_0","tot_param_b":"b_0",
    "adc_fit_slope":"c_0","adc_fit_intercept":"d_0",
})
p0["end"] = p0["end"].astype(int)
p0 = p0[p0["end"]==0].drop(columns=["end"])

p1 = df_params.rename(columns={
    "TOT_threshold_TOTunits":"TOT_thr_TOT_1",
    "tot_asym_slope":"p0_1","tot_asym_intercept":"p1_1",
    "tot_param_a":"a_1","tot_param_b":"b_1",
    "adc_fit_slope":"c_1","adc_fit_intercept":"d_1",
})
p1["end"] = p1["end"].astype(int)
p1 = p1[p1["end"]==1].drop(columns=["end"])

# pedestals per end
ped0 = df_ped[df_ped["end"].astype(int)==0].rename(columns={"new_range":"new_range_0"}).drop(columns=["end"])
ped1 = df_ped[df_ped["end"].astype(int)==1].rename(columns={"new_range":"new_range_1"}).drop(columns=["end"])

# merge (layer,strip)
key = ["layer","strip"]
df = df_real.merge(ped0, on=key, how="left").merge(ped1, on=key, how="left") \
            .merge(p0, on=key, how="left").merge(p1, on=key, how="left")

# ================== PEDESTAL FILTER (per end) ==================
# keep hit if adc_sum_endX >= new_range_X
keep0 = (pd.to_numeric(df["adc_sum_end0"], errors="coerce") >= pd.to_numeric(df["new_range_0"], errors="coerce"))
keep1 = (pd.to_numeric(df["adc_sum_end1"], errors="coerce") >= pd.to_numeric(df["new_range_1"], errors="coerce"))
keep0 = keep0.fillna(False).to_numpy()
keep1 = keep1.fillna(False).to_numpy()

# ================== PER-END SELECTION (vectorized) ==================
adc_sum0 = pd.to_numeric(df["adc_sum_end0"], errors="coerce").to_numpy(dtype=float)
adc_max0 = pd.to_numeric(df["adc_max_end0"], errors="coerce").to_numpy(dtype=float)
tot0     = pd.to_numeric(df["tot_end0"],     errors="coerce").to_numpy(dtype=float)

adc_sum1 = pd.to_numeric(df["adc_sum_end1"], errors="coerce").to_numpy(dtype=float)
adc_max1 = pd.to_numeric(df["adc_max_end1"], errors="coerce").to_numpy(dtype=float)
tot1     = pd.to_numeric(df["tot_end1"],     errors="coerce").to_numpy(dtype=float)

# params arrays; NaN -> no params
TOT_thr0 = pd.to_numeric(df["TOT_thr_TOT_0"], errors="coerce").to_numpy(dtype=float)
p0_0 = pd.to_numeric(df["p0_0"], errors="coerce").to_numpy(dtype=float)
p1_0 = pd.to_numeric(df["p1_0"], errors="coerce").to_numpy(dtype=float)
a_0  = pd.to_numeric(df["a_0"],  errors="coerce").to_numpy(dtype=float)
b_0  = pd.to_numeric(df["b_0"],  errors="coerce").to_numpy(dtype=float)
c_0  = pd.to_numeric(df["c_0"],  errors="coerce").to_numpy(dtype=float)
d_0  = pd.to_numeric(df["d_0"],  errors="coerce").to_numpy(dtype=float)

TOT_thr1 = pd.to_numeric(df["TOT_thr_TOT_1"], errors="coerce").to_numpy(dtype=float)
p0_1 = pd.to_numeric(df["p0_1"], errors="coerce").to_numpy(dtype=float)
p1_1 = pd.to_numeric(df["p1_1"], errors="coerce").to_numpy(dtype=float)
a_1  = pd.to_numeric(df["a_1"],  errors="coerce").to_numpy(dtype=float)
b_1  = pd.to_numeric(df["b_1"],  errors="coerce").to_numpy(dtype=float)
c_1  = pd.to_numeric(df["c_1"],  errors="coerce").to_numpy(dtype=float)
d_1  = pd.to_numeric(df["d_1"],  errors="coerce").to_numpy(dtype=float)

has_params0 = np.isfinite(TOT_thr0) & np.isfinite(p0_0) & np.isfinite(p1_0) & np.isfinite(a_0) & np.isfinite(b_0) & np.isfinite(c_0) & np.isfinite(d_0)
has_params1 = np.isfinite(TOT_thr1) & np.isfinite(p0_1) & np.isfinite(p1_1) & np.isfinite(a_1) & np.isfinite(b_1) & np.isfinite(c_1) & np.isfinite(d_1)

tot_valid0 = np.isfinite(tot0) & (tot0 > 0)
tot_valid1 = np.isfinite(tot1) & (tot1 > 0)

adc_not_sat0 = np.isfinite(adc_max0) & (adc_max0 < SAT_CUTOFF)
adc_not_sat1 = np.isfinite(adc_max1) & (adc_max1 < SAT_CUTOFF)

# default: use adc_sum
val0 = adc_sum0.copy()
val1 = adc_sum1.copy()

# candidate to convert: keep & has_params & tot_valid & (not adc_not_sat) & (tot > TOT_thr)
cand0 = keep0 & has_params0 & tot_valid0 & (~adc_not_sat0) & np.isfinite(TOT_thr0) & (tot0 > TOT_thr0)
cand1 = keep1 & has_params1 & tot_valid1 & (~adc_not_sat1) & np.isfinite(TOT_thr1) & (tot1 > TOT_thr1)

# do vectorized conversion only on candidates
if np.any(cand0):
    val0_conv = vectorized_tot_to_adc(
        tot0[cand0], p0_0[cand0], p1_0[cand0], a_0[cand0], b_0[cand0], c_0[cand0], d_0[cand0],
        fallback=adc_sum0[cand0]
    )
    val0[cand0] = val0_conv

if np.any(cand1):
    val1_conv = vectorized_tot_to_adc(
        tot1[cand1], p0_1[cand1], p1_1[cand1], a_1[cand1], b_1[cand1], c_1[cand1], d_1[cand1],
        fallback=adc_sum1[cand1]
    )
    val1[cand1] = val1_conv

# ends that did NOT pass pedestal are invalid
val0[~keep0] = np.nan
val1[~keep1] = np.nan

# ================== END MERGE (per row -> one bar) ==================
both = np.isfinite(val0) & np.isfinite(val1)
only0 = np.isfinite(val0) & ~np.isfinite(val1)
only1 = np.isfinite(val1) & ~np.isfinite(val0)

bar_adc = np.full_like(val0, np.nan, dtype=float)
bar_adc[both] = 0.5 * (val0[both] + val1[both])
bar_adc[only0] = val0[only0]
bar_adc[only1] = val1[only1]

# ================== SUM PER EVENT (groupby, vectorized) ==================
df["pf_event"] = df["pf_event"].astype("Int64")
df_bars = pd.DataFrame({"pf_event": df["pf_event"].to_numpy(), "bar_adc": bar_adc})
df_evt  = df_bars.dropna().groupby("pf_event", as_index=False)["bar_adc"].sum()
energies = df_evt["bar_adc"].to_numpy()
energies = energies[np.isfinite(energies)]
if energies.size == 0:
    raise RuntimeError("No event energies computed after filtering/merging.")

# ================== BINNING (300 bins) ==================
emin, emax = float(np.min(energies)), float(np.max(energies))
if emin == emax:
    emax = emin + 1.0
bins = np.linspace(emin, emax, 301)
hist, edges = np.histogram(energies, bins=bins)
centers = 0.5 * (edges[:-1] + edges[1:])

# ================== GAUSSIAN FIT ==================
peak_count = hist.max()
mask = hist >= (0.2 * peak_count)
x_fit = centers[mask]
y_fit = hist[mask]

def safe_fit(xf, yf):
    if xf.size < 3 or np.all(yf==0):
        return (np.nan, np.nan, np.nan), (np.nan, np.nan, np.nan), False
    mu0 = centers[np.argmax(hist)]
    A0 = peak_count
    sigma0 = max((emax - emin)/20.0, 1.0)
    try:
        popt, pcov = curve_fit(gaussian, xf, yf, p0=[A0, mu0, sigma0], maxfev=20000)
        perr = np.sqrt(np.diag(pcov)) if pcov is not None else [np.nan, np.nan, np.nan]
        return popt, perr, True
    except Exception:
        return (np.nan, np.nan, np.nan), (np.nan, np.nan, np.nan), False

popt, perr, fit_ok = safe_fit(x_fit, y_fit)
A_fit, mu_fit, sigma_fit = popt
A_err, mu_err, sigma_err = perr

# ================== PLOTTING & FIT (single binning; 3σ points + 2σ-refined fit) ==================
REL_FRACTION = 0.20   # relative threshold for coarse boundaries
MIN_ABS      = 100.0  # absolute floor
RUN_LEN      = 2      # consecutive bins below threshold to define boundary
NBINS_ZOOM   = 200    # ONLY used once to bin [E_L, E_R]
K_SIGMA_OUT  = 3.0    # display & stage-2 window: mu1 ± 3σ1
K_SIGMA_IN   = 2.0    # final refinement window: mu2 ± 2σ2   <-- changed to 2σ

OUT_PNG_FIT = os.path.join(OUT_DIR, "energy_fit.png")
OUT_PNG_ALL = os.path.join(OUT_DIR, "energy_all.png")

emin, emax = float(np.min(energies)), float(np.max(energies))
if emin == emax:
    emax = emin + 1.0

# 1) Coarse scan to get [E_L, E_R]
bins_full = np.linspace(emin, emax, 301)
hist_full, edges_full = np.histogram(energies, bins=bins_full)
peak_full = hist_full.max()
thresh = max(MIN_ABS, REL_FRACTION * peak_full)

left_idx = None
for i in range(0, len(hist_full) - RUN_LEN + 1):
    if np.all(hist_full[i:i+RUN_LEN] < thresh):
        left_idx = i
        break
right_idx = None
for i in range(len(hist_full)-1, RUN_LEN-2, -1):
    if np.all(hist_full[i-RUN_LEN+1:i+1] < thresh):
        right_idx = i
        break

if left_idx is None: left_idx = 0
if right_idx is None: right_idx = len(hist_full) - 1
if right_idx <= left_idx:
    right_idx = min(len(hist_full) - 1, left_idx + 10)

E_L = max(edges_full[left_idx], emin)
E_R = min(edges_full[min(right_idx + 1, len(edges_full)-1)], emax)
if E_R <= E_L:
    E_R = E_L + (emax - emin) / 10.0

# 2) Single binning ONCE in [E_L, E_R] (200 bins)
bins_zoom = np.linspace(E_L, E_R, NBINS_ZOOM + 1)
hist_zoom, edges_zoom = np.histogram(energies, bins=bins_zoom)
centers_zoom = 0.5 * (edges_zoom[:-1] + edges_zoom[1:])

# First fit across [E_L, E_R] with light smoothing (for stable mu1, sigma1)
if len(hist_zoom) >= 3:
    kernel = np.array([1.0, 1.0, 1.0]) / 3.0
    y_fit1 = np.convolve(hist_zoom, kernel, mode="same")
else:
    y_fit1 = hist_zoom.copy()
x_fit1 = centers_zoom

def _safe_gauss_fit(xf, yf, lo, hi):
    """Safe Gaussian fit with guards and initial guesses."""
    m = np.isfinite(xf) & np.isfinite(yf) & (yf > 0)
    xf, yf = xf[m], yf[m]
    if xf.size < 5:
        return (np.nan, np.nan, np.nan), (np.nan, np.nan, np.nan), False
    A0 = float(np.max(yf))
    mu0 = float(xf[np.argmax(yf)])
    sigma0 = max((hi - lo) / 20.0, 1.0)
    try:
        popt, pcov = curve_fit(gaussian, xf, yf, p0=[A0, mu0, sigma0], maxfev=20000)
        perr = np.sqrt(np.diag(pcov)) if pcov is not None else [np.nan, np.nan, np.nan]
        return popt, perr, True
    except Exception:
        return (np.nan, np.nan, np.nan), (np.nan, np.nan, np.nan), False

popt1, perr1, ok1 = _safe_gauss_fit(x_fit1, y_fit1, E_L, E_R)
A1, mu1, sigma1 = popt1

# 3) Stage-2 (3σ window) = select subset of the 200 bins (NO rebin)
if ok1 and np.isfinite(mu1) and np.isfinite(sigma1) and sigma1 > 0:
    win2_L = max(E_L, mu1 - K_SIGMA_OUT * sigma1)
    win2_R = min(E_R, mu1 + K_SIGMA_OUT * sigma1)
else:
    win2_L, win2_R = E_L, E_R
if win2_R <= win2_L:
    win2_R = win2_L + (E_R - E_L) / 20.0

mask_3sig = (centers_zoom >= win2_L) & (centers_zoom <= win2_R)
centers_3sig = centers_zoom[mask_3sig]
hist_3sig    = hist_zoom[mask_3sig]

popt2, perr2, ok2 = _safe_gauss_fit(centers_3sig, hist_3sig, win2_L, win2_R)
A2, mu2, sigma2 = popt2

# 4) Stage-3 (2σ refinement) = select smaller subset from the SAME 200 bins
if ok2 and np.isfinite(mu2) and np.isfinite(sigma2) and sigma2 > 0:
    win3_L = max(win2_L, mu2 - K_SIGMA_IN * sigma2)  # K_SIGMA_IN = 2.0
    win3_R = min(win2_R, mu2 + K_SIGMA_IN * sigma2)
else:
    win3_L, win3_R = win2_L, win2_R
if win3_R <= win3_L:
    win3_R = win3_L + (win2_R - win2_L) / 20.0

mask_1sig = (centers_zoom >= win3_L) & (centers_zoom <= win3_R)
centers_1sig = centers_zoom[mask_1sig]
hist_1sig    = hist_zoom[mask_1sig]

# Final fit uses ONLY those bins inside 2σ window (no rebinning)
popt3, perr3, ok3 = _safe_gauss_fit(centers_1sig, hist_1sig, win3_L, win3_R)
A_fit, mu_fit, sigma_fit = popt3
A_err, mu_err, sigma_err = perr3

# Quality metrics on FINAL (2σ) subset
if ok3 and np.isfinite(mu_fit) and np.isfinite(sigma_fit) and sigma_fit > 0:
    y_hat = gaussian(centers_1sig, A_fit, mu_fit, sigma_fit)
    ss_res = np.sum((hist_1sig - y_hat) ** 2)
    ss_tot = np.sum((hist_1sig - np.mean(hist_1sig)) ** 2)
    R2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan
    var = np.maximum(y_hat, 1.0)  # Poisson approx
    chi2 = np.sum((hist_1sig - y_hat) ** 2 / var)
    ndf = max(len(hist_1sig) - 3, 1)
    chi2_ndf = float(chi2 / ndf)
else:
    R2 = np.nan
    chi2_ndf = np.nan

# 5) Plot FIT: points = 3σ subset; curve = ONLY within the 2σ window; x-axis limited to 3σ window
plt.figure(figsize=(9, 6))
plt.scatter(centers_3sig, hist_3sig, s=12)  # no legend
if ok3 and np.isfinite(mu_fit) and np.isfinite(sigma_fit) and sigma_fit > 0:
    x_dense = np.linspace(max(win3_L, centers_1sig.min()), min(win3_R, centers_1sig.max()), 600)
    plt.plot(x_dense, gaussian(x_dense, A_fit, mu_fit, sigma_fit), linewidth=2)
plt.yscale("log")
plt.xlim(win2_L, win2_R)
plt.xlabel("Total energy (ADC_sum equivalent)")
plt.ylabel("Events per bin")
plt.title("Energy distribution")
# annotation box
txt = []
if ok3:
    txt.append(f"MPV (mu): {mu_fit:.1f} ± {mu_err:.1f} ADC_sum")
    txt.append(f"Std dev (sigma): {sigma_fit:.1f} ± {sigma_err:.1f}")
    txt.append(f"A: {A_fit:.1f} ± {A_err:.1f}")
    txt.append(f"R²: {R2:.3f}   χ²/ndf: {chi2_ndf:.2f}")
else:
    txt.append("Gaussian fit failed")
plt.text(0.98, 0.98, "\n".join(txt), ha="right", va="top",
         transform=plt.gca().transAxes, bbox=dict(boxstyle="round", alpha=0.12))
plt.tight_layout()
plt.savefig(OUT_PNG_FIT, dpi=200)
plt.close()
print(f"Saved plot: {OUT_PNG_FIT}")

# 6) Plot ALL: full energy distribution, 300 bins from 0 to max, no fit line, y >= 1e1 only
bins_all = np.linspace(0.0, emax, 301)  # 300 bins from 0 to max
hist_all, edges_all = np.histogram(energies, bins=bins_all)
centers_all = 0.5 * (edges_all[:-1] + edges_all[1:])
mask_all = hist_all >= 10.0  # only show bins with count >= 10

plt.figure(figsize=(9, 6))
plt.scatter(centers_all[mask_all], hist_all[mask_all], s=10)  # no legend, no line
plt.yscale("log")
plt.ylim(bottom=10.0)  # hide counts < 10
plt.xlabel("Total energy (ADC_sum equivalent)")
plt.ylabel("Events per bin")
plt.title("Energy distribution")
plt.tight_layout()
plt.savefig(OUT_PNG_ALL, dpi=200)
plt.close()
print(f"Saved plot: {OUT_PNG_ALL}")