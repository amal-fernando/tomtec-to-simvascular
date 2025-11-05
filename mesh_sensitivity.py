# import pandas as pd
# import numpy as np
# from math import log
# from scipy.optimize import brentq
#
# # ===============================
# # Parametri delle tre mesh
# #   a = volume/area caratteristica (mm^3) da cui ricaviamo h = a^(1/3)
# #   file: separatore auto (tab o virgola)
# # ===============================
# meshes = {
#     "Coarse": {"file": "coarse.csv", "a": 0.29},
#     "Medium": {"file": "medium.csv", "a": 0.19},
#     "Fine":   {"file": "fine.csv",   "a": 0.08},
# }
#
# # === Funzione per isolare il ciclo centrale ===
# def select_middle_cycle(df):
#     n = len(df)
#     one_third = n // 3
#     start = one_third
#     end = 2 * one_third
#     return df.iloc[start:end].reset_index(drop=True)
#
# # === Funzione per calcolare le metriche principali ===
#
# def extract_metrics(df):
#     """Calcola metriche globali da un ciclo."""
#     metrics = {
#         "Pvol(t)": df["volume_pressure_mmHg"].values,
#         "Qout(t)": df["outlet_flow_mL/s"].values,
#         "KE(t)": df["KE_total_mJ"].values,
#         "Vort(t)": df["Vorticity_mean"],
#         # Valori medi/picco
#         "mean_P": df["volume_pressure_mmHg"].mean(),
#         "max_P": df["volume_pressure_mmHg"].max(),
#         "mean_Qout": df["outlet_flow_mL/s"].mean(),
#         "max_Qout": abs(df["outlet_flow_mL/s"]).max(),
#         "mean_KE": df["KE_total_mJ"].mean(),
#         "max_KE": df["KE_total_mJ"].max(),
#         "mean_vort": df["Vorticity_mean"].mean(),
#         "max_vort": df["Vorticity_mean"].max()
#     }
#     return metrics
#
# # === Funzioni per errore e ordine di convergenza ===
# def observed_order_p(phi_c, phi_m, phi_f, r_cm, r_mf):
#     """
#     Risolve per p:  ( (phi_c - phi_m)/(phi_m - phi_f) ) = (r_cm^p - 1)/(r_mf^p - 1)
#     con root-finding (brentq). Fallback al p "uguale-r" se necessario.
#     """
#     R = (phi_c - phi_m) / (phi_m - phi_f)
#     # funzione da azzerare
#     f = lambda p: (r_cm**p - 1.0)/(r_mf**p - 1.0) - R
#     # guess: formula “uguale-r” con r_eff
#     r_eff = (r_cm*r_mf)**0.5
#     p0 = log((phi_c - phi_m)/(phi_m - phi_f))/log(r_eff)
#     # cerca p in un intervallo ragionevole (0.1–5)
#     # gestisci casi patologici con try/except
#     try:
#         p = brentq(f, 0.1, 8.0)
#     except ValueError:
#         p = p0  # fallback
#     return p
#
# def GCI(phi_f, phi_m, phi_c, p, r_mf, r_cm, Fs=1.25):
#     GCI_mf = Fs * abs(phi_f - phi_m)/abs(phi_f) / (r_mf**p - 1.0) * 100.0
#     GCI_mc = Fs * abs(phi_m - phi_c)/abs(phi_m) / (r_cm**p - 1.0) * 100.0
#     return GCI_mf, GCI_mc
#
# def rel_error(phi1, phi2):
#     return abs(phi2 - phi1) / abs(phi1) * 100
#
# def order_p(phi_c, phi_m, phi_f, r):
#     return np.log((phi_c - phi_m) / (phi_m - phi_f)) / np.log(r)
#
# def rms_error(phi2_t, phi1_t):
#     """Errore RMS percentuale su tutto il ciclo."""
#     return np.sqrt(np.mean(((phi2_t - phi1_t) / phi1_t) ** 2)) * 100
#
# def mean_abs_error(phi2_t, phi1_t):
#     """Errore medio assoluto percentuale su tutto il ciclo."""
#     return np.mean(np.abs((phi2_t - phi1_t) / phi1_t)) * 100
#
# # === Lettura, preprocessing e calcolo ===
# results = {}
# for key, info in meshes.items():
#     df = pd.read_csv(info["file"], sep="\t|,", engine="python")
#     if "HDF_N" in df.columns:
#         df = df.drop(columns=["HDF_N"])
#     df = df.dropna(axis=1, how="all")  # elimina colonne completamente vuote
#     df = (df.apply(pd.to_numeric, errors="coerce")).dropna()  # converte in numerico e rimuove righe con NaN
#     df = select_middle_cycle(df)
#     results[key] = extract_metrics(df)
#
# # === Conversione in DataFrame ===
# df_metrics = pd.DataFrame(results).T
# df_metrics["a(mm^3)"] = [m["a"] for m in meshes.values()]
# df_metrics["h(mm)"] = df_metrics["a(mm^3)"] ** (1 / 3)
# print("\n=== Mean metrics per mesh (central cycle) ===\n")
# print(df_metrics.round(3))
#
# # === Calcolo di epsilon e ordine p ===
# r = (0.29 ** (1 / 3)) / (0.19 ** (1 / 3))
#
# # === Calcolo errori globali ===
# print("\n=== Global (mean/picco) errors and observed order ===\n")
# for metric in ["mean_ΔP", "mean_KE", "mean_EL"]:
#     eps_CM = rel_error(df_metrics.loc["Coarse", metric], df_metrics.loc["Medium", metric])
#     eps_MF = rel_error(df_metrics.loc["Medium", metric], df_metrics.loc["Fine", metric])
#     p_val = order_p(df_metrics.loc["Coarse", metric],
#                     df_metrics.loc["Medium", metric],
#                     df_metrics.loc["Fine", metric], r)
#     print(f"{metric:10s} | ε(C-M)={eps_CM:6.2f}% | ε(M-F)={eps_MF:6.2f}% | p={p_val:4.2f}")
#
# # === Calcolo errori temporali RMS e media assoluta ===
# print("\n=== Time-resolved errors (RMS and mean) ===\n")
# for var in ["ΔP(t)", "Qout(t)", "KE(t)", "EL(t)"]:
#     eps_rms = rms_error(results["Medium"][var], results["Fine"][var])
#     eps_mean = mean_abs_error(results["Medium"][var], results["Fine"][var])
#     print(f"{var:8s} | ε_RMS={eps_rms:5.2f}% | ε_mean={eps_mean:5.2f}%")
#
# print("\n=== End of analysis ===\n")


import pandas as pd
import numpy as np
from math import log
from scipy.optimize import brentq

# ==========================================================
# Parametri delle tre mesh
# ==========================================================
meshes = {
    "Coarse": {"file": "coarse.csv", "a": 0.29},
    "Medium": {"file": "medium.csv", "a": 0.19},
    "Fine":   {"file": "fine.csv",   "a": 0.08},
}

# ==========================================================
# Utility functions
# ==========================================================
def _safe_rel_err(phi_ref, phi_test):
    denom = abs(phi_ref) if abs(phi_ref) > 1e-16 else 1e-16
    return abs(phi_test - phi_ref) / denom * 100.0

def rms_error(phi_ref_t, phi_test_t):
    phi_ref_t = np.asarray(phi_ref_t)
    phi_test_t = np.asarray(phi_test_t)
    denom = np.where(np.abs(phi_ref_t) == 0, 1e-16, np.abs(phi_ref_t))
    return np.sqrt(np.mean(((phi_test_t - phi_ref_t) / denom) ** 2)) * 100

def mean_abs_error(phi_ref_t, phi_test_t):
    phi_ref_t = np.asarray(phi_ref_t)
    phi_test_t = np.asarray(phi_test_t)
    denom = np.where(np.abs(phi_ref_t) == 0, 1e-16, np.abs(phi_ref_t))
    return np.mean(np.abs((phi_test_t - phi_ref_t) / denom)) * 100

def select_middle_cycle(df):
    n = len(df)
    if n < 3:
        return df.reset_index(drop=True)
    one_third = n // 3
    return df.iloc[one_third:2 * one_third].reset_index(drop=True)

# ==========================================================
# Estrazione metriche globali e temporali
# ==========================================================
def extract_metrics(df):
    """Calcola metriche globali da un ciclo."""
    metrics = {
        "Pvol(t)": df["volume_pressure_mmHg"].values,
        "Qout(t)": df["outlet_flow_mL_s"].values,
        "KE(t)": df["KE_total_mJ"].values,
        # Valori medi/picco
        "mean_P": df["volume_pressure_mmHg"].mean(),
        "max_P": df["volume_pressure_mmHg"].max(),
        "mean_Qout": df["outlet_flow_mL_s"].mean(),
        "max_Qout": abs(df["outlet_flow_mL_s"]).max(),
        "mean_KE": df["KE_total_mJ"].mean(),
        "max_KE": df["KE_total_mJ"].max()
    }
    return metrics

# ==========================================================
# NASA GCI functions
# ==========================================================
def observed_order_p(phi_c, phi_m, phi_f, r_cm, r_mf):
    """Osserva p risolvendo la relazione NASA GCI per r non uguali."""
    num = (phi_c - phi_m)
    den = (phi_m - phi_f)
    if den == 0 or np.isnan(num) or np.isnan(den):
        return np.nan
    R = num / den
    def f(p): return (r_cm**p - 1) / (r_mf**p - 1) - R
    r_eff = np.sqrt(r_cm * r_mf)
    try:
        p0 = log(num/den) / log(r_eff)
    except Exception:
        p0 = 1.0
    try:
        p = brentq(f, 0.1, 8.0)
        return p
    except ValueError:
        return p0

def richardson_extrapolated(phi_f, phi_m, r_mf, p):
    return phi_f + (phi_f - phi_m) / (r_mf**p - 1)

def GCI(phi_f, phi_m, p, r_mf, Fs=1.25):
    return Fs * abs((phi_f - phi_m) / phi_f) / (r_mf**p - 1) * 100

# ==========================================================
# Lettura e preprocessing
# ==========================================================
results = {}
for key, info in meshes.items():
    df = pd.read_csv(info["file"], sep=r"\t|,", engine="python")
    if "HDF_N" in df.columns:
        df = df.drop(columns=["HDF_N"])
    df = df.dropna(axis=1, how="all")  # elimina colonne completamente vuote
    df = (df.apply(pd.to_numeric, errors="coerce")).dropna()  # converte in numerico e rimuove righe con NaN
    df = select_middle_cycle(df)
    results[key] = extract_metrics(df)

# ==========================================================
# Tabella riassuntiva delle metriche globali
# ==========================================================
df_metrics = pd.DataFrame({
    k: {
        "mean_P": v["mean_P"],
        "mean_Qout": v["mean_Qout"],
        "mean_KE": v["mean_KE"]
    } for k, v in results.items()
}).T

df_metrics["a_mm3"] = [meshes[idx]["a"] for idx in df_metrics.index]
df_metrics["h_mm"] = df_metrics["a_mm3"] ** (1/3)
print("\n=== Mean metrics per mesh (central cycle) ===\n")
print(df_metrics.round(4).T)

# ==========================================================
# Rapporti di raffinamento
# ==========================================================
h_c, h_m, h_f = df_metrics.loc["Coarse","h_mm"], df_metrics.loc["Medium","h_mm"], df_metrics.loc["Fine","h_mm"]
r_cm = h_c / h_m
r_mf = h_m / h_f

# ==========================================================
# Calcolo p, GCI e Richardson extrapolation per mean values
# ==========================================================
print("\n=== Global (mean) errors, observed order p, Richardson extrapolation, and GCI ===\n")
header = "{:>10s} | {:>8s} | {:>8s} | {:>6s} | {:>12s} | {:>9s} | {:>9s} | {:>8s}"
print(header.format("metric","eps C-M","eps M-F","p","phi_ext(F)","GCI12 %","GCI23 %","AR check"))
print("-"*90)

for metric in ["mean_P","mean_Qout","mean_KE"]:
    phi_c = df_metrics.loc["Coarse", metric]
    phi_m = df_metrics.loc["Medium", metric]
    phi_f = df_metrics.loc["Fine", metric]

    eps_CM = _safe_rel_err(phi_m, phi_c)
    eps_MF = _safe_rel_err(phi_f, phi_m)
    p = observed_order_p(phi_c, phi_m, phi_f, r_cm, r_mf)

    phi_ext = np.nan
    GCI12 = np.nan
    GCI23 = np.nan
    AR = np.nan
    if np.isfinite(p):
        phi_ext = richardson_extrapolated(phi_f, phi_m, r_mf, p)
        GCI12 = GCI(phi_f, phi_m, p, r_mf)
        GCI23 = GCI(phi_m, phi_c, p, r_cm)
        AR = GCI12 / ((r_mf**p) * GCI23) if GCI23 != 0 else np.nan

    line = "{:>10s} | {:8.3f} | {:8.3f} | {:6.3f} | {:12.5g} | {:9.3f} | {:9.3f} | {:8.3f}"
    print(line.format(metric, eps_CM, eps_MF, p, phi_ext, GCI12, GCI23, AR))

# ==========================================================
# Errori temporali (serie: Fine = riferimento)
# ==========================================================
print("\n=== Time-resolved errors (Medium vs Fine) ===\n")
for var in ["Pvol(t)", "Qout(t)", "KE(t)"]:
    eps_rms = rms_error(results["Fine"][var], results["Medium"][var])
    eps_mean = mean_abs_error(results["Fine"][var], results["Medium"][var])
    print(f"{var:>8s} | ε_RMS={eps_rms:6.2f}% | ε_mean={eps_mean:6.2f}%")

print("\n=== Time-resolved errors (Coarse vs Fine) ===\n")
for var in ["Pvol(t)", "Qout(t)", "KE(t)"]:
    eps_rms = rms_error(results["Medium"][var], results["Coarse"][var])
    eps_mean = mean_abs_error(results["Medium"][var], results["Coarse"][var])
    print(f"{var:>8s} | ε_RMS={eps_rms:6.2f}% | ε_mean={eps_mean:6.2f}%")

print("\n=== End of analysis ===\n")


import pandas as pd, numpy as np, matplotlib.pyplot as plt

# --- file (central cycle già uniforme → lo tagliamo a 1/3..2/3) ---
FINE   = "fine.csv"
MEDIUM = "medium.csv"
COARSE = "coarse.csv"  # opzionale nei grafici

# --- colonne disponibili nel tuo CSV ---
COLS = {
    "Pin":   "inlet_pressure_mmHg",
    "Pout":  "outlet_pressure_mmHg",
    "Pvol":  "volume_pressure_mmHg",
    "Qout":  "outlet_flow_mL_s",
    "KE":    "KE_total_mJ",
    "EL":    "Viscous_EL_W",
    "Vort":  "Vorticity_mean",
}

def read_middle_cycle(path):
    df = pd.read_csv(path, sep=r"\t|,", engine="python", decimal=",")
    if "HDF_N" in df.columns:
        df = df.drop(columns=["HDF_N"])
    df = df.dropna(axis=1, how="all")  # elimina colonne completamente vuote
    df = (df.apply(pd.to_numeric, errors="coerce")).dropna()
    n = len(df); k = n//3
    return df.iloc[k:2*k].reset_index(drop=True)

def rel_err_series(ref, test):
    denom = np.where(np.abs(ref)<1e-16, 1e-16, np.abs(ref))
    e = (test - ref)/denom * 100.0
    eps_rms  = float(np.sqrt(np.mean(e**2)))
    eps_mean = float(np.mean(np.abs(e)))
    return e, eps_rms, eps_mean

# --- carica dati ---
dfF = read_middle_cycle(FINE)
dfM = read_middle_cycle(MEDIUM)
dfC = read_middle_cycle(COARSE)  # se vuoi anche coarse

# --- costruisci le serie utili ---
DeltaP_F = (dfF[COLS["Pin"]] - dfF[COLS["Pout"]]).values
DeltaP_M = (dfM[COLS["Pin"]] - dfM[COLS["Pout"]]).values

series = {
    "P̄_V [mmHg]": (dfF[COLS["Pvol"]].values, dfM[COLS["Pvol"]].values),
    "Q_out [mL/s]": (dfF[COLS["Qout"]].values, dfM[COLS["Qout"]].values),
    "KE [mJ]": (dfF[COLS["KE"]].values, dfM[COLS["KE"]].values),
}

# --- calcola errori e fai grafici compatti ---
summary = []
t = np.arange(len(dfF))  # usa l'indice; se vuoi in s: t = dfF["time"].to_numpy()/1000
t = np.linspace(0, 1, len(dfF))

for label, (ref, test) in series.items():
    e, e_rms, e_mean = rel_err_series(ref, test)
    summary.append((label, e_rms, e_mean))

    # plot: curve sovrapposte + errore (secondo asse)
    fig, ax1 = plt.subplots(figsize=(8,3))
    ax1.plot(t, ref, label="Fine (ref)", linewidth=1.5)
    ax1.plot(t, test, '--', label="Medium", linewidth=1.2)
    ax1.set_xlabel("frames (central cycle)")
    ax1.set_ylabel(label)
    ax2 = ax1.twinx()
    ax2.plot(t, e, alpha=0.5)
    ax2.set_ylabel("rel. error [%]")
    ax1.legend(loc="upper left")
    ax1.set_title(f"{label}  |  ε_RMS={e_rms:.2f}%  ε_mean={e_mean:.2f}%")
    plt.tight_layout()
    plt.show()

# bar chart riassuntivo di ε_RMS per variabile
labels = [s[0] for s in summary]
vals   = [s[1] for s in summary]
plt.figure(figsize=(8,3))
plt.bar(labels, vals)
plt.ylabel("ε_RMS [%] (Medium vs Fine)")
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
plt.show()

# stampa tabellina testuale
print("\nSummary (Medium vs Fine):")
for lbl, r, m in summary:
    print(f"{lbl:24s}  ε_RMS={r:6.2f}%   ε_mean={m:6.2f}%")
print("\n=== End of analysis ===\n")