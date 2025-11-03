import pandas as pd
import numpy as np
from math import log
from scipy.optimize import brentq

# ===============================
# Parametri delle tre mesh
#   a = volume/area caratteristica (mm^3) da cui ricaviamo h = a^(1/3)
#   file: separatore auto (tab o virgola)
# ===============================
meshes = {
    "Coarse": {"file": "coarse.csv", "a": 0.29},
    "Medium": {"file": "medium.csv", "a": 0.19},
    "Fine":   {"file": "fine.csv",   "a": 0.08},
}

# === Funzione per isolare il ciclo centrale ===
def select_middle_cycle(df):
    n = len(df)
    one_third = n // 3
    start = one_third
    end = 2 * one_third
    return df.iloc[start:end].reset_index(drop=True)

# === Funzione per calcolare le metriche principali ===

def extract_metrics(df):
    """Calcola metriche globali da un ciclo."""
    metrics = {
        "Pvol(t)": df["volume_pressure_mmHg"].values,
        "Qout(t)": df["outlet_flow_mL/s"].values,
        "KE(t)": df["KE_total_mJ"].values,
        "Vort(t)": df["Vorticity_mean"],
        # Valori medi/picco
        "mean_P": df["volume_pressure_mmHg"].mean(),
        "max_P": df["volume_pressure_mmHg"].max(),
        "mean_Qout": df["outlet_flow_mL/s"].mean(),
        "max_Qout": abs(df["outlet_flow_mL/s"]).max(),
        "mean_KE": df["KE_total_mJ"].mean(),
        "max_KE": df["KE_total_mJ"].max(),
        "mean_vort": df["Vorticity_mean"].mean(),
        "max_vort": df["Vorticity_mean"].max()
    }
    return metrics

# === Funzioni per errore e ordine di convergenza ===
def observed_order_p(phi_c, phi_m, phi_f, r_cm, r_mf):
    """
    Risolve per p:  ( (phi_c - phi_m)/(phi_m - phi_f) ) = (r_cm^p - 1)/(r_mf^p - 1)
    con root-finding (brentq). Fallback al p "uguale-r" se necessario.
    """
    R = (phi_c - phi_m) / (phi_m - phi_f)
    # funzione da azzerare
    f = lambda p: (r_cm**p - 1.0)/(r_mf**p - 1.0) - R
    # guess: formula “uguale-r” con r_eff
    r_eff = (r_cm*r_mf)**0.5
    p0 = log((phi_c - phi_m)/(phi_m - phi_f))/log(r_eff)
    # cerca p in un intervallo ragionevole (0.1–5)
    # gestisci casi patologici con try/except
    try:
        p = brentq(f, 0.1, 8.0)
    except ValueError:
        p = p0  # fallback
    return p

def GCI(phi_f, phi_m, phi_c, p, r_mf, r_cm, Fs=1.25):
    GCI_mf = Fs * abs(phi_f - phi_m)/abs(phi_f) / (r_mf**p - 1.0) * 100.0
    GCI_mc = Fs * abs(phi_m - phi_c)/abs(phi_m) / (r_cm**p - 1.0) * 100.0
    return GCI_mf, GCI_mc

def rel_error(phi1, phi2):
    return abs(phi2 - phi1) / abs(phi1) * 100

def order_p(phi_c, phi_m, phi_f, r):
    return np.log((phi_c - phi_m) / (phi_m - phi_f)) / np.log(r)

def rms_error(phi2_t, phi1_t):
    """Errore RMS percentuale su tutto il ciclo."""
    return np.sqrt(np.mean(((phi2_t - phi1_t) / phi1_t) ** 2)) * 100

def mean_abs_error(phi2_t, phi1_t):
    """Errore medio assoluto percentuale su tutto il ciclo."""
    return np.mean(np.abs((phi2_t - phi1_t) / phi1_t)) * 100

# === Lettura, preprocessing e calcolo ===
results = {}
for key, info in meshes.items():
    df = pd.read_csv(info["file"], sep="\t|,", engine="python")
    if "HDF_N" in df.columns:
        df = df.drop(columns=["HDF_N"])
    df = df.dropna(axis=1, how="all")  # elimina colonne completamente vuote
    df = (df.apply(pd.to_numeric, errors="coerce")).dropna()  # converte in numerico e rimuove righe con NaN
    df = select_middle_cycle(df)
    results[key] = extract_metrics(df)

# === Conversione in DataFrame ===
df_metrics = pd.DataFrame(results).T
df_metrics["a(mm^3)"] = [m["a"] for m in meshes.values()]
df_metrics["h(mm)"] = df_metrics["a(mm^3)"] ** (1 / 3)
print("\n=== Mean metrics per mesh (central cycle) ===\n")
print(df_metrics.round(3))

# === Calcolo di epsilon e ordine p ===
r = (0.29 ** (1 / 3)) / (0.19 ** (1 / 3))

# === Calcolo errori globali ===
print("\n=== Global (mean/picco) errors and observed order ===\n")
for metric in ["mean_ΔP", "mean_KE", "mean_EL"]:
    eps_CM = rel_error(df_metrics.loc["Coarse", metric], df_metrics.loc["Medium", metric])
    eps_MF = rel_error(df_metrics.loc["Medium", metric], df_metrics.loc["Fine", metric])
    p_val = order_p(df_metrics.loc["Coarse", metric],
                    df_metrics.loc["Medium", metric],
                    df_metrics.loc["Fine", metric], r)
    print(f"{metric:10s} | ε(C-M)={eps_CM:6.2f}% | ε(M-F)={eps_MF:6.2f}% | p={p_val:4.2f}")

# === Calcolo errori temporali RMS e media assoluta ===
print("\n=== Time-resolved errors (RMS and mean) ===\n")
for var in ["ΔP(t)", "Qout(t)", "KE(t)", "EL(t)"]:
    eps_rms = rms_error(results["Medium"][var], results["Fine"][var])
    eps_mean = mean_abs_error(results["Medium"][var], results["Fine"][var])
    print(f"{var:8s} | ε_RMS={eps_rms:5.2f}% | ε_mean={eps_mean:5.2f}%")

print("\n=== End of analysis ===\n")