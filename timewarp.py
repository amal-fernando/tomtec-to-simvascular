import pandas as pd
import numpy as np
from tkinter import filedialog, Tk
import os
from pathlib import Path
from scipy.interpolate import Akima1DInterpolator, PchipInterpolator
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

# ===============================
# Parametri
# ===============================
INTERPOLATOR = "pchip"  # 'makima' o 'pchip'
CLAMP = True             # clip dei valori dopo il warping

# ===============================
# Funzioni helper
# ===============================

def build_inverse_warp(knots_src, knots_dst, t_ref):
    """
    Costruisce l'inversa della mappa tempo: t' = warp(t).
    knots_src: landmark del soggetto (crescenti in [0,1])       es. [0, ES_i, 1]
    knots_dst: landmark target   (crescenti in [0,1])           es. [0, ES_mean, 1]
    t_ref:    griglia target (dove vuoi il segnale allineato),  es. np.linspace(0,1,N)

    Ritorna t_inv: per ogni u=t_ref, il tempo originale t tale che warp(t)=u.
    """
    knots_src = np.asarray(knots_src, float)
    knots_dst = np.asarray(knots_dst, float)
    t_ref = np.asarray(t_ref, float)

    # controlli minimi
    if not (np.all(np.diff(knots_src) > 0) and np.all(np.diff(knots_dst) > 0)):
        raise ValueError("knots_src e knots_dst devono essere strettamente crescenti.")
    if (knots_src[0] < 0) or (knots_src[-1] > 1) or (knots_dst[0] < 0) or (knots_dst[-1] > 1):
        raise ValueError("I landmark devono stare in [0,1].")

    # mappa diretta piecewise-lineare: t -> t'
    # poi usiamo l'inversa interpolando al contrario: t' -> t
    # costruiamo funzione t = g(t') sui nodi
    g_x = knots_dst  # ascisse (target)
    g_y = knots_src  # valori (origini)
    # t_inv(u) = g(u) via interpolazione lineare (monotona)
    t_inv = np.interp(t_ref, g_x, g_y)
    return t_inv

def warp_signal(y_orig, t_orig, t_inv, interpolator="makima", extrapolate=False, clamp=False):
    """
    Applica il warping: valuta y_orig in t = t_inv( t_ref ).
    t_orig e y_orig definiscono il segnale originale sul tempo [0,1].
    t_inv è l'inversa precomputata (stessa lunghezza di t_ref).

    interpolator: 'makima' (Akima modificata) o 'pchip'
    extrapolate: se True, abilita extrapolazione (sconsigliato, meglio evitare e clip)
    clamp: se True, clippa i valori al range [min(y_orig), max(y_orig)] come sicurezza.
    """
    if interpolator == "pchip":
        f = PchipInterpolator(t_orig, y_orig, extrapolate=extrapolate)
    else:
        f = Akima1DInterpolator(t_orig, y_orig, method="makima", extrapolate=extrapolate)

    # per sicurezza: evita di valutare fuori dal dominio
    t_eval = np.clip(t_inv, t_orig.min(), t_orig.max())
    y_warp = f(t_eval)

    if clamp:
        y_warp = np.clip(y_warp, np.nanmin(y_orig), np.nanmax(y_orig))
    return y_warp

# ===============================
# Step 1: Seleziona file
# ===============================
Tk().withdraw()
file_path = filedialog.askopenfilename(
    title="Seleziona il file volume_mL.csv",
    filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
)
if not file_path:
    raise SystemExit("❌ Nessun file selezionato.")

print(f"📂 File selezionato: {file_path}")

# Crea cartella per i risultati allineati
save_dir = os.path.join(os.path.dirname(file_path), "aligned")
os.makedirs(save_dir, exist_ok=True)

# ===============================
# Step 2: Leggi il CSV
# ===============================
# Nota: i numeri usano la virgola decimale nel tuo esempio → specifica decimal=","
df = pd.read_csv(file_path, sep=r"[;,]", engine="python", decimal=",")

# La prima colonna è il tempo normalizzato
t = df.iloc[:, 0].to_numpy(dtype=float)
subjects = df.columns[1:]  # tutte le altre colonne

print(f"✅ Letti {len(subjects)} soggetti, {len(t)} punti temporali.")


# ===============================
# Step 3: Trova t_ES (argmin del volume)
# ===============================
t_ES_dict = {}
for subj in subjects:
    y = df[subj].to_numpy(dtype=float)
    idx_min = np.argmin(y)
    t_ES_dict[subj] = t[idx_min]

mean_ES = np.mean(list(t_ES_dict.values()))
mean_ES = 0.405  # valore fisso noto da dati precedenti
print("📊 t_ES per soggetti:")
for k, v in t_ES_dict.items():
    print(f"  {k}: {v:.3f}")
print(f"📍 Media t_ES* = {mean_ES:.3f}")

# ===============================
# Step 4: Applica il warping
# ===============================
aligned = pd.DataFrame(index=t)
aligned.index.name = "T_aligned"

for subj in subjects:
    y = df[subj].to_numpy(dtype=float)
    ES_i = t_ES_dict[subj]
    knots_src = [0.0, ES_i, 1.0]
    knots_dst = [0.0, mean_ES, 1.0]
    t_inv = build_inverse_warp(knots_src, knots_dst, t)
    y_warp = warp_signal(y, t, t_inv, interpolator=INTERPOLATOR, clamp=CLAMP)
    aligned[subj] = y_warp

# ===============================
# Step 5: Plot di verifica
# ===============================
plt.figure(figsize=(8, 5))

# Curva originale (solo per confronto)
for subj in subjects:
    plt.plot(t, df[subj], color='gray', alpha=0.3, linewidth=1)

# Curve riallineate (colori più vivi)
for subj in subjects:
    plt.plot(t, aligned[subj], linewidth=1.5, label=subj)

# Linea verticale al t_ES medio
plt.axvline(mean_ES, color='red', linestyle='--', label='mean t_ES*')

plt.title("Volume ventricolare - Allineamento temporale (t_ES medio)")
plt.xlabel("Tempo normalizzato (T/Tmax)")
plt.ylabel("Volume [mL]")
plt.legend(loc='upper right', fontsize=8, ncol=2)
plt.grid(alpha=0.3)
plt.tight_layout()

plt.show()

# ===============================
# Step 6: Salva il risultato
# ===============================
out_path = os.path.join(save_dir, os.path.basename(file_path).replace(".csv", "_aligned.csv"))
aligned.to_csv(out_path, sep=",", decimal=".")
print(f"✅ File salvato in: {out_path}")
