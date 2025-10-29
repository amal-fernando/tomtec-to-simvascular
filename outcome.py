import pyvista as pv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend instead of Qt
import glob
import os
from pathlib import Path
import tkinter as tk
from tkinter import filedialog
import re
from scipy.stats import pearsonr
from scipy.integrate import simpson as simps

def extract_number(filepath):
    filename = os.path.basename(filepath)  # <-- prendi solo il nome del file
    match = re.search(r'(\d+)', filename)
    return int(match.group(1)) if match else -1

def compute_flow_rate(surface):
    """Calcola la portata integrando la velocità normale sulla superficie"""
    if "Velocity" not in surface.point_data:
        return np.nan
    velocity = surface.point_data["Velocity"]
    normals = surface.point_normals
    normal_component = np.einsum('ij,ij->i', velocity, normals)
    areas = surface.compute_cell_sizes().cell_data["Area"]
    flow = 0.0
    for cell_id in range(surface.n_cells):
        point_ids = surface.get_cell(cell_id).point_ids
        cell_vel = np.mean(normal_component[point_ids])
        cell_area = areas[cell_id]
        flow += cell_vel * cell_area
    return flow/1000  # mL/s

# === CONFIGURAZIONE ===
tk.Tk().withdraw()
file_path = Path(filedialog.askopenfilename(title="Select first result_xxx.vtu file"))
if not file_path.exists():
    raise FileNotFoundError("No file selected or file does not exist.")

path = file_path.parent
base_name = file_path.name
radice_dataset = base_name[:-7]
multiplier = int(base_name[len(radice_dataset):-4])
boundary_result_path = path / "boundary_simulation"
boundary_result_path.mkdir(exist_ok=True)
vtu_files = sorted(glob.glob(os.path.join(path, "result_*.vtu")),
                   key=extract_number)
inlet_files = sorted(glob.glob(os.path.join(boundary_result_path, "result_inlet_*.vtp")), key=extract_number)
outlet_files = sorted(glob.glob(os.path.join(boundary_result_path, "result_outlet_*.vtp")), key=extract_number)
save_path = path.parent / "results"
save_path.mkdir(exist_ok=True)

time_values = []
for f in vtu_files:
    print(os.path.basename(f), "→", extract_number(f))
    time_values.append(extract_number(f))


inlet_pressures = []
outlet_pressures = []
inlet_flows = []
outlet_flows = []
ke_totals = []
ke_means = []
vort_means = []
vort_peaks = []
wss_means = [] # non serve
# tubulent kintetik energy
# energy loss


for i, (vtu_path, inlet_path, outlet_path) in enumerate(zip(vtu_files, inlet_files, outlet_files)):
    print(f"Processing timestep {i+1}...")

    # Caricamento superfici inlet/outlet
    inlet = pv.read(inlet_path)
    outlet = pv.read(outlet_path)

    if i == 0:
        # === POINT DATA ===
        print("Point data arrays:")
        print(inlet.point_data.keys())

        # === CELL DATA ===
        print("Cell data arrays:")
        print(inlet.cell_data.keys())

        # === FIELD DATA (metadata globali) ===
        print("Field data:")
        print(inlet.field_data.keys())

        # === ALTRE INFO ===
        print("Mesh has", inlet.n_points, "points")
        print("Mesh has", inlet.n_cells, "cells")

    # === PRESSIONE MEDIA su inlet ===
    if "Pressure" in inlet.point_data:
        inlet_pressure_array = inlet.point_data["Pressure"]
        inlet_avg_pressure = np.mean(inlet_pressure_array)/133.322 # Converti da Pa a mmHg
    else:
        inlet_avg_pressure = np.nan

    # === PRESSIONE MEDIA su outlet ===
    if "Pressure" in outlet.point_data:
        outlet_pressure_array = outlet.point_data["Pressure"]
        outlet_avg_pressure = np.mean(outlet_pressure_array)/133.322 # Converti da Pa a mmHg
    else:
        outlet_avg_pressure = np.nan

    # === PORTATA su inlet ===
    inlet_flow = compute_flow_rate(inlet)

    # === PORTATA su outlet ===
    outlet_flow = compute_flow_rate(outlet)

    # === CARICA IL VOLUME COMPLETO ===
    vol_mesh = pv.read(vtu_path)

    # --- Energia cinetica ---
    if "Velocity" in vol_mesh.point_data:
        vel = vol_mesh.point_data["Velocity"]/1000 # mm/s --> m/s
        ke_pt = 0.5 * 1060 * np.sum(vel**2, axis=1)  # J/m^3
        ke_mean = np.mean(ke_pt)
        ke_total = ke_mean * abs(vol_mesh.volume*1e-9)  # J (volume in mm^3 --> m^3)
    else:
        ke_mean = np.nan
        ke_total = np.nan
    ke_means.append(ke_mean)
    ke_totals.append(ke_total)

    # --- Vorticità ---
    try:
        vorticity_vector = vol_mesh.point_data['Vorticity']
        # 2. Calcola la magnitudine (norma) del vettore per ogni punto
        vort_mag = np.linalg.norm(vorticity_vector, axis=1)
        # 3. Aggiungi media e picco alle liste
        vort_means.append(np.mean(vort_mag))
        vort_peaks.append(np.max(vort_mag))
    except KeyError:
        # Se il campo 'Vorticity' non esiste in un file, gestisci l'errore
        print("Attenzione: campo 'Vorticity' non trovato. Imposto NaN.")
        vort_means.append(np.nan)
        vort_peaks.append(np.nan)

    # --- WSS (stima semplificata) ---
    try:
        surf = vol_mesh.extract_surface()
        if "WSS" in surf.point_data:
            wss_vector = surf.point_data["WSS"]
            wss_vals = np.linalg.norm(wss_vector, axis=1)
        wss_means.append(np.mean(wss_vals))
    except KeyError:
        wss_means.append(np.nan)



    # Salva dati
    inlet_pressures.append(inlet_avg_pressure)
    outlet_pressures.append(outlet_avg_pressure)
    inlet_flows.append(inlet_flow)
    outlet_flows.append(outlet_flow)

# === SALVA IN CSV ===
df = pd.DataFrame({
    "time": time_values,
    "inlet_pressure": inlet_pressures,
    "outlet_pressure": outlet_pressures,
    "inlet_flow": inlet_flows,
    "outlet_flow": outlet_flows,
    "KE_total": ke_totals,
    "KE_mean": ke_means,
    "Vorticity_mean": vort_means,
    "Vorticity_peak": vort_peaks,
    "WSS_mean": wss_means
})
df.to_csv(save_path / "extended_results.csv", index=False)
print("Saved extended results to 'extended_results.csv'")

# === PLOT ===
plt.plot(df["time"], df["inlet_pressure"], label="Inlet")
plt.plot(df["time"], df["outlet_pressure"], label="Outlet")
plt.xlabel("Time step")
plt.ylabel("Average Pressure")
plt.legend()
plt.grid(True)
plt.title("Pressure over time")
plt.savefig(save_path / "pressure_over_time.png")
plt.show()

plt.plot(df["time"], df["inlet_flow"], label="Inlet Flow")
plt.plot(df["time"], df["outlet_flow"], label="Outlet Flow")
plt.xlabel("Time step")
plt.ylabel("Flow Rate (mL/s)")
plt.legend()
plt.grid(True)
plt.title("Flow Rate over time")
plt.savefig(save_path / "flow_rate_over_time.png")
plt.show()

print("Processing complete. Results saved to 'pressures.csv' and 'pressure_over_time.png'.")

# Definisci qui quanti timestep formano un ciclo
# === Parametri da configurare ===
cycles = 3  # <-- imposta il numero di cicli della tua simulazione
steps_total = len(vtu_files)
steps_per_cycle = steps_total // cycles

print(f"Steps totali: {steps_total}, Steps per ciclo: {steps_per_cycle}, Cicli: {cycles}")

# --- Valutazione periodicità per tutti i cicli ---
def get_cycle_data(arr, cycle_idx):
    start = cycle_idx * steps_per_cycle
    end = start + steps_per_cycle
    return np.array(arr[start:end])

# Funzione RMSE
def rmse(x, y):
    return np.sqrt(np.mean((x - y)**2))

# Funzione per cross correlazione (Pearson)
def cross_corr(x, y):
    corr, _ = pearsonr(x, y)
    return corr


# --- Confronto tra tutti i cicli ---
metrics = {"Cycle_i": [], "Cycle_j": [], "Type": [], "RMSE": [], "CrossCorr": []}

for i in range(cycles):
    for j in range(i+1, cycles):
        for name, data_list in [("Inlet Pressure", inlet_pressures),
                                ("Outlet Pressure", outlet_pressures),
                                ("Inlet Flow", inlet_flows),
                                ("Outlet Flow", outlet_flows)]:
            x = get_cycle_data(data_list, i)
            y = get_cycle_data(data_list, j)
            if np.isnan(x).all() or np.isnan(y).all():
                continue
            r = rmse(x, y)
            c = cross_corr(x, y)
            metrics["Cycle_i"].append(i+1)
            metrics["Cycle_j"].append(j+1)
            metrics["Type"].append(name)
            metrics["RMSE"].append(r)
            metrics["CrossCorr"].append(c)

# --- Salva in CSV ---
df_metrics = pd.DataFrame(metrics)
df_metrics.to_csv(save_path / "cycle_comparison_metrics.csv", index=False)
print("Metriche di periodicità salvate in 'cycle_comparison_metrics.csv'.")


# === RMSE puntuale tra cicli ===

def compute_pointwise_rmse_and_errors(data_list, steps_per_cycle, cycles):
    """
    Calcola RMSE, errore percentuale e RMSE percentuale per ogni step temporale tra i cicli.
    """
    data_matrix = np.array(data_list).reshape(cycles, steps_per_cycle)

    pointwise_rmse = []
    pointwise_pct_error = []
    pointwise_pct_rmse = []

    for t in range(steps_per_cycle):
        values_at_t = data_matrix[:, t]  # valori dei vari cicli in questo istante
        mean_val = np.mean(values_at_t)

        rmse_t = np.sqrt(np.mean((values_at_t - mean_val) ** 2))
        abs_pct_errors = np.abs((values_at_t - mean_val) / mean_val) * 100 if mean_val != 0 else np.zeros_like(
            values_at_t)
        pct_rmse_t = np.sqrt(np.mean(abs_pct_errors ** 2))

        pointwise_rmse.append(rmse_t)
        pointwise_pct_error.append(np.mean(abs_pct_errors))
        pointwise_pct_rmse.append(pct_rmse_t)

    return pointwise_rmse, pointwise_pct_error, pointwise_pct_rmse


# Calcola per ogni variabile
pointwise_metrics = []

for name, data_list in [("Inlet Pressure", inlet_pressures),
                        ("Outlet Pressure", outlet_pressures),
                        ("Inlet Flow", inlet_flows),
                        ("Outlet Flow", outlet_flows)]:

    rmse_vals, mean_pct_errs, pct_rmse_vals = compute_pointwise_rmse_and_errors(data_list, steps_per_cycle, cycles)

    for t in range(steps_per_cycle):
        pointwise_metrics.append({
            "Variable": name,
            "Timestep_in_cycle": t,
            "RMSE": rmse_vals[t],
            "Mean_Pct_Error": mean_pct_errs[t],
            "Pct_RMSE": pct_rmse_vals[t]
        })

# Salva risultati
df_pointwise = pd.DataFrame(pointwise_metrics)
df_pointwise.to_csv(save_path / "pointwise_cycle_metrics.csv", index=False)
print("Salvati RMSE e errori puntuali per timestep in 'pointwise_cycle_metrics.csv'.")

# Plot RMSE e percentuali per ogni variabile
for var in df_pointwise["Variable"].unique():
    sub = df_pointwise[df_pointwise["Variable"] == var]
    plt.figure(figsize=(10, 4))
    plt.plot(sub["Timestep_in_cycle"], sub["RMSE"], label="RMSE (assoluto)")
    plt.plot(sub["Timestep_in_cycle"], sub["Mean_Pct_Error"], label="Errore percentuale medio")
    plt.plot(sub["Timestep_in_cycle"], sub["Pct_RMSE"], label="RMSE percentuale")
    plt.xlabel("Timestep nel ciclo")
    plt.title(f"Errore tra cicli – {var}")
    plt.ylabel("Errore")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path / f"pointwise_errors_{var.replace(' ', '_').lower()}.png")
    plt.show()


# === Errore percentuale in corrispondenza del picco ===

def get_peak_index(data_list, steps_per_cycle, cycle_idx):
    """
    Restituisce l'indice del massimo valore del ciclo indicato.
    """
    cycle_data = get_cycle_data(data_list, cycle_idx)
    peak_idx = np.argmax(cycle_data)
    return peak_idx

def compute_cycle_values_at_index(data_list, steps_per_cycle, index):
    """
    Estrae i valori di tutti i cicli in un certo istante (index relativo al ciclo).
    """
    data_matrix = np.array(data_list).reshape(cycles, steps_per_cycle)
    return data_matrix[:, index]

# Scegli il ciclo di riferimento (es. il primo)
ref_cycle_idx = 0

# Esempio: Outlet Pressure
peak_idx_outlet = get_peak_index(outlet_pressures, steps_per_cycle, ref_cycle_idx)
values_at_peak_outlet = compute_cycle_values_at_index(outlet_pressures, steps_per_cycle, peak_idx_outlet)
mean_val = np.mean(values_at_peak_outlet)

# Errore percentuale rispetto alla media (puoi anche fare rispetto al valore del ciclo di riferimento)
pct_errors = np.abs((values_at_peak_outlet - mean_val) / mean_val) * 100 if mean_val != 0 else np.zeros_like(values_at_peak_outlet)

# Salva e stampa i risultati
print(f"Indice del picco di Outlet Pressure nel ciclo {ref_cycle_idx + 1}: timestep {peak_idx_outlet}")
for i, val in enumerate(values_at_peak_outlet):
    print(f"Ciclo {i+1} → Valore: {val:.2f} mmHg, Errore % rispetto alla media: {pct_errors[i]:.2f}%")

# Salva in CSV
df_peak = pd.DataFrame({
    "Cycle": np.arange(1, cycles+1),
    "OutletPressure_at_peak_idx": values_at_peak_outlet,
    "Pct_Error_vs_mean": pct_errors
})
df_peak.to_csv(save_path / "peak_outlet_pressure_errors.csv", index=False)
print("Salvato errore percentuale al picco di outlet pressure in 'peak_outlet_pressure_errors.csv'.")



# --- Plot delle differenze (facoltativo per ora: confronta tutti contro ciclo 1) ---
import matplotlib.pyplot as plt

reference_cycle = 0
fig, axs = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
variables = [("Inlet Pressure", inlet_pressures),
             ("Outlet Pressure", outlet_pressures),
             ("Inlet Flow", inlet_flows),
             ("Outlet Flow", outlet_flows)]

for ax, (title, data) in zip(axs, variables):
    ref = get_cycle_data(data, reference_cycle)
    for i in range(1, cycles):
        other = get_cycle_data(data, i)
        diff = other - ref
        ax.plot(diff, label=f"Ciclo {i+1} - Ciclo {reference_cycle+1}")
    ax.axhline(0, color='gray', linestyle='--')
    ax.set_title(f"Differenza {title} rispetto al Ciclo {reference_cycle + 1}")
    if "Pressure" in title:
        ax.set_ylabel("Differenza (mmHg)")
    if "Flow" in title:
        ax.set_ylabel("Differenza (mL/s)")
    ax.grid(True)
    ax.legend()

axs[-1].set_xlabel("Time step nel ciclo")
plt.tight_layout()
plt.savefig(save_path / "cycle_differences.png")
plt.show()

import seaborn as sns
import matplotlib.pyplot as plt

# --- Heatmap per ogni variabile e metrica ---
metric_types = ["RMSE", "CrossCorr"]
variables = df_metrics["Type"].unique()

for metric in metric_types:
    for var in variables:
        # Filtra i dati
        subset = df_metrics[(df_metrics["Type"] == var)]

        # Crea matrice NxN (cicli x cicli)
        n = cycles
        mat = np.full((n, n), np.nan)
        for _, row in subset.iterrows():
            i, j = int(row["Cycle_i"]) - 1, int(row["Cycle_j"]) - 1
            val = row[metric]
            mat[i, j] = val
            mat[j, i] = val  # Simmetrica

        # Plot
        plt.figure(figsize=(7, 6))
        sns.heatmap(mat, annot=True, fmt=".3f", cmap="coolwarm" if metric == "CrossCorr" else "viridis",
                    xticklabels=[f"C{i+1}" for i in range(n)],
                    yticklabels=[f"C{i+1}" for i in range(n)],
                    square=True, cbar_kws={'label': metric})
        plt.title(f"{metric} Heatmap – {var}")
        plt.tight_layout()
        fname = save_path / f"heatmap_{metric.lower()}_{var.replace(' ', '_').lower()}.png"
        plt.savefig(fname)
        plt.show()
        print(f"Saved heatmap: {fname}")

print('All done!')