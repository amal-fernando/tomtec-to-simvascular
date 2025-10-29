import pyvista as pv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import glob
import os
import re
from pathlib import Path
from tkinter import filedialog, Tk, filedialog, simpledialog
from scipy.stats import pearsonr
from tqdm import tqdm
import seaborn as sns
from funzioni import pv_to_np, np_to_pv, volume, flow

# Imposta il backend di Matplotlib
matplotlib.use('TkAgg')

# =============================================================================
# --- COSTANTI E CONFIGURAZIONE ---
# =============================================================================
# 💡 IMPOSTA QUESTI VALORI PRIMA DI ESEGUIRE LO SCRIPT
RHO_BLOOD = 1060  # Densità del sangue in kg/m^3
MU_BLOOD = 0.004  # Vicosità del sangue Pa·s
PA_TO_MMHG = 133.322  # Fattore di conversione da Pascal a mmHg
CYCLES = 3  # Numero di cicli cardiaci nella simulazione
SIMULATION_TIME_STEP = 0.001  # IMPORTANTE: Imposta la durata di un timestep in secondi (es. 1 ms = 0.001 s)


FLAG_FLOW = True  # Se True, calcola portate e metriche correlate
FLAG_PRESSURE = True  # Se True, calcola pressioni e metriche correlate
FLAG_VOLUME = True  # Se True, calcola volumi e metriche correlate
FLAG_KE = True  # Se True, calcola energia cinetica e metriche correlate
FLAG_VORTICITY = True  # Se True, calcola vorticità e metriche correlate
FLAG_ENERGY_LOSS = True  # Se True, calcola perdite di energia viscosa e metriche correlate

'''N.B. Le normali alle superfici dei file di simulazione sono orientate verso l'interno del dominio fluido.'''
# =============================================================================
# --- FUNZIONI HELPER E DI CALCOLO ---
# =============================================================================
def automatic_parser():
    Tk().withdraw()

    # Open file dialog to select results file
    root_dir = filedialog.askdirectory(
        title="Select results file directory"
    )
    # root_dir = r'C:\Users\amal1\Desktop\Tesi\Results Cluster\Population'
    if not root_dir:
        print("Nessuna directory selezionata. Uscita.")
        return [], 0

    # Dizionario per salvare i path
    paths = {}

    # Trova tutte le cartelle Pxxxx
    patient_dirs = sorted([
        d for d in os.listdir(root_dir)
        if d.startswith('P') and os.path.isdir(os.path.join(root_dir, d))
    ])

    for patient in patient_dirs:
        paths[patient] = {}

        sim_dir = os.path.join(root_dir, patient, 'simulation')

        if not os.path.isdir(sim_dir):
            print(f'[ATTENZIONE] Cartella non trovata: {sim_dir}')
            continue

        # Trova tutti i file result_XXXX.vtu
        result_files = [
            f for f in os.listdir(sim_dir)
            if re.match(r'result_\d+\.vtu$', f)
        ]

        if not result_files:
            print(f'[ATTENZIONE] Nessun file result_XXXX.vtu in {sim_dir}')
            continue

        # Estrai la parte numerica e ordina per valore numerico
        result_files.sort(key=lambda f: int(re.search(r'result_(\d+)\.vtu', f).group(1)))

        # Prendi il più piccolo
        first_result = result_files[0]
        results_path = os.path.join(sim_dir, first_result)

        if os.path.exists(results_path):
            paths[patient] = results_path
        else:
            print(f'[ATTENZIONE] File mancante: {results_path}')
            paths[patient] = None  # oppure salta, se preferisci

    results_files_path = [
        path for path in paths.values()
        if path is not None
    ]

    # ============================================
    # Permetti all’utente di scegliere da dove iniziare
    # ============================================
    if not results_files_path:
        print("Nessun file results trovato.")
        return [], 0

    # Mostra l’elenco numerato
    print("\nFile trovati:")
    for i, path in enumerate(results_files_path):
        patient_id = Path(path).parts[-3]
        print(f"{i}: {patient_id}")

    # Chiedi l'indice di partenza
    start_index = simpledialog.askinteger(
        "Indice di partenza",
        f"Inserisci l'indice da cui iniziare (0 - {len(results_files_path) - 1}):",
        minvalue=0,
        maxvalue=len(results_files_path) - 1
    )

    if start_index is None:
        print("Nessun indice selezionato. Uscita.")
        exit()

    # ================================================
    # 1. Parse the results file to extract information
    # ================================================
    return results_files_path, start_index


def extract_number(filepath: str) -> int:
    """Estrae il primo numero intero dal nome di un file."""
    filename = os.path.basename(filepath)
    match = re.search(r'(\d+)', filename)
    return int(match.group(1)) if match else -1


def setup_paths_and_files(vtu_file_path):
    """
    Apre una finestra di dialogo per selezionare il file iniziale,
    e prepara tutti i percorsi e le liste di file necessari per l'analisi.
    """
    # Tk().withdraw()
    vtu_file_path = Path(vtu_file_path)

    if not vtu_file_path.exists():
        raise FileNotFoundError("Nessun file selezionato o il file non esiste.")

    path = vtu_file_path.parent
    boundary_result_path = path / "boundary_simulation"
    save_path = path.parent / "results"
    save_path.mkdir(exist_ok=True)

    print(f"Cartella dati: {path}")
    print(f"Cartella risultati: {save_path}")

    vtu_files = sorted(glob.glob(str(path / "result_*.vtu")), key=extract_number)
    inlet_files = sorted(glob.glob(str(boundary_result_path / "result_inlet_*.vtp")), key=extract_number)
    outlet_files = sorted(glob.glob(str(boundary_result_path / "result_outlet_*.vtp")), key=extract_number)

    if not all([vtu_files, inlet_files, outlet_files]):
        raise FileNotFoundError("Non sono stati trovati tutti i file necessari. Controlla i percorsi.")

    time_values = [extract_number(f) for f in vtu_files]

    return {
        "vtu_files": vtu_files,
        "inlet_files": inlet_files,
        "outlet_files": outlet_files,
        "time_values": time_values,
        "save_path": save_path
    }

def compute_flow_rate(surface: pv.PolyData):
    """
    Calcola la portata integrando la velocità normale sulla superficie.
    """
    if "Velocity" not in surface.point_data:
        return np.nan

    # Cell wise integration
    surface = surface.compute_normals(point_normals=False, cell_normals=True, auto_orient_normals=True)
    normals = surface.cell_data["Normals"]
    surface_cell = surface.point_data_to_cell_data(pass_point_data=True)
    velocity = surface_cell.cell_data["Velocity"]
    normal_velocity = np.einsum('ij,ij->i', velocity, normals)
    surface_cell["normal_velocity"] = normal_velocity
    surface_cell.set_active_scalars("normal_velocity")
    integrated = surface_cell.integrate_data()
    Q_mm3_s = integrated["normal_velocity"][0]

    # #Point wise integration
    # velocity = surface.point_data["Velocity"]
    # normals = surface.point_data["Normals"]
    # normal_velocity = np.einsum('ij,ij->i', velocity, normals)
    # surface['normal_velocity'] = normal_velocity
    # surface.set_active_scalars('normal_velocity')
    # integrated = surface.integrate_data()
    # Q_mm3_s = integrated['normal_velocity'][0]

    return Q_mm3_s / 1000.0  # mm^3/s -> mL/s


def process_timestep(vtu_path, inlet_path, outlet_path, timestep_index):
    """
    Carica i dati per un singolo timestep e calcola tutte le metriche quantitative.
    """
    metrics = {}
    inlet_surf = pv.read(inlet_path)
    outlet_surf = pv.read(outlet_path)

    # inlet_surf = inlet_surf.compute_normals(point_normals=True, cell_normals=True, auto_orient_normals=True)
    # outlet_surf = outlet_surf.compute_normals(point_normals=True, cell_normals=True, auto_orient_normals=True)

    # inlet_centers = inlet_surf.cell_centers().points
    # outlet_centers = outlet_surf.cell_centers().points

    # p = pv.Plotter()
    # p.add_mesh(inlet_surf, color="lightgreen", opacity=0.8)
    # p.add_mesh(outlet_surf, color="lightcoral", opacity=0.8)
    # p.add_arrows(inlet_centers, inlet_surf.cell_data["Normals"], mag=2)  # mag scala la lunghezza delle frecce
    # p.add_arrows(outlet_centers, outlet_surf.cell_data["Normals"], mag=2)  # mag scala la lunghezza delle frecce
    # p.add_arrows(inlet_surf.points, inlet_surf.point_data["Normals"], mag=2)  # mag scala la lunghezza delle frecce
    # p.add_arrows(outlet_surf.points, outlet_surf.point_data["Normals"], mag=2)  # mag scala la lunghezza delle frecce
    # p.show()

    if timestep_index == 0:
        print("\n--- Informazioni dal primo timestep ---")
        print(f"Dati puntuali disponibili: {inlet_surf.point_data.keys()}")
        print(f"Dati di cella disponibili: {inlet_surf.cell_data.keys()}")
        print("-------------------------------------\n")

    inlet_surf.points = inlet_surf.points + inlet_surf.point_data["Displacement"]
    outlet_surf.points = outlet_surf.points + outlet_surf.point_data["Displacement"]

    # Pressione (Pa -> mmHg)
    if FLAG_PRESSURE:
        metrics['inlet_pressure'] = np.mean(inlet_surf.point_data["Pressure"]) / PA_TO_MMHG if "Pressure" in inlet_surf.point_data else np.nan
        metrics['outlet_pressure'] = np.mean(outlet_surf.point_data["Pressure"]) / PA_TO_MMHG if "Pressure" in outlet_surf.point_data else np.nan
    else:
        metrics['inlet_pressure'] = np.nan
        metrics['outlet_pressure'] = np.nan

    # Portata (mm^3/s -> mL/s)
    if FLAG_FLOW:
        metrics['inlet_flow'] = compute_flow_rate(inlet_surf)
        metrics['outlet_flow'] = compute_flow_rate(outlet_surf)
    else:
        metrics['inlet_flow'] = np.nan
        metrics['outlet_flow'] = np.nan

    # Carica il volume per le metriche volumetriche
    vol_mesh = pv.read(vtu_path)
    vol_mesh.points = vol_mesh.points + vol_mesh.point_data["Displacement"]

    # surf = vol_mesh.extract_surface()
    # surf = surf.compute_normals(point_normals=True, cell_normals=False)
    # # Crea un plotter
    # p = pv.Plotter()
    # p.add_mesh(surf, color="lightblue", opacity=0.8)
    # # Aggiungi le normali come frecce
    # p.add_arrows(surf.points, surf["Normals"], mag=2)  # mag scala la lunghezza delle frecce
    # p.show()

    if FLAG_VOLUME:
        V = -vol_mesh.volume / 1000  # in mL
        metrics['Volume_mL'] = V
    else:
        metrics['Volume_mL'] = np.nan

    # Energia Cinetica (J)
    if "Velocity" in vol_mesh.point_data and FLAG_KE:
        '''
        # copia della mesh originale
        mesh_cells = vol_mesh.copy()
        mesh_cells.points /= 1000.0

        # 1️⃣ prendi la velocità in m/s (definita per punto)
        vel_point_m = mesh_cells.point_data["Velocity"] / 1000.0  # mm/s → m/s
        mesh_cells.point_data["Velocity_m"] = vel_point_m  # definita per punto

        # vel = np.linalg.norm(mesh_cells.point_data["Velocity"], axis=1)

        # 2️⃣ speed^2 per punto (m^2/s^2)
        speed2_point = np.sum(vel_point_m ** 2, axis=1)  # (Npoints,)
        mesh_cells.point_data["speed2_m2s2"] = speed2_point

        # v2 = np.array(speed2_point*1e6)
        # #  imposta come campo vettoriale attivo
        # mesh_cells.set_active_vectors("Velocity_m")

        # 3️⃣ converti da dati per punto a dati per cella
        #    → calcola la velocità media per ciascuna cella
        mesh_cells = mesh_cells.point_data_to_cell_data(pass_point_data=True)

        # 4️⃣ calcola i volumi delle celle
        mesh_cells = mesh_cells.compute_cell_sizes(length=False, area=False, volume=True)
        cell_vol_m3 = -mesh_cells.cell_data["Volume"] #* 1e-9  # mm³ → m³

        # 5️⃣ estrai la velocità media per cella
        # cell_vel = mesh_cells.cell_data["Velocity_m"]
        speed2_cell = mesh_cells.cell_data["speed2_m2s2"]  # (Ncells,)

        # 6️⃣ calcola KE per cella e totale
        ke_density_cell = 0.5 * RHO_BLOOD * speed2_cell # np.sum(cell_vel ** 2, axis=1)  # J/m³
        ke_cell = ke_density_cell * cell_vol_m3  # J
        KE_total = np.sum(ke_cell) # J
        KE_mean = KE_total / np.sum(cell_vol_m3)  # J/m3
        KE_density = np.sum(ke_density_cell)
        metrics['KE_mean'] = KE_mean
        metrics['KE_total'] = KE_total
        '''
        mesh = vol_mesh.copy()
        mesh.points /= 1000.0  # mm → m

        # 1️⃣ velocità per nodo (in m/s)
        vel_node = mesh.point_data["Velocity"] / 1000.0  # mm/s → m/s
        speed2_node = np.sum(vel_node ** 2, axis=1)

        # 2️⃣ volumi delle celle (in m³)
        mesh = mesh.compute_cell_sizes(length=False, area=False, volume=True)
        vol = abs(mesh.cell_data["Volume"])

        # 3️⃣ connettività (nodi di ciascun tetraedro)
        cells = mesh.cells.reshape(-1, 5)[:, 1:5]  # formato [ncell, 4 nodi]

        vel_cell = vel_node[cells]
        # Calcolo vettorializzato
        # Gram per cella: (ncells, 4, 4)
        G = np.einsum('nik,njk->nij', vel_cell, vel_cell)
        term1 = np.trace(G, axis1=1, axis2=2)  # (ncells,)
        # somma della parte inferiore esclusa diag
        term2 = np.sum(np.tril(G, k=-1), axis=(1, 2))  # (ncells,)

        KE_cell_consistent = 0.5 * RHO_BLOOD * (vol / 10.0) * (term1 + term2)  # (ncells,)
        KE_total_consistent = KE_cell_consistent.sum()
        KE_density_consistent = KE_total_consistent / vol.sum()  # J/m^3

        # print("KE (consistente)  [J]:   ", KE_total_consistent)
        # print("KE/V (consistente)[J/m^3]:", KE_density_consistent)

        # 4️⃣ massa di ciascun tetraedro
        m_cell = RHO_BLOOD * vol  # kg

        # 5️⃣ distribuisci 1/4 della massa di ogni cella ai suoi nodi
        n_points = mesh.points.shape[0]
        m_node = np.zeros(n_points)
        for ci, nodes in enumerate(cells):
            m_node[nodes] += m_cell[ci] / 4.0

        # 6️⃣ energia cinetica per nodo
        KE_node_lumped = 0.5 * m_node * speed2_node
        KE_total_lumped = KE_node_lumped.sum()
        KE_density_lumped = KE_total_lumped / vol.sum()

        # print("KE (lumped)       [J]:   ", KE_total_lumped)
        # print("KE/V (lumped)     [J/m^3]:", KE_density_lumped)  # J/m3 (energia per volume totale)

        metrics["KE_total_consistent"] = KE_total_consistent
        metrics["KE_density_consistent"] = KE_density_consistent
        metrics["KE_total_lumped"] = KE_total_lumped
        metrics["KE_density_lumped"] = KE_density_lumped
    else:
        metrics.update({'KE_density_consistent': np.nan, 'KE_total_consistent': np.nan})

    # Vorticità (1/s)
    if 'Vorticity' in vol_mesh.point_data and FLAG_VORTICITY:
        vort_mag = np.linalg.norm(vol_mesh.point_data['Vorticity'], axis=1)
        metrics['Vorticity_mean'] = np.mean(vort_mag)
        metrics['Vorticity_peak'] = np.max(vort_mag)
    else:
        metrics.update({'Vorticity_mean': np.nan, 'Vorticity_peak': np.nan})

    # try:
    #     wall_surface = vol_mesh.extract_surface()
    #     metrics['WSS_mean'] = np.mean(
    #         np.linalg.norm(wall_surface.point_data["WSS"], axis=1)) if "WSS" in wall_surface.point_data else np.nan
    # except Exception:
    #     metrics['WSS_mean'] = np.nan

    # metrics['TKE_mean'] = np.mean(vol_mesh.point_data['TKE']) if 'TKE' in vol_mesh.point_data else np.nan

    if not np.isnan(metrics['inlet_pressure']) and not np.isnan(metrics['outlet_pressure']):
        # Copia della mesh deformata
        mesh = vol_mesh.copy()

        # Conversione unità: mm → m, mm/s → m/s
        mesh.points /= 1000.0
        mesh.point_data["Velocity_m"] = mesh.point_data["Velocity"] / 1000.0
        # print(f"Dati puntuali disponibili: {grad_mesh.point_data.keys()}")
        # Calcola il gradiente del campo di velocità (∇u)
        # Restituisce un tensore 3x3 per ogni cella (flattened in un vettore 9-componenti)
        grad_mesh = mesh.compute_derivative(scalars="Velocity_m", gradient=True)


        # 2️⃣ imposta come campo vettoriale attivo
        grad_mesh.set_active_tensors("gradient")

        # 3️⃣ converti da dati per punto a dati per cella
        #    → calcola la velocità media per ciascuna cella
        grad_mesh = grad_mesh.point_data_to_cell_data(pass_point_data=True)

        gradients = grad_mesh.cell_data["gradient"]  # shape: (n_cells, 9)

        # Rimodella in (n_cells, 3, 3)
        gradients = gradients.reshape(-1, 3, 3)

        # Calcola la parte simmetrica: S = 0.5 * (∇u + ∇u^T)
        S = 0.5 * (gradients + np.transpose(gradients, (0, 2, 1)))

        # Calcola Phi = 2μ * S_ij * S_ij = 2μ * sum(S^2)
        Phi = 2.0 * MU_BLOOD * np.sum(S ** 2, axis=(1, 2))  # [W/m³]

        # Aggiungiamolo alla mesh per visualizzare
        grad_mesh.cell_data["Viscous_Dissipation"] = Phi

        # Calcolo del volume delle celle (in m³)
        grad_mesh = grad_mesh.compute_cell_sizes(length=False, area=False, volume=True)
        cell_vol = -grad_mesh.cell_data["Volume"]  # già in m³

        # Energia dissipata totale (potenza dissipata)
        EL_total = np.sum(Phi * cell_vol)  # [W]
        EL_mean = np.mean(Phi)  # [W/m³]
        metrics['Viscous_EL_W'] = EL_total
        metrics['Viscous_EL_mean_W/m3'] = EL_mean

        # print(f"\nViscous Energy Loss (total): {EL_total:.4e} W")
        # print(f"Mean viscous dissipation: {EL_mean:.4e} W/m³")
    else:
        metrics['Energy_Loss_mW'] = np.nan

    return metrics


# =============================================================================
# --- FUNZIONI PER VISUALIZZAZIONE QUALITATIVA ---
# =============================================================================

def find_peak_timesteps(df: pd.DataFrame) -> dict:
    """Identifica gli indici dei timestep corrispondenti a picchi di flusso."""
    if df.empty or 'inlet_flow' not in df.columns: return {}

    steps_per_cycle = len(df) // CYCLES
    last_cycle_df = df.iloc[-steps_per_cycle:]

    timesteps = {
        "peak_diastole": last_cycle_df['inlet_flow'].idxmax(),
        "peak_systole": last_cycle_df['outlet_flow'].idxmax(),
    }
    print(f"Identificati timestep chiave per visualizzazioni: {timesteps}")
    return timesteps


def create_streamlines_plot(mesh, save_path, timestep_name):
    """Genera e salva una visualizzazione delle streamlines."""
    if 'Velocity' not in mesh.point_data: return

    inlet_center = mesh.extract_surface().extract_feature_edges(boundary_edges=True, feature_edges=False,
                                                                manifold_edges=False).center
    streamline_source = pv.Line(
        pointa=(inlet_center[0] - 20, inlet_center[1], inlet_center[2]),
        pointb=(inlet_center[0] + 20, inlet_center[1], inlet_center[2]),
        resolution=20)
    streams = mesh.streamlines_from_source(streamline_source, vectors='Velocity', max_time=100.0)

    plotter = pv.Plotter(off_screen=True, window_size=[1024, 768])
    plotter.add_mesh(mesh.extract_surface(), style='wireframe', opacity=0.1, color='gray')
    plotter.add_mesh(streams, scalar_bar_args={'title': 'Velocity (mm/s)'})
    plotter.add_text(f"Streamlines at {timestep_name.replace('_', ' ').title()}", font_size=12)
    plotter.view_isometric()
    plotter.screenshot(save_path / f"streamlines_{timestep_name}.png")
    plotter.close()


def create_slice_with_vectors_plot(mesh, save_path, timestep_name):
    """Genera un piano di taglio con glifi vettoriali."""
    if 'Velocity' not in mesh.point_data: return

    slice_plane = mesh.slice(normal='y', origin=mesh.center)
    slice_plane['Velocity Magnitude'] = np.linalg.norm(slice_plane['Velocity'], axis=1)
    glyphs = slice_plane.glyph(orient='Velocity', scale='Velocity Magnitude', factor=0.05)

    plotter = pv.Plotter(off_screen=True, window_size=[1024, 768])
    plotter.add_mesh(mesh.extract_surface(), style='wireframe', opacity=0.1, color='gray')
    plotter.add_mesh(slice_plane, scalars='Velocity Magnitude', scalar_bar_args={'title': 'Velocity (mm/s)'})
    plotter.add_mesh(glyphs, color='black')
    plotter.add_text(f"Velocity Slice at {timestep_name.replace('_', ' ').title()}", font_size=12)
    plotter.view_isometric()
    plotter.screenshot(save_path / f"velocity_slice_{timestep_name}.png")
    plotter.close()


def create_vortex_plot(mesh, save_path, timestep_name):
    """Identifica e visualizza le strutture vorticose."""
    if 'Velocity' not in mesh.point_data: return

    mesh.compute_q_criterion(inplace=True)
    q_threshold = np.percentile(mesh['Q-criterion'][mesh['Q-criterion'] > 0], 95) if np.any(
        mesh['Q-criterion'] > 0) else 0.01

    plotter = pv.Plotter(off_screen=True, window_size=[1024, 768])
    plotter.add_mesh(mesh.extract_surface(), style='wireframe', opacity=0.1, color='gray')
    if q_threshold > 0:
        plotter.add_mesh(mesh.contour(isosurfaces=[q_threshold], scalars='Q-criterion'), color='royalblue', opacity=0.7)
    plotter.add_text(f"Vortical Structures (Q-criterion) at {timestep_name.replace('_', ' ').title()}", font_size=12)
    plotter.view_isometric()
    plotter.screenshot(save_path / f"vortices_{timestep_name}.png")
    plotter.close()


def create_pathlines_plot(paths_info: dict, df: pd.DataFrame, visuals_path: Path):
    """
    NUOVO: Genera e salva le pathlines tracciando particelle virtuali nel tempo.
    """
    print("\nInizio calcolo Pathlines (potrebbe richiedere tempo)...")
    steps_per_cycle = len(df) // CYCLES
    last_cycle_start_index = len(df) - steps_per_cycle

    # Prendi i file .vtu solo per l'ultimo ciclo
    cycle_vtu_files = paths_info['vtu_files'][last_cycle_start_index:]

    # Usa il primo inlet del ciclo per definire la sorgente delle particelle
    inlet_mesh = pv.read(paths_info['inlet_files'][last_cycle_start_index])
    seeding_points = inlet_mesh.points

    # Crea un plotter
    plotter = pv.Plotter(off_screen=True, window_size=[1024, 768])

    # Aggiungi la geometria esterna come riferimento
    first_mesh = pv.read(cycle_vtu_files[0])
    plotter.add_mesh(first_mesh.extract_surface(), style='wireframe', opacity=0.1, color='gray')

    # Integra le particelle attraverso ogni timestep del ciclo
    current_points = seeding_points
    path_segments = []

    for vtu_file in tqdm(cycle_vtu_files, desc="Tracciando Pathlines"):
        mesh = pv.read(vtu_file)
        if 'Velocity' not in mesh.point_data: continue

        # Esegui un'integrazione per la durata di un timestep
        stream, _ = mesh.streamlines(
            vectors='Velocity',
            source_center=current_points,
            n_points=len(current_points),
            max_time=SIMULATION_TIME_STEP,
            terminal_speed=1e-12,  # Evita terminazione prematura
            initial_step_length=0.1
        )
        if stream.n_points > 0:
            path_segments.append(stream)
            # Aggiorna la posizione delle particelle per il prossimo step
            # Prendiamo il punto finale di ogni streamline generata
            final_points_indices = np.where(stream.point_data['IntegrationTime'] > 0)[0]
            if len(final_points_indices) > 0:
                # Questo è un approccio semplificato; per maggiore precisione, si dovrebbe tracciare ogni particella individualmente
                current_points = stream.points[final_points_indices[-len(current_points):]]

    # Unisci tutti i segmenti di pathline e visualizzali
    if path_segments:
        all_paths = pv.MultiBlock(path_segments).combine()
        plotter.add_mesh(all_paths, scalars='IntegrationTime', cmap='viridis', scalar_bar_args={'title': 'Time (s)'})

    plotter.add_text("Particle Pathlines (Last Cycle)", font_size=12)
    plotter.view_isometric()
    plotter.screenshot(visuals_path / "pathlines_last_cycle.png")
    plotter.close()
    print("Salvato: pathlines_last_cycle.png")


def generate_qualitative_visualizations(paths_info, df):
    """Orchestra la generazione di tutte le visualizzazioni qualitative."""
    print("\n--- Inizio Generazione Visualizzazioni Qualitative ---")
    visuals_path = paths_info['save_path'] / "qualitative_visuals"
    visuals_path.mkdir(exist_ok=True)

    key_timesteps = find_peak_timesteps(df)
    if not key_timesteps: return

    for name, index in key_timesteps.items():
        print(f"\nGenerazione snapshot per: {name} (timestep index {index})")
        mesh = pv.read(paths_info['vtu_files'][index])
        create_streamlines_plot(mesh, visuals_path, name)
        create_slice_with_vectors_plot(mesh, visuals_path, name)
        create_vortex_plot(mesh, visuals_path, name)

    # Genera le pathlines sull'ultimo ciclo
    create_pathlines_plot(paths_info, df, visuals_path)


# =============================================================================
# --- FUNZIONI PER ANALISI E ESPORTAZIONE ---
# =============================================================================

def generate_summary_plots(df, save_path):
    """Genera e salva i grafici riassuntivi delle principali metriche."""
    print("\nGenerazione grafici riassuntivi...")
    plt.style.use('seaborn-v0_8-whitegrid')

    # Pressione
    plt.figure(figsize=(10, 5))
    plt.plot(df["time"], df["inlet_pressure"], label="Inlet Pressure", color='royalblue')
    plt.plot(df["time"], df["outlet_pressure"], label="Outlet Pressure", color='crimson')
    plt.xlabel("Time step")
    plt.ylabel("Pressione Media (mmHg)")
    plt.title("Pressione Media nel Tempo")
    plt.legend()
    plt.savefig(save_path / "pressure_over_time.png")
    plt.close()

    # Portata
    plt.figure(figsize=(10, 5))
    plt.plot(df["time"], df["inlet_flow"], label="Inlet Flow", color='royalblue')
    plt.plot(df["time"], df["outlet_flow"], label="Outlet Flow", color='crimson')
    plt.xlabel("Time step")
    plt.ylabel("Portata (mL/s)")
    plt.title("Portata nel Tempo")
    plt.legend()
    plt.savefig(save_path / "flow_rate_over_time.png")
    plt.close()

    # Energia Cinetica
    plt.figure(figsize=(10, 5))
    plt.plot(df["time"], df["KE_total_consistent"], label="Energia Cinetica Totale (J)", color='darkorange')
    plt.xlabel("Time step")
    plt.ylabel("Energia Cinetica (J)")
    plt.title("Energia Cinetica Totale nel Volume")
    plt.legend()
    plt.savefig(save_path / "kinetic_energy_over_time.png")
    plt.close()

    # NUOVO: Plot Perdita di Energia
    plt.figure(figsize=(10, 5))
    plt.plot(df["time"], df["Viscous_EL_W"], label="Perdita di Energia (mW)", color='purple')
    plt.xlabel("Time step")
    plt.ylabel("Perdita di Energia (mW)")
    plt.title("Perdita di Energia nel Tempo")
    plt.legend()
    plt.savefig(save_path / "energy_loss_over_time.png")
    plt.close()


def analyze_periodicity(df, cycles, save_path):
    """Esegue un'analisi di periodicità completa."""
    print("Inizio analisi di periodicità...")
    if 'time' not in df.columns or len(df) == 0:
        print("DataFrame vuoto o senza colonna 'time'. Salto l'analisi di periodicità.")
        return

    steps_total = len(df)
    if steps_total < cycles or cycles == 0:
        print(f"Numero di step ({steps_total}) insufficiente per {cycles} cicli. Salto analisi.")
        return
    steps_per_cycle = steps_total // cycles
    print(f"Steps totali: {steps_total}, Steps per ciclo: {steps_per_cycle}, Cicli: {cycles}")

    def get_cycle_data(series, cycle_idx):
        start = cycle_idx * steps_per_cycle
        end = start + steps_per_cycle
        return series.iloc[start:end].values

    # --- 1. Confronto tra tutti i cicli (RMSE e Correlazione) ---
    metrics = []
    variables_to_check = [
        "inlet_pressure", "outlet_pressure", "inlet_flow", "outlet_flow",
        "KE_total_consistent", "Vorticity_mean"
    ]

    for var in variables_to_check:
        if var not in df.columns: continue
        for i in range(cycles):
            for j in range(i + 1, cycles):
                x = get_cycle_data(df[var], i)
                y = get_cycle_data(df[var], j)
                if np.isnan(x).all() or np.isnan(y).all(): continue

                rmse = np.sqrt(np.mean((x - y) ** 2))
                corr, _ = pearsonr(x, y)
                metrics.append({
                    "Cycle_i": i + 1, "Cycle_j": j + 1, "Variable": var,
                    "RMSE": rmse, "CrossCorr": corr
                })

    df_metrics = pd.DataFrame(metrics)
    df_metrics.to_csv(save_path / "cycle_comparison_metrics.csv", index=False)
    print("Salvate metriche di confronto tra cicli.")

    # --- 2. Heatmap delle metriche ---
    for metric_type in ["RMSE", "CrossCorr"]:
        for var in df[variables_to_check].columns.unique():
            subset = df_metrics[df_metrics["Variable"] == var]
            if subset.empty: continue

            mat = np.full((cycles, cycles), np.nan if metric_type == "RMSE" else 1.0)
            for _, row in subset.iterrows():
                i, j = int(row["Cycle_i"]) - 1, int(row["Cycle_j"]) - 1
                mat[i, j] = mat[j, i] = row[metric_type]

            plt.figure(figsize=(7, 6))
            sns.heatmap(mat, annot=True, fmt=".3f", cmap="viridis_r",
                        xticklabels=[f"C{i + 1}" for i in range(cycles)],
                        yticklabels=[f"C{i + 1}" for i in range(cycles)],
                        cbar_kws={'label': metric_type})
            plt.title(f"Heatmap {metric_type} - {var}")
            plt.tight_layout()
            plt.savefig(save_path / f"heatmap_{metric_type.lower()}_{var}.png")
            plt.close()

    # --- 3. Errore puntuale tra cicli ---
    pointwise_metrics = []
    for var in variables_to_check:
        if var not in df.columns: continue
        data_matrix = np.array(df[var]).reshape(cycles, steps_per_cycle)
        mean_cycle = np.mean(data_matrix, axis=0)

        for t in range(steps_per_cycle):
            values_at_t = data_matrix[:, t]
            mean_val_t = mean_cycle[t]
            rmse_t = np.sqrt(np.mean((values_at_t - mean_val_t) ** 2))

            # Evita divisione per zero
            pct_rmse_t = (rmse_t / abs(mean_val_t)) * 100 if abs(mean_val_t) > 1e-9 else 0

            pointwise_metrics.append({
                "Variable": var, "Timestep_in_cycle": t,
                "RMSE": rmse_t, "Pct_RMSE": pct_rmse_t
            })

    df_pointwise = pd.DataFrame(pointwise_metrics)
    df_pointwise.to_csv(save_path / "pointwise_cycle_errors.csv", index=False)

    # Plot degli errori puntuali
    for var in df_pointwise["Variable"].unique():
        sub = df_pointwise[df_pointwise["Variable"] == var]
        fig, ax1 = plt.subplots(figsize=(12, 6))
        ax1.plot(sub["Timestep_in_cycle"], sub["RMSE"], color='tab:blue', label="RMSE (assoluto)")
        ax1.set_xlabel("Timestep nel ciclo")
        ax1.set_ylabel("RMSE", color='tab:blue')
        ax1.tick_params(axis='y', labelcolor='tab:blue')

        ax2 = ax1.twinx()
        ax2.plot(sub["Timestep_in_cycle"], sub["Pct_RMSE"], color='tab:red', label="RMSE %")
        ax2.set_ylabel("RMSE %", color='tab:red')
        ax2.tick_params(axis='y', labelcolor='tab:red')

        plt.title(f"Errore Puntuale tra Cicli – {var}")
        fig.tight_layout()
        plt.savefig(save_path / f"pointwise_errors_{var}.png")
        plt.close()

    print("Analisi di periodicità completata.")


def export_for_graphpad(df: pd.DataFrame, save_path: Path):
    """
    NUOVO: Esporta i dati in un formato "lungo" (long-format) ideale per
    GraphPad e altri software statistici, salvando come file di testo (TSV).
    """
    print("\nEsportazione dati per GraphPad...")
    if df.empty: return

    # Aggiungi colonne per ciclo e timestep relativo
    steps_per_cycle = len(df) // CYCLES
    df['cycle'] = (df.index // steps_per_cycle) + 1
    df['timestep_in_cycle'] = df.index % steps_per_cycle

    # Seleziona le colonne di dati da esportare
    columns_to_export = [
        "inlet_pressure", "outlet_pressure", "inlet_flow", "outlet_flow",
        "KE_total_consistent", "Vorticity_mean", "WSS_mean", "TKE_mean", "Energy_Loss_mW"
    ]

    # Riorganizza il DataFrame da "wide" a "long"
    df_long = pd.melt(
        df,
        id_vars=['cycle', 'timestep_in_cycle'],
        value_vars=[col for col in columns_to_export if col in df.columns],
        var_name='variable',
        value_name='value'
    )

    # Salva in un file di testo con tabulazioni come separatore
    output_filename = save_path / "results_for_graphpad.txt"
    df_long.to_csv(output_filename, sep='\t', index=False, float_format='%.6f')

    print(f"Dati per analisi statistica salvati in: '{output_filename}'")


# =============================================================================
# --- FLUSSO PRINCIPALE ---
# =============================================================================
def main():
    """Funzione principale che orchestra l'intero processo di analisi."""
    try:
        results_files_path, start_index = automatic_parser()
        for file_path in results_files_path[start_index:]:
            if not os.path.exists(file_path):
                print(f"[ERROR] File not found: {file_path}")
                continue  # Skip to the next file if the current one does not exist

            print(f"Processing file: {file_path}")
            # Read and parse the results file
            paths_info = setup_paths_and_files(file_path)

            all_metrics = []
            file_iterator = zip(paths_info["vtu_files"], paths_info["inlet_files"], paths_info["outlet_files"])

            for i, (vtu_p, inlet_p, outlet_p) in enumerate(
                    tqdm(file_iterator, total=len(paths_info['vtu_files']), desc="Processing Timesteps")):
                all_metrics.append(process_timestep(vtu_p, inlet_p, outlet_p, i))
                # mesh = pv.read(vtu_p)  # Carica la mesh per eventuali visualizzazioni future
                # mesh.points = mesh.points + mesh.point_data["Displacement"]
                # if i == 0:
                #     v,t = pv_to_np(mesh.extract_surface())
                #     v = v[:, :, np.newaxis]
                # else:
                #     v = np.concatenate((v, pv_to_np(mesh.extract_surface())[0][:, :, np.newaxis]), axis=2)


            V = np.array([m['Volume_mL'] for m in all_metrics])
            Q_in = np.array([m['inlet_flow'] for m in all_metrics])
            Q_out = np.array([m['outlet_flow'] for m in all_metrics])
            p_in = np.array([m['inlet_pressure'] for m in all_metrics])
            p_out = np.array([m['outlet_pressure'] for m in all_metrics])
            KE_total_consistent = np.array([m['KE_total_consistent'] for m in all_metrics])
            KE_density_consistent = np.array([m['KE_density_consistent'] for m in all_metrics])
            EL_total = np.array([m['Viscous_EL_W'] for m in all_metrics])
            EL_mean = np.array([m['Viscous_EL_mean_W/m3'] for m in all_metrics])

            time = np.array(paths_info["time_values"])

            # Finds the indices of end-systolic and end-diastolic frames
            idx_es = np.argmin(V[(len(V) // 3) +1:(len(V) // 3) *2]) + (len(V) // 3) +1 # Only consider the first third of the volume for end-systolic
            idx_ed = np.argmax(V[(len(V) // 3) +1:(len(V) // 3) *2]) + (len(V) // 3) +1 # Only consider the first third of the volume for end-diastolic

            # Check if the end-systolic and end-diastolic times are aligned with the time vector
            ed = time[idx_ed]
            es = time[idx_es]
            is_aligned = np.isclose(time[idx_es], es) and np.isclose(time[idx_ed], ed)
            print(f"These values are aligned with the time vector: {is_aligned}")
            print(f"ESV, EDV: {V[idx_es]:.2f} ml, {V[idx_ed]:.2f} mL")
            print(f"SV, EF: {(V[idx_ed] - V[idx_es]):.2f} mL, {(V[idx_ed] - V[idx_es]) / V[idx_ed] * 100:.2f} %")

            # time = time - time[0]  # Normalize time to start from 0
            T = time[-1] / CYCLES
            plt.figure()
            # plt.plot(time, V, 'o-', label='Volume [mL]')
            # plt.plot(time, Q_in, 'x--', label='Q_in [mL/s]')
            # plt.plot(time, Q_out, 's--', label='Q_out [mL/s]')
            # plt.plot(time, p_in, 'd-.', label='P_in [mmHg]')
            # plt.plot(time, p_out, 'v-.', label='P_out [mmHg]')
            plt.plot(time, KE_total_consistent*1000, 'h-.', label='KE_total_consistent [mJ]')
            # plt.plot(time, KE_density_consistent, '*-.', label='KE_density_consistent [J/m³]')
            plt.plot(time, EL_total*1000, '+-.', label='EL_total [mW]')
            # plt.plot(time, EL_mean, 'D-.', label='EL_mean [W/m³]')
            # plt.plot(time, PL, '<-.', label='Energy Loss [mW]')
            plt.axvline(x=es, color='r', linestyle='--', label='ES Time')
            plt.axvline(x=es-T, color='r', linestyle='--')
            plt.axvline(x=es+T, color='r', linestyle='--')
            plt.axvline(x=ed, color='g', linestyle='--', label='ED Time')
            plt.axvline(x=ed-T, color='g', linestyle='--')
            plt.axvline(x=ed+T, color='g', linestyle='--')
            plt.ylabel('Metrics')
            plt.xlabel('Time [ms]')
            plt.title('Metrics Over Time')
            plt.grid(True)
            plt.legend()
            ticks_T = np.arange(0, CYCLES + 0.5, 0.5) * T
            labels_T = [f"{x:.1f}T" if x % 1 != 0 else f"{int(x)}T" for x in np.arange(0, CYCLES + 0.5, 0.5)]
            plt.xticks(ticks_T, labels_T)
            # plt.savefig(paths_info["save_path"] / "volume_over_time.png")
            plt.show()

            Q = np.gradient(V, time/1000) # flow in mL/s
            plt.figure()
            plt.plot(time, Q, 'o-', label='flowOriginal')
            # plt.plot(time, [m['Flow'] for m in all_metrics], 'x--', label='flowComputed')
            plt.xlabel('Time [ms]')
            plt.ylabel('Flow [mL/s]')
            plt.title('Flow Over Time')
            plt.grid(True)
            plt.legend()
            # plt.savefig(paths_info["save_path"] / "flow_over_time.png")
            plt.show()

            summary = {
                "EDV": float(V[idx_ed]),
                "ESV": float(V[idx_es]),
                "SV": float(V[idx_ed] - V[idx_es]),
                "EF": float((V[idx_ed] - V[idx_es]) / V[idx_ed] * 100),
                "ED": float(ed),
                "IDX-ED": float(idx_ed),
                "ES": float(es),
                "IDX-ES": float(idx_es),
                "RR": float(T)
            }
            with open(paths_info['save_path'] / "volume_summary.txt", "w") as f:
                for k, v in summary.items():
                    f.write(f"{k}\t{v:.6f}\n")

            # dVdt = np.gradient(V, time / 1000.0)  # mL/s
            #
            # qin = np.array([m['inlet_flow'] for m in all_metrics])  # mL/s
            # qout = np.array([m['outlet_flow'] for m in all_metrics])  # mL/s
            #
            # err_sum = np.max(np.abs(dVdt - (qin + qout)))
            # err_diff = np.max(np.abs(dVdt - (qin - qout)))
            #
            # print("||dV/dt - (Qin+Qout)||:", err_sum)
            # print("||dV/dt - (Qin-Qout)||:", err_diff)

            df = pd.DataFrame(all_metrics)
            df.insert(0, "time", paths_info["time_values"])

            save_path = paths_info["save_path"]
            df.to_csv(save_path / "simulation_results_summary.csv", index=False, float_format='%.5f')
            print(f"\nDati riassuntivi salvati in '{save_path / 'simulation_results_summary.csv'}'")

            # Genera grafici riassuntivi
            generate_summary_plots(df, save_path) # Puoi decommentare se li vuoi

            # NUOVO: Esporta dati per analisi statistica
            # export_for_graphpad(df, save_path)

            # Esegui analisi di periodicità
            analyze_periodicity(df, CYCLES, save_path) # Puoi decommentare se la vuoi

            # Genera visualizzazioni qualitative (inclusa la nuova pathlines)
            # generate_qualitative_visualizations(paths_info, df)

            plt.close('all')  # Close all matplotlib plots

            print("\nElaborazione completata con successo! ✨")

    except (FileNotFoundError, IndexError) as e:
        print(f"\nERRORE: {e}")
    except Exception as e:
        print(f"\nSi è verificato un errore inaspettato: {e}")


if __name__ == "__main__":
    main()


# import pyvista as pv
# import numpy as np
#
# points = np.array([
#     [0, 0, 0],  # A
#     [1, 0, 0],  # B
#     [0, 1, 0],  # C
#     [0, 0, 1],  # D
#     [0, 0, -1]  # E
# ])
#
# points = points.astype(float)
#
# # Due celle tetraedriche:
# #  - prima: A-B-C-D
# #  - seconda: E-A-B-C
# cells = np.hstack([
#     [4, 0, 1, 2, 3],
#     [4, 4, 0, 1, 2]
# ])
#
# # Tipo di cella: 2 celle tetraedriche
# cell_type = np.full(2, pv.CellType.TETRA)
#
# # Crea la griglia
# grid = pv.UnstructuredGrid(cells, cell_type, points)
#
# # Controllo
# print(grid.n_cells)   # 2
# print(grid.n_points)  # 5
#
# grid.plot(show_edges=True)
#
# velocity = np.array([
#     [ 0.0,  0.0,  -2.0],   # nodo (0,0,0)
#     [ 3.0, -3.0,  0.0],   # nodo (1,0,0)
#     [-3.0,  3.0,  0.0],   # nodo (0,1,0)
#     [ 0.0,  0.0,  2.0],    # nodo (0,0,1)
#     [ 0.0,  0.0,  6.0]     # nodo (0,0,-1)
# ])
#
# grid.point_data["Velocity"] = velocity
#
# # --- 5. Controllo ---
# print(grid)
# print(grid.point_data)
#
# # --- 6. Visualizzazione ---
# # grid.arrows.plot(show_edges=True)
#
# grid.set_active_vectors("Velocity")
#
# grid = grid.point_data_to_cell_data(pass_point_data=True)
#
# v = np.array(grid.point_data["Velocity"])
# vv = grid.cell_data["Velocity"]
#
# m = np.mean([
#     [0.0, 0.0, -2.0],  # nodo (0,0,0)
#     [3.0, -3.0, 0.0],  # nodo (1,0,0)
#     [-3.0, 3.0, 0.0],  # nodo (0,1,0)
#     [0.0, 0.0, 6.0]  # nodo (0,0,-1)
# ])

