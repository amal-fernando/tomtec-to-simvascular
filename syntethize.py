import pandas as pd
import numpy as np
import glob, os
from scipy.interpolate import Akima1DInterpolator as akima
from pathlib import Path
from tkinter import filedialog, Tk, filedialog, simpledialog

# === PARAMETRI ===
N_SAMPLES = 300           # numero di punti nel tempo normalizzato
TIME_COL = "time"         # nome colonna tempo
SEPARATOR = ","           # separatore CSV (prova anche ";" se necessario)
# DECIMAL = ","             # se i numeri usano la virgola decimale
DECIMAL = "."           # se invece usano il punto decimale

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
        return [], [], 0, None

    root_dir = Path(root_dir)
    # Dizionario per salvare i path
    csv_paths = {}
    txt_paths = {}

    # Trova tutte le cartelle Pxxxx
    # patient_dirs = sorted([
    #     d for d in os.listdir(root_dir)
    #     if d.startswith('P') and os.path.isdir(os.path.join(root_dir, d))
    # ])
    patient_dirs = sorted([
        d for d in root_dir.iterdir()
        if d.is_dir() and d.name.startswith('P')
    ])

    if not patient_dirs:
        print("❌ Nessuna sottocartella 'Pxxxx' trovata.")
        return [], [], 0, None

    # --- Cerca file in ciascuna cartella paziente ---
    for patient_dir in patient_dirs:
        patient_id = patient_dir.name
        results_dir = patient_dir / "results"

        csv_file = results_dir / "simulation_results_summary.csv"
        txt_file = results_dir / "volume_summary.txt"

        if not csv_file.exists():
            print(f"⚠️  CSV mancante per {patient_id}: {csv_file}")
        else:
            csv_paths[patient_id] = csv_file

        if not txt_file.exists():
            print(f"⚠️  TXT mancante per {patient_id}: {txt_file}")
        else:
            txt_paths[patient_id] = txt_file

        # --- Filtra solo i pazienti con entrambi i file ---
    common_patients = [
        p for p in csv_paths.keys()
        if p in txt_paths
    ]

    if not common_patients:
        print("❌ Nessun paziente con entrambi i file trovati.")
        return [], [], 0, None

    # --- Crea liste parallele ordinate ---
    csv_list = [csv_paths[p] for p in common_patients]
    txt_list = [txt_paths[p] for p in common_patients]

    # --- Mostra elenco pazienti ---
    print("\n✅ Pazienti trovati con entrambi i file:")
    for i, p in enumerate(common_patients):
        print(f"{i}: {p}")

    # --- Chiedi indice di partenza ---
    start_index = simpledialog.askinteger(
        "Indice di partenza",
        f"Inserisci l'indice da cui iniziare (0 - {len(common_patients) - 1}):",
        minvalue=0,
        maxvalue=len(common_patients) - 1
    )

    if start_index is None:
        print("❌ Nessun indice selezionato. Uscita.")
        return [], [], 0, None

    # --- Crea cartella di salvataggio ---
    save_path = root_dir / "resampled_csv"
    save_path.mkdir(exist_ok=True)

    return csv_list, txt_list, start_index, save_path

'''
        csv_paths[patient] = {}
        txt_paths[patient] = {}

        csv_path = os.path.join(root_dir, patient, 'results', 'simulation_results_summary.csv')
        txt_path = os.path.join(root_dir, patient, 'results', 'volume_summary.txt')

        if not os.path.isfile(csv_path):
            print(f'[ATTENZIONE] File non trovato: {csv_path}')
            continue

        if not os.path.isfile(txt_path):
            print(f'[ATTENZIONE] File non trovato: {txt_path}')
            continue

        # # Estrai la parte numerica e ordina per valore numerico
        # result_files.sort(key=lambda f: int(re.search(r'result_(\d+)\.vtu', f).group(1)))
        #
        # # Prendi il più piccolo
        # first_result = result_files[0]
        # results_path = os.path.join(sim_dir, first_result)

        if os.path.exists(csv_path):
            csv_paths[patient] = csv_path
        else:
            print(f'[ATTENZIONE] File mancante: {csv_path}')
            csv_paths[patient] = None  # oppure salta, se preferisci

        if os.path.exists(txt_path):
            txt_paths[patient] = txt_path
        else:
            print(f'[ATTENZIONE] File mancante: {txt_path}')
            txt_paths[patient] = None  # oppure salta, se preferisci

    results_files_path = [
        path for path in paths.values()
        if path is not None
    ]

    save_path = Path(root_dir)/ "resampled_csv"
    save_path.mkdir(exist_ok=True)

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

    return results_files_path, start_index, save_path
'''

def setup_paths_and_files(file_path):
    """
    Apre una finestra di dialogo per selezionare il file iniziale,
    e prepara tutti i percorsi e le liste di file necessari per l'analisi.
    """
    # Tk().withdraw()
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError("Nessun file selezionato o il file non esiste.")

    path = file_path.parent

    print(f"Cartella dati: {path}")

    return {
        "subject": path.parent
    }

# === STRUTTURE DATI ===
data_pre, data_post = {}, {}
summary_pre, summary_post = {}, {}  # per i dati TXT
t_common = np.linspace(0, 3, N_SAMPLES)

# === RACCOGLI FILE ===
csv_list, txt_list, start_index, save_path = automatic_parser()
print(f"📂 Cartella salvataggio: {save_path}")

# for file in csv_list[start_index:]:
#     if not file.exists():
#         print(f"[ERROR] File not found: {file}")
#         continue  # Skip to the next file if the current one does not exist
#
#     print(f"Processing file: {file}")
#     # Read and parse the results file
#     paths_info = setup_paths_and_files(file)
#     subject = os.path.splitext(os.path.basename(paths_info["subject"]))[0]
#
#     # determina gruppo (pre o post)
#     if "_pre" in subject.lower():
#         group = "pre"
#     elif "_post" in subject.lower():
#         group = "post"
#     else:
#         print(f"⚠️  File {subject} non riconosciuto come pre/post, salto.")
#         continue
#
#     df = pd.read_csv(file, sep="\t|;|,", engine="python", decimal=DECIMAL)
#     if TIME_COL not in df.columns:
#         raise ValueError(f"La colonna '{TIME_COL}' non è nel file {file}")
#
#     # normalizza tempo su 0–1
#     t = df[TIME_COL].to_numpy()
#     t_norm = 3*((t - t.min()) / (t.max() - t.min()))
#
#     # per ogni variabile
#     for col in df.columns:
#         if col == TIME_COL:
#             continue
#         y = df[col].to_numpy(dtype=float)
#         f_interp = akima(t_norm, y, method="makima", extrapolate=True)
#         y_resamp = f_interp(t_common)
#
#         if group == "pre":
#             data_dict = data_pre
#         else:
#             data_dict = data_post
#
#         if col not in data_dict:
#             data_dict[col] = {}
#         data_dict[col][subject] = y_resamp

# === FUNZIONE DI SALVATAGGIO ===
def save_group(data_dict, group_name, base_folder):
    folder = os.path.join(base_folder, group_name)
    os.makedirs(folder, exist_ok=True)

    for var, subj_data in data_dict.items():
        df_out = pd.DataFrame(index=t_common)
        df_out.index.name = "time (T/Tmax)"
        for subj, y_resamp in subj_data.items():
            df_out[subj] = y_resamp
        out_name = os.path.join(folder, f"{var}.csv")
        df_out.to_csv(out_name, sep=f"{SEPARATOR}", decimal=f"{DECIMAL}")
        print(f"✅ Salvato: {out_name}")

# === SALVA I DUE GRUPPI ===
# if data_pre:
#     save_group(data_pre, "pre", save_path)
# if data_post:
#     save_group(data_post, "post", save_path)

# --- PROCESSA FILE TXT ---
for file in txt_list[start_index:]:
    pid = file.parents[1].name
    print(f"📄 Processing (TXT): {pid}")

    df_txt = pd.read_csv(file, sep=r"\s+", header=None, names=["Variable", pid], engine="python", decimal=DECIMAL)

    # Determina gruppo pre/post
    if "_pre" in pid.lower():
        target = summary_pre
    elif "_post" in pid.lower():
        target = summary_post
    else:
        print(f"⚠️ {pid} non riconosciuto come pre/post, salto TXT.")
        continue

    for _, row in df_txt.iterrows():
        var = row["Variable"]
        val = float(row[pid])
        if var not in target:
            target[var] = {}
        target[var][pid] = val

def save_summary_txt(summary_dict, group_name, base_folder):
    """Salva un file CSV con le variabili statiche (EDV, ESV, EF, ...) per tutti i soggetti."""
    if not summary_dict:
        return
    folder = Path(base_folder)
    folder.mkdir(exist_ok=True)

    df_out = pd.DataFrame(summary_dict)
    df_out = df_out.reset_index().rename(columns={"index": "Variable"})
    df_out = df_out.set_index("Variable").transpose()

    out_name = folder / f"summary_txt_{group_name}.csv"
    df_out.to_csv(out_name, sep=SEPARATOR, decimal=DECIMAL)
    print(f"✅ Salvato file TXT: {out_name}")

# SALVATAGGIO summary_txt
save_summary_txt(summary_pre, "pre", save_path)
save_summary_txt(summary_post, "post", save_path)

print("\n🎉 Tutti i file salvati in ./resampled_csv/pre/ e ./resampled_csv/post/")