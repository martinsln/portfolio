"""
Composite Fiber Orientation Optimizer
======================================
Projet ENSAM 2026 - Jeanne, Turki, Saulnier, Hénon

CE QUE FAIT CE SCRIPT :
------------------------
1. PARSE les fichiers .rpt Abaqus (un par matériau)
2. FILTRE les 15 éléments les plus chargés (Von Mises max) — zones critiques
3. ENTRAÎNE un Gaussian Process par composante de contrainte :
     GP_S11 : prédit σ_xx  en fonction de (E, ν, element_id)
     GP_S22 : prédit σ_yy  en fonction de (E, ν, element_id)
     GP_S12 : prédit σ_xy  en fonction de (E, ν, element_id)
4. VALIDE par Leave-One-Simulation-Out cross-validation
5. OPTIMISE : pour chaque matériau × orientation θ, prédit les contraintes,
   calcule Tsai-Wu, trouve θ_opt qui minimise f(σ)
6. AFFICHE le meilleur matériau + orientation optimale

UTILISATION :
  1. Mettre tous les .rpt dans le même dossier que ce script
  2. Vérifier que RPT_FILES correspond bien aux noms de vos fichiers
  3. Lancer : python composite_optimizer.py

DÉPENDANCES :
  pip install numpy pandas scikit-learn matplotlib
"""

import numpy as np
import pandas as pd
import os
import re
import warnings
warnings.filterwarnings("ignore")

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# =============================================================================
# 0. CONFIGURATION — ADAPTER ICI
# =============================================================================

# Dossier contenant les .rpt (. = même dossier que le script)
RPT_FOLDER = "tensor"

# Mapping : nom_simulation → fichier .rpt
# Vérifier que les noms correspondent exactement à vos fichiers
RPT_FILES = {
    'hayon_basalt_PC':    'tensor_basalt_PC.rpt',
    'hayon_basalto_epoxy':'tensor_basalto_epoxy.rpt',
    'hayon_basalto_PE':   'tensor_basalt_PE.rpt',
    'hayon_carb_epoxy':   'tensor_carb_epoxy.rpt',
    'hayon_carb_PC':      'tensor_carb_PC.rpt',
    'hayon_carb_PE':      'tensor_carb_PE.rpt',
    'hayon_PP':           'tensor_PP.rpt',
    'hayon_vitre_epoxy':  'tensor_vitre_epoxy.rpt',
    'hayon_vitre_PC':     'tensor_vitre_pc.rpt',
    'hayon_vitre_PE':     'tensor_vitre_PE.rpt',
}

# Nombre de mailles critiques à garder par simulation
N_CRITICAL = 30

# Grille d'orientations à tester (degrés)
THETA_RANGE = np.arange(0, 180, 5)

# Filtre spatial — exclure les bords (boundary conditions artificielles)
# .inp utilisé pour récupérer les coordonnées Y de chaque élément
INP_FILE = "inp/hayon.inp"
Y_MIN = -250.0   # mm — exclure Y < -250
Y_MAX =  250.0   # mm — exclure Y > +250

# =============================================================================
# 1. PROPRIÉTÉS MATÉRIAU — Tsai-Wu (MPa)
# =============================================================================

# Résistances verre/époxy typiques
X_t =  1000.0   # traction sens fibre
X_c =   600.0   # compression sens fibre
Y_t =    30.0   # traction transverse
Y_c =   120.0   # compression transverse
S   =    70.0   # cisaillement plan

F1  =  1/X_t - 1/X_c
F2  =  1/Y_t - 1/Y_c
F11 =  1/(X_t * X_c)
F22 =  1/(Y_t * Y_c)
F66 =  1/(S**2)
F12 = -0.5 * np.sqrt(F11 * F22)

# Catalogue matériaux — valeurs à 23°C
MATERIAL_CATALOG = [
    {'name': 'PP',              'simulation': 'hayon_PP',           'E': 1500.0, 'nu': 0.42, 'rho': 1210},
    {'name': 'Verre + PE',      'simulation': 'hayon_vitre_PE',     'E': 1727.0, 'nu': 0.34, 'rho': 1210},
    {'name': 'Verre + PC',      'simulation': 'hayon_vitre_PC',     'E': 3135.0, 'nu': 0.30, 'rho': 1210},
    {'name': 'Verre + époxy',   'simulation': 'hayon_vitre_epoxy',  'E': 6434.0, 'nu': 0.26, 'rho': 1420},
    {'name': 'Basalte + PE',    'simulation': 'hayon_basalto_PE',   'E': 1735.0, 'nu': 0.34, 'rho': 1510},
    {'name': 'Basalte + PC',    'simulation': 'hayon_basalt_PC',    'E': 3158.0, 'nu': 0.30, 'rho': 1210},
    {'name': 'Basalte + époxy', 'simulation': 'hayon_basalto_epoxy','E': 6527.0, 'nu': 0.26, 'rho': 1420},
    {'name': 'Carbone + PE',    'simulation': 'hayon_carb_PE',      'E': 1755.0, 'nu': 0.34, 'rho': 1210},
    {'name': 'Carbone + PC',    'simulation': 'hayon_carb_PC',      'E': 3223.0, 'nu': 0.30, 'rho': 1210},
    {'name': 'Carbone + époxy', 'simulation': 'hayon_carb_epoxy',   'E': 6775.0, 'nu': 0.27, 'rho': 1420},
]

# =============================================================================
# 2. ROTATION TENSEUR + TSAI-WU
# =============================================================================

def rotate_stress(s11, s22, s12, theta_deg):
    """Rotation du tenseur plan vers le repère matériau (angle θ en degrés)."""
    t = np.radians(theta_deg)
    c, s = np.cos(t), np.sin(t)
    sigma_1  =  c**2*s11 + s**2*s22 + 2*c*s*s12
    sigma_2  =  s**2*s11 + c**2*s22 - 2*c*s*s12
    sigma_12 = -c*s*s11  + c*s*s22  + (c**2 - s**2)*s12
    return sigma_1, sigma_2, sigma_12


def tsai_wu(s1, s2, s12):
    """Critère de Tsai-Wu. f < 1 = sûr, f ≥ 1 = rupture."""
    return (F1*s1 + F2*s2
            + F11*s1**2 + F22*s2**2
            + F66*s12**2 + 2*F12*s1*s2)


def tsai_wu_global(s11, s22, s12, theta_deg):
    """Tsai-Wu depuis tenseur global + angle."""
    s1, s2, s12r = rotate_stress(s11, s22, s12, theta_deg)
    return tsai_wu(s1, s2, s12r)


# =============================================================================
# 3. PARSER .rpt
# =============================================================================

def parse_float(s):
    """
    Convertit une chaîne Abaqus en float.
    Gère les formats : '1.234', '934.179E-03', '-3.456', '0.'
    """
    s = s.strip()
    if s in ('0.', '0'):
        return 0.0
    try:
        return float(s)
    except ValueError:
        return float(s.replace('E', 'e'))


def parse_rpt(filepath):
    """
    Parse un fichier .rpt Abaqus et retourne un DataFrame avec :
    element, S11, S22, S33, S12
    
    Format attendu (colonnes) :
    Element Label | Int Pt | S11@Loc3 | S11@Loc4 | S22@Loc3 | S22@Loc4 |
                             S33@Loc3 | S33@Loc4 | S12@Loc3 | S12@Loc4

    On prend @Loc4 (SPOS = face extérieure, plus critique en flexion).
    On moyenne les 4 points d'intégration par élément.
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()

    # Trouver la ligne de tirets qui précède les données
    data_start = None
    for i, line in enumerate(lines):
        if line.strip().startswith('---'):
            data_start = i + 1
            break

    if data_start is None:
        raise ValueError(f"Format inattendu dans {filepath} — ligne de tirets introuvable")

    # Parser les lignes de données
    # Format : "   66001045   1   val1   val2   val3   val4   val5   val6   val7   val8"
    raw = {}   # {element_id: [(s11, s22, s33, s12), ...]}

    for line in lines[data_start:]:
        line = line.strip()
        if not line or line.startswith('*') or line.startswith('-'):
            continue

        parts = line.split()
        if len(parts) < 10:
            continue

        try:
            elem_id  = int(parts[0])
            # int_pt = int(parts[1])  — pas utilisé
            # On prend @Loc4 (index 3, 5, 7, 9)
            s11 = parse_float(parts[3])
            s22 = parse_float(parts[5])
            s33 = parse_float(parts[7])
            s12 = parse_float(parts[9])
        except (ValueError, IndexError):
            continue

        if elem_id not in raw:
            raw[elem_id] = []
        raw[elem_id].append((s11, s22, s33, s12))

    if not raw:
        raise ValueError(f"Aucune donnée parsée dans {filepath}")

    # Moyenne des points d'intégration par élément
    records = []
    for elem_id, pts in raw.items():
        arr = np.array(pts)
        records.append({
            'element': elem_id,
            'S11': arr[:, 0].mean(),
            'S22': arr[:, 1].mean(),
            'S33': arr[:, 2].mean(),
            'S12': arr[:, 3].mean(),
        })

    df = pd.DataFrame(records)
    print(f"  Parsed {len(df)} elements from {os.path.basename(filepath)}")
    return df


# =============================================================================
# 3b. PARSER .inp — coordonnées Y pour filtrage spatial
# =============================================================================

def load_element_y(inp_filepath):
    """
    Extrait la coordonnée Y du barycentre de chaque élément depuis le .inp.
    Retourne dict {elem_id: y}
    """
    nodes = {}
    elements = {}
    mode = None

    with open(inp_filepath, 'r') as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if not line or line.startswith('**'):
            continue
        if line.upper().startswith('*NODE'):
            mode = 'node'; continue
        elif line.upper().startswith('*ELEMENT'):
            mode = 'element'; continue
        elif line.startswith('*'):
            mode = None; continue

        if mode == 'node':
            parts = line.replace(',', ' ').split()
            if len(parts) >= 4:
                try:
                    nodes[int(parts[0])] = float(parts[2])  # Y = colonne index 2
                except ValueError:
                    continue
        elif mode == 'element':
            parts = line.replace(',', ' ').split()
            if len(parts) >= 4:
                try:
                    eid = int(parts[0])
                    nids = [int(p) for p in parts[1:4]]
                    elements[eid] = nids
                except ValueError:
                    continue

    elem_y = {}
    for eid, nids in elements.items():
        ys = [nodes[n] for n in nids if n in nodes]
        if ys:
            elem_y[eid] = sum(ys) / len(ys)

    print(f"  Coordonnées Y chargées pour {len(elem_y)} éléments depuis {os.path.basename(inp_filepath)}")
    return elem_y


# =============================================================================
# 4. CONSTRUCTION DU DATASET
# =============================================================================

def build_dataset(rpt_folder, rpt_files, material_catalog, n_critical=15,
                  inp_file=None, y_min=None, y_max=None):
    """
    Charge tous les .rpt, filtre les n_critical mailles les plus chargées
    (par Von Mises), associe E et ν de chaque matériau.
    Si inp_file + y_min/y_max fournis, exclut les éléments hors de la plage Y
    (pour éviter les artefacts des conditions aux limites).

    Retourne un DataFrame : simulation, element, S11, S22, S12, E, nu
    """
    mat_map = {m['simulation']: m for m in material_catalog}
    all_records = []
    missing_files = []

    # Charger le filtre spatial Y si demandé
    elem_y = {}
    if inp_file and (y_min is not None or y_max is not None):
        print(f"\nChargement filtre spatial depuis {inp_file} ...")
        elem_y = load_element_y(inp_file)
        print(f"  Filtre Y : [{y_min}, {y_max}] mm")

    for sim_name, rpt_file in rpt_files.items():
        filepath = os.path.join(rpt_folder, rpt_file)

        if not os.path.exists(filepath):
            print(f"  [MANQUANT] {rpt_file}")
            missing_files.append(rpt_file)
            continue

        try:
            df = parse_rpt(filepath)
        except Exception as e:
            print(f"  [ERREUR] {rpt_file} : {e}")
            continue

        # Von Mises comme proxy pour identifier les mailles les plus chargées
        df['vm'] = np.sqrt(
            df['S11']**2 + df['S22']**2 - df['S11']*df['S22'] + 3*df['S12']**2
        )

        # Filtre spatial Y — exclure les bords
        if elem_y and (y_min is not None or y_max is not None):
            before = len(df)
            df['_y'] = df['element'].map(elem_y)
            if y_min is not None:
                df = df[df['_y'] >= y_min]
            if y_max is not None:
                df = df[df['_y'] <= y_max]
            df = df.drop(columns='_y')
            print(f"  Filtre Y : {before} → {len(df)} éléments conservés")

        df = df.nlargest(n_critical, 'vm').reset_index(drop=True)
        df.drop(columns='vm', inplace=True)

        mat = mat_map.get(sim_name, {})
        df['simulation'] = sim_name
        df['E']  = mat.get('E',  np.nan)
        df['nu'] = mat.get('nu', np.nan)

        all_records.append(df)

    if not all_records:
        raise RuntimeError("Aucun fichier .rpt chargé. Vérifier RPT_FOLDER et RPT_FILES.")

    dataset = pd.concat(all_records, ignore_index=True)

    if missing_files:
        print(f"\n[ATTENTION] {len(missing_files)} fichier(s) manquant(s) : {missing_files}")
        print("  Le modèle sera entraîné sur les simulations disponibles uniquement.\n")

    print(f"\nDataset : {len(dataset)} points "
          f"({len(all_records)} simulations × ~{n_critical} éléments)")
    return dataset


# =============================================================================
# 5. SURROGATE — 3 Gaussian Processes
# =============================================================================

class StressSurrogate:
    """
    3 GPs indépendants : GP_S11, GP_S22, GP_S12
    Features : E, nu, element_id (encodé numériquement)
    
    On utilise element_id car les .rpt n'ont pas de coordonnées x,y,z.
    Le GP apprend comment les contraintes varient selon le matériau (E, ν)
    pour chaque position de maille.
    """

    TARGETS = ['S11', 'S22', 'S12']

    def __init__(self):
        self.scalers    = {}
        self.gps        = {}
        self.elem_index = {}   # mapping element_id → entier
        self.trained    = False

        kernel = (ConstantKernel(1.0, (1e-3, 1e3))
                  * RBF(length_scale=np.ones(3),
                        length_scale_bounds=[(1e-3, 1e3)]*3)
                  + WhiteKernel(noise_level=1e-2,
                                noise_level_bounds=(1e-6, 1e1)))

        for t in self.TARGETS:
            self.scalers[t] = StandardScaler()
            self.gps[t] = GaussianProcessRegressor(
                kernel=kernel, n_restarts_optimizer=5,
                normalize_y=True, random_state=42)

    def _encode_elements(self, df):
        """Encode les element_id en entiers pour le GP."""
        if not self.elem_index:
            unique_elems = sorted(df['element'].unique())
            self.elem_index = {e: i for i, e in enumerate(unique_elems)}
        return df['element'].map(self.elem_index).fillna(-1).astype(int)

    def _make_X(self, df):
        elem_enc = self._encode_elements(df)
        return np.column_stack([df['E'].values, df['nu'].values, elem_enc.values])

    def fit(self, df):
        X = self._make_X(df)
        print(f"\nEntraînement surrogate sur {len(df)} points ...")
        for t in self.TARGETS:
            y = df[t].values
            Xs = self.scalers[t].fit_transform(X)
            self.gps[t].fit(Xs, y)
            print(f"  GP_{t} OK")
        self.trained = True
        return self

    def predict(self, E_vals, nu_vals, elem_ids, return_std=False):
        """
        Prédit S11, S22, S12 pour des vecteurs (E, nu, element_id).
        Retourne dict {S11: array, S22: array, S12: array}
        """
        elem_enc = np.array([self.elem_index.get(e, 0) for e in elem_ids])
        X = np.column_stack([E_vals, nu_vals, elem_enc])
        preds, stds = {}, {}
        for t in self.TARGETS:
            Xs = self.scalers[t].transform(X)
            mu, sigma = self.gps[t].predict(Xs, return_std=True)
            preds[t] = mu
            stds[t]  = sigma
        if return_std:
            return preds, stds
        return preds

    def cross_validate(self, df):
        """Leave-One-Simulation-Out CV."""
        print("\nCross-validation (Leave-One-Simulation-Out) ...")
        sims = df['simulation'].unique()
        results = {}

        for t in self.TARGETS:
            y_true_all, y_pred_all = [], []
            for sim in sims:
                mask_test  = df['simulation'] == sim
                mask_train = ~mask_test
                if mask_train.sum() == 0:
                    continue

                df_train = df[mask_train].copy()
                df_test  = df[mask_test].copy()

                # Recalculer l'encodage sur train seulement
                surr_cv = StressSurrogate()
                surr_cv.fit(df_train)

                # Prédire sur test
                p = surr_cv.predict(
                    df_test['E'].values, df_test['nu'].values,
                    df_test['element'].values)

                y_true_all.extend(df_test[t].values)
                y_pred_all.extend(p[t])

            rmse = np.sqrt(mean_squared_error(y_true_all, y_pred_all))
            results[t] = rmse
            print(f"  RMSE {t}: {rmse:.3f} MPa")

        return results


# =============================================================================
# 6. OPTIMISATION
# =============================================================================

def optimize(surrogate, df_critical, material_catalog,
             theta_range=None, top_n=5):
    """
    Pour chaque matériau × orientation θ :
      - Prédit S11, S22, S12 sur les mailles critiques
      - Calcule Tsai-Wu sur chaque maille
      - Retient le max (maille la plus critique)
    
    Retourne le classement trié par Tsai-Wu croissant.
    """
    if theta_range is None:
        theta_range = THETA_RANGE

    # Mailles critiques (element_ids fixes)
    elem_ids = df_critical['element'].unique()
    n_elems  = len(elem_ids)

    records = []
    for mat in material_catalog:
        E_arr  = np.full(n_elems, mat['E'])
        nu_arr = np.full(n_elems, mat['nu'])

        preds = surrogate.predict(E_arr, nu_arr, elem_ids)

        for theta in theta_range:
            tw_vals = [
                tsai_wu_global(preds['S11'][i], preds['S22'][i],
                               preds['S12'][i], theta)
                for i in range(n_elems)
            ]
            records.append({
                'Matériau':    mat['name'],
                'ρ (kg/m³)':  mat['rho'],
                'E (MPa)':    mat['E'],
                'ν':          mat['nu'],
                'θ_opt (°)':  theta,
                'Tsai-Wu max': max(tw_vals),
                'Tsai-Wu moy': np.mean(tw_vals),
                'Sûr':         max(tw_vals) < 1.0,
            })

    results = pd.DataFrame(records)
    results.sort_values('Tsai-Wu max', inplace=True)
    results.reset_index(drop=True, inplace=True)

    # Affichage top N
    print(f"\n{'='*65}")
    print(f"TOP {top_n} — Configurations les plus sûres (Tsai-Wu le plus bas)")
    print(f"{'='*65}")
    cols = ['Matériau', 'ρ (kg/m³)', 'θ_opt (°)', 'Tsai-Wu max', 'Sûr']
    print(results.head(top_n)[cols].to_string(index=False,
          float_format=lambda x: f"{x:.4f}"))

    # Meilleure orientation par matériau
    print(f"\n{'='*65}")
    print("RÉSULTAT FINAL — Orientation optimale par matériau")
    print(f"{'='*65}")
    best = results.loc[results.groupby('Matériau')['Tsai-Wu max'].idxmin()]
    best = best.sort_values('Tsai-Wu max')
    print(best[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\n[Rappel] Tsai-Wu < 1 → sûr  |  ≥ 1 → rupture")

    return results, best


# =============================================================================
# 7. VISUALISATION
# =============================================================================

def plot_results(results, best):
    """Deux graphiques : courbes Tsai-Wu vs θ + barplot résumé."""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # --- Graphique 1 : Tsai-Wu vs orientation par matériau ---
    colors = plt.cm.tab10(np.linspace(0, 1, len(results['Matériau'].unique())))
    for i, (mat_name, group) in enumerate(results.groupby('Matériau')):
        group = group.sort_values('θ_opt (°)')
        ax1.plot(group['θ_opt (°)'], group['Tsai-Wu max'],
                 label=mat_name, color=colors[i], linewidth=1.8)

    ax1.axhline(y=1.0, color='red', linestyle='--', linewidth=2,
                label='Seuil de rupture (f = 1)')
    ax1.set_xlabel('Orientation des fibres θ (°)', fontsize=11)
    ax1.set_ylabel('Tsai-Wu max (sur mailles critiques)', fontsize=11)
    ax1.set_title('Tsai-Wu vs orientation — tous matériaux', fontsize=12)
    ax1.legend(fontsize=7, loc='upper right')
    ax1.grid(True, alpha=0.3)

    # --- Graphique 2 : Barplot orientation optimale + Tsai-Wu ---
    mats   = best['Matériau'].values
    thetas = best['θ_opt (°)'].values
    tw     = best['Tsai-Wu max'].values
    safe   = best['Sûr'].values
    colors_bar = ['#2ecc71' if s else '#e74c3c' for s in safe]

    x = np.arange(len(mats))
    bars = ax2.barh(x, tw, color=colors_bar, edgecolor='white', height=0.6)
    ax2.axvline(x=1.0, color='red', linestyle='--', linewidth=2,
                label='Seuil de rupture')

    # Annotation : orientation optimale sur chaque barre
    for i, (bar, theta) in enumerate(zip(bars, thetas)):
        ax2.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                 f'θ = {theta:.0f}°', va='center', fontsize=8)

    ax2.set_yticks(x)
    ax2.set_yticklabels(mats, fontsize=9)
    ax2.set_xlabel('Tsai-Wu max à θ_opt', fontsize=11)
    ax2.set_title('Orientation optimale par matériau\n(vert = sûr, rouge = rupture)',
                  fontsize=12)

    patch_safe   = mpatches.Patch(color='#2ecc71', label='Sûr (Tsai-Wu < 1)')
    patch_unsafe = mpatches.Patch(color='#e74c3c', label='Rupture (Tsai-Wu ≥ 1)')
    ax2.legend(handles=[patch_safe, patch_unsafe], fontsize=9)
    ax2.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    out = 'img/resultats_optimisation.png'
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"\nGraphique sauvegardé : {out}")


# =============================================================================
# 8. MAIN
# =============================================================================

if __name__ == "__main__":

    print("=" * 65)
    print("COMPOSITE FIBER OPTIMIZER — Projet ENSAM 2026")
    print("=" * 65)

    # --- Chargement des données ---
    print("\n--- Chargement des .rpt ---")
    df = build_dataset(RPT_FOLDER, RPT_FILES, MATERIAL_CATALOG, N_CRITICAL,
                       inp_file=INP_FILE, y_min=Y_MIN, y_max=Y_MAX)

    print(f"\nAperçu dataset :")
    print(df[['simulation', 'E', 'nu', 'S11', 'S22', 'S12']].describe().round(3))

    # --- Surrogate ---
    surrogate = StressSurrogate()
    surrogate.fit(df)
    surrogate.cross_validate(df)

    # --- Optimisation ---
    print("\n--- Optimisation ---")
    results, best = optimize(
        surrogate, df, MATERIAL_CATALOG,
        theta_range=THETA_RANGE, top_n=10)

    # --- Sauvegarde ---
    os.makedirs('results', exist_ok=True)
    os.makedirs('img', exist_ok=True)
    results.to_csv('results/optimization_results.csv', index=False)
    best.to_csv('results/best_per_material.csv', index=False)
    print("\nFichiers sauvegardés : results/optimization_results.csv, results/best_per_material.csv")

    # --- Plots ---
    plot_results(results, best)

    print("\nTerminé.")