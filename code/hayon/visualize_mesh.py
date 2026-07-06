"""
Visualisation 2D du maillage + zones critiques
===============================================
Projet ENSAM 2026

CE QUE FAIT CE SCRIPT :
  1. Parse le fichier .inp → coordonnées des nœuds + connectivité des éléments
  2. Parse un fichier .rpt → contraintes Von Mises par élément
  3. Projette la pièce en 2D (vue choisie : XY, XZ ou YZ)
  4. Colorie chaque élément par niveau de Von Mises
  5. Marque les 15 mailles critiques avec des étoiles

UTILISATION :
  Mettre ce script dans le même dossier que le .inp et le .rpt
  puis lancer : python visualize_mesh.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import matplotlib.colors as mcolors
import os

# =============================================================================
# CONFIGURATION
# =============================================================================

INP_FILE = 'inp/hayon.inp'   # fichier géométrie
RPT_FILE = 'tensor/tensor_basalt_PC.rpt'  # fichier contraintes (un au choix)
N_CRITICAL = 15                     # nombre de mailles critiques à marquer
VIEW = 'YZ'                         # vue : 'XY', 'XZ' ou 'YZ'

# Filtre spatial — exclure les bords (boundary conditions artificielles)
Y_MIN = -250.0   # mm
Y_MAX =  250.0   # mm

# =============================================================================
# 1. PARSER .inp
# =============================================================================

def parse_inp(filepath):
    """
    Extrait les nœuds et éléments du fichier .inp Abaqus.
    
    Retourne :
      nodes : dict {node_id: (x, y, z)}
      elements : dict {elem_id: [node1, node2, node3]}
    """
    nodes    = {}
    elements = {}

    with open(filepath, 'r') as f:
        lines = f.readlines()

    mode = None  # 'node' ou 'element'

    for line in lines:
        line = line.strip()
        if not line or line.startswith('**'):
            continue

        if line.upper().startswith('*NODE'):
            mode = 'node'
            continue
        elif line.upper().startswith('*ELEMENT'):
            mode = 'element'
            continue
        elif line.startswith('*'):
            mode = None
            continue

        if mode == 'node':
            parts = line.replace(',', ' ').split()
            if len(parts) >= 4:
                try:
                    nid = int(parts[0])
                    x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                    nodes[nid] = (x, y, z)
                except ValueError:
                    continue

        elif mode == 'element':
            parts = line.replace(',', ' ').split()
            if len(parts) >= 4:
                try:
                    eid = int(parts[0])
                    # Éléments S3 : 3 nœuds
                    nids = [int(p) for p in parts[1:4]]
                    elements[eid] = nids
                except ValueError:
                    continue

    print(f"  .inp : {len(nodes)} nœuds, {len(elements)} éléments")
    return nodes, elements


def compute_centroids(nodes, elements):
    """
    Calcule le barycentre (x, y, z) de chaque élément.
    Retourne dict {elem_id: (cx, cy, cz)}
    """
    centroids = {}
    for eid, nids in elements.items():
        coords = [nodes[n] for n in nids if n in nodes]
        if len(coords) == 3:
            cx = sum(c[0] for c in coords) / 3
            cy = sum(c[1] for c in coords) / 3
            cz = sum(c[2] for c in coords) / 3
            centroids[eid] = (cx, cy, cz)
    return centroids


# =============================================================================
# 2. PARSER .rpt (Von Mises)
# =============================================================================

def parse_float(s):
    s = s.strip()
    if s in ('0.', '0'):
        return 0.0
    try:
        return float(s)
    except ValueError:
        return float(s.replace('E', 'e'))


def parse_rpt_vm(filepath):
    """
    Parse le .rpt et calcule Von Mises par élément.
    Retourne dict {elem_id: vm}
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()

    data_start = None
    for i, line in enumerate(lines):
        if line.strip().startswith('---'):
            data_start = i + 1
            break

    if data_start is None:
        raise ValueError(f"Format inattendu dans {filepath}")

    raw = {}
    for line in lines[data_start:]:
        line = line.strip()
        if not line or line.startswith('*') or line.startswith('-'):
            continue
        parts = line.split()
        if len(parts) < 10:
            continue
        try:
            eid = int(parts[0])
            s11 = parse_float(parts[3])
            s22 = parse_float(parts[5])
            s12 = parse_float(parts[9])
            vm  = np.sqrt(s11**2 + s22**2 - s11*s22 + 3*s12**2)
            if eid not in raw:
                raw[eid] = []
            raw[eid].append(vm)
        except (ValueError, IndexError):
            continue

    # Moyenne des points d'intégration
    vm_map = {eid: np.mean(vals) for eid, vals in raw.items()}
    print(f"  .rpt : {len(vm_map)} éléments avec Von Mises")
    return vm_map


def parse_rpt_full(filepath):
    """
    Parse le .rpt et retourne S11, S22, S12 par élément (moyennés).
    Retourne dict {elem_id: (s11, s22, s12)}
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()

    data_start = None
    for i, line in enumerate(lines):
        if line.strip().startswith('---'):
            data_start = i + 1
            break

    raw = {}
    for line in lines[data_start:]:
        line = line.strip()
        if not line or line.startswith('*') or line.startswith('-'):
            continue
        parts = line.split()
        if len(parts) < 10:
            continue
        try:
            eid = int(parts[0])
            s11 = parse_float(parts[3])
            s22 = parse_float(parts[5])
            s12 = parse_float(parts[9])
            if eid not in raw:
                raw[eid] = []
            raw[eid].append((s11, s22, s12))
        except (ValueError, IndexError):
            continue

    result = {}
    for eid, pts in raw.items():
        arr = np.array(pts)
        result[eid] = (arr[:, 0].mean(), arr[:, 1].mean(), arr[:, 2].mean())
    return result


# Tsai-Wu coefficients (verre/époxy)
_X_t, _X_c, _Y_t, _Y_c, _S = 1000., 600., 30., 120., 70.
_F1  =  1/_X_t - 1/_X_c
_F2  =  1/_Y_t - 1/_Y_c
_F11 =  1/(_X_t * _X_c)
_F22 =  1/(_Y_t * _Y_c)
_F66 =  1/(_S**2)
_F12 = -0.5 * np.sqrt(_F11 * _F22)


def tsai_wu_map(stress_map, theta_deg):
    """
    Calcule Tsai-Wu pour chaque élément à l'angle theta_deg.
    Retourne dict {elem_id: f}
    """
    t = np.radians(theta_deg)
    c, s = np.cos(t), np.sin(t)
    result = {}
    for eid, (s11, s22, s12) in stress_map.items():
        s1  =  c**2*s11 + s**2*s22 + 2*c*s*s12
        s2  =  s**2*s11 + c**2*s22 - 2*c*s*s12
        s12r = -c*s*s11 + c*s*s22 + (c**2 - s**2)*s12
        f = (_F1*s1 + _F2*s2 + _F11*s1**2 + _F22*s2**2
             + _F66*s12r**2 + 2*_F12*s1*s2)
        result[eid] = f
    return result


def plot_vm_comparison(nodes, elements, centroids,
                       rpt_mat1, rpt_mat2,
                       name_mat1, name_mat2,
                       view='YZ', y_min=None, y_max=None):
    """
    Deux cartes Von Mises côte à côte pour comparer deux matériaux.
    Même échelle de couleur pour que la comparaison soit honnête.
    """
    axes = {'XY': (0, 1, 'X (mm)', 'Y (mm)'),
            'XZ': (0, 2, 'X (mm)', 'Z (mm)'),
            'YZ': (1, 2, 'Y (mm)', 'Z (mm)')}
    ax1_idx, ax2_idx, xlabel, ylabel = axes[view]

    node_ids = list(nodes.keys())
    node_idx = {nid: i for i, nid in enumerate(node_ids)}
    all_x = np.array([nodes[nid][ax1_idx] for nid in node_ids])
    all_y = np.array([nodes[nid][ax2_idx] for nid in node_ids])

    # Éléments filtrés
    common = [eid for eid in elements
              if eid in centroids
              and (y_min is None or centroids[eid][1] >= y_min)
              and (y_max is None or centroids[eid][1] <= y_max)]

    triangles, tri_eids = [], []
    for eid in common:
        nids = elements[eid]
        if all(n in node_idx for n in nids):
            triangles.append([node_idx[n] for n in nids])
            tri_eids.append(eid)
    triangles = np.array(triangles)
    tri_eids  = np.array(tri_eids)
    triang    = mtri.Triangulation(all_x, all_y, triangles)

    def vm_from_stress(stress_map):
        return {eid: np.sqrt(s11**2 + s22**2 - s11*s22 + 3*s12**2)
                for eid, (s11, s22, s12) in stress_map.items()}

    def node_values(val_map):
        nv = np.zeros(len(node_ids))
        nc = np.zeros(len(node_ids))
        for eid in tri_eids:
            v = val_map.get(eid, 0)
            for nid in elements[eid]:
                if nid in node_idx:
                    nv[node_idx[nid]] += v
                    nc[node_idx[nid]] += 1
        nc[nc == 0] = 1
        return nv / nc

    vm1 = vm_from_stress(rpt_mat1)
    vm2 = vm_from_stress(rpt_mat2)

    # Même échelle pour les deux cartes
    all_vm = list(vm1.values()) + list(vm2.values())
    vmin = np.percentile(all_vm, 5)
    vmax = np.percentile(all_vm, 98)
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    nv1 = node_values(vm1)
    nv2 = node_values(vm2)

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(18, 7))

    for ax, nv, name in [(ax_l, nv1, name_mat1), (ax_r, nv2, name_mat2)]:
        tcf = ax.tripcolor(triang, nv, cmap='jet', norm=norm,
                           shading='gouraud', alpha=0.9)
        ax.triplot(triang, color='k', linewidth=0.08, alpha=0.15)
        ax.set_title(name, fontsize=13, fontweight='bold')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_aspect('equal')

    # Colorbar commune
    fig.subplots_adjust(right=0.88)
    cbar_ax = fig.add_axes([0.91, 0.15, 0.02, 0.7])
    sm = plt.cm.ScalarMappable(cmap='jet', norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Von Mises (MPa)', fontsize=11)

    # Annotation max sur chaque carte
    for ax, vm_d in [(ax_l, vm1), (ax_r, vm2)]:
        max_eid = max((e for e in vm_d if e in set(tri_eids)), key=lambda e: vm_d[e])
        mx = centroids[max_eid][ax1_idx]
        my = centroids[max_eid][ax2_idx]
        ax.annotate(f'Max\n{vm_d[max_eid]:.1f} MPa',
                    xy=(mx, my), xytext=(mx+20, my+20),
                    fontsize=8, color='white',
                    arrowprops=dict(arrowstyle='->', color='white', lw=1.2),
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))

    plt.suptitle(f'Comparaison Von Mises — {name_mat1} vs {name_mat2}',
                 fontsize=14, fontweight='bold')

    os.makedirs('img', exist_ok=True)
    out = 'img/comparaison_von_mises.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Graphique sauvegardé : {out}")


    """
    Deux cartes côte à côte :
      - Gauche  : Von Mises avec le pire matériau
      - Droite  : Tsai-Wu avec le meilleur matériau à θ_opt
    """
    axes = {'XY': (0, 1, 'X (mm)', 'Y (mm)'),
            'XZ': (0, 2, 'X (mm)', 'Z (mm)'),
            'YZ': (1, 2, 'Y (mm)', 'Z (mm)')}
    ax1_idx, ax2_idx, xlabel, ylabel = axes[view]

    # Coordonnées nœuds projetées
    node_ids = list(nodes.keys())
    node_idx = {nid: i for i, nid in enumerate(node_ids)}
    all_x = np.array([nodes[nid][ax1_idx] for nid in node_ids])
    all_y = np.array([nodes[nid][ax2_idx] for nid in node_ids])

    # Éléments filtrés
    common = [eid for eid in elements
              if eid in centroids
              and (y_min is None or centroids[eid][1] >= y_min)
              and (y_max is None or centroids[eid][1] <= y_max)]

    # Triangulation
    triangles, tri_eids = [], []
    for eid in common:
        nids = elements[eid]
        if all(n in node_idx for n in nids):
            triangles.append([node_idx[n] for n in nids])
            tri_eids.append(eid)
    triangles = np.array(triangles)
    tri_eids  = np.array(tri_eids)
    triang    = mtri.Triangulation(all_x, all_y, triangles)



def plot_mesh_2d(nodes, elements, centroids, vm_map, n_critical=15,
                 view='XY', title='Carte Von Mises — Hayon',
                 y_min=None, y_max=None):
    """
    Trace la carte 2D du maillage colorié par Von Mises.
    
    view : 'XY' vue de face, 'XZ' vue de dessus, 'YZ' vue de côté
    """

    # Axes selon la vue choisie
    axes = {'XY': (0, 1, 'X (mm)', 'Y (mm)'),
            'XZ': (0, 2, 'X (mm)', 'Z (mm)'),
            'YZ': (1, 2, 'Y (mm)', 'Z (mm)')}
    ax1_idx, ax2_idx, xlabel, ylabel = axes[view]

    # Coordonnées projetées des nœuds
    node_ids = list(nodes.keys())
    node_pos = {nid: (nodes[nid][ax1_idx], nodes[nid][ax2_idx])
                for nid in node_ids}

    # Identifier les éléments communs inp/rpt
    common_elems = [eid for eid in elements if eid in vm_map and eid in centroids]
    print(f"  Éléments communs inp/rpt : {len(common_elems)}")

    # Filtre spatial Y — exclure les bords
    if y_min is not None or y_max is not None:
        before = len(common_elems)
        common_elems = [
            eid for eid in common_elems
            if (y_min is None or centroids[eid][1] >= y_min)
            and (y_max is None or centroids[eid][1] <= y_max)
        ]
        print(f"  Filtre Y [{y_min}, {y_max}] mm : {before} → {len(common_elems)} éléments")

    # Trouver les N_CRITICAL éléments les plus chargés
    vm_sorted = sorted(common_elems, key=lambda e: vm_map[e], reverse=True)
    critical_elems = set(vm_sorted[:n_critical])

    # --- Construire la triangulation ---
    # Indices dans le tableau de nœuds
    node_idx = {nid: i for i, nid in enumerate(node_ids)}
    all_x = np.array([node_pos[nid][0] for nid in node_ids])
    all_y = np.array([node_pos[nid][1] for nid in node_ids])

    triangles = []
    tri_elem_ids = []
    for eid in common_elems:
        nids = elements[eid]
        if all(n in node_idx for n in nids):
            triangles.append([node_idx[n] for n in nids])
            tri_elem_ids.append(eid)

    triangles    = np.array(triangles)
    tri_elem_ids = np.array(tri_elem_ids)

    if len(triangles) == 0:
        print("[ERREUR] Aucun triangle à tracer — vérifier les IDs d'éléments")
        return

    # Valeur VM par triangle
    vm_vals = np.array([vm_map[eid] for eid in tri_elem_ids])

    # --- Figure ---
    fig, ax = plt.subplots(figsize=(14, 8))

    # Colormap
    norm = mcolors.Normalize(vmin=np.percentile(vm_vals, 5),
                             vmax=np.percentile(vm_vals, 98))
    cmap = plt.cm.jet

    # Tracer chaque triangle colorié
    triang = mtri.Triangulation(all_x, all_y, triangles)

    # Interpolation de la couleur sur les nœuds (moyenne des triangles adjacents)
    node_vm = np.zeros(len(node_ids))
    node_count = np.zeros(len(node_ids))
    for i, eid in enumerate(tri_elem_ids):
        for nid in elements[eid]:
            if nid in node_idx:
                idx = node_idx[nid]
                node_vm[idx]    += vm_vals[i]
                node_count[idx] += 1
    node_count[node_count == 0] = 1
    node_vm /= node_count

    tcf = ax.tripcolor(triang, node_vm, cmap=cmap, norm=norm,
                       shading='gouraud', alpha=0.9)

    # Contours du maillage (légers)
    ax.triplot(triang, color='k', linewidth=0.08, alpha=0.2)

    # Colorbar
    cbar = fig.colorbar(tcf, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label('Von Mises (MPa)', fontsize=11)

    # Marquer les mailles critiques
    crit_x = [centroids[eid][ax1_idx] for eid in critical_elems]
    crit_y = [centroids[eid][ax2_idx] for eid in critical_elems]
    crit_vm = [vm_map[eid] for eid in critical_elems]

    ax.scatter(crit_x, crit_y, c='white', s=120, marker='*',
               edgecolors='black', linewidths=0.8, zorder=5,
               label=f'Mailles critiques (top {n_critical})')

    # Annoter la maille la plus chargée
    max_eid = max(critical_elems, key=lambda e: vm_map[e])
    max_x = centroids[max_eid][ax1_idx]
    max_y = centroids[max_eid][ax2_idx]
    ax.annotate(f'Max VM\n{vm_map[max_eid]:.1f} MPa',
                xy=(max_x, max_y),
                xytext=(max_x + 20, max_y + 20),
                fontsize=8, color='white',
                arrowprops=dict(arrowstyle='->', color='white', lw=1.2),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='black',
                          alpha=0.7))

    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.set_aspect('equal')
    ax.grid(False)

    plt.tight_layout()
    os.makedirs('img', exist_ok=True)
    out = f'img/mesh_von_mises_{view}.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Graphique sauvegardé : {out}")

    # Résumé des mailles critiques
    print(f"\n  Top {n_critical} mailles critiques :")
    print(f"  {'Element':>10}  {'VM (MPa)':>10}  {'x':>10}  {'y':>10}  {'z':>10}")
    for eid in sorted(critical_elems, key=lambda e: vm_map[e], reverse=True):
        cx, cy, cz = centroids[eid]
        print(f"  {eid:>10}  {vm_map[eid]:>10.3f}  {cx:>10.1f}  {cy:>10.1f}  {cz:>10.1f}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":

    print("=" * 55)
    print("VISUALISATION MAILLAGE — Projet ENSAM 2026")
    print("=" * 55)

    print(f"\nParsing {INP_FILE} ...")
    nodes, elements = parse_inp(INP_FILE)
    centroids = compute_centroids(nodes, elements)
    print(f"  Barycentres calculés : {len(centroids)} éléments")

    print(f"\nParsing {RPT_FILE} ...")
    vm_map = parse_rpt_vm(RPT_FILE)

    print(f"\nGénération de la carte Von Mises (vue {VIEW}) ...")
    plot_mesh_2d(
        nodes, elements, centroids, vm_map,
        n_critical=N_CRITICAL,
        view=VIEW,
        title=f'Carte Von Mises — Hayon (vue {VIEW})\nMailles critiques marquées (★)',
        y_min=Y_MIN, y_max=Y_MAX,
    )

    # ------------------------------------------------------------------
    # Comparaison Von Mises : PP (pire) vs Verre + époxy (meilleur)
    # ------------------------------------------------------------------
    print(f"\nGénération comparaison Von Mises PP vs Verre + époxy ...")

    rpt_pp    = parse_rpt_full('tensor/tensor_PP.rpt')
    rpt_epoxy = parse_rpt_full('tensor/tensor_vitre_epoxy.rpt')

    plot_vm_comparison(
        nodes, elements, centroids,
        rpt_pp, rpt_epoxy,
        'PP (pire matériau)', 'Verre + époxy (matériau optimal)',
        view=VIEW, y_min=Y_MIN, y_max=Y_MAX,
    )

    print("\nTerminé.")

