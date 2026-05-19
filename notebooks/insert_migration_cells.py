"""
Script robusto para insertar celdas de análisis de migración de clústeres
en task3_v2.ipynb.

Estrategia: busca la ÚLTIMA celda de código que contenga 'flatten' y
'spectral_k' en su source (los experimentos k=2,3,4 de la sección flatten),
e inserta las dos nuevas celdas justo después de ella.

Si ya existen celdas con id 'migration_md_flatten' o 'migration_code_flatten',
no hace nada (idempotente).
"""

import json
import re

NOTEBOOK_PATH = (
    r"c:\Users\javip\OneDrive - Universidad Técnica Federico Santa María"
    r"\6to año\elo308\ML_NPF\notebooks\task3_v2.ipynb"
)

# ── Celda Markdown de explicación ─────────────────────────────────────────────
markdown_cell = {
    "cell_type": "markdown",
    "id": "migration_md_flatten",
    "metadata": {},
    "source": [
        "### Análisis de migración de clústeres — Capa Flatten, Spectral Clustering\n",
        "\n",
        "Se compara cómo los datos migran entre las configuraciones k=2, k=3 y k=4.  \n",
        "La **matriz de contingencia** cuenta cuántas muestras compartidas hay entre cada par de clústeres.  \n",
        "Las filas/columnas se reordenan con el **algoritmo húngaro** (`linear_sum_assignment`) para maximizar la diagonal y facilitar la lectura.  \n",
        "Comparaciones realizadas: k=2 → k=3, k=3 → k=4, y k=2 → k=4."
    ]
}

# ── Celda de código ───────────────────────────────────────────────────────────
code_cell = {
    "cell_type": "code",
    "execution_count": None,
    "id": "migration_code_flatten",
    "metadata": {},
    "outputs": [],
    "source": [
        "import numpy as np\n",
        "import matplotlib.pyplot as plt\n",
        "import seaborn as sns\n",
        "from sklearn.cluster import SpectralClustering\n",
        "from sklearn.metrics.cluster import contingency_matrix\n",
        "from scipy.optimize import linear_sum_assignment\n",
        "\n",
        "# ── Función para graficar la matriz de migración ──────────────────────────────\n",
        "def plot_migration_matrix(labels_a, labels_b, ka, kb, title=None, figsize=(6, 5)):\n",
        "    \"\"\"\n",
        "    Genera un heatmap de la matriz de contingencia entre dos asignaciones\n",
        "    de clústeres. Reordena filas/columnas con linear_sum_assignment para\n",
        "    maximizar la diagonal.\n",
        "\n",
        "    Parámetros\n",
        "    ----------\n",
        "    labels_a : array-like   etiquetas del clustering con k=ka\n",
        "    labels_b : array-like   etiquetas del clustering con k=kb\n",
        "    ka, kb   : int          valores de k usados\n",
        "    title    : str          título del gráfico (opcional)\n",
        "    \"\"\"\n",
        "    cm = contingency_matrix(labels_a, labels_b)\n",
        "    row_ind, col_ind = linear_sum_assignment(-cm)\n",
        "    cm_reordered = cm[row_ind, :][:, col_ind]\n",
        "\n",
        "    fig, ax = plt.subplots(figsize=figsize)\n",
        "    sns.heatmap(\n",
        "        cm_reordered,\n",
        "        annot=True,\n",
        "        fmt='d',\n",
        "        cmap='Blues',\n",
        "        xticklabels=[f'C{col_ind[j]}' for j in range(cm_reordered.shape[1])],\n",
        "        yticklabels=[f'C{row_ind[i]}' for i in range(cm_reordered.shape[0])],\n",
        "        ax=ax\n",
        "    )\n",
        "    ax.set_xlabel(f'Clústeres k={kb}', fontsize=11)\n",
        "    ax.set_ylabel(f'Clústeres k={ka}', fontsize=11)\n",
        "    _title = title if title else f'Migración k={ka} → k={kb}  (Flatten, WDA)'\n",
        "    ax.set_title(_title, fontsize=12, fontweight='bold')\n",
        "    plt.tight_layout()\n",
        "    plt.show()\n",
        "\n",
        "\n",
        "# ── Recomputar Spectral Clustering para k=2, 3, 4 sobre la capa flatten ──────\n",
        "X_flatten = repr_wda_32['flatten']   # representación 32-D de la capa flatten\n",
        "\n",
        "print('Ejecutando Spectral Clustering en flatten...')\n",
        "sc2 = SpectralClustering(n_clusters=2, affinity='nearest_neighbors',\n",
        "                          n_neighbors=10, assign_labels='kmeans',\n",
        "                          random_state=42, n_init=10)\n",
        "labels_flatten_k2 = sc2.fit_predict(X_flatten)\n",
        "\n",
        "sc3 = SpectralClustering(n_clusters=3, affinity='nearest_neighbors',\n",
        "                          n_neighbors=10, assign_labels='kmeans',\n",
        "                          random_state=42, n_init=10)\n",
        "labels_flatten_k3 = sc3.fit_predict(X_flatten)\n",
        "\n",
        "sc4 = SpectralClustering(n_clusters=4, affinity='nearest_neighbors',\n",
        "                          n_neighbors=10, assign_labels='kmeans',\n",
        "                          random_state=42, n_init=10)\n",
        "labels_flatten_k4 = sc4.fit_predict(X_flatten)\n",
        "\n",
        "print(f'Muestras por clúster — k=2: {np.bincount(labels_flatten_k2)}')\n",
        "print(f'Muestras por clúster — k=3: {np.bincount(labels_flatten_k3)}')\n",
        "print(f'Muestras por clúster — k=4: {np.bincount(labels_flatten_k4)}')\n",
        "\n",
        "# ── Visualizar matrices de migración ─────────────────────────────────────────\n",
        "plot_migration_matrix(labels_flatten_k2, labels_flatten_k3, ka=2, kb=3)\n",
        "plot_migration_matrix(labels_flatten_k3, labels_flatten_k4, ka=3, kb=4)\n",
        "plot_migration_matrix(labels_flatten_k2, labels_flatten_k4, ka=2, kb=4)\n"
    ]
}

# ── Cargar el notebook ────────────────────────────────────────────────────────
with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
    nb = json.load(f)

cells = nb["cells"]
print(f"Total de celdas en el notebook: {len(cells)}")

# ── Verificar idempotencia ────────────────────────────────────────────────────
already_inserted = any(
    c.get("id") in ("migration_md_flatten", "migration_code_flatten")
    for c in cells
)
if already_inserted:
    print("Las celdas de migración ya existen. No se realizaron cambios.")
    exit(0)

# ── Buscar la celda correcta donde insertar ───────────────────────────────────
# Buscamos la ÚLTIMA celda de código que mencione 'flatten' y 'spectral_k'
# (los experimentos k=2, k=3, k=4 de la sección flatten).
# Si no se encuentra ese patrón exacto, buscamos la última celda que mencione
# 'repr_wda_32["flatten"]' o "repr_wda_32['flatten']".

def source_as_str(cell):
    src = cell.get("source", [])
    if isinstance(src, list):
        return "".join(src)
    return src

# Estrategia 1: celda con 'flatten' + 'spectral_k'
insert_after = None
for i, cell in enumerate(cells):
    if cell.get("cell_type") != "code":
        continue
    src = source_as_str(cell)
    if ("flatten" in src) and ("spectral_k" in src):
        insert_after = i

# Estrategia 2: si no, celda que llame cluster_plot_Spect con flatten
if insert_after is None:
    for i, cell in enumerate(cells):
        if cell.get("cell_type") != "code":
            continue
        src = source_as_str(cell)
        if ("flatten" in src) and ("cluster_plot_Spect" in src):
            insert_after = i

# Estrategia 3: celda que use repr_wda_32 con flatten
if insert_after is None:
    for i, cell in enumerate(cells):
        if cell.get("cell_type") != "code":
            continue
        src = source_as_str(cell)
        if ('repr_wda_32' in src) and ('flatten' in src.lower()):
            insert_after = i

# Estrategia 4: fallback — buscar el markdown "Resultados capa flatten"
# y usar la última celda de código antes del siguiente markdown de sección
if insert_after is None:
    for i, cell in enumerate(cells):
        if cell.get("cell_type") != "markdown":
            continue
        src = source_as_str(cell).lower()
        if "flatten" in src and ("spectral" in src or "resultados" in src):
            # buscar la última celda de código hasta el próximo markdown de sección
            for j in range(i + 1, len(cells)):
                if cells[j].get("cell_type") == "code":
                    insert_after = j
                elif cells[j].get("cell_type") == "markdown":
                    sec_src = source_as_str(cells[j]).lower()
                    # si es un nuevo encabezado de sección, parar
                    if sec_src.startswith("#"):
                        break

# Mostrar diagnóstico
print("\n=== DIAGNÓSTICO ===")
print(f"Posición de inserción encontrada: {insert_after}")
if insert_after is not None:
    cell_src = source_as_str(cells[insert_after])
    print(f"Celda en posición {insert_after}:")
    print(f"  Tipo: {cells[insert_after].get('cell_type')}")
    print(f"  ID  : {cells[insert_after].get('id')}")
    print(f"  Src (primeros 200 chars): {repr(cell_src[:200])}")

# Buscar también por contexto: listar todas las celdas con 'flatten'
print("\n=== Celdas que contienen 'flatten' ===")
for i, cell in enumerate(cells):
    src = source_as_str(cell)
    if "flatten" in src.lower():
        print(f"  [{i}] tipo={cell.get('cell_type')} id={cell.get('id')} | {repr(src[:120])}")

if insert_after is None:
    print("\nERROR: No se pudo determinar la posición de inserción.")
    print("Revisa el diagnóstico de arriba e indica manualmente la posición.")
    exit(1)

# ── Insertar celdas ───────────────────────────────────────────────────────────
cells.insert(insert_after + 1, code_cell)
cells.insert(insert_after + 1, markdown_cell)

with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"\n✅ Celdas insertadas exitosamente.")
print(f"  markdown_cell en posición : {insert_after + 1}")
print(f"  code_cell     en posición : {insert_after + 2}")
