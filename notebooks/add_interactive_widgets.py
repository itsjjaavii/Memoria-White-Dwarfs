"""
Agrega a exp_wd.ipynb, exp_wda.ipynb y exp_wdb.ipynb:
  - Explorador DBSCAN (tSNE+PCA con cache dbscan_store)
  - Explorador de espectros por cluster (Spectral + DBSCAN, con toggles)

6 celdas nuevas al final de cada notebook.
"""
import json, re

NOTEBOOKS = {
    "exp_wd.ipynb":  {"cls": "WD",  "suffix": "wd",  "repr": "repr_wd_32",
                      "spec": "X_wd_spec",  "wav": "W_wd[0]"},
    "exp_wda.ipynb": {"cls": "WDA", "suffix": "wda", "repr": "repr_wda_32",
                      "spec": "X_wda_spec", "wav": "W_wda[0]"},
    "exp_wdb.ipynb": {"cls": "WDB", "suffix": "wdb", "repr": "repr_wdb_32",
                      "spec": "X_wdb_spec", "wav": "W_wdb[0]"},
}

# ── helpers ────────────────────────────────────────────────────────────────────
def md(src):  return {"cell_type": "markdown", "metadata": {}, "source": [src]}
def code(src):
    return {"cell_type": "code", "execution_count": None,
            "metadata": {}, "outputs": [], "source": [src]}

# ══════════════════════════════════════════════════════════════════════════════
# CELDA 1 – cabecera DBSCAN
# ══════════════════════════════════════════════════════════════════════════════
MD_DBSCAN_HEADER = """\
## Explorador interactivo — DBSCAN (tSNE + PCA)

Selecciona la **capa**, **eps** y **min\\_samples** en los tres dropdowns.

- Si el experimento `(capa, eps, min_samples)` ya se corrió, se muestra \
desde caché (sin recalcular el tSNE).
- Si no existe, se calcula y guarda en:
  - `dbscan_store[(capa, eps, min_samples)]`
  - `cluster_labels_all[class_name][capa]["dbscan"]`
  - Variable global `labels_dbs_<capa>_e<eps>_ms<min_samples>_<sufijo>`
- El checkbox **Forzar recálculo** ignora el caché y vuelve a correr."""

# ══════════════════════════════════════════════════════════════════════════════
# CELDA 2 – funciones DBSCAN explorer
# ══════════════════════════════════════════════════════════════════════════════
CODE_DBSCAN_FUNCS = '''\
# ============================================================
# Explorador interactivo de DBSCAN (tSNE + PCA)
# ============================================================

dbscan_store = {}   # cache: (layer, eps, min_samples) -> {labels, emb_tsne, emb_pca, metrics, dbcv}


def _run_dbscan_cached(features_32, eps, min_samples):
    """Corre DBSCAN + embeddings tSNE y PCA y los cachea."""
    Xs       = StandardScaler().fit_transform(features_32)
    emb_tsne = TSNE(n_components=2, random_state=42,
                    perplexity=30, init=\'pca\').fit_transform(Xs)
    emb_pca  = PCA(n_components=2, random_state=42).fit_transform(Xs)
    labels   = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(Xs)
    metrics  = compute_cluster_metrics(Xs, labels)
    n_pts    = int(np.sum(labels != -1))
    dbcv     = dbcv_score(Xs, labels) if 0 < n_pts <= 5000 else None
    if n_pts > 5000:
        print(f\'  [DBCV omitido: {n_pts} puntos no-ruido > 5000]\')
    return {"labels": labels, "emb_tsne": emb_tsne,
            "emb_pca": emb_pca, "metrics": metrics, "dbcv": dbcv}


def _plot_dbscan_views(data, class_name, layer_name, eps, min_samples):
    """Dibuja scatter tSNE y PCA a partir de datos cacheados."""
    labels = data["labels"]
    views  = [("tsne", data["emb_tsne"]), ("pca", data["emb_pca"])]
    for vis_name, emb in views:
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.scatter(emb[:, 0], emb[:, 1], c=labels,
                   s=30, alpha=0.4, cmap=\'viridis\')
        ax.set_title(
            f"{class_name} \\u2014 {layer_name} \\u2014 DBSCAN "
            f"(eps={eps}, ms={min_samples}, vis={vis_name}, n={len(labels)})",
            fontsize=14)
        _add_cluster_legend(ax, labels)
        ax.set_xlabel("Comp. 1"); ax.set_ylabel("Comp. 2")
        plt.tight_layout(); plt.show()
    _print_metrics_box(class_name, layer_name, "dbscan",
                       f"eps={eps} ms={min_samples}",
                       data["metrics"], dbcv=data["dbcv"])


def dbscan_explorer(features_dict, class_name, suffix,
                    store=None,
                    layers_avail=("flatten", "dense", "dense_1", "dense_2"),
                    eps_values=None,
                    ms_values=None):
    """
    Explorador con tres dropdowns (capa, eps, min_samples) para DBSCAN.

    Cachea en dbscan_store[(capa, eps, min_samples)] y guarda:
      * cluster_labels_all[class_name][capa]["dbscan"]
      * variable global  labels_dbs_<capa>_e<eps>_ms<min_samples>_<suffix>
        (el punto del eps se reemplaza por \'p\', ej. 1.5 -> \'1p5\')
    """
    if store is None:
        store = dbscan_store
    if eps_values is None:
        eps_values = [0.3, 0.5, 0.7, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0]
    if ms_values is None:
        ms_values = [3, 5, 8, 10, 12, 15, 20, 25, 30]

    def _show(capa, eps, min_samples, recalcular):
        key    = (capa, eps, min_samples)
        hechos = sorted(f"{l}/eps={e}/ms={m}" for (l, e, m) in store)
        print("Experimentos guardados:",
              ", ".join(hechos) if hechos else "ninguno")
        print("-" * 60)

        if key not in store or recalcular:
            print(f"Calculando {class_name} | {capa} | eps={eps} ms={min_samples} ...")
            store[key] = _run_dbscan_cached(features_dict[capa], eps, min_samples)
            save_cluster_labels(cluster_labels_all, class_name,
                                capa, "dbscan", store[key]["labels"])
            eps_str = str(eps).replace(".", "p")
            varname = f"labels_dbs_{capa}_e{eps_str}_ms{min_samples}_{suffix}"
            globals()[varname] = store[key]["labels"]
            print(f"Guardado en store[{key}], en "
                  f"cluster_labels_all['{class_name}']['{capa}']['dbscan'] "
                  f"y en la variable global `{varname}`")
        else:
            print(f"Mostrando cacheado: {class_name} | {capa} | eps={eps} ms={min_samples}")
        print("-" * 60)
        _plot_dbscan_views(store[key], class_name, capa, eps, min_samples)

    interact(
        _show,
        capa=widgets.Dropdown(options=list(layers_avail), description="Capa:"),
        eps=widgets.Dropdown(options=eps_values, value=1.5, description="eps:"),
        min_samples=widgets.Dropdown(options=ms_values, value=10, description="min_samples:"),
        recalcular=widgets.Checkbox(value=False, description="Forzar recalculo"),
    )


def get_dbscan_labels(layer, eps, min_samples, store=None):
    """Devuelve etiquetas de un experimento guardado por el explorador DBSCAN."""
    if store is None:
        store = dbscan_store
    key = (layer, float(eps), int(min_samples))
    if key not in store:
        raise KeyError(
            f"No existe el experimento ({layer}, eps={eps}, ms={min_samples}). "
            f"Correlo primero en el explorador DBSCAN.")
    return store[key]["labels"]
'''

# ══════════════════════════════════════════════════════════════════════════════
# CELDA 3 – launch DBSCAN explorer  (se parametriza por notebook)
# ══════════════════════════════════════════════════════════════════════════════
CODE_DBSCAN_LAUNCH = "dbscan_explorer({repr}, class_name=\"{cls}\", suffix=\"{suffix}\")"

# ══════════════════════════════════════════════════════════════════════════════
# CELDA 4 – cabecera explorador de espectros
# ══════════════════════════════════════════════════════════════════════════════
MD_SPECTRA_HEADER = """\
## Explorador de espectros por cluster (Spectral + DBSCAN)

Visualiza los espectros promedio de cada cluster a partir de \
experimentos ya corridos en los exploradores anteriores.

**Controles:**
- **Método** / **Capa** / **k** (Spectral) o **eps + min\\_samples** (DBSCAN)
- **Media clase** — muestra/oculta el espectro promedio de la clase en el gráfico combinado
- **Balmer** — superpone las líneas de Balmer
- **±1σ individual** — banda de desviación estándar en los gráficos individuales

> Si el experimento seleccionado no existe aún, el widget muestra un \
aviso en lugar de fallar."""

# ══════════════════════════════════════════════════════════════════════════════
# CELDA 5 – definiciones spectra_viewer
# ══════════════════════════════════════════════════════════════════════════════
CODE_SPECTRA_FUNCS = '''\
# ============================================================
# Explorador de espectros por cluster (Spectral + DBSCAN)
# ============================================================

def _plot_cluster_spectra_interactive(
    spectra_2d, labels, wavelength, total_mean,
    class_name, layer_name, method_name, param_str,
    show_media=True, show_balmer=False, show_std=False,
    figsize=(11, 5), linewidth_mean=2.4
):
    """Dibuja graficos de espectro por cluster + grafico combinado."""
    labels     = np.asarray(labels)
    unique_cls = np.sort(np.unique(labels))
    n_clusters = len(unique_cls)
    cmap       = plt.cm.get_cmap("tab10", n_clusters)

    # --- graficos individuales por cluster ---
    for i, c in enumerate(unique_cls):
        mask   = labels == c
        g_spec = spectra_2d[mask]
        g_mean = np.mean(g_spec, axis=0)
        n      = int(mask.sum())
        color  = "gray" if c == -1 else cmap(i)
        lbl    = "Ruido (-1)" if c == -1 else f"Cluster {c}"

        fig, ax = plt.subplots(figsize=figsize)
        if show_std:
            std = np.std(g_spec, axis=0)
            ax.fill_between(wavelength, g_mean - std, g_mean + std,
                            alpha=0.18, color=color, label="\\u00b11\\u03c3")
        ax.plot(wavelength, g_mean, linewidth=linewidth_mean,
                color=color, label=f"{lbl} mean (n={n})")
        ax.set_xlabel("Wavelength [\\u00c5]")
        ax.set_ylabel("Normalized flux")
        ax.set_title(
            f"{class_name} \\u2014 {layer_name} \\u2014 "
            f"{method_name} ({param_str})\\n{lbl} (n={n})")
        if show_balmer:
            add_balmer_lines(ax)
        ax.legend()
        plt.tight_layout()
        plt.show()

    # --- grafico combinado ---
    fig, ax    = plt.subplots(figsize=figsize)
    all_curves = []
    for i, c in enumerate(unique_cls):
        mask   = labels == c
        g_spec = spectra_2d[mask]
        g_mean = np.mean(g_spec, axis=0)
        n      = int(mask.sum())
        color  = "gray" if c == -1 else cmap(i)
        leg    = "Ruido (-1)" if c == -1 else f"Cluster {c} (n={n})"
        if show_std:
            std = np.std(g_spec, axis=0)
            ax.fill_between(wavelength, g_mean - std, g_mean + std,
                            alpha=0.12, color=color)
        ax.plot(wavelength, g_mean, linewidth=2.2, color=color, label=leg)
        all_curves.append(g_mean)

    if show_media and len(all_curves) > 0:
        ax.plot(wavelength, total_mean, linewidth=2.4, linestyle="--",
                color="black", label=f"Media clase (n={len(labels)})")
        all_curves.append(total_mean)

    ax.set_xlabel("Wavelength [\\u00c5]")
    ax.set_ylabel("Normalized flux")
    ax.set_title(
        f"{class_name} \\u2014 {layer_name} \\u2014 "
        f"{method_name} ({param_str})\\nAll cluster means")

    if all_curves:
        stacked    = np.vstack(all_curves)
        p_lo, p_hi = np.percentile(stacked, [1, 99])
        margin     = 0.10 * (p_hi - p_lo + 1e-8)
        ax.set_ylim(p_lo - margin, p_hi + margin)

    if show_balmer:
        add_balmer_lines(ax)
    ax.legend()
    plt.tight_layout()
    plt.show()


def spectra_viewer(spectra_2d, wavelength, class_name, suffix,
                   spectral_store_ref=None, dbscan_store_ref=None):
    """
    Widget interactivo para visualizar espectros de clusters ya calculados.

    Metodos soportados: spectral (usa spectral_store) y dbscan (usa dbscan_store).
    Cuando se selecciona \'spectral\' aparece el dropdown de k;
    cuando se selecciona \'dbscan\' aparecen los dropdowns de eps y min_samples.

    Controles de visualizacion:
      - Media clase : muestra/oculta el espectro promedio en el grafico combinado
      - Balmer      : superpone lineas de Balmer en todos los graficos
      - +/-1sigma   : banda de desviacion estandar en los graficos individuales
    """
    if spectral_store_ref is None:
        spectral_store_ref = spectral_store
    if dbscan_store_ref is None:
        dbscan_store_ref = dbscan_store

    spectra_2d = np.asarray(spectra_2d)
    wavelength = np.asarray(wavelength)
    total_mean = np.mean(spectra_2d, axis=0)

    W = widgets.Layout

    metodo_dd = widgets.Dropdown(
        options=["spectral", "dbscan"], description="Metodo:",
        layout=W(width="190px"))
    capa_dd = widgets.Dropdown(
        options=["flatten", "dense", "dense_1", "dense_2"],
        description="Capa:", layout=W(width="200px"))
    k_dd = widgets.Dropdown(
        options=list(range(2, 13)), value=3,
        description="k:", layout=W(width="140px"))
    eps_dd = widgets.Dropdown(
        options=[0.3, 0.5, 0.7, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0],
        value=1.5, description="eps:", layout=W(width="150px"))
    ms_dd = widgets.Dropdown(
        options=[3, 5, 8, 10, 12, 15, 20, 25, 30],
        value=10, description="min_s:", layout=W(width="160px"))

    media_btn  = widgets.ToggleButton(
        value=True,  description="Media clase",
        button_style="info", layout=W(width="150px"))
    balmer_btn = widgets.ToggleButton(
        value=False, description="Balmer",
        button_style="", layout=W(width="110px"))
    std_btn    = widgets.ToggleButton(
        value=False, description="\\u00b11\\u03c3 individual",
        button_style="", layout=W(width="150px"))

    out = widgets.Output()

    def _update_visibility(change=None):
        is_sp = metodo_dd.value == "spectral"
        k_dd.layout.display   = ""     if is_sp else "none"
        eps_dd.layout.display = "none" if is_sp else ""
        ms_dd.layout.display  = "none" if is_sp else ""

    metodo_dd.observe(_update_visibility, names="value")
    _update_visibility()

    def _render(change=None):
        with out:
            out.clear_output(wait=True)
            metodo = metodo_dd.value
            capa   = capa_dd.value

            if metodo == "spectral":
                k   = int(k_dd.value)
                key = (capa, k)
                if key not in spectral_store_ref:
                    print(f"No hay experimento spectral ({capa}, k={k}).\\n"
                          "Correlo primero en el explorador Spectral Clustering.")
                    return
                labels    = spectral_store_ref[key]["labels"]
                param_str = f"k={k}"
            else:
                eps = float(eps_dd.value)
                ms  = int(ms_dd.value)
                key = (capa, eps, ms)
                if key not in dbscan_store_ref:
                    print(f"No hay experimento DBSCAN ({capa}, eps={eps}, ms={ms}).\\n"
                          "Correlo primero en el explorador DBSCAN.")
                    return
                labels    = dbscan_store_ref[key]["labels"]
                param_str = f"eps={eps} ms={ms}"

            _plot_cluster_spectra_interactive(
                spectra_2d, labels, wavelength, total_mean,
                class_name, capa, metodo, param_str,
                show_media  = media_btn.value,
                show_balmer = balmer_btn.value,
                show_std    = std_btn.value,
            )

    for w in [metodo_dd, capa_dd, k_dd, eps_dd, ms_dd,
              media_btn, balmer_btn, std_btn]:
        w.observe(_render, names="value")

    row1 = widgets.HBox([metodo_dd, capa_dd, k_dd, eps_dd, ms_dd])
    row2 = widgets.HBox([media_btn, balmer_btn, std_btn])
    display(widgets.VBox([row1, row2, out]))
    _render()
'''

# ══════════════════════════════════════════════════════════════════════════════
# CELDA 6 – launch spectra_viewer  (se parametriza por notebook)
# ══════════════════════════════════════════════════════════════════════════════
CODE_SPECTRA_LAUNCH = (
    "spectra_viewer({spec}, {wav}, class_name=\"{cls}\", suffix=\"{suffix}\")"
)

# ──────────────────────────────────────────────────────────────────────────────

def build_cells(cfg):
    cls    = cfg["cls"]
    suffix = cfg["suffix"]
    repr_  = cfg["repr"]
    spec   = cfg["spec"]
    wav    = cfg["wav"]

    return [
        md(MD_DBSCAN_HEADER),
        code(CODE_DBSCAN_FUNCS),
        code(CODE_DBSCAN_LAUNCH.format(repr=repr_, cls=cls, suffix=suffix)),
        md(MD_SPECTRA_HEADER),
        code(CODE_SPECTRA_FUNCS),
        code(CODE_SPECTRA_LAUNCH.format(spec=spec, wav=wav, cls=cls, suffix=suffix)),
    ]


import os

base = os.path.join(os.path.dirname(__file__))

for nb_name, cfg in NOTEBOOKS.items():
    path = os.path.join(base, nb_name)
    with open(path, encoding="utf-8") as f:
        nb = json.load(f)

    nb["cells"].extend(build_cells(cfg))

    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)

    print(f"{nb_name}: {len(nb['cells'])} celdas totales")

print("Listo.")
