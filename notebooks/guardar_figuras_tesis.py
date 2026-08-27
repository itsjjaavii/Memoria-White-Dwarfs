# -*- coding: utf-8 -*-
"""
Genera y guarda las figuras de la seccion 6.9 de la tesis con los nombres exactos
que espera ch6.tex.

Uso desde el notebook exp_wda, despues de haber ejecutado las celdas que definen
las funciones de graficado y que cargan repr_wda_32, X_wda_spec y W_wda:

    %run guardar_figuras_tesis.py

    # espectros de Spectral Clustering
    et = etiquetas_eigengap()
    generar_espectros(et)

    # espectros de DBSCAN
    etiquetas_dbscan = {
        "flatten": labels_dbs_flatten,
        "dense":   labels_dbs_dense,
        "dense_1": labels_dbs_dense1,
        "dense_2": labels_dbs_dense2,
    }
    generar_espectros_dbscan(etiquetas_dbscan)

    revisar_faltantes()
"""

import contextlib
from pathlib import Path

import matplotlib.pyplot as plt

# ---------------------------------------------------------------- configuracion

FIGS = Path(r"C:\Users\javip\Downloads\Formato_ELO_Tesis (1)\Formato_ELO_Tesis\ch6\figs")

DPI = 200

# capa -> valores de k evaluados (deben coincidir con ch6.tex)
COMBOS = {
    "flatten": [2, 3, 4],
    "dense":   [3, 4, 5],
    "dense_1": [2, 3, 4],
    "dense_2": [3, 8],
}

# capa -> sufijo usado en los nombres de archivo
SLUG = {
    "flatten": "flatten",
    "dense":   "dense",
    "dense_1": "dense1",
    "dense_2": "dense2",
}

# capa -> k seleccionado por el eigengap del Laplaciano.
# Es la unica particion para la que se reportan espectros sintetizados en la tesis.
K_EIGENGAP = {
    "flatten": 3,
    "dense":   5,
    "dense_1": 2,
    "dense_2": 8,
}

# capa -> transiciones de k para las matrices de migracion
PARES = {
    "flatten": [(2, 3), (3, 4), (2, 4)],
    "dense":   [(3, 4), (4, 5), (3, 5)],
    "dense_1": [(2, 3), (3, 4), (2, 4)],
    # dense_2 no lleva matriz de migracion
}


# ------------------------------------------------- acceso al espacio del notebook

def _nb(nombre):
    """
    Busca un nombre en el espacio de trabajo del notebook.

    Hace falta porque `%run archivo.py` ejecuta el archivo en un espacio de
    nombres propio: las funciones definidas aqui no ven por si solas las
    variables (repr_wda_32, X_wda_spec, ...) ni las funciones de graficado
    (cluster_plot_Spect, plot_spectra_spectral, ...) que viven en el cuaderno.
    """
    try:
        from IPython import get_ipython
        ip = get_ipython()
        if ip is not None and nombre in ip.user_ns:
            return ip.user_ns[nombre]
    except Exception:                                  # noqa: BLE001
        pass
    if nombre in globals():
        return globals()[nombre]
    raise NameError(
        f"No se encontro '{nombre}' en el notebook. "
        f"Ejecuta antes la celda que lo define y vuelve a intentar."
    )


# ---------------------------------------------------------------- utilidad base

@contextlib.contextmanager
def guardar_figuras(nombres, dpi=DPI, carpeta=FIGS):
    """
    Intercepta plt.show() y guarda cada figura generada dentro del bloque,
    usando los nombres entregados en orden.

        with guardar_figuras(["mi_figura.png"]):
            alguna_funcion_que_grafica(...)
    """
    carpeta.mkdir(parents=True, exist_ok=True)
    nombres = list(nombres)
    estado = {"i": 0}
    show_original = plt.show

    def show_interceptado(*args, **kwargs):
        fig = plt.gcf()
        i = estado["i"]
        if i < len(nombres):
            fig.savefig(carpeta / nombres[i], dpi=dpi, bbox_inches="tight")
            print(f"    guardada  {nombres[i]}")
        else:
            print(f"    [figura {i + 1} sin nombre asignado: no se guardo]")
        estado["i"] += 1
        plt.close(fig)

    plt.show = show_interceptado
    try:
        yield estado
    finally:
        plt.show = show_original
        if estado["i"] < len(nombres):
            print(f"    aviso: sobraron {len(nombres) - estado['i']} nombres sin usar")


# ---------------------------------------------------------------- generadores

def etiquetas_eigengap(repr_32=None):
    """
    Calcula solo las etiquetas que hacen falta para los espectros de la tesis
    (el k del eigengap en cada capa), sin graficar nada.
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import SpectralClustering

    if repr_32 is None:
        repr_32 = _nb("repr_wda_32")

    etiquetas = {}
    for capa, k in K_EIGENGAP.items():
        Xs = StandardScaler().fit_transform(repr_32[capa])
        lab = SpectralClustering(n_clusters=k, random_state=42,
                                 affinity="nearest_neighbors").fit_predict(Xs)
        etiquetas[capa] = {k: lab}
        print(f"  {capa}: k={k} listo")
    return etiquetas


def generar_clustering(repr_32=None, class_name="WDA"):
    """
    Ejecuta Spectral Clustering para cada capa y cada k, guarda las proyecciones
    PCA y t-SNE, y devuelve las etiquetas por (capa, k).
    """
    if repr_32 is None:
        repr_32 = _nb("repr_wda_32")
    cluster_plot_Spect = _nb("cluster_plot_Spect")

    etiquetas = {}
    for capa, ks in COMBOS.items():
        etiquetas[capa] = {}
        for k in ks:
            print(f"  {capa}  k={k}")
            res = None
            for proj in ("pca", "tsne"):
                nombre = f"wda_{SLUG[capa]}_spectral_k{k}_{proj}.png"
                with guardar_figuras([nombre]):
                    res = cluster_plot_Spect(
                        repr_32[capa], class_name, capa,
                        spectral_k=k, vis_method=proj,
                    )
            etiquetas[capa][k] = res["spectral"]
    return etiquetas


def generar_migracion(etiquetas, class_name="WDA"):
    """Guarda una matriz de migracion por cada transicion definida en PARES."""
    plot_migration_matrix = _nb("plot_migration_matrix")

    for capa, pares in PARES.items():
        for a, b in pares:
            if a not in etiquetas.get(capa, {}) or b not in etiquetas.get(capa, {}):
                print(f"  {capa}: faltan etiquetas para k={a} o k={b}, se omite")
                continue
            nombre = f"wda_{SLUG[capa]}_migracion_k{a}k{b}.png"
            print(f"  {capa}  k={a} -> k={b}")
            with guardar_figuras([nombre]):
                plot_migration_matrix(
                    etiquetas[capa][a], etiquetas[capa][b],
                    ka=a, kb=b,
                    title=f"Migracion k={a} -> k={b}  ({capa}, {class_name})",
                )


def _nombres_espectros(slug, k, n_grupos, metodo="spectral"):
    """
    Nombres en el orden exacto en que _plot_cluster_spectra_base emite figuras:

        1 .. n          promedio de cada grupo por separado
        n+1             promedio total de la clase
        n+2 .. 2n+1     diferencia de cada grupo respecto del promedio total
        2n+2            todos los promedios superpuestos (comparacion)
    """
    if metodo == "spectral":
        base = f"wda_{slug}_espectro_k{k}"
        comp = f"wda_{slug}_espectros_k{k}_comparacion.png"
        sufijos = [f"c{i}" for i in range(n_grupos)]
    else:
        base = f"wda_{slug}_espectro_dbscan"
        comp = f"wda_{slug}_espectros_dbscan_comparacion.png"
        # DBSCAN: np.unique ordena el ruido (-1) primero, luego 0, 1, 2, ...
        sufijos = ["ruido"] + [f"c{i}" for i in range(n_grupos - 1)]

    nombres = [f"{base}_{suf}.png" for suf in sufijos]
    nombres.append(f"{base}_promedio_clase.png")
    nombres += [f"{base}_dif_{suf}.png" for suf in sufijos]
    nombres.append(comp)
    return nombres


def generar_espectros(etiquetas, class_name="WDA", spectra=None, wavelength=None):
    """
    Guarda los espectros sintetizados de la particion que el eigengap selecciona
    en cada capa. Los archivos quedan con el nombre definitivo que espera
    ch6.tex: no hace falta renombrar nada.
    """
    import numpy as np

    plot_spectra_spectral = _nb("plot_spectra_spectral")
    if spectra is None:
        spectra = _nb("X_wda_spec")
    if wavelength is None:
        wavelength = _nb("W_wda")[0]

    for capa, k_ref in K_EIGENGAP.items():
        if k_ref not in etiquetas.get(capa, {}):
            print(f"  {capa}: sin etiquetas para k={k_ref}, se omite")
            continue
        lab = etiquetas[capa][k_ref]
        n_grupos = len(np.unique(lab))
        print(f"  {capa}  espectros spectral k={k_ref}  ({n_grupos} grupos)")
        with guardar_figuras(_nombres_espectros(SLUG[capa], k_ref, n_grupos)):
            plot_spectra_spectral(spectra, lab, wavelength, class_name, capa)


def generar_espectros_dbscan(etiquetas_dbscan, class_name="WDA",
                             spectra=None, wavelength=None):
    """
    Guarda los espectros de la particion de DBSCAN por capa.

    etiquetas_dbscan es un dict {capa: labels}, por ejemplo:

        {"flatten": labels_dbs_flatten, "dense": labels_dbs_dense,
         "dense_1": labels_dbs_dense1, "dense_2": labels_dbs_dense2}
    """
    import numpy as np

    plot_spectra_dbscan = _nb("plot_spectra_dbscan")
    if spectra is None:
        spectra = _nb("X_wda_spec")
    if wavelength is None:
        wavelength = _nb("W_wda")[0]

    for capa, lab in etiquetas_dbscan.items():
        n_grupos = len(np.unique(lab))       # incluye el grupo de ruido (-1)
        print(f"  {capa}  espectros dbscan  ({n_grupos} grupos, ruido incluido)")
        with guardar_figuras(_nombres_espectros(SLUG[capa], None, n_grupos,
                                                metodo="dbscan")):
            plot_spectra_dbscan(spectra, lab, wavelength, class_name, capa)


# ------------------------------------------------- guardado desde los widgets

_ESTADO_WIDGET = {"activo": False, "show": None, "prefijo": None, "n": 0}


def activar_guardado(prefijo, carpeta=FIGS, dpi=DPI):
    """
    Guarda toda figura que se dibuje desde este momento, numerandola.

    Sirve para los exploradores interactivos, donde no se puede usar el bloque
    `with guardar_figuras`: esas funciones dibujan dentro de un callback que se
    ejecuta despues, al mover un control.

        activar_guardado("prueba_flatten")
        spectra_viewer(X_wda_spec, W_wda[0], class_name="WDA", suffix="wda")
        # ...mueve los controles...
        desactivar_guardado()
    """
    if _ESTADO_WIDGET["activo"]:
        print("El guardado ya estaba activo. Usa desactivar_guardado() primero.")
        return

    carpeta.mkdir(parents=True, exist_ok=True)
    _ESTADO_WIDGET.update(activo=True, show=plt.show, prefijo=prefijo, n=0)
    show_original = _ESTADO_WIDGET["show"]

    def show_y_guardar(*args, **kwargs):
        _ESTADO_WIDGET["n"] += 1
        nombre = f"{prefijo}_{_ESTADO_WIDGET['n']:02d}.png"
        try:
            plt.gcf().savefig(carpeta / nombre, dpi=dpi, bbox_inches="tight")
        except Exception as exc:                       # noqa: BLE001
            print(f"    no se pudo guardar {nombre}: {exc}")
        return show_original(*args, **kwargs)          # se sigue mostrando

    plt.show = show_y_guardar
    print(f"Guardado activo. Prefijo: {prefijo}  |  Carpeta: {carpeta}")


def desactivar_guardado():
    """Restaura plt.show y reporta cuantas figuras se guardaron."""
    if not _ESTADO_WIDGET["activo"]:
        print("El guardado no estaba activo.")
        return
    plt.show = _ESTADO_WIDGET["show"]
    print(f"Guardado desactivado. Figuras guardadas: {_ESTADO_WIDGET['n']} "
          f"(prefijo {_ESTADO_WIDGET['prefijo']})")
    _ESTADO_WIDGET.update(activo=False, show=None, prefijo=None, n=0)


# ---------------------------------------------------------------- diagnostico

def revisar_faltantes():
    """Lista los archivos que ch6.tex espera y que aun no existen."""
    esperados = []
    for capa, ks in COMBOS.items():
        for k in ks:
            for proj in ("pca", "tsne"):
                esperados.append(f"wda_{SLUG[capa]}_spectral_k{k}_{proj}.png")
        k_e = K_EIGENGAP[capa]
        esperados += [f"wda_{SLUG[capa]}_espectro_k{k_e}_c{i}.png" for i in range(k_e)]
        esperados.append(f"wda_{SLUG[capa]}_espectros_k{k_e}_comparacion.png")
        esperados.append(f"wda_{SLUG[capa]}_espectros_dbscan_comparacion.png")
    for capa, pares in PARES.items():
        for a, b in pares:
            esperados.append(f"wda_{SLUG[capa]}_migracion_k{a}k{b}.png")

    faltan = [n for n in esperados if not (FIGS / n).exists()]
    print(f"esperados: {len(esperados)}  |  faltan: {len(faltan)}")
    for n in faltan:
        print("  ", n)
    return faltan


def revisar_entorno():
    """
    Comprueba que esten disponibles los objetos que necesitan los generadores.
    Util para diagnosticar antes de correr nada.
    """
    requeridos = [
        ("repr_wda_32", "datos"), ("X_wda_spec", "datos"), ("W_wda", "datos"),
        ("cluster_plot_Spect", "funcion"), ("plot_migration_matrix", "funcion"),
        ("plot_spectra_spectral", "funcion"), ("plot_spectra_dbscan", "funcion"),
        ("labels_dbs_flatten", "etiquetas DBSCAN"),
        ("labels_dbs_dense", "etiquetas DBSCAN"),
        ("labels_dbs_dense1", "etiquetas DBSCAN"),
        ("labels_dbs_dense2", "etiquetas DBSCAN"),
    ]
    faltan = []
    for nombre, tipo in requeridos:
        try:
            _nb(nombre)
            print(f"  OK     {nombre}  ({tipo})")
        except NameError:
            print(f"  FALTA  {nombre}  ({tipo})")
            faltan.append(nombre)
    if faltan:
        print("\nEjecuta las celdas que definen lo que falta antes de continuar.")
    return faltan
