#!/usr/bin/env python3
""" astro_spots.py — Encuentra zonas oscuras accesibles en coche cerca de un origen.
Requisitos: pip install rasterio numpy shapely folium osmnx scikit-image matplotlib tqdm
Entrada: - GeoTIFF VIIRS VNL v2.* (average-masked o median-masked), p.ej. 2022/2024.
Uso típico:
# 1) Ver candidatos por píxel (recomendado) sin filtrar carreteras
python astro_spots.py --lat 40.4168 --lon -3.7038 --radius_km 40 \
--viirs_path VNL.tif --out_prefix madrid_viirs --dark_thr 300 \
--pixel_mode --verbose
# 2) Además filtrar por carreteras cercanas (OSMnx)
python astro_spots.py --lat 40.4168 --lon -3.7038 --radius_km 40 \
--viirs_path VNL.tif --out_prefix madrid_viirs --dark_thr 300 \
--pixel_mode --check_drive --drive_search_m 2000 --verbose
Salidas:
- <prefix>_candidates.csv (todos los candidatos)
- <prefix>_map_candidates.html (mapa con candidatos)
- <prefix>_drive.csv (solo accesibles; si procede)
- <prefix>_map_drive.html (mapa accesibles; si procede)
"""
import argparse
import math
import os
import csv
import concurrent.futures
from typing import List, Tuple
import numpy as np
import rasterio
from rasterio.transform import rowcol, xy
from rasterio.windows import from_bounds
from shapely.geometry import Point
import folium
from skimage.transform import resize
# OSMnx: sólo si se pide validar accesibilidad
try:
    import osmnx as ox
except Exception:
    ox = None
# tqdm opcional
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

# ------------------ utilidades geodésicas ------------------
def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0088
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = phi2 - phi1
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return R * (2 * math.atan2(math.sqrt(a), math.sqrt(1 - a)))

def deg_per_km_lat():
    return 1.0 / 110.574  # ~ km/deg lat

def deg_per_km_lon(lat):
    return 1.0 / (111.320 * math.cos(math.radians(lat)) + 1e-12)

# ------------------ muestreo en rejilla (modo clásico) ------------------
def generate_grid(center_lat, center_lon, radius_km, step_km):
    lat_step = step_km * deg_per_km_lat()
    lat_range = int(radius_km / step_km)
    for i in range(-lat_range, lat_range + 1):
        lat = center_lat + i * lat_step
        lon_step_i = step_km * deg_per_km_lon(lat)
        lon_range = int(radius_km / step_km)
        for j in range(-lon_range, lon_range + 1):
            lon = center_lon + j * lon_step_i
            if haversine_km(center_lat, center_lon, lat, lon) <= radius_km:
                yield (lat, lon)

def process_point(args_tuple):
    viirs_path, lat, lon, band, center_lat, center_lon, dark_thr = args_tuple
    # función aislada para ejecución en procesos
    import rasterio
    import numpy as np
    from rasterio.transform import rowcol
    import math

    def haversine_km_(lat1, lon1, lat2, lon2):
        R = 6371.0088
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        dphi = phi2 - phi1
        dlambda = math.radians(lon2 - lon1)
        a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
        return R * (2 * math.atan2(math.sqrt(a), math.sqrt(1 - a)))

    try:
        with rasterio.open(viirs_path) as raster:
            row, col = rowcol(raster.transform, lon, lat)
            if (0 <= row < raster.height) and (0 <= col < raster.width):
                val = raster.read(band)[row, col]
                if raster.nodata is not None and val == raster.nodata:
                    return None
                if np.isnan(val):
                    return None
                if val > dark_thr:
                    return None
                dist_km = haversine_km_(center_lat, center_lon, lat, lon)
                return (lat, lon, float(val), dist_km)
            else:
                return None
    except Exception:
        return None

# ------------------ NUEVO: muestreo por píxel del raster ------------------
def sample_pixels(raster, center_lat, center_lon, radius_km, band, dark_thr):
    """ Recorre los píxeles VIIRS dentro del radio y devuelve candidatos oscuros.
    Devuelve lista de tuplas: (lat, lon, radiancia, distancia_km) """
    results = []
    # bounds en grados
    delta_deg_lat = radius_km * deg_per_km_lat()
    delta_deg_lon = radius_km * deg_per_km_lon(center_lat)
    min_lat, max_lat = center_lat - delta_deg_lat, center_lat + delta_deg_lat
    min_lon, max_lon = center_lon - delta_deg_lon, center_lon + delta_deg_lon
    window = from_bounds(min_lon, min_lat, max_lon, max_lat, raster.transform)
    arr = raster.read(band, window=window)
    transform = raster.window_transform(window)
    for row in range(arr.shape[0]):
        for col in range(arr.shape[1]):
            val = arr[row, col]
            if raster.nodata is not None and val == raster.nodata:
                continue
            if np.isnan(val):
                continue
            if val > dark_thr:
                continue
            # centro del píxel (x=lon, y=lat)
            lon, lat = xy(transform, row, col)
            dist_km = haversine_km(center_lat, center_lon, lat, lon)
            if dist_km <= radius_km:
                results.append((lat, lon, float(val), dist_km))
    return results

# ------------------ otros helpers ------------------
def is_drivable_near(G, coord: Tuple[float, float], search_m=1200) -> bool:
    """Comprueba si hay red de carreteras 'drive' cerca del punto (lat, lon) usando grafo pre-cargado."""
    if G is None or ox is None:
        return False
    lat, lon = coord
    try:
        # Encuentra el edge más cercano y su distancia en metros (great-circle)
        _, dist = ox.distance.nearest_edges(G, lon, lat, return_dist=True)
        return dist <= search_m
    except Exception:
        return False  # Si falla (e.g., punto fuera del grafo), asume no accesible

def thin_by_distance(rows, min_sep_km):
    """Reduce puntos muy agrupados conservando primero los más oscuros y cercanos."""
    if min_sep_km <= 0:
        return rows
    selected = []
    for lat, lon, rad, dist in rows:
        ok = True
        for slat, slon, _, _ in selected:
            if haversine_km(lat, lon, slat, slon) < min_sep_km:
                ok = False
                break
        if ok:
            selected.append((lat, lon, rad, dist))
    return selected

def add_viirs_overlay(m, raster, args):
    """Añade como overlay el recorte del raster para situarse visualmente."""
    import matplotlib.pyplot as plt
    import tempfile
    # bounds de interés
    delta_deg = args.radius_km * deg_per_km_lat()
    min_lat = args.lat - delta_deg
    max_lat = args.lat + delta_deg
    delta_deg_lon = args.radius_km * deg_per_km_lon(args.lat)
    min_lon = args.lon - delta_deg_lon
    max_lon = args.lon + delta_deg_lon
    window = from_bounds(min_lon, min_lat, max_lon, max_lat, raster.transform)
    arr = raster.read(args.band, window=window)
    # visual: recorte al 99p y normalización
    arr_vis = np.clip(arr, np.nanmin(arr[arr > -np.inf]), np.nanpercentile(arr, 99))
    arr_vis = arr_vis / arr_vis.max() if np.nanmax(arr_vis) > 0 else arr_vis
    arr_vis_resized = resize(arr_vis, (600, 600), preserve_range=True, anti_aliasing=True)
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        plt.imsave(tmp.name, arr_vis_resized, cmap="inferno")
    img_path = tmp.name
    folium.raster_layers.ImageOverlay(
        image=img_path,
        bounds=[[min_lat, min_lon], [max_lat, max_lon]],
        opacity=0.5,
        interactive=True,
        cross_origin=False,
        name="Contaminación lumínica"
    ).add_to(m)

def build_map(rows, raster, args, title_marker="Origen"):
    m = folium.Map(location=[args.lat, args.lon], zoom_start=10, control_scale=True)
    folium.Marker(
        [args.lat, args.lon],
        tooltip=title_marker,
        icon=folium.Icon(icon="home"),
    ).add_to(m)
    # puntos (top_n)
    topN = rows[: args.top_n] if args.top_n > 0 else rows
    for i, (lat, lon, rad, dist_km) in enumerate(topN, start=1):
        folium.CircleMarker(
            [lat, lon],
            radius=6,
            tooltip=f"#{i} — {dist_km:.1f} km — {rad:.3f} nW/cm²/sr",
            fill=True,
        ).add_to(m)
    # overlay de VIIRS
    add_viirs_overlay(m, raster, args)
    folium.LayerControl().add_to(m)
    return m

# ------------------ MAIN ------------------
def main():
    parser = argparse.ArgumentParser(description="Buscar zonas oscuras accesibles en coche usando VIIRS + OSM.")
    parser.add_argument("--lat", type=float, required=True, help="Latitud del origen")
    parser.add_argument("--lon", type=float, required=True, help="Longitud del origen")
    parser.add_argument("--radius_km", type=float, default=60, help="Radio de búsqueda en km (por defecto 60)")
    parser.add_argument("--grid_km", type=float, default=2.0, help="Paso de la rejilla en km (modo rejilla)")
    parser.add_argument("--viirs_path", type=str, required=True, help="Ruta al GeoTIFF de VIIRS (average/median masked)")
    parser.add_argument("--band", type=int, default=1, help="Banda del GeoTIFF (por defecto 1)")
    parser.add_argument("--dark_thr", type=float, default=0.3, help="Umbral de oscuridad en nW/cm^2/sr (más bajo = más oscuro)")
    parser.add_argument("--max_candidates", type=int, default=300000, help="Límite de candidatos (modo rejilla)")
    parser.add_argument("--check_drive", action="store_true", help="Validar accesibilidad por coche con OSMnx")
    parser.add_argument("--drive_search_m", type=int, default=1500, help="Radio (m) para buscar carreteras en OSMnx")
    parser.add_argument("--min_sep_km", type=float, default=0.0, help="Separación mínima entre puntos (0 = sin filtrar)")
    parser.add_argument("--top_n", type=int, default=50, help="Cuántos puntos dibujar en los mapas (CSV guarda todos)")
    parser.add_argument("--pixel_mode", action="store_true", help="Usar muestreo por píxel VIIRS (recomendado)")
    parser.add_argument("--out_prefix", type=str, default="astro_spots", help="Prefijo de archivos de salida")
    parser.add_argument("--verbose", action="store_true", help="Modo detallado")
    parser.add_argument("--output_dir", type=str, default=".", help="Directorio de salida para mapas y CSV")
    parser.add_argument("--cache_graph", action="store_true", help="Guardar/cargar grafo OSMnx en disco para reutilizar")
    args = parser.parse_args()

    # Crear carpeta de salida si no existe
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Abrir raster VIIRS
    if not os.path.exists(args.viirs_path):
        raise FileNotFoundError(f"No existe el archivo VIIRS: {args.viirs_path}")
    raster = rasterio.open(args.viirs_path)
    if args.verbose:
        print("CRS del raster:", raster.crs)
        band1 = raster.read(args.band)
        print("Rango de valores:", np.nanmin(band1), np.nanmax(band1))

    # -------- 1) Muestreo de candidatos --------
    if args.pixel_mode:
        if args.verbose:
            print("[DEBUG] Modo = píxel VIIRS")
        rows = sample_pixels(raster, args.lat, args.lon, args.radius_km, args.band, args.dark_thr)
        if args.verbose:
            print(f"[DEBUG] Píxeles candidatos bajo umbral: {len(rows)}")
    else:
        if args.verbose:
            print("[DEBUG] Modo = rejilla")
        grid_points = list(generate_grid(args.lat, args.lon, args.radius_km, args.grid_km))
        if args.verbose:
            print(f"[DEBUG] Puntos en la rejilla: {len(grid_points)}")
        rows = []
        args_list = [
            (args.viirs_path, lat, lon, args.band, args.lat, args.lon, args.dark_thr)
            for lat, lon in grid_points
        ]
        iterator = tqdm(args_list, desc="Muestreando VIIRS") if args.verbose and tqdm else args_list
        with concurrent.futures.ProcessPoolExecutor() as executor:
            for result in executor.map(process_point, iterator):
                if result:
                    rows.append(result)
                if len(rows) >= args.max_candidates:
                    break
        if args.verbose:
            print(f"[DEBUG] Candidatos bajo umbral: {len(rows)}")

    if not rows:
        print("No se encontraron candidatos bajo el umbral. Prueba a subir --dark_thr o el radio.")
        raster.close()
        return

    # -------- 2) Orden + separación mínima opcional --------
    if args.verbose:
        print("[DEBUG] Ordenando candidatos por oscuridad y distancia...")
    rows.sort(key=lambda x: (x[2], x[3]))  # primero más oscuro, luego más cercano
    if args.min_sep_km > 0:
        before = len(rows)
        rows = thin_by_distance(rows, args.min_sep_km)
        if args.verbose:
            print(f"[DEBUG] Filtro separación {args.min_sep_km} km: {before} -> {len(rows)}")

    # -------- 3) Guardar CSV de candidatos --------
    csv_path = os.path.join(args.output_dir, f"{args.out_prefix}_candidates.csv")
    if args.verbose:
        print(f"[DEBUG] Guardando CSV en {csv_path} con {len(rows)} puntos...")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["lat", "lon", "viirs_radiance_nW_cm2_sr", "distance_km"])
        w.writerows(rows)

    # -------- 4) Mapa de candidatos --------
    if args.verbose:
        print(f"[DEBUG] Generando mapa Folium de candidatos (top {args.top_n})...")
    m_candidates = build_map(rows, raster, args, title_marker="Origen")
    html_candidates = os.path.join(args.output_dir, f"{args.out_prefix}_map_candidates.html")
    m_candidates.save(html_candidates)

    # -------- 5) (Opcional) Filtrado por accesibilidad drive --------
    if args.check_drive:
        if ox is None:
            print("OSMnx no disponible; omitiendo validación de accesibilidad. Instala 'osmnx' y vuelve a ejecutar con --check_drive.")
            filtered = rows
        else:
            G = None
            graph_cache_path = os.path.join(args.output_dir, f"{args.out_prefix}_graph.graphml")
            buffer_m = args.radius_km * 1000 + args.drive_search_m  # Buffer en metros para cubrir todo

            if args.cache_graph and os.path.exists(graph_cache_path):
                if args.verbose:
                    print(f"[DEBUG] Cargando grafo cacheado desde {graph_cache_path}...")
                G = ox.load_graphml(graph_cache_path)
            else:
                if args.verbose:
                    print(f"[DEBUG] Descargando grafo grande (buffer {buffer_m}m)...")
                try:
                    G = ox.graph_from_point((args.lat, args.lon), dist=buffer_m, network_type="drive")
                    if args.cache_graph:
                        if args.verbose:
                            print(f"[DEBUG] Guardando grafo en {graph_cache_path} para reutilizar...")
                        ox.save_graphml(G, graph_cache_path)
                except Exception as e:
                    print(f"Error al descargar grafo: {e}. Omitiendo accesibilidad.")
                    G = None

            if args.verbose:
                print("[DEBUG] Validando accesibilidad (OSMnx, multitarea)...")
            filtered = []

            def check_access(row, G):
                lat, lon, rad, dist_km = row
                accesible = is_drivable_near(G, (lat, lon), search_m=args.drive_search_m)
                if args.verbose:
                    print(f"[DEBUG] ({lat:.5f}, {lon:.5f}) accesible={accesible}")
                return (lat, lon, rad, dist_km) if accesible else None

            iterator = tqdm(rows, desc="Validando accesibilidad") if args.verbose and tqdm else rows
            with concurrent.futures.ThreadPoolExecutor() as executor:
                results = list(executor.map(lambda row: check_access(row, G), iterator))
            filtered = [r for r in results if r is not None]

            if not filtered:
                print("No hay puntos con carreteras cercanas dentro del radio seleccionado.")
                # mantenemos 'filtered' vacío para que el mapa de drive se omita

        # Guardar CSV y mapa de accesibles si hay
        if filtered:
            csv_drive = os.path.join(args.output_dir, f"{args.out_prefix}_drive.csv")
            with open(csv_drive, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["lat", "lon", "viirs_radiance_nW_cm2_sr", "distance_km"])
                w.writerows(filtered)
            # mapa de accesibles
            if args.verbose:
                print(f"[DEBUG] Generando mapa Folium de accesibles (top {args.top_n})...")
            filtered.sort(key=lambda x: (x[2], x[3]))
            html_drive = os.path.join(args.output_dir, f"{args.out_prefix}_map_drive.html")
            m_drive = build_map(filtered, raster, args, title_marker="Origen (accesibles)")
            m_drive.save(html_drive)
            print(f"Generado: {csv_drive}")
            print(f"Generado: {html_drive}")

    print(f"Generado: {csv_path}")
    print(f"Generado: {html_candidates}")
    raster.close()

if __name__ == "__main__":
    main()