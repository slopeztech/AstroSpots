#!/usr/bin/env python3
""" astro_spots.py — Find accessible dark zones by car near an origin.
Requirements: pip install rasterio numpy shapely folium osmnx scikit-image matplotlib tqdm
Input: - GeoTIFF VIIRS VNL v2.* (average-masked or median-masked), e.g., 2022/2024.
Typical usage:
# 1) View candidates per pixel (recommended) without filtering roads
python astro_spots.py --lat 40.4168 --lon -3.7038 --radius_km 40 \
--viirs_path VNL.tif --out_prefix madrid_viirs --dark_thr 300 \
--pixel_mode --verbose
# 2) Additionally filter by nearby roads (OSMnx)
python astro_spots.py --lat 40.4168 --lon -3.7038 --radius_km 40 \
--viirs_path VNL.tif --out_prefix madrid_viirs --dark_thr 300 \
--pixel_mode --check_drive --drive_search_m 2000 --verbose
# 3) Include all points of interest with any 'amenity', 'tourism', or 'leisure' tag within the radius
python astro_spots.py --lat 40.4168 --lon -3.7038 --radius_km 40 \
--viirs_path VNL.tif --out_prefix madrid_viirs --dark_thr 300 \
--pixel_mode --pois all --verbose
Outputs:
- <prefix>_candidates.csv (all candidates)
- <prefix>_map_candidates.html (map with candidates and POIs)
- <prefix>_drive.csv (only accessible; if applicable)
- <prefix>_map_drive.html (map of accessible with POIs; if applicable)
"""

import argparse
import math
import os
import csv
from typing import List, Tuple
import numpy as np
import rasterio
from rasterio.transform import rowcol, xy
from rasterio.windows import from_bounds
from shapely.geometry import Point
import folium
from skimage.transform import resize
import logging

# OSMnx: only if validating accessibility or fetching POIs
try:
    import osmnx as ox
    # Check OSMnx version for compatibility
    import pkg_resources
    osmnx_version = pkg_resources.get_distribution("osmnx").version
    logging.info(f"Using OSMnx version {osmnx_version}")
except Exception:
    ox = None
# tqdm optional
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# ------------------ Geodetic utilities ------------------
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

# ------------------ Grid sampling (classic mode) ------------------
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
    # Isolated function for process execution
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
    except Exception as e:
        logging.error(f"Error in process_point for {lat}, {lon}: {e}")
        return None

# ------------------ NEW: Pixel-based sampling ------------------
def sample_pixels(raster, center_lat, center_lon, radius_km, band, dark_thr):
    """ Iterates through VIIRS pixels within the radius and returns dark candidates.
    Returns a list of tuples: (lat, lon, radiance, distance_km) """
    results = []
    # Bounds in degrees
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
            # Pixel center (x=lon, y=lat)
            lon, lat = xy(transform, row, col)
            dist_km = haversine_km(center_lat, center_lon, lat, lon)
            if dist_km <= radius_km:
                results.append((lat, lon, float(val), dist_km))
    return results

# ------------------ NEW: Fetch Points of Interest ------------------
def fetch_pois(center_lat, center_lon, radius_km, poi_type):
    """Fetch points of interest from OSM within the radius using osmnx."""
    if ox is None:
        logging.warning("OSMnx not available; cannot fetch POIs. Install 'osmnx' to use --pois.")
        return []
    
    try:
        # Convert radius to meters for OSMnx
        dist_m = radius_km * 1000
        # Define tags based on poi_type
        if poi_type.lower() == 'all':
            # Fetch all features with 'amenity', 'tourism', or 'leisure' tags
            tags = {
                'amenity': True,  # Any amenity (e.g., park, observatory, place_of_worship)
                'tourism': True,  # Any tourism feature (e.g., viewpoint, attraction)
                'leisure': True    # Any leisure feature (e.g., nature_reserve, park)
            }
        elif '=' in poi_type:
            # Custom key=value pair (e.g., amenity=park)
            key, value = poi_type.split('=', 1)
            tags = {key: value}
        else:
            # Assume poi_type is an amenity value (e.g., observatory)
            tags = {'amenity': poi_type}
        
        # Fetch geometries from OSM (compatible with OSMnx v1.x and v2.x)
        if hasattr(ox, 'features') and hasattr(ox.features, 'features_from_point'):
            # For newer versions (v2.x)
            gdf = ox.features.features_from_point((center_lat, center_lon), tags=tags, dist=dist_m)
        else:
            # For older versions (v1.x or earlier), fallback to geometries_from_point if available
            if hasattr(ox, 'geometries_from_point'):
                gdf = ox.geometries_from_point((center_lat, center_lon), tags=tags, dist=dist_m)
            else:
                raise AttributeError("OSMnx version does not support POI fetching. Please update to v1.2+.")
        
        pois = []
        for _, row in gdf.iterrows():
            # Get geometry (centroid for polygons)
            geom = row['geometry']
            if geom.geom_type == 'Point':
                lon, lat = geom.x, geom.y
            else:
                lon, lat = geom.centroid.x, geom.centroid.y
            
            # Check if within radius
            dist_km = haversine_km(center_lat, center_lon, lat, lon)
            if dist_km <= radius_km:
                # Get label (try 'name', then specific tag, then default to 'POI')
                label = row.get('name', None)
                if not label or not isinstance(label, str) or not label.strip():
                    # Try the tag value for the relevant key
                    for key in ['amenity', 'tourism', 'leisure']:
                        if key in row and isinstance(row[key], str) and row[key].strip():
                            label = row[key]
                            break
                    else:
                        label = 'POI'
                pois.append((lat, lon, label, dist_km))
        
        if not pois:
            logging.info(f"No POIs of type '{poi_type}' found within {radius_km} km.")
        else:
            logging.info(f"Found {len(pois)} POIs of type '{poi_type}' within {radius_km} km.")
        
        return pois
    
    except Exception as e:
        logging.error(f"Error fetching POIs: {e}")
        return []

# ------------------ Other helpers ------------------
def is_drivable_near(G, coord: Tuple[float, float], search_m=150) -> Tuple[bool, float]:
    """Checks if there is a 'drive' road network near the point (lat, lon) using a pre-loaded graph.
    Returns (accessible, distance_m)"""
    if G is None or ox is None:
        logging.warning("Graph or OSMnx is None")
        return False, float('inf')
    lat, lon = coord
    try:
        # Project the point to the graph's CRS
        point = Point(lon, lat)
        proj_point, proj_crs = ox.projection.project_geometry(point, crs='epsg:4326', to_crs=G.graph['crs'])
        x, y = proj_point.x, proj_point.y
        # Find nearest edge and distance in the projected graph
        nearest_data, dist = ox.distance.nearest_edges(G, x, y, return_dist=True)
        logging.debug(f"Nearest edge distance for {lat}, {lon}: {dist} m")
        return dist <= search_m, dist if dist is not None else float('inf')
    except Exception as e:
        logging.error(f"Error in is_drivable_near for {lat}, {lon}: {e}")
        return False, float('inf')  # If it fails, assume not accessible with infinite distance

def thin_by_distance(rows, min_sep_km):
    """Reduces clustered points, prioritizing darker and closer ones."""
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
    """Adds the raster clip as an overlay for visual reference."""
    import matplotlib.pyplot as plt
    import tempfile
    # Bounds of interest
    delta_deg = args.radius_km * deg_per_km_lat()
    min_lat = args.lat - delta_deg
    max_lat = args.lat + delta_deg
    delta_deg_lon = args.radius_km * deg_per_km_lon(args.lat)
    min_lon = args.lon - delta_deg_lon
    max_lon = args.lon + delta_deg_lon
    window = from_bounds(min_lon, min_lat, max_lon, max_lat, raster.transform)
    arr = raster.read(args.band, window=window)
    # Visual: clip to 99th percentile and normalize
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
        name="Light pollution"
    ).add_to(m)

def build_map(rows, raster, args, title_marker="Origin", show_road_distance=False):
    m = folium.Map(location=[args.lat, args.lon], zoom_start=10, control_scale=True)
    folium.Marker(
        [args.lat, args.lon],
        tooltip=title_marker,
        icon=folium.Icon(icon="home"),
    ).add_to(m)
    # Points (top_n)
    topN = rows[: args.top_n] if args.top_n > 0 else rows
    for i, row in enumerate(topN, start=1):
        if show_road_distance and len(row) == 5:
            lat, lon, rad, dist_km, dist_road = row
            tooltip = f"#{i} — {dist_road:.6f} m to road — {rad:.3f} nW/cm²/sr"
            popup_html = f"""
            <b>#{i} — Distance to road: {dist_road:.6f} m</b><br>
            Radiance: {rad:.3f} nW/cm²/sr<br>
            Distance from origin: {dist_km:.1f} km<br>
            <ul>
            <li><a href="https://www.google.com/maps/search/?api=1&query={lat},{lon}" target="_blank">Open in Google Maps</a></li>
            </ul>
            """
        else:
            lat, lon, rad, dist_km = row[:4]
            tooltip = f"#{i} — {dist_km:.1f} km from origin — {rad:.3f} nW/cm²/sr"
            popup_html = f"""
            <b>#{i}</b><br>
            Radiance: {rad:.3f} nW/cm²/sr<br>
            Distance from origin: {dist_km:.1f} km<br>
            <ul>
            <li><a href="https://www.google.com/maps/search/?api=1&query={lat},{lon}" target="_blank">Open in Google Maps</a></li>
            </ul>
            """
        folium.CircleMarker(
            [lat, lon],
            radius=6,
            tooltip=tooltip,
            fill=True,
            popup=popup_html,
        ).add_to(m)

    # Add points of interest if --pois is specified
    if args.pois:
        pois = fetch_pois(args.lat, args.lon, args.radius_km, args.pois)
        for poi_lat, poi_lon, poi_label, dist_km in pois:
            tooltip = f"{poi_label} — {dist_km:.1f} km from origin"
            popup_html = f"""
            <b>{poi_label}</b><br>
            Distance from origin: {dist_km:.1f} km<br>
            <ul>
            <li><a href="https://www.google.com/maps/search/?api=1&query={poi_lat},{poi_lon}" target="_blank">Open in Google Maps</a></li>
            </ul>
            """
            folium.Marker(
                [poi_lat, poi_lon],
                tooltip=tooltip,
                popup=popup_html,
                icon=folium.Icon(color='green', icon='star'),
            ).add_to(m)

    # VIIRS overlay
    add_viirs_overlay(m, raster, args)
    folium.LayerControl().add_to(m)
    return m

# ------------------ MAIN ------------------
def main():
    parser = argparse.ArgumentParser(description="Find dark zones using VIIRS + OSM.")
    parser.add_argument("--lat", type=float, required=True, help="Latitude of the origin")
    parser.add_argument("--lon", type=float, required=True, help="Longitude of the origin")
    parser.add_argument("--radius_km", type=float, default=25, help="Search radius in km (default 25)")
    parser.add_argument("--grid_km", type=float, default=2.0, help="Grid step in km (grid mode)")
    parser.add_argument("--viirs_path", type=str, required=True, help="Path to VIIRS GeoTIFF (average/median masked)")
    parser.add_argument("--band", type=int, default=1, help="GeoTIFF band (default 1)")
    parser.add_argument("--dark_thr", type=float, default=0.550, help="Darkness threshold in nW/cm^2/sr (lower = darker)")
    parser.add_argument("--max_candidates", type=int, default=300000, help="Candidate limit (grid mode)")
    parser.add_argument("--check_drive", action="store_true", help="Validate accessibility by car with OSMnx")
    parser.add_argument("--drive_search_m", type=int, default=150, help="Radius (m) to search for roads in OSMnx")
    parser.add_argument("--min_sep_km", type=float, default=0.0, help="Minimum separation between points (0 = no filtering)")
    parser.add_argument("--top_n", type=int, default=50, help="Number of points to draw on maps (CSV saves all)")
    parser.add_argument("--pixel_mode", action="store_true", help="Use VIIRS pixel sampling (recommended)")
    parser.add_argument("--out_prefix", type=str, default="astro_spots", help="Output file prefix")
    parser.add_argument("--verbose", action="store_true", help="Verbose mode")
    parser.add_argument("--output_dir", type=str, default=".", help="Output directory for maps and CSV")
    parser.add_argument("--cache_graph", action="store_true", help="Save/load OSMnx graph to/from disk for reuse")
    parser.add_argument("--network_type", type=str, default="drive",
                        choices=["drive", "walk", "all"],
                        help="OSMnx network type: drive (roads), walk (pedestrian), all (all)")
    parser.add_argument("--pois", type=str, default=None,
                        help="Type of points of interest to fetch from OSM within the radius (e.g., 'observatory', 'park', 'amenity=place_of_worship', 'all' for all amenities/tourism/leisure). Requires OSMnx.")

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Open VIIRS raster
    if not os.path.exists(args.viirs_path):
        raise FileNotFoundError(f"VIIRS file does not exist: {args.viirs_path}")
    raster = rasterio.open(args.viirs_path)
    if args.verbose:
        print("Raster CRS:", raster.crs)
        band1 = raster.read(args.band)
        print("Value range:", np.nanmin(band1), np.nanmax(band1))

    # -------- 1) Candidate sampling --------
    if args.pixel_mode:
        if args.verbose:
            print("[DEBUG] Mode = VIIRS pixel")
        rows = sample_pixels(raster, args.lat, args.lon, args.radius_km, args.band, args.dark_thr)
        if args.verbose:
            print(f"[DEBUG] Candidate pixels below threshold: {len(rows)}")
    else:
        if args.verbose:
            print("[DEBUG] Mode = grid")
        grid_points = list(generate_grid(args.lat, args.lon, args.radius_km, args.grid_km))
        if args.verbose:
            print(f"[DEBUG] Grid points: {len(grid_points)}")
        rows = []
        args_list = [
            (args.viirs_path, lat, lon, args.band, args.lat, args.lon, args.dark_thr)
            for lat, lon in grid_points
        ]
        for args_tuple in args_list:  # Sequential processing instead of executor
            result = process_point(args_tuple)
            if result:
                rows.append(result)
            if len(rows) >= args.max_candidates:
                break
        if args.verbose:
            print(f"[DEBUG] Candidates below threshold: {len(rows)}")

    if not rows:
        print("No candidates found below the threshold. Try increasing --dark_thr or the radius.")
        raster.close()
        return

    # -------- 2) Sort + optional minimum separation --------
    if args.verbose:
        print("[DEBUG] Sorting candidates by darkness and distance...")
    rows.sort(key=lambda x: (x[2], x[3]))  # Darker first, then closer
    if args.min_sep_km > 0:
        before = len(rows)
        rows = thin_by_distance(rows, args.min_sep_km)
        if args.verbose:
            print(f"[DEBUG] Separation filter {args.min_sep_km} km: {before} -> {len(rows)}")

    # -------- 3) Save candidates CSV --------
    csv_path = os.path.join(args.output_dir, f"{args.out_prefix}_candidates.csv")
    if args.verbose:
        print(f"[DEBUG] Saving CSV to {csv_path} with {len(rows)} points...")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["lat", "lon", "viirs_radiance_nW_cm2_sr", "distance_km"])
        w.writerows(rows)

    # -------- 4) Candidates map --------
    if args.verbose:
        print(f"[DEBUG] Generating Folium map of candidates (top {args.top_n})...")
    m_candidates = build_map(rows, raster, args, title_marker="Origin")
    html_candidates = os.path.join(args.output_dir, f"{args.out_prefix}_map_candidates.html")
    m_candidates.save(html_candidates)

    # -------- 5) (Optional) Drive accessibility filtering --------
    filtered = rows  # Default to all rows if no check_drive
    if args.check_drive:
        if ox is None:
            print("OSMnx not available; skipping accessibility validation. Install 'osmnx' and rerun with --check_drive.")
        else:
            G = None
            graph_cache_path = os.path.join(args.output_dir, f"{args.out_prefix}_graph.graphml")
            buffer_m = args.radius_km * 1000 + args.drive_search_m  # Buffer in meters to cover all

            if args.cache_graph and os.path.exists(graph_cache_path):
                if args.verbose:
                    print(f"[DEBUG] Loading cached graph from {graph_cache_path}...")
                G = ox.load_graphml(graph_cache_path)
            else:
                if args.verbose:
                    print(f"[DEBUG] Downloading large graph (buffer {buffer_m}m, type {args.network_type})...")
                try:
                    G = ox.graph_from_point((args.lat, args.lon), dist=buffer_m, network_type=args.network_type)
                    if args.cache_graph or not os.path.exists(graph_cache_path):  # Save if it doesn't exist
                        if args.verbose:
                            print(f"[DEBUG] Saving graph to {graph_cache_path} for reuse...")
                        ox.save_graphml(G, graph_cache_path)
                except Exception as e:
                    print(f"Error downloading graph: {e}. Skipping accessibility.")
                    G = None

            if G is None:
                print("Graph not available; skipping accessibility validation.")
            else:
                # Project the graph for accurate Euclidean distance calculations
                if args.verbose:
                    print("[DEBUG] Projecting graph to local UTM for accurate distances...")
                G = ox.project_graph(G)

                if args.verbose:
                    print("[DEBUG] Validating accessibility (sequential)...")
                filtered = []
                iterator = tqdm(rows) if args.verbose and tqdm else rows
                for row in iterator:
                    lat, lon, rad, dist_km = row
                    accessible, dist_road = is_drivable_near(G, (lat, lon), search_m=args.drive_search_m)
                    if accessible and dist_road is not None and dist_road != float('inf'):
                        logging.debug(f"Accessible point at {lat}, {lon} with distance to road: {dist_road} m")
                        filtered.append((lat, lon, rad, dist_km, dist_road))
                    else:
                        logging.debug(f"Non-accessible or invalid point at {lat}, {lon}, dist_road: {dist_road}")

                if not filtered:
                    print("No points with nearby roads within the selected radius.")

        # Save CSV and map of accessible points if any
        if len(filtered) > 0 and filtered != rows:  # Only if filtered differently
            csv_drive = os.path.join(args.output_dir, f"{args.out_prefix}_drive.csv")
            with open(csv_drive, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["lat", "lon", "viirs_radiance_nW_cm2_sr", "distance_km_origin", "distance_to_road_m"])
                w.writerows(filtered)
            # Accessible points map
            if args.verbose:
                print(f"[DEBUG] Generating Folium map of accessible points (top {args.top_n})...")
            filtered.sort(key=lambda x: (x[2], x[3]))
            html_drive = os.path.join(args.output_dir, f"{args.out_prefix}_map_drive.html")
            m_drive = build_map(filtered, raster, args, title_marker="Origin (accessible)", show_road_distance=True)
            m_drive.save(html_drive)
            print(f"Generated: {csv_drive}")
            print(f"Generated: {html_drive}")

    print(f"Generated: {csv_path}")
    print(f"Generated: {html_candidates}")
    raster.close()

if __name__ == "__main__":
    main()