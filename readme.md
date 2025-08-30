# Astro Spots: Find Accessible Dark Sky Zones

<!-- Replace with an actual screenshot or demo image if available -->

## Overview

Astro Spots is a Python script designed to identify dark sky zones (low light pollution areas) within a specified radius from a given origin location, using NASA/NOAA VIIRS Nighttime Light (VNL) data. It can optionally filter these zones for accessibility by car via nearby roads, leveraging OpenStreetMap data through OSMnx.

This tool is ideal for astronomers, astrophotographers, or anyone seeking dark spots for stargazing, while ensuring they are reachable by vehicle.

### Key Features
- **Pixel-based or Grid Sampling**: Scans VIIRS GeoTIFF for low-radiance (dark) areas.
- **Road Accessibility Check**: Uses OSMnx to filter points near drivable roads.
- **Outputs**: CSV files with candidate points and interactive Folium maps (HTML) for visualization.
- **Customizable**: Adjust radius, darkness threshold, road search distance, and more.

Data Source: VIIRS VNL v2 (average-masked or median-masked GeoTIFF, e.g. 2024). Download from [NASA Earthdata](https://earthdata.nasa.gov/), [Earth Observation Group](https://eogdata.mines.edu/products/vnl/) or similar sources.

## Requirements

- Python 3.6+
- Required packages: `rasterio numpy shapely folium osmnx scikit-image matplotlib tqdm`

Note: OSMnx is optional but required for `--check_drive` mode.

## Installation

1. Clone the repository:

   git clone https://github.com/yourusername/astro-spots.git
   cd astro-spots

2. Install dependencies:

   pip install -r requirements.txt


3. Download a VIIRS GeoTIFF file (e.g., `VNL_npp_2024_global_vcmslcfg_v2_c202502261200.average.dat.tif`) and place it in your working directory.

## Usage

Run the script with command-line arguments. Basic structure:

python astro_spots.py --lat <latitude> --lon <longitude> --radius_km <radius> --viirs_path <path_to_tif> --out_prefix <prefix> --dark_thr <threshold> [options]

### Typical Usage Examples

1. **Find candidates without road filtering (pixel mode recommended)**:

   python astro_spots.py --lat 40.4168 --lon -3.7038 --radius_km 40 
   --viirs_path VNL.tif --out_prefix madrid_viirs --dark_thr 300 
   --pixel_mode --verbose

2. **Filter by nearby roads (using OSMnx)**:

   python astro_spots.py --lat 40.4168 --lon -3.7038 --radius_km 40 
   --viirs_path VNL.tif --out_prefix madrid_viirs --dark_thr 300 
   --pixel_mode --check_drive --drive_search_m 2000 --verbose

### Command-Line Arguments

- `--lat`: Latitude of the origin (required).
- `--lon`: Longitude of the origin (required).
- `--radius_km`: Search radius in km (default: 25).
- `--viirs_path`: Path to VIIRS GeoTIFF file (required).
- `--dark_thr`: Darkness threshold in nW/cm²/sr (lower = darker; default: 0.550).
- `--pixel_mode`: Use pixel-based sampling from VIIRS (recommended; flag).
- `--check_drive`: Enable road accessibility check with OSMnx (flag).
- `--drive_search_m`: Max distance (m) to nearest road (default: 150).
- `--top_n`: Number of top points to show on maps (default: 50; 0 for all).
- `--out_prefix`: Output file prefix (default: "astro_spots").
- `--verbose`: Enable debug logging (flag).
- `--cache_graph`: Cache OSM graph for reuse (flag).
- `--network_type`: OSM network type (`drive`, `walk`, `all`; default: `drive`).

For full list: `python astro_spots.py --help`

## Outputs

- `<prefix>_candidates.csv`: All candidate dark points (lat, lon, radiance, distance_km).
- `<prefix>_map_candidates.html`: Interactive Folium map of candidates.
- `<prefix>_drive.csv`: Accessible points only (if `--check_drive` enabled).
- `<prefix>_map_drive.html`: Map of accessible points (if applicable).

Open HTML maps in a web browser to view interactive results with VIIRS overlay.

## Example

(lat 42, lon -2), running with a 40km radius and dark_thr=0.500 might yield dark spots in nearby rural areas, filtered for road access.

## Troubleshooting

- **No candidates found**: Increase `--dark_thr` or `--radius_km`.
- **OSMnx errors**: Ensure OSMnx is installed; check internet for graph download.
- **Raster issues**: Verify VIIRS file path and format (EPSG:4326 expected).
- **Performance**: For large radii, use `--min_sep_km` to thin points or reduce `--max_candidates`.

## Contributing

Contributions welcome! Fork the repo, make changes, and submit a pull request. Issues and feature requests are appreciated.

## License

No license. Just enjoy.

## Acknowledgments

- Built with [rasterio](https://rasterio.readthedocs.io/), [OSMnx](https://osmnx.readthedocs.io/), [Folium](https://python-visualization.github.io/folium/), and other open-source libraries.
- Inspired by dark sky mapping for astronomy enthusiasts.

