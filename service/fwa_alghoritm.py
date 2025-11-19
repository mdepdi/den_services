import os
import sys
import numpy as np
import geopandas as gpd
import pandas as pd
import zipfile
import sys

sys.path.append(r"D:\Data Analytical\SERVICE\API")

from shapely.geometry import Polygon
from datetime import datetime
from time import time
from glob import glob
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import multiprocessing as mp
from modules.table import excel_styler, sanitize_header
from modules.data import read_gdf
from core.config import settings

MAINDATA_DIR = settings.MAINDATA_DIR
DATA_DIR = settings.DATA_DIR
EXPORT_DIR = settings.EXPORT_DIR

def identify_hexagon(data_gdf, resolution=5, buffer=10000, type="bound"):
    """
    Identify hexagon identifiers in a GeoDataFrame.

    Args:
        data_gdf (GeoDataFrame): GeoDataFrame containing geometries with 'hex_id' column.
        resolution (int): Resolution of the hexagon grid.
        buffer (int): Buffer distance in meters to expand the bounding box.
        type (str): Type of hexagon to identify. Options are ['bound', 'convex'].

    Returns:
        list: List of unique hexagon identifiers.
    """
    from shapely.geometry import box

    hex_path = f"{MAINDATA_DIR}/22. H3 Hex/Hex_{resolution}.parquet"
    if not os.path.exists(hex_path):
        raise FileNotFoundError(f"Hexagon file not found at {hex_path}")
    hex_gdf = gpd.read_parquet(hex_path)

    # CONVERT TO 3857
    if hex_gdf.crs != "EPSG:3857":
        hex_gdf = hex_gdf.to_crs("EPSG:3857")
    if data_gdf.crs != "EPSG:3857":
        data_gdf = data_gdf.to_crs("EPSG:3857")

    match type:
        case "bound":
            bounding_box = data_gdf.total_bounds
            bbox_polygon = box(*bounding_box).buffer(buffer)
            hex_gdf = hex_gdf[hex_gdf.intersects(bbox_polygon)]
        case "convex":
            convex_hull = data_gdf.geometry.union_all().convex_hull.buffer(buffer)
            hex_gdf = hex_gdf[hex_gdf.intersects(convex_hull)]
        case _:
            raise ValueError("Invalid type specified. Use 'bound' or 'convex'.")
    if hex_gdf.empty:
        raise ValueError("No hexagons found for the given bounding box or convex hull.")

    hex_list = hex_gdf[f"hex_{resolution}"].unique().tolist()
    if not hex_list:
        raise ValueError("No hexagons found for the given bounding box.")
    return hex_list


def retrieve_building(hex_list, centroid=True, hex_dir=None, **kwargs):
    """
    Retrieve building data from hex files.
    Args:
        hex_list (list): List of hex identifiers.
        hex_dir (str, optional): Directory containing hex files. Defaults to None.
    Returns:
        list: List of GeoDataFrames containing building data.
    """
    import shutil
    from concurrent.futures import ThreadPoolExecutor, as_completed

    one_unit = kwargs.get("one_unit", False)
    area_building = kwargs.get("area_building", True)
    aspect_ratio = kwargs.get("aspect_ratio", True)
    parameters = kwargs.get("parameters",
        {
            "aspect_ratio_value": 0.25,
            "area_building_value": {
                "min": 25,
                "max": 500,
            },
        },
    )

    # print("ℹ️ Retrieve Building:")
    # print(f"Centroid        : {centroid}")
    # print(f"One unit        : {one_unit}")
    # print(f"Area building   : {area_building}")
    # print(f"Aspect ratio    : {aspect_ratio}\n")

    if hex_dir is None:
        hex_dir = f"{MAINDATA_DIR}/02. Building/Hexed Building 2024"

    all_data = []
    def load_hex_file(hex_id):
        try:
            hex_path = os.path.join(hex_dir, f"{hex_id}_buildings.parquet")
            if os.path.exists(hex_path):
                data = gpd.read_parquet(hex_path)
                data = data.to_crs(epsg=3857)
                if centroid:
                    data["geometry"] = data.geometry.centroid
                if aspect_ratio:
                    data = data[data["asp_ratio"] > parameters["aspect_ratio_value"]]
                if area_building:
                    data = data[(data["area_in_meters"] > parameters["area_building_value"]["min"]) & (data["area_in_meters"] < parameters["area_building_value"]["max"])]
                if one_unit:
                    data = data[data["one_unit"] == 1]

                if not data.empty and "geometry" in data.columns:
                    return data
            else:
                return None
        except Exception as e:
            print(f"Error Hex {hex_id}: {e}")
            source_dir = r"Z:\01. DATABASE\02. Building\Adm 2024\Hexed Building 2024"
            target_dir = (
                f"{MAINDATA_DIR}/02. Building/Adm 2024/Hexed Building 2024"
            )
            source_file = os.path.join(source_dir, f"{hex_id}_buildings.parquet")
            target_file = os.path.join(target_dir, f"{hex_id}_buildings.parquet")
            shutil.copy(source_file, target_file)
            print(f"ℹ️ Copied {hex_id} from Z.")
            return load_hex_file(hex_id)

    all_data = []
    with ThreadPoolExecutor() as executor:
        future_to_hex = {
            executor.submit(load_hex_file, hex_id): hex_id for hex_id in hex_list
        }

        for future in as_completed(future_to_hex):
            id = future_to_hex[future]
            result = future.result()
            if result is not None:
                all_data.append(result)

    if not all_data:
        print(f"⚠️ No data found for hex list: {hex_list}.")
        data = gpd.GeoDataFrame(geometry=pd.Series([], dtype='geometry'), crs='epsg:4326').set_geometry('geometry')
        data = data.to_crs(epsg=3857)
        return data

    # print(f"ℹ️ Concating building data.")
    all_data = pd.concat(all_data, ignore_index=True)
    all_data = all_data.drop_duplicates(subset="geometry").reset_index(drop=True)
    all_data = gpd.GeoDataFrame(all_data, geometry="geometry", crs="EPSG:3857")

    if "geom_point" in all_data.columns:
        all_data = all_data.drop(columns="geom_point")
    if centroid:
        all_data["geometry"] = all_data.geometry.centroid
    return all_data


def auto_group(data_gdf: gpd.GeoDataFrame, expand_gdf: gpd.GeoDataFrame = None, distance=10000):
    if data_gdf.crs != "EPSG:3857":
        data_gdf = data_gdf.to_crs(epsg=3857)

    data_gdf = data_gdf[['geometry']]
    if expand_gdf and not expand_gdf.empty:
        expand_gdf = expand_gdf[['geometry']]
        data_gdf = pd.concat([data_gdf, expand_gdf])

    groups = data_gdf.copy()
    groups["geometry"] = groups.geometry.buffer(distance)
    groups = groups.dissolve().explode(ignore_index=True)

    groups["region"] = groups.index + 1
    print(f"ℹ️ Total Group generated: {len(groups)}")

    return groups

def count_homepass(site_gdf: gpd.GeoDataFrame, building: gpd.GeoDataFrame, distance : float = 500):
    site_gdf = site_gdf.to_crs(epsg=3857)
    building = building.to_crs(epsg=3857)
    building['geometry'] = building.geometry.centroid
    site_gdf['buffer'] = site_gdf.geometry.buffer(distance)
    site_gdf = site_gdf.set_geometry('buffer')

    joined = gpd.sjoin(building, site_gdf, how='inner', predicate='intersects').rename(columns={'index_right':'site_idx'})
    count_homepass_class = joined.groupby(['site_idx', 'homepass_class']).size().unstack(fill_value=0)
    count_total = joined.groupby('site_idx').size().rename('total_homepass')
    for col in count_homepass_class.columns:
        site_gdf[f'hp_{col}'] = site_gdf.index.to_series().map(count_homepass_class[col]).fillna(0).astype(int)
    
    site_gdf['total_homepass'] = site_gdf.index.to_series().map(count_total).fillna(0).astype(int)
    site_gdf = site_gdf.set_geometry('geometry').drop(columns='buffer')
    return site_gdf

def clean_overlaps(
    site_tbg: gpd.GeoDataFrame,
    max_distance: float = 300,
    tolerance: float = 10.0
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Clean overlapping sites (points) by buffer.
    Priority:
      1. Greenfield sites win over non-greenfield
      2. If same type -> higher total_homepass wins
    Returns:
      site_kept (survivors), site_drop (dropped)
    """
    if site_tbg.crs is None:
        site_tbg = site_tbg.set_crs(4326)
    if site_tbg.crs.to_string() != "EPSG:3857":
        site_tbg = site_tbg.to_crs(3857)

    site_tbg = site_tbg.reset_index(drop=True)
    site_tbg["total_homepass"] = (site_tbg.get("total_homepass", pd.Series(0, index=site_tbg.index)).fillna(0))

    if "site_type" in site_tbg.columns:
        stype = site_tbg["site_type"].astype(str).str.lower().str.replace(" ", "")
        site_tbg["__isgreenfield"] = stype.str.contains("greenfield")
    else:
        site_tbg["__isgreenfield"] = False

    site_tbg['geometry'] = site_tbg.geometry.buffer(max_distance)
    site_tbg["__buf"] = site_tbg.geometry.buffer(max_distance)
    site_tbg["__area"] = site_tbg["__buf"].area

    # Spatial index
    sidx = site_tbg.sindex
    L, R = sidx.query(site_tbg["__buf"], predicate="intersects")
    mask = L < R
    L, R = L[mask], R[mask]

    dropped_pos = set()
    for i, j in zip(L.tolist(), R.tolist()):
        if i in dropped_pos or j in dropped_pos:
            continue

        inter_area = site_tbg["__buf"].iloc[i].intersection(site_tbg["__buf"].iloc[j]).area
        if inter_area <= 0:
            continue

        pct_i = 100.0 * inter_area / site_tbg.at[i, "__area"]
        pct_j = 100.0 * inter_area / site_tbg.at[j, "__area"]
        if max(pct_i, pct_j) < tolerance:
            continue

        # Scoring rule
        def score(idx):
            return (
                1 if site_tbg.at[idx, "__isgreenfield"] else 0,
                int(site_tbg.at[idx, "total_homepass"])
            )

        score_i, score_j = score(i), score(j)
        winner = i if score_i > score_j else j
        loser  = j if winner == i else i
        dropped_pos.add(loser)

    site_tbg['geometry'] = site_tbg.geometry.centroid
    all_pos = set(range(len(site_tbg)))
    keep_pos = sorted(all_pos - dropped_pos)

    site_kept = site_tbg.iloc[keep_pos].copy()
    site_drop = site_tbg.iloc[sorted(dropped_pos)].copy()

    # Keep original geometries
    site_kept["geometry"] = site_tbg.loc[site_kept.index, "geometry"]

    # Cleanup
    drop_cols = ["__buf", "__area", "__isgreenfield"]
    site_kept = site_kept.drop(columns=drop_cols, errors="ignore").reset_index(drop=True)
    site_drop = site_drop.drop(columns=drop_cols, errors="ignore").reset_index(drop=True)
    return site_kept, site_drop


def clean_sector_overlaps(
    sectors: gpd.GeoDataFrame,
    accept_list: list | None = None,
    tolerance: float = 10.0
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Clean overlapping sector polygons.
    Priority:
      1. Greenfield sectors win over non-greenfield
      2. If same type -> higher total_homepass wins
    Returns:
      kept_sectors, dropped_sectors
    """
    if sectors.empty:
        return sectors, sectors

    if sectors.crs is None:
        sectors = sectors.set_crs(3857)
    elif sectors.crs.to_string() != "EPSG:3857":
        sectors = sectors.to_crs(3857)

    sectors = sectors.reset_index(drop=True)
    sectors["total_homepass"] = (sectors.get("total_homepass", pd.Series(0, index=sectors.index)).fillna(0))

    # Parameters
    # Accepted
    if accept_list:
        sectors["__isaccepted"] = sectors["site_id"].isin(accept_list)
    else:
        sectors["__isaccepted"] = False

    # Greenfield
    if "site_type" in sectors.columns:
        stype = sectors["site_type"].astype(str).str.lower().str.replace(" ", "")
        sectors["__isgreenfield"] = stype.str.contains("greenfield")
    else:
        sectors["__isgreenfield"] = False
    
    # Total Utilized
    utils_sector = sectors.groupby('site_id').size().rename('sector_used')
    sectors['__sector_utilized'] = sectors["site_id"].map(utils_sector)
    sectors["__area"] = sectors.geometry.area

    # Spatial index
    sidx = sectors.sindex
    L, R = sidx.query(sectors.geometry, predicate="intersects")
    mask = L < R
    L, R = L[mask], R[mask]

    dropped_pos = set()
    for i, j in zip(L.tolist(), R.tolist()):
        if i in dropped_pos or j in dropped_pos:
            continue

        inter_area = sectors.geometry.iloc[i].intersection(sectors.geometry.iloc[j]).area
        if inter_area <= 0:
            continue

        pct_i = 100.0 * inter_area / sectors.at[i, "__area"]
        pct_j = 100.0 * inter_area / sectors.at[j, "__area"]
        if max(pct_i, pct_j) < tolerance:
            continue

        # Same scoring logic as sites
        def score(idx):
            if accept_list:
                return (
                    1 if sectors.at[idx, "__isaccepted"] else 0,
                    1 if sectors.at[idx, "__isgreenfield"] else 0,
                    int(sectors.at[idx, "__sector_utilized"]),
                    int(sectors.at[idx, "total_homepass"])
                )
            else:
                return(1 if sectors.at[idx, "__isgreenfield"] else 0, int(sectors.at[idx, "total_homepass"]))
                

        score_i, score_j = score(i), score(j)
        winner = i if score_i > score_j else j
        loser  = j if winner == i else i
        dropped_pos.add(loser)

    all_pos = set(range(len(sectors)))
    keep_pos = sorted(all_pos - dropped_pos)

    kept   = sectors.iloc[keep_pos].copy()
    dropped = sectors.iloc[sorted(dropped_pos)].copy()

    drop_cols = ["__area", "__isgreenfield", "__isaccepted", "__sector_utilized"]
    kept = kept.drop(columns=drop_cols, errors="ignore").reset_index(drop=True)
    dropped = dropped.drop(columns=drop_cols, errors="ignore").reset_index(drop=True)
    return kept, dropped

def classify_market(x):
    match x:
        case x if x > 700:
            return "P1"
        case x if x > 500:
            return "P2"
        case x if x > 300:
            return "P3"
        case _:
            return "P4"
        
def homepass_class(x: int|float):
    match x:
        case x if x > 120:
            return "high"
        case x if x > 70:
            return "high"
        case x if x > 45:
            return "medium"
        case x if x > 36:
            return "low"
        case _:
            return "very low"

# =========
# SECTORIZE
# =========
        
def generate_sector(center_point, buffer_distance, sector_angle=90, rotation_angle=0):
    x = center_point.x
    y = center_point.y
    full_buffer = center_point.buffer(buffer_distance)

    # Number of sectors
    num_sectors = 360 // sector_angle
    buffer_distance = buffer_distance + (buffer_distance * 0.25)
    sectors = []

    for i in range(num_sectors):
        start_angle = i * sector_angle + rotation_angle
        end_angle = start_angle + sector_angle
        angles = np.linspace(np.radians(start_angle), np.radians(end_angle), num=50)
        x_arc = x + buffer_distance * np.sin(angles)
        y_arc = y + buffer_distance * np.cos(angles)
        curve = list(zip(x_arc, y_arc))

        # Azimuth sector
        azimuth_start = start_angle
        azimuth_end = end_angle

        # Sector coordinates
        sector = Polygon([(x, y), *curve, (x, y)])
        sector_data = {
            "geometry": full_buffer.intersection(sector),
            "azimuth_start": azimuth_start,
            "azimuth_end": azimuth_end,
        }

        sectors.append(sector_data)
    sectors = gpd.GeoDataFrame(sectors, geometry="geometry", crs="EPSG:3857")
    return sectors

def sectorize_sites(args):
    import pandas as pd
    import geopandas as gpd

    (
        site_data,
        homepass_data,
        distance_fwa,
        sector_angle,
        group_factor,
        threshold_sector,
    ) = args

    # Site Info
    site_id = site_data["site_id"]
    site_geom = site_data["geometry"]

    valid_operation = {"iteration": -1, "valid_count": 0, "sectors": None}
    iteration = 0

    for offset in range(0, sector_angle, 5):
        sectors = generate_sector(
            site_geom,
            distance_fwa,
            sector_angle=sector_angle,
            rotation_angle=offset,
        )

        group = 0
        temp_sector = []
        for i, (_, sec) in enumerate(sectors.iterrows()):
            if i % group_factor == 0:
                group += 1

            temp_sector.append(
                {
                    "site_id": site_id,
                    "sector": i + 1,
                    "sector_id": f"{site_id}_{i+1}",
                    "sector_group": f"{site_id}_{group}",
                    "azimuth": f"{sec['azimuth_start']}-{sec['azimuth_end']-360 if sec['azimuth_end'] > 360 else sec['azimuth_end']}",
                    "geometry": sec["geometry"],
                }
            )

        temp_sector_gdf = gpd.GeoDataFrame(temp_sector, geometry="geometry", crs="EPSG:3857")
        temp_sector_gdf = temp_sector_gdf.drop_duplicates(subset=["geometry"]).reset_index(drop=True)

        if not homepass_data.empty:
            homepass_site_joined = (
                gpd.sjoin(
                    homepass_data,
                    temp_sector_gdf[["geometry", "sector_id", "sector_group"]],
                    how="inner",
                    predicate="intersects",
                )
                .drop_duplicates(subset=["geometry"])
                .drop(columns=["index_right"])
            )
            sum_sector_homepass = (homepass_site_joined.groupby(["sector_id"]).size().reset_index(name="total_homepass"))
            sum_dict = dict(zip(sum_sector_homepass["sector_id"],sum_sector_homepass["total_homepass"]))
        else:
            sum_dict = {}

        temp_sector_gdf["total_homepass"] = (temp_sector_gdf["sector_id"].map(sum_dict).fillna(0))
        temp_sector_gdf["83_percent"] = temp_sector_gdf["total_homepass"].apply(lambda x: np.ceil(x * 0.83))

        # Check total homepass
        temp_sector_gdf["status"] = temp_sector_gdf["83_percent"].apply(
            lambda x: (
                f"Over {threshold_sector}HP"
                if x >= threshold_sector
                else f"Under {threshold_sector}HP"
            )
        )
        valid_sectors = temp_sector_gdf[temp_sector_gdf["status"] == f"Over {threshold_sector}HP"]
        # Check if valid sectors is more than 0
        if valid_operation["iteration"] == -1:
            valid_operation["iteration"] = iteration
            valid_operation["valid_count"] = len(valid_sectors)
            valid_operation["sectors"] = temp_sector_gdf
        elif len(valid_sectors) > valid_operation["valid_count"]:
            valid_operation["iteration"] = iteration
            valid_operation["valid_count"] = len(valid_sectors)
            valid_operation["sectors"] = temp_sector_gdf
        iteration += 1

    return valid_operation["sectors"]

def parallel_sectorize(
    sites_gdf: gpd.GeoDataFrame,
    homepass_final: gpd.GeoDataFrame,
    distance_fwa=300,
    sector_angle=120,
    sector_group=120,
    threshold_sector=120,
    max_workers=4,
):
    # print("🧩 Parallel Sector Processing Started!")

    group_factor = sector_group // sector_angle
    homepass_final = homepass_final.reset_index(drop=True)
    homepass_sindex = homepass_final.sindex
    sites_gdf["site_id"] = sites_gdf["site_id"].astype(str)
    site_args = []
    for _, site in sites_gdf.iterrows():
        site_id = site["site_id"]
        site_geom = site['geometry']
        site_data = {
            "site_id": site_id,
            "geometry": site_geom,
        }
        site_buff = site_geom.buffer(distance_fwa)

        try:
            idx = homepass_sindex.query(site_buff, predicate="intersects")
            homepass_data = homepass_final.iloc[idx].reset_index(drop=True)
        except KeyError:
            print(f"⚠️ No homepass data found for site_id: {site_id}")
            homepass_data = pd.DataFrame(columns=homepass_final.columns)
            continue

        homepass_data = homepass_data.reset_index(drop=True)
        args = (
            site_data,
            homepass_data,
            distance_fwa,
            sector_angle,
            group_factor,
            threshold_sector,
        )
        site_args.append(args)

    site_sectors = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(sectorize_sites, args): site_data["site_id"]
            for args, site_data in zip(site_args, sites_gdf.to_dict(orient="records"))
        }

        for future in as_completed(futures):
            try:
                sector_result = future.result()
                if sector_result is not None and not sector_result.empty:
                    site_sectors.append(sector_result)
            except Exception as e:
                site_id = futures[future]
                print(f"⚠️ Error processing site {site_id}: {e}")

    if site_sectors:
        site_sectors_df = pd.concat(site_sectors, ignore_index=True)
        site_sectors_df = site_sectors_df.drop_duplicates(subset=["geometry"]).reset_index(drop=True)
        site_sectors_gdf = gpd.GeoDataFrame(site_sectors_df, geometry="geometry", crs="EPSG:3857")
    else:
        site_sectors_gdf = gpd.GeoDataFrame(
            columns=[
                "site_id",
                "sector",
                "sector_id",
                "sector_group",
                "azimuth",
                "geometry",
                "total_homepass",
                "83_percent",
                "status",
            ],
            crs="EPSG:3857",
        )
    # print(f"🧩 Parallel Sector Processing Completed - Generated {len(site_sectors_gdf)} sectors")
    return site_sectors_gdf

# ==============
# EXTEND SECTORS
# ==============
def extend_sectors(
    accepted_sites: gpd.GeoDataFrame,
    dropped_sites: gpd.GeoDataFrame,
    sectors: gpd.GeoDataFrame,
    tolerance: float = 10.0
) -> tuple[gpd.GeoDataFrame, list]:
    """
    Returns:
      result_gdf (accepted + extended sectors, dedup by sector_id),
      extend_id (list of site_id that got at least one sector extended)
    """
    def is_gf(site_id):
        try:
            st = str(drp_sites_meta.at[site_id, "site_type"]).upper().strip()
            return st == "GREEN FIELD"
        except Exception:
            return False
        
    # Split sectors
    acc_ids = set(accepted_sites["site_id"].astype(str))
    drp_ids = set(dropped_sites["site_id"].astype(str))

    acc_secs = sectors[sectors["site_id"].astype(str).isin(acc_ids)].copy()
    drp_secs = sectors[sectors["site_id"].astype(str).isin(drp_ids)].copy()

    # Normalize CRS
    for g in (acc_secs, drp_secs):
        if g.crs is None:
            g.set_crs(3857, inplace=True)
        elif g.crs.to_string() != "EPSG:3857":
            g.to_crs(3857, inplace=True)

    # Precompute Area
    drp_secs["__area"] = drp_secs.geometry.area

    # ---------- DROPPED SECTOR ----------
    if not drp_secs.empty:
        sidx_drp = drp_secs.sindex
        L, R = sidx_drp.query(drp_secs.geometry, predicate="intersects")
        m = L < R
        L, R = L[m], R[m]
    else:
        L = R = np.array([], dtype=int)
    drp_sites_meta = dropped_sites.set_index("site_id")

    losers = set()
    for i, j in zip(L.tolist(), R.tolist()):
        if i in losers or j in losers:
            continue
        gi = drp_secs.geometry.iat[i]
        gj = drp_secs.geometry.iat[j]
        inter_area = gi.intersection(gj).area
        if inter_area <= 0: 
            continue
        area_i = drp_secs["__area"].iat[i] or 1.0
        area_j = drp_secs["__area"].iat[j] or 1.0
        if max(100.0 * inter_area / area_i, 100.0 * inter_area / area_j) < tolerance:
            continue

        si = str(drp_secs["site_id"].iat[i])
        sj = str(drp_secs["site_id"].iat[j])
        hp_i = float(drp_secs["total_homepass"].iat[i]) if "total_homepass" in drp_secs.columns else 0.0
        hp_j = float(drp_secs["total_homepass"].iat[j]) if "total_homepass" in drp_secs.columns else 0.0

        def score(site_id, hp):
            return (0 if not is_gf(site_id) else 1, hp)

        score_i = score(si, hp_i)
        score_j = score(sj, hp_j)
        winner = i if score_i > score_j else j
        loser  = j if winner == i else i
        losers.add(loser)

    # Cleaned Dropped Sectors
    cand_idx = drp_secs.index.difference(drp_secs.index[list(losers)])
    cand = drp_secs.loc[cand_idx].copy()

    if cand.empty:
        result = acc_secs.copy()
        return result.reset_index(drop=True), []

    # ---------- ACCEPTED SECTOR ----------
    if not cand.empty and not isinstance(cand.geometry, gpd.GeoSeries):
        cand = gpd.GeoDataFrame(cand, geometry=gpd.GeoSeries(cand["geometry"], crs=drp_secs.crs))
    if not acc_secs.empty and not isinstance(acc_secs.geometry, gpd.GeoSeries):
        acc_secs = gpd.GeoDataFrame(acc_secs, geometry=gpd.GeoSeries(acc_secs["geometry"], crs=sectors.crs))

    if acc_secs.empty:
        final_extend = cand
    else:
        sidx_acc = acc_secs.sindex
        keep_mask = []
        for idx, row in cand.iterrows():
            geom = row.geometry
            sector_area = row["__area"] or geom.area or 1.0
            hits = list(sidx_acc.query(geom))

            keep = True
            if hits:
                overlap_area = 0.0
                for h in hits:
                    g2 = acc_secs.geometry.iat[h]
                    if not geom.intersects(g2): 
                        continue
                    sector_overlap = geom.intersection(g2).area
                    overlap_area += sector_overlap

                    if 100 * sector_overlap / sector_area >= 70 or \
                    100.0 * overlap_area / sector_area >= tolerance:
                        keep = False
                        break
            keep_mask.append(keep)
        final_extend = cand.iloc[keep_mask].copy()

    # ---------- FINALIZE EXTEND ----------
    result = pd.concat([acc_secs, final_extend], ignore_index=True)
    result = gpd.GeoDataFrame(result, geometry="geometry", crs=sectors.crs)

    if "sector_id" in result.columns:
        result = result.drop_duplicates(subset=["sector_id"])
    else:
        result["_k"] = result.geometry.apply(lambda g: g.wkt if g is not None else None)
        result = result.drop_duplicates("_k").drop(columns="_k")

    # Clean internal cols
    for col in ["__area", "_k"]:
        if col in result.columns:
            result = result.drop(columns=col)

    result_gdf = gpd.GeoDataFrame(result, geometry="geometry", crs=sectors.crs)
    extend_id = final_extend["site_id"].astype(str).unique().tolist()
    return result_gdf.reset_index(drop=True), extend_id


def compile_geodashboard(directory: str):
    filelist = os.listdir(directory)
    filelist = [os.path.join(directory, file) for file in filelist if file.startswith("TBG") and file.endswith(".xlsx")]

    all_data = []
    for data in tqdm(filelist, total=len(filelist), desc="Process Excel File"):
        excel_data = pd.read_excel(data)
        if "Prov_Name" in excel_data.columns:
            excel_data = excel_data.rename(columns={"Prov_Name": "Province", "Kab_Name": "City"})
        all_data.append(excel_data)

    all_data = pd.concat(all_data)
    all_data_geom = gpd.points_from_xy(all_data['Long'], all_data['Lat'], crs="EPSG:4326")
    all_data_gdf = gpd.GeoDataFrame(all_data, geometry=all_data_geom)

    export_dir = f"{directory}/Compiled"
    os.makedirs(export_dir, exist_ok=True)

    if "Site_ID" in all_data_gdf.columns:
        all_data_gdf['Site_ID'] = all_data_gdf['Site_ID'].astype(str)

    all_data.to_excel(os.path.join(export_dir, "Compile Data GeoDashboard.xlsx"))
    all_data_gdf.to_parquet(os.path.join(export_dir, "Compile Data GeoDashboard.parquet"))
    print(f"ℹ️ Total compiled data : {len(all_data_gdf):,}")
    return all_data_gdf

def identify_nearest(source_gdf, distance=50):
    sitelist = gpd.read_parquet(f"{MAINDATA_DIR}/10. TBG Sitelist/Colopriming_Aug 2025/Sitelist TBG_Aug 2025_v.1.0.parquet")
    source_gdf = source_gdf.to_crs(epsg=3857)
    sitelist = sitelist.to_crs(epsg=3857)
    print(f"ℹ️ Total GeoDashboard: {len(source_gdf):,}")
    print(f"ℹ️ Total Sitelist    : {len(sitelist):,}")

    if "SiteId TBG" in sitelist.columns:
        sitelist = sitelist.rename(columns={"SiteId TBG": "site_id"})
    if "Sitename TBG" in sitelist.columns:
        sitelist = sitelist.rename(columns={"Sitename TBG": "site_name"})

    if 'index_right' in source_gdf.columns:
        source_gdf = source_gdf.drop(columns='index_right')

    if 'index_right' in sitelist.columns:
        sitelist = sitelist.drop(columns='index_right')

    sitelist = gpd.sjoin_nearest(sitelist, source_gdf[['Site_ID', 'geometry']], max_distance=distance, distance_col="dist_to_tbg")
    print(f"ℹ️ Total Sitelist Joined Nearest     : {len(sitelist):,}")
    sitelist = sitelist.drop_duplicates('geometry').reset_index(drop=True)
    print(f"ℹ️ Total Sitelist Dropped Duplicates : {len(sitelist):,}")

    list_id = set(source_gdf['Site_ID'].astype(str))
    sitelist['note'] = sitelist['site_id'].astype(str).apply(lambda x: "Match ID" if x in list_id else "Need Identify")
    return sitelist

def parallel_region(
    region_data:gpd.GeoDataFrame, 
    distance_fwa:int=500, 
    sector_angle:int=120, 
    sector_group:int=120, 
    threshold_sector:int=100, 
    max_workers:int=8, 
    method:str='maximize',
    clean_overlap:bool=False,
    extend_sector:bool=False,
    ):
    region = region_data['region'].mode().values[0]
    print(f"ℹ️ Region {region} | Total sites {len(region_data):,} | Processing...")

    hex_list    = identify_hexagon(region_data, type='convex')
    hex_list    = set(hex_list)
    # print(f"ℹ️ Total Hex Building: {len(hex_list):,}")

    buildings   = retrieve_building(hex_list)
    if not buildings.empty:
        buildings   = buildings.to_crs(epsg=3857)
        buffered = region_data.copy()
        buffered['geometry'] = buffered.geometry.buffer(distance_fwa)
        buildings = gpd.sjoin(buildings, buffered[['geometry']]).drop(columns='index_right').drop_duplicates('geometry').reset_index(drop=True)
        buildings['homepass_class'] = buildings["area_in_meters"].map(homepass_class)
        region_calc = count_homepass(region_data, buildings)
    else:
        region_calc = region_data.copy()
        region_calc['total_homepass'] = 0

    # CLEAN OVERLAPS
    # print(f"🌏 Region {region} | Clean Overlaps.")
    site_accept, site_dropped = clean_overlaps(region_calc, max_distance=distance_fwa)
    accept_list = site_accept['site_id'].unique().tolist()
    region_calc['note'] = region_calc['site_id'].apply(lambda site: "Accept" if site in accept_list else "Dropped")
    sites_point = region_calc.copy()

    # print(f"🌏 Sectorizing.")
    sectors = parallel_sectorize(
        region_calc,
        buildings,
        distance_fwa=distance_fwa,
        sector_angle=sector_angle,
        sector_group=sector_group,
        threshold_sector=threshold_sector,
        max_workers=max_workers,
    )

    # COUNT HOMEPASS SECTOR INITIAL
    homepass_sectors = gpd.sjoin(buildings, sectors[['geometry', 'sector_id']]).drop(columns='index_right')
    grouped_sectors = homepass_sectors.groupby('sector_id').size().reset_index(name='total_homepass')
    sectors = sectors.drop(columns='total_homepass').merge(grouped_sectors, on='sector_id')

    # CLEAN SECTOR OVERLAP
    if clean_overlap or extend_sector:
        sectors, dropped_secs = clean_sector_overlaps(sectors, accept_list, tolerance=10.0)

    # EXTEND SECTORS
    if extend_sector:
        # print(f"🌏 Region {region} | Extend Sectors.")
        extended_sectors, extend_ids = extend_sectors(site_accept, site_dropped, sectors)
        # print(f"🌏 Region {region} | Homepass Extended.")
        sites_point.loc[sites_point['site_id'].isin(extend_ids), "note"] = "Extend"
        sectors = extended_sectors.copy()

    # COUNT HOMEPASS SECTOR EXTENDED
    match method:
        case 'maximize':
            homepass_sectors = gpd.sjoin(buildings, sectors[['geometry', 'sector_id']]).drop(columns='index_right')
        case 'unique':
            homepass_sectors = gpd.sjoin(buildings, sectors[['geometry', 'sector_id']]).drop(columns='index_right')
            homepass_sectors = homepass_sectors.drop_duplicates('geometry').reset_index(drop=True)

    # AGGREGATE HOMEPASS CLASS TO SECTOR LEVEL
    grouped_sectors = homepass_sectors.groupby(['sector_id', 'homepass_class']).size().unstack(fill_value=0)
    count_total = homepass_sectors.groupby('sector_id').size().rename('total_homepass')
    for col in grouped_sectors.columns:
        sectors[f'hp_{col}'] = sectors['sector_id'].map(grouped_sectors[col]).fillna(0).astype(int)
    
    sectors = sectors.drop(columns='total_homepass')
    sectors['total_homepass'] = sectors['sector_id'].map(count_total).fillna(0).astype(int)

    # AGGREGATE SECTOR TO SITE LEVEL
    col_hp_class = [col for col in sectors.columns if col.startswith('hp_')]
    sector_summary = sectors.groupby('site_id').size().rename('total_sectors_utilized')
    sector_summary = sector_summary.to_frame()
    for col in col_hp_class:
        sector_summary_col = sectors.groupby('site_id')[col].sum().rename(f'{col}')
        sector_summary = sector_summary.join(sector_summary_col)
        if col in sites_point.columns:
            sites_point = sites_point.drop(columns=col)

    if 'total_homepass' in sites_point.columns:
        sites_point = sites_point.drop(columns='total_homepass')
    total_homepass_col = sectors.groupby('site_id')['total_homepass'].sum().rename('total_homepass')
    sector_summary = sector_summary.join(total_homepass_col)
    sites_point = sites_point.set_index('site_id').join(sector_summary).reset_index()
    
    
    # CALCULATE MARKET CLASS
    sites_point['market_class'] = sites_point['total_homepass'].map(classify_market)
    sectors['market_class'] = sectors['total_homepass'].map(classify_market)
    if (not sites_point.empty) & (not sectors.empty) & (not buildings.empty):
        # print(f"🟢 Region {region} done.")
        return sites_point, sectors, buildings
    else:
        print(f"🔴 Result empty for {region}. Please check the input data.")
    
# ===========
# WRAP IN API
# ===========
def validate_fwa(excel_file: str|pd.DataFrame|gpd.GeoDataFrame):
    used_col = ["site_id", "long", "lat"]
    if isinstance(excel_file, str):
        excel_df = pd.read_excel(excel_file)
    else:
        excel_df = excel_file

    excel_df = sanitize_header(excel_df, lowercase=True)
    for col in used_col:
        if col not in excel_df.columns:
            raise ValueError(f"{col} is not provided in Excel file. File columns {excel_df.columns}")
    
    return excel_df

def main_fwa(
    data_gdf: gpd.GeoDataFrame, 
    export_dir:str,
    export_building:bool=False, 
    method:str='maximize', 
    clean_overlap:bool=False,
    extend_sector:bool=False,
    sector_angle:int = 120, 
    sector_group:int = 120, 
    threshold:int = 300, 
    threshold_sector:int = 50, 
    distance_fwa:int = 500, 
    max_workers:int = 8,
    **kwargs
    ):

    # VALIDATE INPUT DATA
    data_gdf = validate_fwa(data_gdf)
    data_gdf = data_gdf.to_crs(epsg=3857)

    task_celery = kwargs.get("task_celery", None)

    start_time = time()
    group_factor = sector_group // sector_angle
    total_sectors = 360 // sector_group
    threshold_sector = threshold // total_sectors

    print(f"===============")
    print(f"CONFIGURATIONS:")
    print(f"===============")
    print(f"⚙️ Max Workers       : {max_workers}")
    print(f"⚙️ Extend Sectors    : {extend_sector}")
    print()
    print(f"===============")
    print(f"PARAMETERS:")
    print(f"===============")
    print(f"ℹ️ Distance FWA      : {distance_fwa}m")
    print(f"ℹ️ Sector Angle      : {sector_angle}°")
    print(f"ℹ️ Sector Group      : {sector_group}°")
    print(f"ℹ️ Threshold         : {threshold} HP")
    print(f"ℹ️ Threshold Sector  : {threshold_sector} HP")
    print(f"ℹ️ Group Factor      : {group_factor}")
    print(f"ℹ️ Total Sectors     : {total_sectors}")
    print()

    # GROUPS
    print(f"🌏 Grouping.")
    print(f"ℹ️ Total Sites to Process: {len(data_gdf):,}")
    groups = auto_group(data_gdf, distance=1000)
    if task_celery:
        task_celery.update_state(
            state="PROGRESS",
            meta={
                "status": f"Hold on, Grouping data for parallel processing.",
                "step": 1
            },
        )
    if "region" in data_gdf.columns:
        data_gdf = data_gdf.drop(columns='region')
    data_gdf = gpd.sjoin(data_gdf, groups[["geometry", "region"]]).drop(columns="index_right")
    grouped_data = data_gdf.groupby('region').size().reset_index(name='total_sites').sort_values('total_sites', ascending=False)
    print()

    # PROCESSING BY REGION
    region_list = grouped_data['region'].unique().tolist()
    mp_ctx = mp.get_context("spawn")

    with ProcessPoolExecutor(max_workers=max_workers, mp_context=mp_ctx) as executor:
        tasks_region = {}
        for region in region_list:
            region_data = data_gdf[data_gdf['region'] == region].copy()    
            future = executor.submit(parallel_region, region_data, distance_fwa=distance_fwa, sector_angle=sector_angle, sector_group=sector_group, threshold_sector=threshold_sector, max_workers=max_workers, method=method, clean_overlap=clean_overlap, extend_sector=extend_sector)
            tasks_region[future]=region
        print(f"ℹ️ All Task submitted.")
        if task_celery:
            task_celery.update_state(
                state="PROGRESS",
                meta={
                    "status": f"ℹ️ All task submitted.",
                    "step": 2
                },
            )
        site_result = []
        sector_result = []
        building_result = []
        processed = 0
        for future in tqdm(as_completed(tasks_region), total=len(tasks_region), desc="Process Region"):
            region = tasks_region[future]
            if task_celery:
                task_celery.update_state(
                    state="PROGRESS",
                    meta={
                        "status": f"✅ Processed region {region}.",
                        "step": 2,
                        "detail": f"Processed {processed}/{len(tasks_region)} regions."
                    },
                )
            try:
                result = future.result()
                processed += 1

                if result is None:
                    print(f"⚠️ Region {region} returned no result.")
                    continue
                
                if result:
                    sites_point, extended_sectors, buildings = result
                    site_result.append(sites_point)
                    sector_result.append(extended_sectors)
                    
                    if export_building:
                        building_result.append(buildings)
                    print(f"✅ Region {region} completed.")
            except Exception as e:
                print(f"🔴 Error in region {region}: {e}")

    # CONCAT
    print(f"🧩 Concate results.")
    site_result = pd.concat(site_result)
    sector_result = pd.concat(sector_result)
    if export_building:
        building_result = pd.concat(building_result)

    # EXPORT
    os.makedirs(export_dir, exist_ok=True)

    if task_celery:
        task_celery.update_state(
            state="PROGRESS",
            meta={
                "status": f"ℹ️ Export processed data into Parquet and Excel.",
                "step": 3,
            },
        )
    site_result.to_parquet(os.path.join(export_dir, f"Sitelist Processed_FWA_{distance_fwa}m.parquet"))
    sector_result.to_parquet(os.path.join(export_dir, f"Sectorized_FWA_{distance_fwa}m.parquet"))
    
    if export_building:
        building_result.to_parquet(os.path.join(export_dir, f"Buildings_FWA_{distance_fwa}m.parquet"))
    
    excel_path = os.path.join(export_dir, f"Summary FWA_{distance_fwa}m.xlsx")
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        site_result.drop(columns='geometry').to_excel(writer, sheet_name="Sitelist", index=False)
        sector_result.drop(columns='geometry').to_excel(writer, sheet_name="Sectors", index=False)

    # ZIPFILE
    if task_celery:
        task_celery.update_state(
            state="PROGRESS",
            meta={
                "status": f"ℹ️ Zipping final result, please wait.",
                "step": 4,
            },
        )
    zip_filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_FWA Alghorithm.zip"
    zip_filepath = os.path.join(export_dir, zip_filename)
    with zipfile.ZipFile(zip_filepath, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(export_dir):
            for file in files:
                if file != zip_filename and not file.endswith('.zip'):
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, export_dir)
                    zipf.write(file_path, arcname)
    print(f"📦 Result files zipped.")
    print(f"✅ All FWA process completed.")
    end_time = time()
    process_time = end_time - start_time

    if task_celery:
        task_celery.update_state(
            state="SUCCESS",
            meta={
                "status": f"✅ Fixed Wireless Access (FWA) processing completed in {round(process_time/60, 2)} minutes.",
                "step": 5,
                "zip_file": zip_filepath
            },
        )
    return zip_filepath


if __name__ == "__main__":
    directory = r"D:\JACOBS\TASK\SEPTEMBER\WEEK 5\FWA SURGE CLEAN OVERLAP\Data"
    # distance = 100

    # FWA PARAMS
    sector_angle = 120
    sector_group = 120
    threshold = 300
    distance_fwa = 500
    max_workers = 4
    group_factor = sector_group // sector_angle
    total_sectors = 360 // sector_group
    # threshold_sector = threshold // total_sectors
    threshold_sector = 100
    method = 'maximize' # 'maximize'/'unique'
    extend_sector = True
    clean_overlap = True
    export_building = True

    # EXPORT
    export_dir = r"D:\JACOBS\TASK\OKTOBER\Week 3\FWA SURGE\SITE_7K\Site 7k Sectorize Result"
    os.makedirs(export_dir, exist_ok=True)

    print(f"ℹ️ Using 7k Data.")
    sitelist = r"D:\JACOBS\TASK\OKTOBER\Week 3\FWA SURGE\Sitelist_7160.xlsx"
    output_path = os.path.join(export_dir, f"Sitelist FWA 7k_to_Process.parquet")
    # output_path = os.path.join(export_dir, f"TBG Sitelist_Identify Nearest GeoDashboard {distance}m Indonesia.parquet")
    if os.path.exists(output_path):
        print(f"🟢 Compiled data already exist. Load data.")
        sitelist = gpd.read_parquet(output_path)
    else:
        print(f"🟡 Compiled data not exist. Process compiling...")
        # geodata = compile_geodashboard(directory)
        # sitelist = identify_nearest(geodata, distance=distance)
        # sitelist = sitelist[sitelist['site_id'].astype(str).isin(data_v7)]
        # print(f"ℹ️ Sitelist in V7: {len(sitelist):,}")
        # sitelist.to_parquet(os.path.join(export_dir, f"TBG Sitelist_Identify Nearest GeoDashboard {distance}m Indonesia.parquet"))

        sitelist = read_gdf(sitelist)
        sitelist.columns = sitelist.columns.str.lower()
        # sitelist = sitelist[sitelist['island'].str.contains('JAWA|BALI|PUMA')]
        if 'sitename' in sitelist.columns:
            sitelist['sitename'] = sitelist['sitename'].astype(str)
        if 'tower height' in sitelist.columns:
            sitelist['tower height'] = sitelist['tower height'].astype(str)

        sitelist['site_id'] = sitelist['site_id'].astype(str)
        sitelist['long'] = sitelist.geometry.to_crs(epsg=4326).x
        sitelist['lat'] = sitelist.geometry.to_crs(epsg=4326).y
        sitelist.to_parquet(output_path)

    data_gdf = sitelist.copy()
    data_gdf['site_id'] = data_gdf['site_id'].astype(str)
    data_gdf = data_gdf.to_crs(epsg=3857)

    if 'index_right' in data_gdf.columns:
        data_gdf = data_gdf.drop(columns='index_right')
    if 'Site_ID' in data_gdf.columns:
        data_gdf = data_gdf.drop(columns='Site_ID')
    if "Site Type" in data_gdf.columns:
        data_gdf = data_gdf.rename(columns={"Site Type":"site_type"})

    # FILTER SALES SURGE
    # data_gdf = gpd.sjoin_nearest(data_gdf, surge_sales_gdf[['geometry']], max_distance=50, distance_col='dist_surge_sales').drop(columns='index_right')
    print(f"ℹ️ Total Sites to Process: {len(data_gdf):,}")
    
    start_time = time()
    fwa_result = main_fwa(
        data_gdf=data_gdf,
        export_dir=export_dir,
        export_building=export_building,
        method=method,
        sector_angle=sector_angle,
        sector_group=sector_group,
        threshold=threshold,
        threshold_sector=threshold_sector,
        distance_fwa=distance_fwa,
        max_workers=max_workers,
        extend_sector=extend_sector,
        clean_overlap=clean_overlap
    )
    end_time = time()
    process_time = round((end_time-start_time)/60,2)
    print(f"⌛ Process Time: {process_time} minutes.")