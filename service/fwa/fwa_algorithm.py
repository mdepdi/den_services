# ======================================================
# FIXED WIRELESS ACCESS (FWA) MAIN SCRIPT - CLEAN VERSION
# ======================================================
# Author  : Yakub Hariana (refactored)
# Purpose : FWA site clustering, sectorization, and
#           building coverage analysis (simplified logic)
# ======================================================

import os
import sys
import zipfile
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import multiprocessing as mp

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon, box
from tqdm import tqdm

# ======================================================
# PATHS & MODULE IMPORTS
# ======================================================
sys.path.append(r"D:\JACOBS\SERVICE\API")

from modules.table import excel_styler, sanitize_header  # noqa: F401 (styler optional)
from modules.data import read_gdf
from core.config import settings

MAINDATA_DIR = settings.MAINDATA_DIR
DATA_DIR = settings.DATA_DIR
EXPORT_DIR = settings.EXPORT_DIR


# ======================================================
# GEOMETRY UTILITIES
# ======================================================
def identify_hexagon(data_gdf: gpd.GeoDataFrame,
                     resolution: int = 5,
                     buffer: float = 10_000,
                     type: str = "bound") -> list[str]:
    """
    Identify intersected hexagon IDs covering the input data area.

    Parameters
    ----------
    data_gdf : GeoDataFrame (any CRS)
    resolution : int
        H3 resolution used in the parquet file name.
    buffer : float
        Buffer distance around bounds / convex hull (in meters, once projected).
    type : {"bound", "convex"}
        - "bound"  : use total_bounds bbox + buffer
        - "convex" : use convex hull of all geometries + buffer

    Returns
    -------
    list of hex IDs (strings)
    """
    hex_path = f"{MAINDATA_DIR}/22. H3 Hex/Hex_{resolution}.parquet"
    if not os.path.exists(hex_path):
        raise FileNotFoundError(f"Hexagon file not found at {hex_path}")

    hex_gdf = gpd.read_parquet(hex_path).to_crs("EPSG:3857")
    data_gdf = data_gdf.to_crs("EPSG:3857")

    if type == "bound":
        bbox = box(*data_gdf.total_bounds).buffer(buffer)
        mask = hex_gdf.intersects(bbox)
    elif type == "convex":
        hull = data_gdf.geometry.union_all().convex_hull.buffer(buffer)
        mask = hex_gdf.intersects(hull)
    else:
        raise ValueError("Invalid type: choose 'bound' or 'convex'.")

    hex_clip = hex_gdf[mask]
    if hex_clip.empty:
        raise ValueError("No hexagons found for the given area.")

    return hex_clip[f"hex_{resolution}"].unique().tolist()


def retrieve_building(hex_list: list[str],
                      centroid: bool = True,
                      hex_dir: str | None = None,
                      **kwargs) -> gpd.GeoDataFrame:
    """
    Retrieve building data per hex and filter by area/aspect ratio/one_unit.

    Filters (all optional via kwargs):
      - one_unit : bool        (default False)
      - area_building : bool   (default True)
      - aspect_ratio : bool    (default True)
      - parameters : dict
          {
            "aspect_ratio_value": 0.25,
            "area_building_value": {"min": 25, "max": 500}
          }
    """
    import shutil

    one_unit = kwargs.get("one_unit", False)
    area_building = kwargs.get("area_building", True)
    aspect_ratio = kwargs.get("aspect_ratio", True)
    parameters = kwargs.get(
        "parameters",
        {"aspect_ratio_value": 0.25, "area_building_value": {"min": 25, "max": 500}},
    )

    if hex_dir is None:
        hex_dir = f"{MAINDATA_DIR}/02. Building/Adm 2024/Hexed Building 2024"

    def load_hex(hex_id: str) -> gpd.GeoDataFrame | None:
        try:
            path = os.path.join(hex_dir, f"{hex_id}_buildings.parquet")
            if not os.path.exists(path):
                raise FileNotFoundError
            data = gpd.read_parquet(path).to_crs(epsg=3857)

            if centroid:
                data["geometry"] = data.geometry.centroid

            if aspect_ratio and "asp_ratio" in data.columns:
                data = data[data["asp_ratio"] > parameters["aspect_ratio_value"]]

            if area_building and "area_in_meters" in data.columns:
                amin = parameters["area_building_value"]["min"]
                amax = parameters["area_building_value"]["max"]
                data = data[
                    (data["area_in_meters"] > amin) &
                    (data["area_in_meters"] < amax)
                ]

            if one_unit and "one_unit" in data.columns:
                data = data[data["one_unit"] == 1]

            return data if not data.empty else None

        except Exception:
            # Try copy from central network folder if missing
            src_dir = r"Z:\01. DATABASE\02. Building\Adm 2024\Hexed Building 2024"
            src = os.path.join(src_dir, f"{hex_id}_buildings.parquet")
            dst = os.path.join(hex_dir, f"{hex_id}_buildings.parquet")
            if os.path.exists(src):
                shutil.copy(src, dst)
                print(f"ℹ️ Copied {hex_id} from Z:")
                return load_hex(hex_id)
            return None

    results: list[gpd.GeoDataFrame] = []
    with ThreadPoolExecutor() as exe:
        futures = {exe.submit(load_hex, h): h for h in hex_list}
        for f in as_completed(futures):
            r = f.result()
            if r is not None:
                results.append(r)

    if not results:
        print("⚠️ No building data found.")
        return gpd.GeoDataFrame(columns=["geometry"], crs="EPSG:3857")

    all_data = pd.concat(results, ignore_index=True).drop_duplicates("geometry")
    return gpd.GeoDataFrame(all_data, geometry="geometry", crs="EPSG:3857")


# ======================================================
# SITE UTILITIES
# ======================================================
def auto_group(data_gdf: gpd.GeoDataFrame,
               expand_gdf: gpd.GeoDataFrame | None = None,
               distance: float = 10_000) -> gpd.GeoDataFrame:
    """
    Auto-group sites based on buffered overlap.

    - Buffer each point by `distance`
    - Dissolve overlapping buffers into region polygons
    - Each polygon becomes a 'region'
    """
    if data_gdf.crs != "EPSG:3857":
        data_gdf = data_gdf.to_crs(epsg=3857)

    base = data_gdf[["geometry"]].copy()

    if expand_gdf is not None and not expand_gdf.empty:
        expand_gdf = expand_gdf[["geometry"]]
        base = pd.concat([base, expand_gdf])

    groups = base.copy()
    groups["geometry"] = groups.geometry.buffer(distance)
    groups = groups.dissolve().explode(ignore_index=True)

    groups["region"] = groups.index + 1
    print(f"ℹ️ Total Group generated: {len(groups)}")

    return groups


def count_homepass(site_gdf: gpd.GeoDataFrame,
                   building: gpd.GeoDataFrame,
                   distance: float = 500) -> gpd.GeoDataFrame:
    """
    Count building points within a buffer of each site (homepass).
    """
    site_gdf = site_gdf.to_crs(3857).copy()
    building = building.to_crs(3857).copy()

    building["geometry"] = building.geometry.centroid
    site_gdf["__buffer"] = site_gdf.geometry.buffer(distance)

    joined = gpd.sjoin(
        building,
        site_gdf.set_geometry("__buffer"),
        predicate="intersects",
        how="inner",
    )

    count = joined.groupby("index_right").size().rename("total_homepass")
    site_gdf["total_homepass"] = (
        site_gdf.index.map(count).fillna(0).astype(int)
    )

    site_gdf = site_gdf.set_geometry("geometry").drop(columns="__buffer")
    return site_gdf

# =====
# SCORE
# =====
def compute_priority_score(
    row: pd.Series,
    score_map: dict,
    numeric_cols: dict | None = None,
) -> tuple:
    """
    Build a score tuple for a row based on:
    - score_map: dict[column] -> { value: weight, "__default__": weight }
    - numeric_cols: dict[column] -> multiplier

    Higher tuple is more prioritized (lexicographic compare).
    """
    parts = []

    # 1) categorical mapped scores
    if "__protected" in row:
        protect = row.get("__protected", 0)
        parts.append(protect)

    for col, mapping in score_map.items():
        default_weight = mapping.get("__default__", 0)
        value = row.get(col)

        # use mapped weight or default
        weight = mapping.get(value, default_weight)
        parts.append(weight)

    # 2) numeric columns directly (e.g. total_homepass)
    if numeric_cols:
        for col, multiplier in numeric_cols.items():
            val = row.get(col, 0)
            try:
                val = float(val)
            except Exception:
                val = 0.0
            parts.append(val * multiplier)

    return tuple(parts)


def clean_sites_overlaps(
    sites: gpd.GeoDataFrame,
    max_distance: float = 300,
    tolerance: float = 10.0,
    protect_list=None,
    hp_col: str = "total_homepass",
    score_map: dict = {}
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Clean overlapping site buffers.

    Rules:
    - Build a buffer around each site (max_distance).
    - If two buffers overlap more than `tolerance` % (on either buffer),
      one site is dropped.
    - Priority to KEEP:
        1) Protected sites (in protect_list)
        2) Higher homepass (hp_col)

    Returns
    -------
    kept : GeoDataFrame
        Sites that survive the cleaning.
    dropped : GeoDataFrame
        Sites that are removed.
    """
    if sites.empty:
        return sites.copy(), sites.iloc[0:0].copy()

    gdf = sites.to_crs(3857).copy().reset_index(drop=True)

    if hp_col not in gdf.columns:
        gdf[hp_col] = 0

    if "site_id" not in gdf.columns:
        raise ValueError("clean_sites_overlaps requires a 'site_id' column")

    protect_set = set(str(x) for x in (protect_list or []))
    gdf["__protected"] = gdf["site_id"].astype(str).isin(protect_set)

    # buffers & area
    gdf["__buf"] = gdf.geometry.buffer(max_distance)
    gdf["__area"] = gdf["__buf"].area

    buf_gdf = gpd.GeoDataFrame(
        {"site_idx": gdf.index, "geometry": gdf["__buf"]},
        geometry="geometry",
        crs=gdf.crs,
    )

    joined = gpd.sjoin(
        buf_gdf,
        buf_gdf,
        how="inner",
        predicate="intersects",
        lsuffix="l",
        rsuffix="r",
    )
    joined = joined[joined["site_idx_l"] < joined["site_idx_r"]]

    if joined.empty:
        print("ℹ️ No overlapping sites found.")
        kept = gdf.drop(columns=["__buf", "__area", "__protected"])
        return kept.reset_index(drop=True), gdf.iloc[0:0].copy()

    conflict_pairs: list[tuple[int, int]] = []
    def site_row_score(idx):
        row = gdf.iloc[idx]
        return compute_priority_score(
            row,
            score_map=score_map,
            numeric_cols={"total_homepass" : 1},
        )

    for _, row in joined.iterrows():
        i = int(row["site_idx_l"])
        j = int(row["site_idx_r"])

        inter = gdf.at[i, "__buf"].intersection(gdf.at[j, "__buf"])
        if inter.is_empty:
            continue

        inter_area = inter.area
        if inter_area == 0:
            continue

        area_i = gdf.at[i, "__area"]
        area_j = gdf.at[j, "__area"]

        pct_i = 100 * inter_area / area_i if area_i > 0 else 0
        pct_j = 100 * inter_area / area_j if area_j > 0 else 0

        if max(pct_i, pct_j) >= tolerance:
            conflict_pairs.append((i, j))

    if not conflict_pairs:
        print("ℹ️ Overlaps exist but below tolerance.")
        kept = gdf.drop(columns=["__buf", "__area", "__protected"])
        return kept.reset_index(drop=True), gdf.iloc[0:0].copy()



    dropped_idx: set[int] = set()
    for i, j in conflict_pairs:
        if i in dropped_idx or j in dropped_idx:
            continue

        # if both protected, keep both
        if gdf.at[i, "__protected"] and gdf.at[j, "__protected"]:
            continue

        score_i = site_row_score(i)
        score_j = site_row_score(j)
        
        loser = j if  score_i>= score_j else i
        dropped_idx.add(loser)

    keep_idx = [i for i in gdf.index if i not in dropped_idx]

    kept = gdf.loc[keep_idx].drop(columns=["__buf", "__area", "__protected"])
    dropped = gdf.loc[list(dropped_idx)].drop(columns=["__buf", "__area", "__protected"])

    print(f"ℹ️ Sites kept: {len(kept)}, Sites dropped: {len(dropped)}")
    return kept.reset_index(drop=True), dropped.reset_index(drop=True)


# ======================================================
# CLASSIFICATION UTILITIES
# ======================================================
def classify_market(x: int | float) -> str:
    if x > 700:
        return "P1"
    if x > 500:
        return "P2"
    if x > 300:
        return "P3"
    return "P4"


def homepass_class(x: int | float) -> str:
    if x > 120:
        return "high"
    if x > 70:
        return "high"
    if x > 45:
        return "medium"
    if x > 36:
        return "low"
    return "very low"


# ======================================================
# SECTOR UTILITIES
# ======================================================
def generate_sector(center, buffer_distance: float,
                    sector_angle: int = 90,
                    rotation_angle: int = 0) -> gpd.GeoDataFrame:
    """
    Generate radial sector polygons around a point (center) with given angle.
    """
    x, y = center.x, center.y
    full_buffer = center.buffer(buffer_distance)
    num_sectors = 360 // sector_angle
    radius = buffer_distance * 1.25  # slightly larger to ensure coverage

    sectors = []
    for i in range(num_sectors):
        start = i * sector_angle + rotation_angle
        end = (i + 1) * sector_angle + rotation_angle

        angles = np.linspace(np.radians(start), np.radians(end), 50)
        arc = list(zip(
            x + radius * np.sin(angles),
            y + radius * np.cos(angles),
        ))
        poly = Polygon([(x, y), *arc, (x, y)])
        sectors.append(
            {
                "geometry": full_buffer.intersection(poly),
                "azimuth": ((start + end) / 2) - 360 if ((start + end) / 2) >=360 else ((start + end) / 2),
                "azimuth_start": start,
                "azimuth_end": end - 360 if end >=360 else end,
            }
        )

    return gpd.GeoDataFrame(sectors, geometry="geometry", crs="EPSG:3857")


def sectorize_site(args):
    """
    Sectorize a single site by rotating sectors and choosing the best HP balance.
    """
    site_data, homepass, dist, angle, threshold = args
    site_id = site_data["site_id"]
    geom = site_data["geometry"]
    tower_type = site_data.get("tower_type", "NA")

    best_score = -np.inf
    best_sectors = None

    for offset in range(0, angle, 1):
        sectors = generate_sector(geom, dist, angle, offset)
        sectors["site_id"] = site_id
        sectors["tower_type"] = tower_type
        sectors["sector_id"] = [f"{site_id}_{i+1}" for i in range(len(sectors))]

        if not homepass.empty:
            joined = gpd.sjoin(
                homepass,
                sectors[["geometry", "sector_id"]],
                predicate="intersects",
                how="inner",
            )
            counts = joined.groupby("sector_id").size()
            sectors["total_homepass"] = sectors["sector_id"].map(counts).fillna(0)
        else:
            sectors["total_homepass"] = 0

        sectors["__differ"] = abs(sectors["total_homepass"] - threshold)
        valid = sectors["total_homepass"] >= threshold
        valid_count = valid.sum()
        max_differ = sectors.loc[valid, "__differ"].max() if valid.any() else 0
        score = (valid_count * 1000) - max_differ

        if score > best_score:
            best_score = score
            best_sectors = sectors.copy()

    best_sectors.drop(columns="__differ", inplace=True, errors="ignore")
    return best_sectors


def parallel_sectorize(
    sites_gdf: gpd.GeoDataFrame,
    homepass: gpd.GeoDataFrame,
    distance: float = 300,
    angle: int = 120,
    group: int = 120,
    threshold: int = 100,
    max_workers: int = 4,
) -> gpd.GeoDataFrame:
    """
    Parallel sector generation for all sites.
    """
    if sites_gdf.empty:
        return gpd.GeoDataFrame(columns=["geometry"], crs="EPSG:3857")

    sites = sites_gdf.to_crs(3857).reset_index(drop=True)
    homepass = homepass.to_crs(3857).reset_index(drop=True)

    sindex = homepass.sindex
    site_args = []

    for _, s in sites.iterrows():
        buff = s.geometry.buffer(distance)
        idx = sindex.query(buff, predicate="intersects")
        hp = homepass.iloc[idx].reset_index(drop=True)
        site_args.append(
            (s, hp, distance, angle, threshold)
        )

    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as exe:
        fut_map = {exe.submit(sectorize_site, a): a[0]["site_id"] for a in site_args}
        for f in tqdm(
            as_completed(fut_map),
            total=len(fut_map),
            desc="Sectorizing",
        ):
            r = f.result()
            if r is not None and not r.empty:
                results.append(r)

    if not results:
        return gpd.GeoDataFrame(columns=["geometry"], crs="EPSG:3857")

    all_sectors = pd.concat(results, ignore_index=True)
    return gpd.GeoDataFrame(all_sectors, geometry="geometry", crs="EPSG:3857")


def _assign_buildings_unique(
    buildings: gpd.GeoDataFrame,
    sectors: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """
    Assign each building to exactly one sector.

    Rule:
    - If a building intersects multiple sectors, assign it
      to the sector with highest total_homepass.
    """
    if buildings.empty or sectors.empty:
        return gpd.GeoDataFrame(columns=["geometry"], crs=buildings.crs if not buildings.empty else None)

    join = gpd.sjoin(
        buildings,
        sectors[["geometry", "sector_id", "site_id", "total_homepass"]],
        how="inner",
        predicate="intersects",
    )

    if join.empty:
        return gpd.GeoDataFrame(columns=["geometry"], crs=buildings.crs)

    join = join.reset_index().rename(columns={"index": "building_idx"})
    join_sorted = join.sort_values(
        by=["building_idx", "total_homepass"],
        ascending=[True, False],
    )
    chosen = join_sorted.drop_duplicates(subset="building_idx")

    building_acc = gpd.GeoDataFrame(
        chosen.drop(columns=["index_right"]),
        geometry="geometry",
        crs=buildings.crs,
    )
    return building_acc.reset_index(drop=True)


def clean_sectors_overlaps(
    site_data: gpd.GeoDataFrame,
    sectors: gpd.GeoDataFrame,
    buildings: gpd.GeoDataFrame,
    accepted_list: list,
    score_map: dict,
    tolerance: float = 10.0,
    **_,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Simplified sector overlap cleaning.

    Rules:
    - Sectors whose site_id ∈ accepted_list are "protected".
    - Compute total_homepass per sector from buildings if needed.
    - For each pair of overlapping sectors from different sites:
        * If overlap % (vs each) > tolerance:
            - Compare scores:
              (protected flag, total_homepass, sector_index)
              where sector_index: 3 > 2 > 1
            - Keep higher score, drop the other.
    - Then assign each building uniquely to one sector.

    Result:
    - No remaining cross-site sector pair overlaps above tolerance.
    - For the same site, sectors are allowed to overlap.
    """
    if sectors.empty:
        return sectors, sectors, buildings

    sectors = sectors.to_crs(3857).reset_index(drop=True)
    buildings = buildings.to_crs(3857).reset_index(drop=True)
    site_data = site_data.to_crs(3857).reset_index(drop=True)
    score_keys = ["site_id"] + [key for key in list(score_map.keys()) if key not in sectors.columns]
    sectors = sectors.merge(site_data[score_keys], on="site_id", how="inner")

    # -------------------------
    # 1) Protected flag
    # -------------------------
    accepted_set = set(str(s) for s in (accepted_list or []))
    if "site_id" not in sectors.columns:
        raise ValueError("clean_sectors_overlaps requires 'site_id' in sectors")

    sectors["__protected"] = sectors["site_id"].astype(str).isin(accepted_set)
    sectors["__sst"] = np.where(sectors["tower_type"].str.lower().str.contains("sst"), 1, 0)

    # -------------------------
    # 2) Ensure total_homepass exists
    # -------------------------
    if "total_homepass" not in sectors.columns or sectors["total_homepass"].isna().all():
        join_all = gpd.sjoin(
            buildings,
            sectors[["geometry", "sector_id"]],
            how="inner",
            predicate="intersects",
        ).drop(columns="index_right", errors="ignore")

        if join_all.empty:
            sectors["total_homepass"] = 0
        else:
            hp_per_sector = (
                join_all.groupby("sector_id")
                .size()
                .rename("total_homepass")
                .astype(int)
            )
            sectors["total_homepass"] = (
                sectors["sector_id"].map(hp_per_sector).fillna(0).astype(int)
            )

    # -------------------------
    # 3) Extract sector index (1/2/3) from sector_id
    # -------------------------
    def _parse_sector_index(s):
        try:
            return int(str(s).split("_")[-1])
        except Exception:
            return 1

    sectors["__sector_index"] = sectors["sector_id"].map(_parse_sector_index)

    # -------------------------
    # 4) Build pair list (cross-site overlaps)
    # -------------------------
    sec_gdf = gpd.GeoDataFrame(
        {"sec_idx": sectors.index, "geometry": sectors["geometry"]},
        geometry="geometry",
        crs=sectors.crs,
    )

    joined = gpd.sjoin(
        sec_gdf,
        sec_gdf,
        how="inner",
        predicate="intersects",
        lsuffix="l",
        rsuffix="r",
    )
    # unique unordered pairs
    joined = joined[joined["sec_idx_l"] < joined["sec_idx_r"]]

    if joined.empty:
        building_acc = _assign_buildings_unique(buildings, sectors)
        return (
            sectors.drop(columns=["__protected", "__sector_index"]),
            sectors.iloc[0:0].copy(),
            building_acc,
        )

    conflict_pairs: list[tuple[int, int]] = []

    for _, row in joined.iterrows():
        i = int(row["sec_idx_l"])
        j = int(row["sec_idx_r"])

        site_i = sectors.at[i, "site_id"]
        site_j = sectors.at[j, "site_id"]

        # Same-site sectors are allowed to overlap
        if site_i == site_j:
            continue

        geom_i = sectors.at[i, "geometry"]
        geom_j = sectors.at[j, "geometry"]

        inter = geom_i.intersection(geom_j)
        if inter.is_empty:
            continue

        inter_area = inter.area
        if inter_area == 0:
            continue

        area_i = geom_i.area
        area_j = geom_j.area

        pct_i = 100 * inter_area / area_i if area_i > 0 else 0
        pct_j = 100 * inter_area / area_j if area_j > 0 else 0

        if max(pct_i, pct_j) >= tolerance:
            conflict_pairs.append((i, j))

    # -------------------------
    # 5) Resolve conflicts
    #    Score: protected > total_homepass > sector_index
    #    sector_index: 3 > 2 > 1
    # -------------------------
    dropped_idx: set[int] = set()

    def sector_row_score(row: pd.Series) -> tuple:
        site_score = compute_priority_score(
            row,
            score_map=score_map,
            numeric_cols={"hp_site": 1, "total_homepass" : 1},
        )
        sector_score = (int(row['__protected']), *site_score)
        return sector_score

    for i, j in conflict_pairs:
        if i in dropped_idx or j in dropped_idx:
            continue

        row_i = sectors.loc[i]
        row_j = sectors.loc[j]
        if row_i["__protected"] == 1 and row_j["__protected"] == 1:
            continue

        score_i = sector_row_score(row_i)
        score_j = sector_row_score(row_j)

        # keep higher score, drop lower
        loser = j if score_i >= score_j else i
        dropped_idx.add(loser)

    keep_idx = [i for i in sectors.index if i not in dropped_idx]
    accepted = sectors.loc[keep_idx].drop(columns="__protected")
    dropped = sectors.loc[list(dropped_idx)].drop(columns="__protected")

    accepted = accepted.reset_index(drop=True)
    dropped = dropped.reset_index(drop=True)

    # clean up helper column not needed outside
    # accepted = accepted.drop(columns="__sector_index", errors="ignore")
    # dropped = dropped.drop(columns="__sector_index", errors="ignore")

    # -------------------------
    # 6) Unique building assignment
    # -------------------------
    building_acc = _assign_buildings_unique(buildings, accepted)

    print(f"✅ Final accepted sectors: {len(accepted)} | Dropped: {len(dropped)} ")
    return accepted, dropped, building_acc


# ======================================================
# REGION PROCESSING
# ======================================================
def parallel_region(
    region_data: gpd.GeoDataFrame,
    distance_fwa: float = 500,
    sector_angle: int = 120,
    sector_group: int = 120,
    threshold_sector: int = 100,
    max_workers: int = 8,
    clean_overlap: bool = False,
    accepted_ids: set[str] | None = None,
    export_path: str | None = None,
    score_map: dict = {}
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Process a single region:
    - Load buildings by hexagon
    - Filter buildings within FWA buffer
    - Count homepass per site
    - Clean site overlaps
    - Sectorize sites
    - Optionally clean sector overlaps
    - Aggregate sector stats back to site level
    """
    region = region_data["region"].mode().values[0]
    print(f"🧩 Region {region} | {len(region_data):,} sites")

    try:
        # 1) Get buildings by hex coverage
        hex_list = identify_hexagon(region_data, type="convex")
        buildings = retrieve_building(hex_list).to_crs(3857)

        if buildings.empty:
            print(f"⚠️ Region {region}: no buildings found.")
            return region_data, gpd.GeoDataFrame(), gpd.GeoDataFrame()

        # 2) Filter buildings to FWA buffer around region sites
        buffered = region_data.to_crs(3857).copy()
        buffered["geometry"] = buffered.geometry.buffer(distance_fwa)
        buildings = gpd.sjoin(
            buildings,
            buffered[["geometry"]],
            how="inner",
            predicate="intersects",
        ).drop(columns="index_right", errors="ignore").drop_duplicates("geometry")

        # 3) Count homepass per site & clean sites
        region_calc = count_homepass(region_data, buildings, distance=distance_fwa)
        if accepted_ids is not None:
            site_keep, site_drop = clean_sites_overlaps(
                region_calc,
                max_distance=distance_fwa,
                protect_list=accepted_ids,
                score_map=score_map
            )
            print(
                f"🌏 Region {region} | Using provided accepted sites: "
                f"{len(site_keep):,} accepted, {len(site_drop):,} dropped."
            )
        else:
            site_keep, site_drop = clean_sites_overlaps(
                region_calc,
                max_distance=distance_fwa,
                score_map=score_map
            )
            print(
                f"🌏 Region {region} | Accepted Sites: {len(site_keep):,}, "
                f"Dropped Sites: {len(site_drop):,}"
            )

        accepted_list = set(site_keep["site_id"].astype(str))
        region_calc["note"] = np.where(
            region_calc["site_id"].astype(str).isin(accepted_list),
            "Main Selected",
            "Overlap with Others",
        )
        region_calc['hp_site'] = region_calc['total_homepass']
        
        # 4) Sectorize
        sectors = parallel_sectorize(
            sites_gdf=region_calc,
            homepass=buildings,
            distance=distance_fwa,
            angle=sector_angle,
            group=sector_group,
            threshold=threshold_sector,
            max_workers=max_workers,
        )
        sectors = sectors.merge(region_calc[['site_id', 'hp_site']], how='left', on='site_id')

        if sectors.empty:
            print(f"⚠️ Region {region}: no sectors generated.")
            return region_calc, sectors, buildings

        # 5) Sector overlap cleaning + building assignment
        if clean_overlap:
            sectors_accept, sectors_dropped, _ = clean_sectors_overlaps(
                site_data=region_calc,
                sectors=sectors,
                buildings=buildings,
                accepted_list=list(accepted_list),
                sector_total=360 // sector_group,
                tolerance=10.0,
                score_map=score_map
            )
            building_join = gpd.sjoin(
                buildings,
                sectors[["geometry", "site_id", "sector_id"]],
                how="inner",
                predicate="intersects",
            ).drop(columns="index_right", errors="ignore")
        else:
            building_join = gpd.sjoin(
                buildings,
                sectors[["geometry", "site_id", "sector_id"]],
                how="inner",
                predicate="intersects",
            ).drop(columns="index_right", errors="ignore")

        # 6) Aggregate homepass class to sector level
        accepted = set(sectors_accept['sector_id'].astype(str))
        accepted_ids = set(sectors_accept['site_id'].astype(str))
        sectors['sector_note'] = np.where(
            sectors["sector_id"].astype(str).isin(accepted),
            "Accepted Sector",
            "Dropped Sector",
        )
        region_calc['note'] = np.where(
            region_calc["site_id"].astype(str).isin(accepted_ids),
            "Main Selected",
            "Overlap with Others",
        )
        total_hp_sector = (building_join.groupby("sector_id").size().rename("total_homepass"))
        sectors["total_homepass"] = (sectors["sector_id"].map(total_hp_sector).fillna(0).astype(int))

        # 7) Aggregate sector stats to site level
        agg_dict = {"sector_id": "count", "total_homepass": "sum"}
        sector_summary = (
            sectors[sectors['sector_note'] == 'Accepted Sector'].groupby("site_id")
            .agg(agg_dict)
            .rename(columns={"sector_id": "total_sectors"})
            .reset_index()
        )

        # Drop old total_homepass from region_calc and join new one
        if "total_homepass" in region_calc.columns:
            region_calc = region_calc.drop(columns="total_homepass")

        region_calc = (region_calc.merge(sector_summary, on="site_id", how="left"))

        # 8) Classification
        region_calc["market_class"] = region_calc["total_homepass"].map(classify_market)
        sectors["market_class"] = sectors["total_homepass"].map(classify_market)

        # 9) Optional checkpoint export
        if export_path:
            checkpoint_dir = os.path.join(export_path, "Checkpoint")
            os.makedirs(checkpoint_dir, exist_ok=True)
            region_calc.to_parquet(os.path.join(checkpoint_dir, f"Sitelist_Region_{region}.parquet"))
            sectors.to_parquet(os.path.join(checkpoint_dir, f"Sectors_Region_{region}.parquet"))
            building_join.to_parquet(os.path.join(checkpoint_dir, f"Buildings_Region_{region}.parquet"))

        return region_calc, sectors, building_join

    except Exception as e:
        print(f"❌ Exception in Region {region}: {e}")
        return region_data, gpd.GeoDataFrame(), gpd.GeoDataFrame()


# ======================================================
# VALIDATION & MAIN FWA PIPELINE
# ======================================================
def validate_fwa(excel_file: str | pd.DataFrame) -> pd.DataFrame:
    """
    Check if file or dataframe contains minimum FWA input columns.
    Required: site_id, long, lat
    """
    df = pd.read_excel(excel_file) if isinstance(excel_file, str) else excel_file
    df = sanitize_header(df, lowercase=True)
    for col in ["site_id", "long", "lat"]:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")
    return df


def main_fwa(
    data_gdf: gpd.GeoDataFrame,
    export_dir: str,
    export_building: bool = False,
    clean_overlap: bool = False,
    sector_angle: int = 120,
    sector_group: int = 120,
    threshold: int = 300,
    threshold_sector: int | None = None,
    distance_fwa: float = 500,
    max_workers: int = 8,
    accepted_ids: set[str] | None = None,
    score_map: dict = {}
) -> str:
    """
    Main FWA pipeline:
    - Group sites into regions
    - Process each region in parallel
    - Export Parquet + Excel + Zip

    Returns
    -------
    zip_path : str
        Path to zipped FWA result.
    """
    start = time.time()
    total_sites = len(data_gdf)

    total_sectors = 360 // sector_group
    if threshold_sector is None:
        threshold_sector = threshold // total_sectors

    print(f"ℹ️ Total Sitelist to Process : {total_sites:,}")
    print(f"ℹ️ Clean Sector Overlaps     : {clean_overlap}")

    # 1) Group sites into regions
    print("🧩 GROUPING SITES...")
    groups = auto_group(data_gdf, distance=distance_fwa)
    data_gdf = gpd.sjoin(
        data_gdf,
        groups[["geometry", "region"]],
        how="inner",
        predicate="intersects",
    ).drop(columns="index_right")

    regions = data_gdf["region"].unique().tolist()

    # 2) Parallel region processing
    mp_ctx = mp.get_context("spawn")
    site_result, sector_result, building_result = [], [], []

    with ProcessPoolExecutor(max_workers=max_workers, mp_context=mp_ctx) as executor:
        futures = {
            executor.submit(
                parallel_region,
                data_gdf[data_gdf["region"] == r],
                distance_fwa,
                sector_angle,
                sector_group,
                threshold_sector,
                max_workers,
                clean_overlap,
                accepted_ids,
                export_dir,
                score_map
            ): r
            for r in regions
        }

        for f in tqdm(as_completed(futures), total=len(futures), desc="Process Region"):
            region_id = futures[f]
            try:
                region_sites, region_sectors, region_buildings = f.result()
                site_result.append(region_sites)
                sector_result.append(region_sectors)
                if export_building and not region_buildings.empty:
                    building_result.append(region_buildings)
                print(f"✅ Region {region_id} done.")
            except Exception as e:
                print(f"🔴 Error in region {region_id}: {e}")

    # 3) Concatenate results
    site_df = pd.concat(site_result, ignore_index=True)
    sector_df = pd.concat(sector_result, ignore_index=True)

    if export_building and building_result:
        building_df = pd.concat(building_result, ignore_index=True)
    else:
        building_df = None

    os.makedirs(export_dir, exist_ok=True)
    site_df.to_parquet(f"{export_dir}/Sitelist_FWA.parquet")
    sector_df.to_parquet(f"{export_dir}/Sectors_FWA.parquet")
    if building_df is not None:
        building_df.to_parquet(f"{export_dir}/Buildings_FWA.parquet")

    # 4) Excel export (no geometry)
    with pd.ExcelWriter(f"{export_dir}/Summary_FWA.xlsx", engine="openpyxl") as w:
        site_df.drop(columns="geometry", errors="ignore").to_excel(
            w, sheet_name="Sites", index=False
        )
        sector_df.drop(columns="geometry", errors="ignore").to_excel(
            w, sheet_name="Sectors", index=False
        )

    # 5) Zip all outputs
    zip_path = os.path.join(export_dir, f"FWA_{datetime.now():%Y%m%d_%H%M}.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(export_dir):
            for f in files:
                if not f.endswith(".zip"):
                    full = os.path.join(root, f)
                    zipf.write(full, arcname=os.path.relpath(full, export_dir))

    elapsed_min = round((time.time() - start) / 60, 2)
    print(f"📦 Results zipped to {zip_path}")
    print(f"✅ FWA completed for {total_sites:,} sites in {elapsed_min} mins.")
    return zip_path


# ======================================================
# EXECUTION SAMPLE
# ======================================================
if __name__ == "__main__":
    sector_angle = 120
    sector_group = 120
    distance_fwa = 500
    threshold = 800
    threshold_sector = 300
    max_workers = 16 

    score_map = {
        "batch_list": {
            "Batch 1": 10,
            "Batch 2": 10,
            "Not Yet": 5,
            "Non TBG":2,
            "__default__":1
        },
        "status_sites_2": {
            "Ready for Integration (RFI)": 10,
            "Construction in Progress (CIP)": 7,
            "Ready for Construction (RFC) - Sitac": 5,
            "Ready for Construction (RFC) - Alfa​": 3,
            "Ready for Construction (RFC) - Non Alfa": 2,
            "Non TBG": 1,
            "Dismantle": 1,
            "__default__":1
        },
        "tower_type": {
            "SST": 5,
            "MONO_POLE": 1,
            "MONOPOLE": 1,
            "POLE":1,
            "__default__":1
        },
        "site_type": {
            "GREEN FIELD": 3,
            "GREENFIELD": 3,
            "ROOF TOP": 1,
            "__default__":1
        },
        "remark_data": {
            "Sitelist 37k": 3,
            "STIP TBG 2025": 2,
            "STIP PKP 2025": 2,
            "Plan B2S (FWA)": 2,
            "__default__":1
        },
        "company": {
            "TBG":5,
            "PKP": 2,
            "PKP (ALFA)": 2,
            "GIHON": 2,
            "__default__":1
        },
        # "remark_sitelist": {
        #     "Main Selected": 10,
        #     "Overlaping with Others": 1,
        # },
    }

    # SECTOR_SCORE_MAP = {
    #     "Site Type": {
    #         "GREEN FIELD": 3,
    #         "ROOF TOP": 1,
    #     },
    #     "Tower Type": {
    #         "SST": 3,
    #         "MONO_POLE": 1,
    #     },
    # }

    # SITES_NUMERIC_COLS = {"total_homepass": 1.0}

    export_dir = r"D:\JACOBS\TASK\DESEMBER\Week 5\FWA Surge 45k Sites\FWA Sectorization 45k v3"
    os.makedirs(export_dir, exist_ok=True)

    sitelist_path = r"D:\JACOBS\TASK\DESEMBER\Week 5\FWA Surge 45k Sites\Sitelist Only FWA_45k_v3.xlsx"
    sitelist = read_gdf(sitelist_path, sheet_name='Sitelist')
    sitelist = sanitize_header(sitelist, lowercase=True)
    print(sitelist.columns)
    sitelist["company"] = sitelist["company"].astype(str).str.upper().str.strip()
    sitelist["tower_type"] = sitelist["tower_type"].astype(str).str.upper().str.strip()
    sitelist["site_id"] = sitelist["site_id"].astype(str)
    sitelist["long"] = sitelist.geometry.to_crs(4326).x
    sitelist["lat"] = sitelist.geometry.to_crs(4326).y
    sitelist["site_type"] = sitelist["site_type"].fillna("unknown")
    # sitelist["tower_type"] = sitelist["tower_type"].fillna("unknown")

    sitelist = sitelist.drop_duplicates("site_id").reset_index(drop=True)
    print(f"ℹ️ Total Sitelist to Process: {len(sitelist):,}")

    accepted_ids = None

    data_gdf = sitelist.to_crs(3857)
    main_fwa(
        data_gdf=data_gdf,
        export_dir=export_dir,
        export_building=True,
        clean_overlap=True,
        sector_angle=sector_angle,
        sector_group=sector_group,
        threshold=threshold,
        threshold_sector=threshold_sector,
        distance_fwa=distance_fwa,
        max_workers=max_workers,
        accepted_ids=accepted_ids,
        score_map=score_map
    )
