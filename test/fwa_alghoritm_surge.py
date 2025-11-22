# ======================================================
# FIXED WIRELESS ACCESS (FWA) MAIN SCRIPT
# ======================================================
# Author  : Yakub Hariana
# Purpose : Full workflow for FWA site clustering, sectorization,
#           and building coverage analysis using GeoPandas.
# ======================================================

import os
import sys
import zipfile
import time
import numpy as np
import pandas as pd
import geopandas as gpd
from datetime import datetime
from glob import glob
from tqdm import tqdm
from shapely.geometry import Polygon, box
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import multiprocessing as mp

# ======================================================
# PATHS & MODULE IMPORTS
# ======================================================
sys.path.append(r"D:\JACOBS\SERVICE\API")

from modules.table import excel_styler, sanitize_header
from modules.data import read_gdf
from core.config import settings

MAINDATA_DIR = settings.MAINDATA_DIR
DATA_DIR = settings.DATA_DIR
EXPORT_DIR = settings.EXPORT_DIR


# ======================================================
# GEOMETRY UTILITIES
# ======================================================
def identify_hexagon(data_gdf, resolution=5, buffer=10000, type="bound"):
    """Identify intersected hexagons within a bounding or convex buffer."""
    hex_path = f"{MAINDATA_DIR}/22. H3 Hex/Hex_{resolution}.parquet"
    if not os.path.exists(hex_path):
        raise FileNotFoundError(f"Hexagon file not found at {hex_path}")
    hex_gdf = gpd.read_parquet(hex_path)

    # Ensure projection
    hex_gdf = hex_gdf.to_crs("EPSG:3857")
    data_gdf = data_gdf.to_crs("EPSG:3857")

    if type == "bound":
        bbox = box(*data_gdf.total_bounds).buffer(buffer)
        hex_gdf = hex_gdf[hex_gdf.intersects(bbox)]
    elif type == "convex":
        hull = data_gdf.geometry.union_all().convex_hull.buffer(buffer)
        hex_gdf = hex_gdf[hex_gdf.intersects(hull)]
    else:
        raise ValueError("Invalid type: choose 'bound' or 'convex'.")

    if hex_gdf.empty:
        raise ValueError("No hexagons found for the given area.")
    return hex_gdf[f"hex_{resolution}"].unique().tolist()


def retrieve_building(hex_list, centroid=True, hex_dir=None, **kwargs):
    """Retrieve building data per hex and filter by area/aspect ratio."""
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

    def load_hex(hex_id):
        try:
            path = os.path.join(hex_dir, f"{hex_id}_buildings.parquet")
            if not os.path.exists(path):
                raise FileNotFoundError
            data = gpd.read_parquet(path).to_crs(epsg=3857)
            if centroid:
                data["geometry"] = data.geometry.centroid
            if aspect_ratio:
                data = data[data["asp_ratio"] > parameters["aspect_ratio_value"]]
            if area_building:
                data = data[
                    (data["area_in_meters"] > parameters["area_building_value"]["min"])
                    & (data["area_in_meters"] < parameters["area_building_value"]["max"])
                ]
            if one_unit:
                data = data[data["one_unit"] == 1]
            return data if not data.empty else None
        except:
            # Auto copy missing files from network drive
            src_dir = r"Z:\01. DATABASE\02. Building\Adm 2024\Hexed Building 2024"
            src = os.path.join(src_dir, f"{hex_id}_buildings.parquet")
            dst = os.path.join(hex_dir, f"{hex_id}_buildings.parquet")
            if os.path.exists(src):
                shutil.copy(src, dst)
                print(f"ℹ️ Copied {hex_id} from Z:")
                return load_hex(hex_id)
            return None

    results = []
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


def count_homepass(site_gdf, building, distance=500):
    """Count building points within site buffer (homepass)."""
    site_gdf, building = site_gdf.to_crs(3857), building.to_crs(3857)
    building["geometry"] = building.geometry.centroid
    site_gdf["buffer"] = site_gdf.geometry.buffer(distance)
    joined = gpd.sjoin(building, site_gdf.set_geometry("buffer"), predicate="intersects")
    count = joined.groupby("index_right").size().rename("total_homepass")
    site_gdf["total_homepass"] = site_gdf.index.map(count).fillna(0)
    site_gdf['total_homepass'] = site_gdf['total_homepass'].astype(int)
    return site_gdf.set_geometry("geometry").drop(columns="buffer")


def clean_sites_overlaps(site_tbg, max_distance=300, tolerance=10.0, protect_list=None):
    """Remove overlapping site buffers using GeoPandas sjoin (robust and parallel-safe)."""
    import geopandas as gpd
    import numpy as np

    site_tbg = site_tbg.to_crs(3857).copy()
    site_tbg = site_tbg.reset_index(drop=True)
    site_tbg["total_homepass"] = site_tbg.get("total_homepass", 0)
    if "site_type" not in site_tbg.columns:
        site_tbg["site_type"] = ""

    # Helper columns
    site_tbg["__isgreenfield"] = site_tbg["site_type"].str.lower().str.replace(" ", "").str.contains("greenfield").astype(int)
    site_tbg["__buf"] = site_tbg.geometry.buffer(max_distance)
    site_tbg["__area"] = site_tbg["__buf"].area
    if protect_list is not None:
        protect_set = set(protect_list)
        site_tbg["__isprotected"] = site_tbg["site_id"].astype(str).isin(protect_set).astype(int)
    else:
        site_tbg["__isprotected"] = 0

    copied = site_tbg.copy()
    copied = copied.reset_index()
    copied["geometry"] = copied.geometry.buffer(max_distance)

    # Self join with suffixes for clarity
    joined = gpd.sjoin(
        copied,
        copied,
        predicate="intersects",
        how="inner",
        lsuffix="left",
        rsuffix="right",
    )

    left_idx = joined["index_left"].to_numpy()
    right_idx = joined["index_right"].to_numpy()

    # Remove self-pairs
    mask = left_idx != right_idx
    left_idx, right_idx = left_idx[mask], right_idx[mask]

    if len(left_idx) == 0:
        print("ℹ️ No overlapping sites found.")
        return (
            site_tbg.drop(columns=["__buf", "__area", "__isgreenfield"], errors="ignore"),
            site_tbg.iloc[0:0],
        )

    # Compute intersection areas safely
    inter_areas = []
    for i, j in zip(left_idx, right_idx):
        geom_i = site_tbg.at[i, "__buf"]
        geom_j = site_tbg.at[j, "__buf"]
        inter = geom_i.intersection(geom_j)
        inter_areas.append(inter.area if not inter.is_empty else 0.0)
    inter_areas = np.array(inter_areas)

    valid_mask = inter_areas > 0
    left_idx, right_idx, inter_areas = (
        left_idx[valid_mask],
        right_idx[valid_mask],
        inter_areas[valid_mask],
    )

    dropped = set()
    for i, j, area in zip(left_idx, right_idx, inter_areas):
        # convert to native ints before adding/checking
        i, j = int(i), int(j)
        if i in dropped or j in dropped:
            continue
        
        if site_tbg.at[i, "__isprotected"] and site_tbg.at[j, "__isprotected"]:
            continue

        area_i = site_tbg.at[i, "__area"]
        area_j = site_tbg.at[j, "__area"]
        pct_i = 100 * area / area_i if area_i > 0 else 0
        pct_j = 100 * area / area_j if area_j > 0 else 0
        
        if max(pct_i, pct_j) < tolerance:
            continue

        def score(idx):
            return (site_tbg.at[idx, "__isprotected"], site_tbg.at[idx, "__isgreenfield"], site_tbg.at[idx, "total_homepass"])

        winner = i if score(i) > score(j) else j
        loser = j if winner == i else i
        dropped.add(int(loser))

    keep = [i for i in site_tbg.index if int(i) not in dropped]
    site_kept = site_tbg.loc[keep].drop(columns=["__buf", "__area", "__isgreenfield"], errors="ignore")
    site_drop = site_tbg.loc[list(dropped)].drop(columns=["__buf", "__area", "__isgreenfield"], errors="ignore")

    print(f"ℹ️ Sites kept: {len(site_kept)}, Sites dropped: {len(site_drop)}")
    return site_kept.reset_index(drop=True), site_drop.reset_index(drop=True)

def clean_sectors_overlaps(
    site_data: gpd.GeoDataFrame,
    sectors: gpd.GeoDataFrame,
    building: gpd.GeoDataFrame,
    accepted_list: list,
    sector_total: int = 3,
    tolerance: float = 10.0,
    **kwargs
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Clean overlapping sector polygons iteratively.
    - Sites in accepted_list are fully protected (all 3 sectors kept)
    - Overlaps resolved by total_homepass priority
    - Buildings assigned uniquely
    """
    if sectors.empty:
        return sectors, sectors, building

    # --- kwargs ---
    tolerance = kwargs.get("tolerance", tolerance)
    distance = kwargs.get("distance", 500)
    sector_angle = kwargs.get("sector_angle", 120)
    sector_group = kwargs.get("sector_group", 120)
    threshold = kwargs.get("threshold", 100)

    # --- CRS normalize ---
    sectors = sectors.to_crs(3857).reset_index(drop=True)
    building = building.to_crs(3857).reset_index(drop=True)
    site_data = site_data.to_crs(3857).reset_index(drop=True)

    # --- Protected accepted sites ---
    protected = sectors[sectors["site_id"].astype(str).isin(accepted_list)].copy()
    protected["__protected"] = True
    protected["__sector_utilized"] = sector_total

    dropped = sectors[~sectors["site_id"].astype(str).isin(accepted_list)].copy()
    dropped["__protected"] = False

    # --- Building assignment for protected ---
    building_acc = gpd.sjoin(
        building, protected[["geometry", "sector_id"]],
        predicate="intersects"
    ).drop(columns="index_right", errors="ignore")

    count_acc = building_acc.groupby("sector_id").size().rename("total_homepass")
    protected["total_homepass"] = protected["sector_id"].map(count_acc).fillna(0).astype(int)

    # --- Remove used buildings ---
    used_geom = building_acc.geometry
    building_remain = building.loc[~building.geometry.isin(used_geom)].reset_index(drop=True)

    # --- Accepted baseline ---
    accepted = protected.copy()
    accepted_union = accepted.geometry.unary_union.buffer(0)

    current_sector = sector_total - 1

    while current_sector > 0:
        if dropped.empty or building_remain.empty:
            print("ℹ️ No dropped sectors or buildings remaining. Ending loop.")
            break

        # 1️⃣ Filter by overlap vs accepted
        new_accepts = []
        for _, row in dropped.iterrows():
            geom = row.geometry.buffer(0)
            inter_area = geom.intersection(accepted_union).area
            overlap_pct = 100 * inter_area / (geom.area or 1)
            if overlap_pct <= tolerance and row.get("total_homepass", 0) > 0:
                new_accepts.append(row)

        if not new_accepts:
            print(f"ℹ️ No new sectors accepted at utilization {current_sector}.")
            break

        new_acc = gpd.GeoDataFrame(new_accepts, geometry="geometry", crs=dropped.crs)
        new_acc["__sector_utilized"] = current_sector

        # 2️⃣ Clean internal overlaps inside new_acc (keep higher homepass)
        if len(new_acc) > 1:
            joined = gpd.sjoin(new_acc, new_acc, predicate="intersects", how="inner", lsuffix="l", rsuffix="r")
            joined = joined[joined["sector_id_l"] != joined["sector_id_r"]]
            to_drop = set()
            for _, rowj in joined.iterrows():
                sid_l, sid_r = rowj["sector_id_l"], rowj["sector_id_r"]
                if sid_l in to_drop or sid_r in to_drop:
                    continue
                geom_l = new_acc.loc[new_acc["sector_id"] == sid_l, "geometry"].values[0]
                geom_r = new_acc.loc[new_acc["sector_id"] == sid_r, "geometry"].values[0]
                inter_area = geom_l.intersection(geom_r).area
                pct_overlap = 100 * inter_area / (geom_l.area or 1)
                if pct_overlap > tolerance:
                    hp_l = new_acc.loc[new_acc["sector_id"] == sid_l, "total_homepass"].values[0]
                    hp_r = new_acc.loc[new_acc["sector_id"] == sid_r, "total_homepass"].values[0]
                    loser = sid_l if hp_l < hp_r else sid_r
                    to_drop.add(loser)
            if to_drop:
                print(f"⚠️ Removed {len(to_drop)} overlapping sectors inside new batch (kept higher homepass).")
                new_acc = new_acc[~new_acc["sector_id"].isin(to_drop)].reset_index(drop=True)

        # 3️⃣ Building association for new accepted
        building_new = gpd.sjoin(
            building_remain, new_acc[["geometry", "sector_id"]],
            predicate="intersects"
        ).drop(columns="index_right", errors="ignore")

        if not building_new.empty:
            count_new = building_new.groupby("sector_id").size().rename("total_homepass")
            new_acc["total_homepass"] = new_acc["sector_id"].map(count_new).fillna(0).astype(int)

            building_acc = pd.concat([building_acc, building_new], ignore_index=True)
            used_geom = pd.concat([used_geom, building_new.geometry], ignore_index=True)
            building_remain = building.loc[~building.geometry.isin(used_geom)].reset_index(drop=True)

            # 4️⃣ Merge into accepted
            accepted = pd.concat([accepted, new_acc], ignore_index=True)
            accepted_union = accepted_union.union(new_acc.geometry.unary_union.buffer(0))

            # 5️⃣ Post-merge cleanup (cross-batch overlap)
            if len(accepted) > 1:
                joined = gpd.sjoin(accepted, accepted, predicate="intersects", how="inner", lsuffix="l", rsuffix="r")
                joined = joined[joined["sector_id_l"] != joined["sector_id_r"]]
                to_drop = set()
                for _, rowj in joined.iterrows():
                    sid_l, sid_r = rowj["sector_id_l"], rowj["sector_id_r"]
                    if sid_l in to_drop or sid_r in to_drop:
                        continue

                    site_l = accepted.loc[accepted["sector_id"] == sid_l, "site_id"].values[0]
                    site_r = accepted.loc[accepted["sector_id"] == sid_r, "site_id"].values[0]
                    prot_l = accepted.loc[accepted["sector_id"] == sid_l, "__protected"].values[0]
                    prot_r = accepted.loc[accepted["sector_id"] == sid_r, "__protected"].values[0]
                    if site_l == site_r:
                        continue  # same site
                    geom_l = accepted.loc[accepted["sector_id"] == sid_l, "geometry"].values[0]
                    geom_r = accepted.loc[accepted["sector_id"] == sid_r, "geometry"].values[0]
                    inter_area = geom_l.intersection(geom_r).area
                    pct_overlap = 100 * inter_area / (geom_l.area or 1)
                    if pct_overlap > tolerance:
                        hp_l = accepted.loc[accepted["sector_id"] == sid_l, "total_homepass"].values[0]
                        hp_r = accepted.loc[accepted["sector_id"] == sid_r, "total_homepass"].values[0]
                        # Protected always win
                        if prot_l and not prot_r:
                            loser = sid_r
                        elif prot_r and not prot_l:
                            loser = sid_l
                        else:
                            loser = sid_l if hp_l < hp_r else sid_r
                        to_drop.add(loser)
                if to_drop:
                    print(f"⚠️ Removed {len(to_drop)} cross-site overlaps after merge (kept protected/higher homepass).")
                    accepted = accepted[~accepted["sector_id"].isin(to_drop)].reset_index(drop=True)
                    accepted_union = accepted.geometry.unary_union.buffer(0)

        else:
            print(f"ℹ️ No buildings found for new accepted sectors at utilization {current_sector}.")
            break

        # 6️⃣ Clean dropped before next iteration
        site_dropped = site_data[~site_data["site_id"].isin(accepted["site_id"])]
        if site_dropped.empty:
            print("ℹ️ No dropped sites remaining.")
            break

        sectors_new = parallel_sectorize(
            sites_gdf=site_dropped,
            homepass=building_remain,
            distance=distance,
            angle=sector_angle,
            group=sector_group,
            threshold=threshold,
            max_workers=4
        )
        dropped = sectors_new.copy()

        # Internal dropped cleanup
        if len(dropped) > 1:
            joined = gpd.sjoin(dropped, dropped, predicate="intersects", how="inner", lsuffix="l", rsuffix="r")
            joined = joined[joined["sector_id_l"] != joined["sector_id_r"]]
            to_drop = set()
            for _, rowj in joined.iterrows():
                sid_l, sid_r = rowj["sector_id_l"], rowj["sector_id_r"]
                if sid_l in to_drop or sid_r in to_drop:
                    continue
                geom_l = dropped.loc[dropped["sector_id"] == sid_l, "geometry"].values[0]
                geom_r = dropped.loc[dropped["sector_id"] == sid_r, "geometry"].values[0]
                inter_area = geom_l.intersection(geom_r).area
                pct_overlap = 100 * inter_area / (geom_l.area or 1)
                if pct_overlap > tolerance:
                    hp_l = dropped.loc[dropped["sector_id"] == sid_l, "total_homepass"].values[0]
                    hp_r = dropped.loc[dropped["sector_id"] == sid_r, "total_homepass"].values[0]
                    loser = sid_l if hp_l < hp_r else sid_r
                    to_drop.add(loser)
            if to_drop:
                print(f"⚠️ Removing {len(to_drop)} overlapped sectors inside dropped pool (kept higher homepass).")
                dropped = dropped[~dropped["sector_id"].isin(to_drop)].reset_index(drop=True)

        current_sector -= 1

    # 7️⃣ Ensure all protected sites keep full 3 sectors
    for sid in accepted_list:
        subset = accepted[accepted["site_id"].astype(str) == str(sid)]
        if len(subset) < sector_total:
            needed = sector_total - len(subset)
            extra = sectors[
                (sectors["site_id"].astype(str) == str(sid))
                & (~sectors["sector_id"].isin(subset["sector_id"]))
            ].head(needed)
            if not extra.empty:
                print(f"🔒 Restoring {len(extra)} missing protected sectors for site {sid}.")
                accepted = pd.concat([accepted, extra], ignore_index=True)

    # --- Cleanup ---
    for col in ["__sector_utilized", "__protected"]:
        for df in (accepted, dropped):
            if col in df.columns:
                df.drop(columns=col, inplace=True, errors="ignore")

    building_acc = gpd.GeoDataFrame(building_acc, geometry="geometry", crs=building.crs)
    accepted = accepted.reset_index(drop=True)
    dropped = dropped.reset_index(drop=True)
    building_acc = building_acc.reset_index(drop=True)

    print(f"✅ Final accepted sectors: {len(accepted)} | Dropped: {len(dropped)} (Protected sites kept {sector_total} sectors)")
    return accepted, dropped, building_acc


#V3
def generate_overlap_area(gdf):
    """
    Generate GeoDataFrame of pairwise intersection geometries between input geometries.
    - Uses spatial index to limit pair checks.
    - Only computes for j > i to avoid duplicate pairs.
    - Filters out empty / zero-area intersections and returns unique geometries.
    """
    geoms = gdf.geometry.reset_index(drop=True)
    overlaps = []
    if geoms.empty:
        return gpd.GeoDataFrame(geometry=[], crs=gdf.crs)

    # build spatial index (may be None for some backends)
    sindex = geoms.sindex
    for i, geom1 in enumerate(geoms):
        if geom1 is None or geom1.is_empty:
            continue

        # candidates using spatial index if available, otherwise brute-force
        if sindex is not None:
            candidate_idxs = list(sindex.intersection(geom1.bounds))
        else:
            candidate_idxs = range(len(geoms))

        for j in candidate_idxs:
            if j <= i:
                continue
            geom2 = geoms.iloc[j]
            if geom2 is None or geom2.is_empty:
                continue
            inter = geom1.intersection(geom2)
            if inter.is_empty:
                continue
            # skip zero-area intersections (e.g., touching at a point)
            try:
                if inter.area == 0:
                    continue
            except Exception:
                # some geometry types may not have .area; include if not empty
                pass
            overlaps.append(inter)

    if not overlaps:
        return gpd.GeoDataFrame(geometry=[], crs=gdf.crs)

    overlaps_gdf = gpd.GeoDataFrame(geometry=overlaps, crs=gdf.crs)
    overlaps_gdf = overlaps_gdf[~overlaps_gdf.geometry.is_empty].reset_index(drop=True)
    # remove duplicate geometries (exact matches)
    overlaps_gdf = overlaps_gdf.drop_duplicates(subset="geometry").reset_index(drop=True)
    return overlaps_gdf

#V3
def select_buffers_max_coverage(
    gdf,
    overlap_threshold=0.1,
    require_connectivity=False,
    area_eps=1e-9,
    hp_col="hp_count",
    tower_col = 'tower type',
    site_col = 'site type'):

    """
    Select buffers to maximize total unique coverage area, with minimal overlap, and prioritize higher hp_count, tower_score, and site_score.

    This function performs a greedy selection of buffer geometries (e.g., site buffers) from a GeoDataFrame, aiming to maximize the total area covered by the selected buffers, while:
      - Enforcing a maximum allowed overlap between selected buffers (overlap_threshold).
      - Prioritizing candidates with larger unique (non-overlapping) area.
      - Using site_score, tower_score, and hp_count as secondary tie-breakers.
      - Optionally preferring candidates that are spatially connected to the current selection.

    Parameters
    ----------
    gdf : GeoDataFrame
        Input GeoDataFrame containing buffer geometries and relevant attributes.
    overlap_threshold : float, optional
        Maximum allowed fractional overlap between a candidate buffer and the current union of selected buffers (%).
    require_connectivity : bool, optional
        If True, only consider candidates that intersect/touch the current selection (chain method).
    area_eps : float, optional
        Minimum unique area threshold to consider a candidate (default 1e-9) in degree.
    hp_col : str, optional
        Column name for homepass count (default "hp_count").
    tower_col : str, optional
        Column name for tower type like SST, POLE.
    site_col : str, optional
        Column name for site type GREEN FIELD, ROOFTOP.

    Returns
    -------
    kept : GeoDataFrame
        GeoDataFrame of selected (kept) buffers.
    removed : GeoDataFrame
        GeoDataFrame of removed (not selected) buffers.

    Selection Logic
    --------------
    - At each step, candidates are filtered by overlap and unique area.
    - The candidate with the largest unique_area is selected.
    - If there is a tie, candidates with higher site_score, then tower_score, then hp_count are preferred.
    - After selection, any remaining buffers that overlap too much with the new selection are dropped (preferring to keep higher hp_count or larger area in case of ties).

    Example
    -------
    kept, removed = select_buffers_max_coverage(
        gdf,
        overlap_threshold=0.1,
        require_connectivity=True
    )
    """

    gdf['tower_score'] = gdf[tower_col].apply(lambda x: 2 if x == 'SST' else 1)
    gdf['site_score'] = gdf[site_col].apply(lambda x: 2 if 'GREEN' in x else 1)

    g = gdf.copy()
    overlaps_gdf = generate_overlap_area(gdf)
    if len(overlaps_gdf) > 0:
        overlaps_geom = overlaps_gdf.dissolve().reset_index(drop = True).loc[0,'geometry']
    g["buf_area"] = g["geometry"].area
    remaining = g.copy()
    kept_idxs = []
    union_sel = None

    while not remaining.empty:
        candidates = []
        for idx, row in remaining.iterrows():
            geom = row.geometry
            buf_area = row["buf_area"]

            if union_sel is None and len(overlaps_gdf) == 0:
                inter_area = 0.0
            elif union_sel is None and len(overlaps_gdf) > 0:
                inter = geom.intersection(overlaps_geom)
                inter_area = inter.area if not inter.is_empty else 0.0
            else:
                inter = geom.intersection(union_sel)
                inter_area = inter.area if not inter.is_empty else 0.0

            overlap_ratio = inter_area / buf_area if buf_area > 0 else 1.0
            unique_area = buf_area - inter_area

            candidates.append({
                "idx": idx,
                "unique_area": unique_area,
                "overlap_ratio": overlap_ratio,
                "hp_count": row[hp_col],
                'tower_score': row['tower_score'],
                'site_score': row['site_score'],
                "intersects_union": (union_sel is not None) and geom.intersects(union_sel),
                "geom": geom
            })

        if union_sel == None:
            allowed = [c for c in candidates]
        else:
            allowed = [c for c in candidates if c["overlap_ratio"] <= overlap_threshold and c["unique_area"] > area_eps]
        if not allowed:
            break

        if require_connectivity and union_sel is not None:
            allowed_conn = [c for c in allowed if c["intersects_union"]]
            if allowed_conn:
                allowed = allowed_conn

        best = max(allowed, key=lambda c: (c["unique_area"], c['site_score'], c['tower_score'], c["hp_count"]))
        best_idx = best["idx"]
        kept_idxs.append(best_idx)

        if union_sel is None:
            union_sel = best["geom"]
        else:
            union_sel = union_sel.union(best["geom"])

        overlaps_to_drop = []
        for idx2, row2 in remaining.iterrows():
            if idx2 == best_idx:
                continue
            inter_area = row2.geometry.intersection(best["geom"]).area
            overlap_ratio = inter_area / row2.geometry.area if row2.geometry.area > 0 else 0.0
            if overlap_ratio > overlap_threshold:
                # Drop the one with lower hp_count
                if row2[hp_col] < best["hp_count"]:
                    overlaps_to_drop.append(idx2)
                elif row2[hp_col] == best["hp_count"]:
                    # Drop smaller area if hp_count ties
                    if row2.geometry.area < best["geom"].area:
                        overlaps_to_drop.append(idx2)

        remaining = remaining.drop(overlaps_to_drop + [best_idx])

    kept = g.loc[kept_idxs].copy()
    removed = g.drop(kept_idxs).copy()

    kept = kept.set_geometry("geometry")
    removed = removed.set_geometry("geometry")
    return kept, removed


# ======================================================
# CLASSIFICATION UTILITIES
# ======================================================
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

# ======================================================
# SECTOR UTILITIES
# ======================================================

def generate_sector(center, buffer_distance, sector_angle=90, rotation_angle=0):
    """Generate radial polygons around a point."""
    x, y = center.x, center.y
    full_buffer = center.buffer(buffer_distance)
    num_sectors = 360 // sector_angle
    buffer_distance *= 1.25

    sectors = []
    for i in range(num_sectors):
        start, end = i * sector_angle + rotation_angle, (i + 1) * sector_angle + rotation_angle
        angles = np.linspace(np.radians(start), np.radians(end), 50)
        arc = list(zip(x + buffer_distance * np.sin(angles), y + buffer_distance * np.cos(angles)))
        poly = Polygon([(x, y), *arc, (x, y)])
        sectors.append({"geometry": full_buffer.intersection(poly),
                        "azimuth_start": start, "azimuth_end": end})
    return gpd.GeoDataFrame(sectors, geometry="geometry", crs="EPSG:3857")


def sectorize_sites(args):
    """Sectorize a single site based on threshold homepass balance."""
    site_data, homepass, dist, angle, threshold = args
    site_id, geom = site_data["site_id"], site_data["geometry"]

    best = {"score": -np.inf, "sectors": None}

    for offset in range(0, angle, 1):
        sectors = generate_sector(geom, dist, angle, offset)
        sectors["site_id"] = site_id
        sectors["sector_id"] = [f"{site_id}_{i+1}" for i in range(len(sectors))]

        if not homepass.empty:
            joined = gpd.sjoin(homepass, sectors, predicate="intersects")
            counts = joined.groupby("sector_id").size()
            sectors["total_homepass"] = sectors["sector_id"].map(counts).fillna(0)
        else:
            sectors["total_homepass"] = 0

        # Deviation
        sectors["__differ"] = abs(sectors["total_homepass"] - threshold)

        # Score
        valid_count = (sectors["total_homepass"] >= threshold).sum()
        max_differ = sectors[sectors['total_homepass'] >= threshold]['__differ'].max()
        score = (valid_count * 1000) - max_differ

        if score > best["score"]:
            # print(f"Threshold: {threshold} | Valid Count: {valid_count} | Max Differ {max_differ} | New Best Score: {score}")
            best = {"score": score, "sectors": sectors.copy()}

    return best["sectors"]


def parallel_sectorize(sites_gdf, homepass, distance=300, angle=120, group=120, threshold=100, max_workers=4):
    """Parallel sector generation for all sites."""
    site_args = []
    sindex = homepass.sindex

    for _, s in sites_gdf.iterrows():
        buff = s.geometry.buffer(distance)
        idx = sindex.query(buff, predicate="intersects")
        hp = homepass.iloc[idx].reset_index(drop=True)
        site_args.append(({"site_id": s["site_id"], "geometry": s.geometry}, hp, distance, angle, threshold))

    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as exe:
        for f in tqdm(as_completed({exe.submit(sectorize_sites, a): a[0]["site_id"] for a in site_args}),
                      total=len(site_args), desc="Sectorizing"):
            r = f.result()
            if r is not None and not r.empty:
                results.append(r)
    if not results:
        return gpd.GeoDataFrame(columns=["geometry"], crs="EPSG:3857")
    all_sectors = pd.concat(results, ignore_index=True)
    all_sectors = gpd.GeoDataFrame(all_sectors, geometry="geometry", crs="EPSG:3857")
    return all_sectors


# ======================================================
# REGION PROCESSING
# ======================================================
def parallel_region(region_data, distance_fwa=500, sector_angle=120, sector_group=120,
                    threshold_sector=100, max_workers=8, method="maximize",
                    clean_overlap=False, accepted_ids=None, export_path=None):
    region = region_data["region"].mode().values[0]
    print(f"🧩 Region {region} | {len(region_data):,} sites")
    
    try:
        hex_list = identify_hexagon(region_data, type="convex")
        buildings = retrieve_building(hex_list).to_crs(3857)

        buffered = region_data.copy()
        buffered["geometry"] = buffered.geometry.buffer(distance_fwa)
        buildings = gpd.sjoin(buildings, buffered[["geometry"]]).drop(columns="index_right").drop_duplicates("geometry")
        buildings['homepass_class'] = buildings['area_in_meters'].map(homepass_class)
        region_calc = count_homepass(region_data, buildings)
        if accepted_ids is not None:
            site_accept, site_dropped = clean_sites_overlaps(region_calc, max_distance=distance_fwa, protect_list=accepted_ids)
            accepted_list = set(site_accept["site_id"].astype(str))      
            print(f"🌏 Region {region} | Using provided accepted sites: {len(site_accept):,} accepted, {len(site_dropped):,} dropped.")
        else:
            site_accept, site_dropped = clean_sites_overlaps(region_calc, max_distance=distance_fwa)
            accepted_list = set(site_accept["site_id"].astype(str))
            print(f"🌏 Region {region} | Accepted Sites: {len(site_accept):,}, Dropped Sites: {len(site_dropped):,}")

        region_calc['note'] = np.where(region_calc['site_id'].isin(accepted_list), 'Accepted', 'Dropped')
        sectors = parallel_sectorize(region_calc, buildings, distance_fwa, sector_angle, sector_group, threshold_sector, max_workers)

        if clean_overlap and len(region_calc) > 1 and not sectors.empty:
            sectors, sectors_dropped, buildings = clean_sectors_overlaps(
                site_data=region_calc,
                sectors=sectors,
                building=buildings,
                accepted_list=accepted_list,
                sector_total=360 // sector_group,
                tolerance=10.0,
                sector_group=sector_group,
                sector_angle=sector_angle,
                threshold=threshold_sector,
                distance=distance_fwa
            )
            accepted_list = sectors["site_id"].unique().tolist()
            region_calc['note'] = np.where(region_calc['site_id'].isin(accepted_list), 'Accepted', 'Dropped')
        else:
            region_calc['note'] = np.where(region_calc['site_id'].isin(accepted_list), 'Accepted', 'Dropped')
            buildings = gpd.sjoin(buildings, sectors[["geometry", "site_id", "sector_id"]], predicate="intersects").drop(columns="index_right")

        # AGGREGATE HOMEPASS CLASS TO SECTOR LEVEL
        grouped_sectors = buildings.groupby(['sector_id', 'homepass_class']).size().unstack(fill_value=0)
        count_total = buildings.groupby('sector_id').size().rename('total_homepass')
        for col in grouped_sectors.columns:
            sectors[f'hp_{col}'] = sectors['sector_id'].map(grouped_sectors[col]).fillna(0).astype(int)
        
        sectors = sectors.drop(columns='total_homepass')
        sectors['total_homepass'] = sectors['sector_id'].map(count_total).fillna(0).astype(int)

        # AGGREGATE SECTOR TO SITE LEVEL
        col_hp_class = [col for col in sectors.columns if col.startswith('hp_')]
        sector_summary = sectors.groupby('site_id').size().rename('total_sectors')
        sector_summary = sector_summary.to_frame()
        for col in col_hp_class:
            sector_summary_col = sectors.groupby('site_id')[col].sum().rename(f'{col}')
            sector_summary = sector_summary.join(sector_summary_col)
            if col in region_calc.columns:
                region_calc = region_calc.drop(columns=col)

        if 'total_homepass' in region_calc.columns:
            region_calc = region_calc.drop(columns='total_homepass')
        total_homepass_col = sectors.groupby('site_id')['total_homepass'].sum().rename('total_homepass')
        sector_summary = sector_summary.join(total_homepass_col)
        region_calc = region_calc.set_index('site_id').join(sector_summary).reset_index()
        region_calc = region_calc.fillna(0)

        region_calc["market_class"] = region_calc["total_homepass"].map(classify_market)
        sectors["market_class"] = sectors["total_homepass"].map(classify_market)

        if export_path:
            export_path = os.path.join(export_path, "Checkpoint")
            os.makedirs(export_path, exist_ok=True)
            region_calc.to_parquet(os.path.join(export_path, f"Sitelist_Region_{region}.parquet"))
            sectors.to_parquet(os.path.join(export_path, f"Sectors_Region_{region}.parquet"))
            buildings.to_parquet(os.path.join(export_path, f"Buildings_Region_{region}.parquet"))
            # print(f"💾 Region {region} results exported to {export_path}")

        return region_calc, sectors, buildings
    except Exception as e:
        print(f"❌ Exception in Region {region}: {e}")
        return region_data, gpd.GeoDataFrame(), gpd.GeoDataFrame()


# ======================================================
# MAIN FWA PIPELINE
# ======================================================
def validate_fwa(excel_file):
    """Check if file or dataframe contains minimum FWA input columns."""
    df = pd.read_excel(excel_file) if isinstance(excel_file, str) else excel_file
    df = sanitize_header(df, lowercase=True)
    for col in ["site_id", "long", "lat"]:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")
    return df


def main_fwa(data_gdf, export_dir, export_building=False, method="maximize",
             clean_overlap=False, sector_angle=120,
             sector_group=120, threshold=300, threshold_sector=50,
             distance_fwa=500, max_workers=8, accepted_ids=None):
    start = time.time()
    group_factor = sector_group // sector_angle
    total_sectors = 360 // sector_group
    threshold_sector = threshold // total_sectors if threshold_sector is None else threshold_sector
    total_sites = len(data_gdf)

    print(f"ℹ️ Total Sitelist to Process : {total_sites:,}")
    print(f"ℹ️ Clean Sector Overlaps     : {clean_overlap}")

    print("🧩 GROUPING SITES...")
    groups = auto_group(data_gdf, distance=distance_fwa)
    data_gdf = gpd.sjoin(data_gdf, groups[["geometry", "region"]]).drop(columns="index_right")
    regions = data_gdf["region"].unique()

    mp_ctx = mp.get_context("spawn")
    site_result, sector_result, building_result = [], [], []

    with ProcessPoolExecutor(max_workers=max_workers, mp_context=mp_ctx) as executor:
        futures = {executor.submit(parallel_region,
                                   data_gdf[data_gdf["region"] == r],
                                   distance_fwa, sector_angle, sector_group,
                                   threshold_sector, max_workers, method,
                                   clean_overlap, accepted_ids, export_dir): r
                   for r in regions}

        for f in tqdm(as_completed(futures), total=len(futures), desc="Process Region"):
            r = futures[f]
            try:
                res = f.result()
                if res:
                    s, sec, bld = res
                    site_result.append(s)
                    sector_result.append(sec)
                    if export_building:
                        building_result.append(bld)
                    print(f"✅ Region {r} done.")
            except Exception as e:
                print(f"🔴 Error in region {r}: {e}")

    site_df = pd.concat(site_result)
    sector_df = pd.concat(sector_result)
    if export_building:
        building_df = pd.concat(building_result)
    else:
        building_df = None

    os.makedirs(export_dir, exist_ok=True)
    site_df.to_parquet(f"{export_dir}/Sitelist_FWA.parquet")
    sector_df.to_parquet(f"{export_dir}/Sectors_FWA.parquet")
    if building_df is not None:
        building_df.to_parquet(f"{export_dir}/Buildings_FWA.parquet")

    # Remove Checkoint folder
    # checkpoint_dir = os.path.join(export_dir, "Checkpoint")
    # if os.path.exists(checkpoint_dir):
    #     os.remove(checkpoint_dir)

    # Excel export
    with pd.ExcelWriter(f"{export_dir}/Summary_FWA.xlsx", engine="openpyxl") as w:
        site_df.drop(columns="geometry").to_excel(w, sheet_name="Sites", index=False)
        sector_df.drop(columns="geometry").to_excel(w, sheet_name="Sectors", index=False)

    # Zip
    zip_path = os.path.join(export_dir, f"FWA_{datetime.now():%Y%m%d_%H%M}.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(export_dir):
            for f in files:
                if not f.endswith(".zip"):
                    zipf.write(os.path.join(root, f), arcname=f)
    print(f"📦 Results zipped to {zip_path}")
    print(f"✅ FWA completed for {total_sites:,} sites in {round((time.time()-start)/60, 2)} mins.")
    return zip_path

# ======================================================
# EXECUTION
# ======================================================
if __name__ == "__main__":
    sector_angle = 120
    sector_group = 120
    distance_fwa = 500
    threshold = 800
    threshold_sector = 250
    max_workers = 8
    
    export_dir = r"D:\JACOBS\PROJECT\TASK\OKTOBER\Week 4\FWA Surge\Export\New"
    os.makedirs(export_dir, exist_ok=True)

    sitelist_path = r"D:\JACOBS\PROJECT\TASK\OKTOBER\Week 4\FWA Surge\Sample 125413126.xlsx"
    sitelist = read_gdf(sitelist_path)

    sitelist.columns = sitelist.columns.str.lower()
    sitelist["site_id"] = sitelist["site_id"].astype(str)
    sitelist["long"] = sitelist.geometry.to_crs(4326).x
    sitelist["lat"] = sitelist.geometry.to_crs(4326).y
    sitelist["site_type"] = sitelist["site_type"].fillna("unknown").str.lower()
    sitelist = sitelist[["site_id", "long", "lat", "site_type", "geometry"]]
    sitelist = sitelist.drop_duplicates("site_id").reset_index(drop=True)

    # ACC IDS 701
    # accepted_paths = r"D:\JACOBS\TASK\OKTOBER\Week 3\FWA SURGE\Protect_List_Dec_900.xlsx"
    # accepted_paths = read_gdf(accepted_paths)
    # accepted_ids = set(accepted_paths["site_id"].astype(str))
    accepted_ids = None

    data_gdf = sitelist.to_crs(3857)
    main_fwa(
        data_gdf=data_gdf,
        export_dir=export_dir,
        export_building=True,
        method="maximize",
        sector_angle=sector_angle,
        sector_group=sector_group,
        threshold=threshold,
        threshold_sector=threshold_sector,
        distance_fwa=distance_fwa,
        max_workers=max_workers,
        clean_overlap=False,
        accepted_ids=accepted_ids,
    )
