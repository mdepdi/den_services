import numpy as np
import pandas as pd
import geopandas as gpd
import os
from scipy.spatial import cKDTree

from modules.data import read_gdf, get_unique_col

def nearest_targets_kdtree(
    gdf_source: gpd.GeoDataFrame,
    gdf_target: gpd.GeoDataFrame,
    target_id_col: str | None = None,
    to_crs: str | int | None = None,
    k: int = 5,
    prefix: str = "nn",
) -> gpd.GeoDataFrame:
    src = gdf_source.copy()
    tgt = gdf_target.copy()

    if src.crs is None or tgt.crs is None:
        raise ValueError("Both GeoDataFrames must have a CRS set (gdf.crs).")

    if to_crs is not None:
        src = src.to_crs(to_crs)
        tgt = tgt.to_crs(to_crs)
    else:
        if src.crs.is_geographic or tgt.crs.is_geographic:
            raise ValueError("Geographic CRS (lat/lon). Provide to_crs for metric distances.")

    # requires Points
    if not (src.geometry.geom_type == "Point").all() or not (tgt.geometry.geom_type == "Point").all():
        raise ValueError("KDTree approach requires Point geometries.")

    src_xy = np.c_[src.geometry.x.to_numpy(), src.geometry.y.to_numpy()]
    tgt_xy = np.c_[tgt.geometry.x.to_numpy(), tgt.geometry.y.to_numpy()]

    tree = cKDTree(tgt_xy)

    # If source and target are same set, ask for k+1 then drop self-match
    same_object = gdf_source is gdf_target
    kk = k + 1 if same_object else k

    dists, idxs = tree.query(src_xy, k=kk)

    # normalize shapes when k==1
    if kk == 1:
        dists = dists[:, None]
        idxs = idxs[:, None]

    if same_object:
        # drop the first neighbor (usually itself distance 0)
        dists = dists[:, 1:k+1]
        idxs = idxs[:, 1:k+1]
    else:
        dists = dists[:, :k]
        idxs = idxs[:, :k]

    if target_id_col is None:
        target_ids = tgt.index.to_numpy(dtype=object)
    else:
        target_ids = tgt[target_id_col].to_numpy(dtype=object)

    out = src.copy()

    for r in range(k):
        out[f"{prefix}_id_{r+1}"] = target_ids[idxs[:, r]]
        out[f"{prefix}_dist_{r+1}"] = dists[:, r]
        out[f"{prefix}_range_{r+1}"] = np.select(
            [
                out[f"{prefix}_dist_{r+1}"] < 100,
                out[f"{prefix}_dist_{r+1}"] < 200,
                out[f"{prefix}_dist_{r+1}"] < 300,
                out[f"{prefix}_dist_{r+1}"] < 500,
                out[f"{prefix}_dist_{r+1}"] < 1000,
                out[f"{prefix}_dist_{r+1}"] < 2500,
            ],
            ["<100", "100-200", "200-300", "300-500", "500-1000", "1000-2500"],
            default=">2500"
        )

    print(f"🟢 Identify nearest {k} target done.")
    return out

if __name__ == "__main__":
    excel_path = r"D:\JACOBS\PROJECT\TASK\2026\FEB\W2\SURGE SITE ASESSMENT\Sitelist Compiled 725 Not Found_5K_DRM.xlsx"
    filename = os.path.splitext(os.path.basename(excel_path))[0]
    directory = os.path.dirname(excel_path)

    sitelist = pd.read_excel(excel_path)
    print(f"ℹ️ Total Sitelist to Process: {len(sitelist):,} records.")

    sitelist = read_gdf(sitelist, long_col="Longitude", lat_col="Latitude")
    sitelist = sitelist.drop_duplicates(subset="Site ID")
    print(sitelist['Source'].value_counts())

    unique_col = get_unique_col(sitelist)
    sitelist[unique_col] = sitelist[unique_col].astype(str)

    # Start Distance Matrix
    identified_nearest = nearest_targets_kdtree(sitelist, sitelist, unique_col, to_crs=3857, k=3, prefix="nearest")

    export_dir = os.path.join(directory, "Distance_Matrix")
    os.makedirs(export_dir, exist_ok=True)
    
    identified_nearest.to_parquet(os.path.join(export_dir, f"DM_{filename}.parquet"), index=False)
    identified_nearest.to_excel(os.path.join(export_dir, f"DM_{filename}.xlsx"), index=False)