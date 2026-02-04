import os
import time
import geopandas as gpd
import pandas as pd
import numpy as np
import sys
import shutil
from pathlib import Path
from tqdm import tqdm
from enum import Enum
from openpyxl import load_workbook
from openpyxl.styles import NamedStyle, Border, Side, Font, Alignment

root = Path(__file__).resolve().parents[2]
sys.path.append(root)

from modules.kml import export_kml, sanitize_kml, validate_kmz_design, validate_kmz_ipl
from modules.table import excel_styler
from modules.utils import auto_group, admin_information
from modules.geometry import relative_intersection, geodesic_length
from core.config import settings

MAINDATA_DIR = settings.MAINDATA_DIR
DATA_DIR = settings.DATA_DIR

# ------------------------------------------------------
# LOGGER
# ------------------------------------------------------
from core.logger import create_logger
logger = create_logger(__file__)

# ENUMS
# ------------------------------------------------------
class Operator(str, Enum):
    IOH = "ioh"
    XL = "xl"
    SURGE = "surge"
    TSEL = "tsel"


class DeviceType(str, Enum):
    OTB = "OTB"
    ODP = "ODP"


class ConnectorType(str, Enum):
    SC = "SC"
    FC = "FC"

class BoQType(str, Enum):
    INTERSITE = "intersite"
    MMP = "mmp"

def drm_format(
    kmz_path: str,
    export_dir: str,
    operator: Operator | str = Operator.XL,
    sep: str = ";",
    program_name: str = "Intersite FO"
):
        
    validated = validate_kmz_design(kmz_path, sep=sep)
    if validated is None:
        logger.info(f"❌ Invalid KMZ Design: {kmz_path}")
        return

    # ---------------------------
    # Inputs (GeoDataFrames)
    # ---------------------------
    points_kmz, lines_kmz = validated
    points_kmz = admin_information(points_kmz, level="kabkot")
    lines_kmz = admin_information(lines_kmz, level="kabkot")
    lines_kmz['length'] = lines_kmz['geometry'].apply(geodesic_length)
    colopriming_data = pd.read_excel(f"{DATA_DIR}/Sitelist Dec 2025.xlsx")
    colopriming_data['site_id'] = colopriming_data['site_id'].astype(str)
    colopriming_data = colopriming_data.set_index("site_id")

    target_crs = 3857
    points_kmz = points_kmz.to_crs(epsg=target_crs)
    lines_kmz = lines_kmz.to_crs(epsg=target_crs)

    points_grouped = points_kmz.groupby("ring_name")

    # Container
    segments_records = []
    sites_records = []
    ring_records = []
    for ring, ring_points in points_grouped:
        ring_lines = lines_kmz[lines_kmz['ring_name'] == ring].copy()
        
        if ring_lines.empty:
            logger.error(f"🔴 Ring {ring} lines data not found.")
            continue

        # Ring Metadata
        province = ring_points['Provinsi'].mode()[0]
        city = ring_points['Kabkot'].mode()[0]
        vendor = "TBG"
        ne_ids = set(ring_lines['near_end'].astype(str))
        fe_ids = set(ring_lines['far_end'].astype(str))

        # Route
        routes_length = ring_lines['length'].sum()

        # Hub Sitelist
        hubs = ring_points[ring_points['site_type'].astype(str).str.lower().str.contains("hub")].copy()
        sitelist = ring_points[~(ring_points.index.isin(hubs.index))].copy()
        hubs_ids = set(hubs['site_id'].astype(str))
        sites_ids = set(sitelist['site_id'].astype(str))
        qty_hubs = len(hubs)
        qty_sites = len(sitelist)

        match qty_hubs:
            case 0:
                logger.error(f"🔴 Ring {ring} FO Hub not found.")
                continue
            case 1:
                hub_1 = hubs['site_id'].astype(str).values[0]
                hub_2 = None
            case 2:
                hub_1 = hubs['site_id'].astype(str).values[0]
                hub_2 = hubs['site_id'].astype(str).values[-1]
            case _:
                logger.error(f"🔴 Ring {ring} FO Hub exceeds, found {qty_hubs}.")
                continue
            
        # Enrich Metadata
        route_type = None
        if qty_hubs == 1 and qty_sites == 1:
            route_type = "Star"
        elif qty_hubs == 1 and qty_sites > 1:
            route_type = "Chain"
        elif qty_hubs > 1 and qty_sites >= 1:
            route_type = "Ring"

        # Segments
        for idx, route in ring_lines.iterrows():
            ne_id = route['near_end']
            fe_id = route['far_end']
            ne_site = ring_points[ring_points['site_id'].astype(str) == str(ne_id)].squeeze().copy()
            fe_site = ring_points[ring_points['site_id'].astype(str) == str(fe_id)].squeeze().copy()

            # Enrich Metadata
            match route_type:
                case "Star":
                    ne_status = ne_site['site_name'].str.contains(r'\w', regex=True, na=False)
                    fe_status = fe_site['site_name'].str.contains(r'\w', regex=True, na=False)
                    ne_status = "Station" if ne_status else "Direct to Station"
                    fe_status = "Station" if fe_status else "Direct to Station"
                case "Chain":
                    ne_status = ne_site['site_name'].str.contains(r'\w', regex=True, na=False)
                    fe_status = fe_site['site_name'].str.contains(r'\w', regex=True, na=False)
                    ne_status = "Station" if ne_status else "Direct to Station"
                    fe_status = "Station" if fe_status else "Direct to Station"
                case "Ring":
                    ne_status = ne_site['site_type']
                    fe_status = fe_site['site_type']
            
            segment = {
                "no": None,
                "segment": route['segment'],
                "ne_site": ne_site['site_id'],
                "ne_long": ne_site['long'],
                "ne_lat": ne_site['lat'],
                "fe_site": fe_site['site_id'],
                "fe_long": fe_site['long'],
                "fe_lat": fe_site['lat'],
                "ring_name": ring,
                "route_type": None,
                "length": route['length'],
                "hub_1": hub_1,
                "hub_2": hub_2,
                "pop_type": None,
                "status_ne": ne_status,
                "status_fe": fe_status,
                "rfs_plan":None,
                "province": province,
                "city": city,
                "vendor": vendor
            }
            segments_records.append(segment)
        
            # Site
            colo_ne = colopriming_data.loc[ne_site['site_id']].copy() if ne_site['site_id'] in colopriming_data.index else pd.DataFrame()
            colo_fe = colopriming_data.loc[fe_site['site_id']].copy() if fe_site['site_id'] in colopriming_data.index else pd.DataFrame()
            ne_record = {
                "no": None,
                "site_id": ne_site['site_id'],
                "site_name": ne_site['site_name'],
                "site_id_cust": None,
                "province": ne_site['Provinsi'],
                "city": ne_site['Kabkot'],
                "kecamatan": colo_ne.get("kecamatan", None),
                "kelurahan": colo_ne.get("kelurahan", None),
                "lat": ne_site['lat'],
                "long": ne_site['long'],
                "tower_owner": colo_ne.get("tp", None),
                "field_type": colo_ne.get("field_type", None),
                "pop_interconnection": (",").join(hubs['site_id'].unique().tolist()),
                "location_pop": None,
                "antenna_height": colo_ne.get("total_tower_height", None),
                "total_sector": None,
                "fo_type": route_type,
                "ring_id": ring,
                "distance_per_site": route['length'] if str(ne_site['site_id']) not in hubs_ids else None
            }
            fe_record = {
                "no": None,
                "site_id": fe_site['site_id'],
                "site_name": fe_site['site_name'],
                "site_id_cust": None,
                "province": fe_site['Provinsi'],
                "city": fe_site['Kabkot'],
                "kecamatan": colo_fe.get("kecamatan", None),
                "kelurahan": colo_fe.get("kelurahan", None),
                "lat": fe_site['lat'],
                "long": fe_site['long'],
                "tower_owner": colo_fe.get("tp", None),
                "field_type": colo_fe.get("field_type", None),
                "pop_interconnection": (",").join(hubs['site_id'].unique().tolist()),
                "location_pop": None,
                "antenna_height": colo_fe.get("total_tower_height", None),
                "total_sector": None,
                "fo_type": route_type,
                "ring_id": ring,
                "distance_per_site": route['length'] if str(fe_site['site_id']) not in hubs_ids else None
            }

            if str(ne_site['site_id']) not in hubs_ids:
                sites_records.append(ne_record)

            if str(fe_site['site_id']) not in hubs_ids:
                sites_records.append(fe_record)
    
        # Ring Record
        ring_record = {
            "no": None,
            "province": province,
            "ring_id": ring,
            "total_site": qty_sites,
            "route_type": route_type,
            "total_distance": routes_length,
            "avg_per_site": round(routes_length / qty_sites, 3),
        }
        ring_records.append(ring_record)
    
    # Compile DRM Summary
    segments_records = pd.DataFrame(segments_records)
    sites_records = pd.DataFrame(sites_records)
    ring_records = pd.DataFrame(ring_records)

    segments_records['no'] = segments_records.index + 1
    sites_records['no'] = sites_records.index + 1
    ring_records['no'] = ring_records.index + 1

    # ---------------------------
    # Write Excel
    # ---------------------------
    template_path = os.path.join(DATA_DIR, "template", "drm", "Template_DRM_Report.xlsx")
    output_path = os.path.join(export_dir, "DRM Report.xlsx")

    if not os.path.exists(template_path):
        raise ValueError("BOQ template file not found in template directory.")

    shutil.copy2(template_path, output_path)

    # ---------------------------
    # PROCESS BOQ EXCEL
    # ---------------------------
    wb = load_workbook(output_path)

    # Style
    named_style = NamedStyle(name="RowStyle")
    side_style = Side(style="thin", border_style="thin")
    border = Border(left=side_style, right=side_style, top=side_style, bottom=side_style)
    named_style.font = Font(name="Arial", size=10)
    named_style.border = border

    if "RowStyle" not in [s for s in wb.named_styles]:
        wb.add_named_style(named_style)

    # Segment Sheet
    segment_sheet = wb["Lampiran Segment"]
    start_data_row = 2
    col_index = {col: num for num, col in enumerate(segments_records.columns, start=1)}
    for idx, record in enumerate(segments_records.to_dict("records")):
        excel_row = start_data_row + idx

        for key, value in record.items():
            cell = segment_sheet.cell(
                row = excel_row,
                column = col_index[key],
                value = value
            )
            cell.style = "RowStyle"

            if isinstance(value, (int, float)) and value is not None:
                if ('long' in str(key).lower()) or ('lat' in str(key).lower()):
                    continue

                cell.number_format = "#,##0"
                cell.alignment = Alignment(horizontal="center", vertical="center")

    # Sites Sheet
    segment_sheet = wb["Lampiran Site"]
    start_data_row = 4
    col_index = {col: num for num, col in enumerate(sites_records.columns, start=1)}
    for idx, record in enumerate(sites_records.to_dict("records")):
        excel_row = start_data_row + idx

        for key, value in record.items():
            cell = segment_sheet.cell(
                row = excel_row,
                column = col_index[key],
                value = value
            )
            cell.style = "RowStyle"

            if isinstance(value, (int, float)) and value is not None:
                if ('long' in str(key).lower()) or ('lat' in str(key).lower()):
                    continue

                cell.number_format = "#,##0"
                cell.alignment = Alignment(horizontal="center", vertical="center")

    # Summary Ring Sheet
    segment_sheet = wb["Summary Ring"]
    start_data_row = 3
    col_index = {col: num for num, col in enumerate(ring_records.columns, start=1)}
    for idx, record in enumerate(ring_records.to_dict("records")):
        excel_row = start_data_row + idx

        for key, value in record.items():
            cell = segment_sheet.cell(
                row = excel_row,
                column = col_index[key],
                value = value
            )
            cell.style = "RowStyle"

            if isinstance(value, (int, float)) and value is not None:
                if ('long' in str(key).lower()) or ('lat' in str(key).lower()):
                    continue

                cell.number_format = "#,##0"
                cell.alignment = Alignment(horizontal="center", vertical="center")

    wb.save(output_path)
    logger.info("✅ DRM Excel format saved.")

if __name__ == "__main__":
    kmz_path = r"D:\JACOBS\PROJECT\TASK\2026\FEB\W1\DRM FORMAT\20250716-H2B2NewSiteCoverage-TBG-v9 (BoQ).kmz"
    export_dir = r"D:\JACOBS\PROJECT\TASK\2026\FEB\W1\DRM FORMAT\20250716-H2B2NewSiteCoverage-TBG-v9"
    sep = "-"

    os.makedirs(export_dir, exist_ok=True)
    
    drm_format(kmz_path=kmz_path, export_dir=export_dir, sep=sep)