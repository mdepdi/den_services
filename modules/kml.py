import simplekml
import html
import os
import re
import io
import zipfile
import pandas as pd
import numpy as np
import geopandas as gpd
from bs4 import BeautifulSoup
from shapely.geometry import Point, LineString, Polygon
from modules.table import sanitize_header

def export_kml(
    gdf,
    kml_obj=None,
    folder_name="Features",
    subfolder=None,
    name_col=None,
    color="#FFFFFF",
    color_col=None,
    color_map=None,
    icon='http://maps.google.com/mapfiles/kml/shapes/donut.png',
    size=1.0,
    opacity=1.0,
    popup=True,
    schema_name="Schema"

):
    kml = kml_obj if kml_obj is not None else simplekml.Kml(name=folder_name)

    # --- create/attach schema once ---
    fields = [c for c in gdf.columns if c not in ("geometry", "description")]
    schema = kml.newschema(name=schema_name)
    for col in fields:
        schema.newsimplefield(name=col, type="string", displayname=col)
    schema_url = f"{schema.id}"
    schema_template = schema.name

    # --- folders (keep clean so they don’t turn blue) ---
    def sanitize_folder(folder):
        try: folder.description = None
        except: pass
        try: folder.extendeddata = None
        except: pass
        try: folder.styleurl = None
        except: pass
        try: folder.style.balloonstyle = None
        except: pass
        try: folder.gxballoonvisibility = 0
        except: pass

    container = kml
    if subfolder:
        for part in subfolder.split('/'):
            found = next((f for f in getattr(container, 'features', [])
                          if isinstance(f, simplekml.Folder) and f.name == part), None)
            container = found or container.newfolder(name=part)
            sanitize_folder(container)
    else:
        sanitize_folder(container)

    # --- balloon template (on placemarks only) ---
    if popup:
        rows = "".join(
            f"<tr><td><b>{html.escape(f)}</b></td>"
            f"<td>$[{html.escape(schema_template)}/{html.escape(f)}]</td></tr>"
            for f in fields
        )
        balloon_html = f"<![CDATA[<table border='0'>{rows}</table>]]>"
    else:
        balloon_html = None

    # --- small helpers ---
    def hex_color(hex_color: str, opacity: float = 1.0) -> str:
        s = hex_color.lstrip('#')
        if len(s) != 6: return "ffffffff"
        r, g, b = s[0:2], s[2:4], s[4:6]
        a = f'{int(255*opacity):02x}'
        return f"{a}{b}{g}{r}"

    # --- iterate features ---
    for _, row in gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue

        # style color
        feature_hex = color_map.get(row[color_col], color) if (color_col and color_map) else color
        kml_color = hex_color(feature_hex, opacity)

        # make a new placemark (inline style; simple & robust)
        def stylize(pm):
            pm.style.iconstyle.icon.href = icon
            pm.style.iconstyle.scale = size
            pm.style.iconstyle.color = kml_color
            pm.style.linestyle.color = kml_color
            pm.style.linestyle.width = max(1, int(size))
            pm.style.polystyle.color = kml_color
            pm.style.labelstyle.scale = size
            if balloon_html:
                pm.style.balloonstyle.text = balloon_html

            if popup:
                sd = simplekml.SchemaData(schemaurl=schema_url)
                for col in fields:
                    val = row[col]
                    sd.newsimpledata(col, "" if val is None else str(val))
                pm.extendeddata.schemadata = sd

        name = str(row.get(name_col, "")) if name_col else ""

        if geom.geom_type in ("Point", "MultiPoint"):
            geoms = geom.geoms if geom.geom_type == "MultiPoint" else [geom]
            if len(geoms) > 1:
                pm = container.newmultigeometry(name=name)
                for p in geoms:
                    pm.newpoint(coords=[p.coords[0]])
            else:
                pm = container.newpoint(name=name, coords=[geoms[0].coords[0]])
            stylize(pm)

        elif geom.geom_type in ("LineString", "MultiLineString"):
            geoms = geom.geoms if geom.geom_type == "MultiLineString" else [geom]
            if len(geoms) > 1:
                pm = container.newmultigeometry(name=name)
                for line in geoms:
                    pm.newlinestring(coords=list(line.coords))
            else:
                pm = container.newlinestring(name=name, coords=list(geoms[0].coords))
            stylize(pm)

        elif geom.geom_type in ("Polygon", "MultiPolygon"):
            geoms = geom.geoms if geom.geom_type == "MultiPolygon" else [geom]
            if len(geoms) > 1:
                pm = container.newmultigeometry(name=name)
                for poly in geoms:
                    p = pm.newpolygon(outerboundaryis=list(poly.exterior.coords))
                    p.innerboundaryis = [list(i.coords) for i in poly.interiors]
            else:
                pm = container.newpolygon(name=name, outerboundaryis=list(geoms[0].exterior.coords))
                pm.innerboundaryis = [list(i.coords) for i in geoms[0].interiors]
            stylize(pm)

        else:
            continue

    return kml

def sanitize_kml(kml):
    # Recursively remove description, style, and balloonstyle from all folders
    def clean_folder(folder):
        if hasattr(folder, 'description'):
            folder.description = None
        # Remove style only if it exists and is not None
        if hasattr(folder, 'style') and folder.style is not None:
            # Instead of setting to None, assign a new empty Style
            folder.style = simplekml.Style()
        if hasattr(folder, 'balloonstyle'):
            folder.balloonstyle = None
        if hasattr(folder, 'features'):
            for feat in folder.features:
                if isinstance(feat, simplekml.Folder):
                    clean_folder(feat)
    if kml is not None and hasattr(kml, 'features'):
        for feat in kml.features:
            if isinstance(feat, simplekml.Folder):
                clean_folder(feat)

def hex_to_kml_color(hex_color, alpha=255, opacity=1.0):
    """
    Convert a CSS-style hex color ('#RRGGBB' or 'RRGGBB') plus
    an alpha (0–255) into a KML color string 'AABBGGRR'.
    """
    # strip leading '#'
    s = hex_color.lstrip("#")
    if len(s) != 6:
        raise ValueError("Expected 6-digit hex (e.g. '#ff8800')")
    # parse
    r = int(s[0:2], 16)
    g = int(s[2:4], 16)
    b = int(s[4:6], 16)
    # clamp alpha
    a = max(0, min(int(alpha * opacity), 255))
    # format as AABBGGRR
    return f"{a:02x}{b:02x}{g:02x}{r:02x}"


def parse_extdata(placemark):
    attributes = {}
    ext = placemark.find("ExtendedData")
    if not ext:
        return attributes

    # <Data> format
    skip_col = ['name', 'folders', 'folder_name', 'description']
    for d in ext.find_all("Data"):
        key = d.get("name")
        val = d.find("value").text.strip() if d.find("value") else None
        if key in skip_col:
            continue

        attributes[key] = val

    # <SimpleData> format
    for d in ext.find_all("SimpleData"):
        key = d.get("name")
        val = d.text.strip()
        if key in skip_col:
            continue

        attributes[key] = val

    return attributes


def parse_geom(coords, geom_type):
    if not coords:
        return None

    coords = [tuple(map(float, c.split(","))) for c in coords.split() if c.strip()]
    if not coords:
        return None

    if geom_type == "Point":
        return Point(coords[0][0], coords[0][1])

    elif geom_type in ["LineString", "MultiLineString"]:
        if len(coords) < 2:
            print(f"Invalid Linestring: {coords}")
            return None

        return LineString([(x, y) for x, y, *_ in coords])

    elif geom_type in ["Polygon", "MultiPolygon"]:
        if len(coords) < 3:
            return None
        return Polygon([(x, y) for x, y, *_ in coords])

    return None

def parse_placemark(pm, folder_name, full_path):
    rows = []
    name = pm.find("name").text if pm.find("name") else "Unnamed"
    desc = pm.find("description").text if pm.find("description") else ""
    data = parse_extdata(pm)

    for geom_type in ["Point", "LineString", "Polygon"]:
        geom_tag = pm.find(geom_type)
        if geom_tag and geom_tag.find("coordinates"):
            coords = geom_tag.find("coordinates").text.strip()
            geometry = parse_geom(coords, geom_type)
            if geometry:
                row_data = {
                    "name": name,
                    "folders": full_path,
                    "folder_name": folder_name,
                    "description": desc,
                    **data,
                    "geometry": geometry,
                }
                rows.append(row_data)
    return rows



def parse_folder(folder, parent_name=None):
    results = []
    folder_name = folder.find("name").text if folder.find("name") else "Unnamed Folder"
    folder_name = folder_name.strip()
    full_path = f"{parent_name};{folder_name}" if parent_name else folder_name

    # Parse Placemark
    for pm in folder.find_all("Placemark", recursive=False):
        results.extend(parse_placemark(pm, folder_name, full_path))

    # Recursive folders
    for sub in folder.find_all("Folder", recursive=False):
        results.extend(parse_folder(sub, parent_name=full_path))
    return results


def parse_doc(doc, parent=None, source_prefix=""):
    result = []
    doc_name = doc.find("name").text if doc.find("name") else None

    full_doc = f"{parent};{doc_name}" if parent else doc_name
    print(f"ℹ️ Parsing {source_prefix}{full_doc}")

    for pm in doc.find_all("Placemark", recursive=False):
        result.extend(parse_placemark(pm, folder_name=doc_name, full_path=full_doc))

    for f in doc.find_all("Folder", recursive=False):
        result.extend(parse_folder(f))

    for sub_doc in doc.find_all("Document", recursive=False):
        result.extend(parse_doc(sub_doc, parent=full_doc, source_prefix=source_prefix))

    return result


def parse_kml(kml_file, source_prefix=""):
    soup = BeautifulSoup(kml_file.read(), "xml")
    doc = soup.find("Document")
    if doc is None:
        doc = soup.find("kml") or soup

    parsed = parse_doc(doc, source_prefix=source_prefix)
    return pd.DataFrame(parsed) if parsed else pd.DataFrame()

def read_kml(file: str):
    ext = os.path.splitext(file)[1].lower()
    basename = os.path.basename(file)
    print(f"🌏 Extracting KMZ File: {basename}")

    result = []

    def walk_kmz(zip_obj: zipfile.ZipFile, prefix: str):
        entries = [e for e in zip_obj.namelist()]
        kml_files = [e for e in entries if e.lower().endswith(".kml")]

        for kml_path in kml_files:
            with zip_obj.open(kml_path) as kml_fp:
                print(f"KML File ({prefix}): {kml_path}")
                parsed_kml = parse_kml(kml_fp, source_prefix=f"{prefix}{kml_path}::")
                result.append(parsed_kml)

        kmz_files = [e for e in entries if e.lower().endswith(".kmz")]
        if kmz_files:
            print(f"List of nested KMZ ({prefix}): {kmz_files}")

        for kmz_path in kmz_files:
            with zip_obj.open(kmz_path) as kmz_fp:
                kmz_bytes = kmz_fp.read()

            try:
                with zipfile.ZipFile(io.BytesIO(kmz_bytes), "r") as nested_zip:
                    nested_prefix = f"{prefix}{kmz_path}::"
                    print(f"➡️ Enter nested KMZ: {nested_prefix}")
                    walk_kmz(nested_zip, nested_prefix)
            except zipfile.BadZipFile:
                print(f"⚠️ Skipped invalid nested KMZ: {prefix}{kmz_path}")

    # --- entrypoint ---
    if ext == ".kmz":
        with zipfile.ZipFile(file, "r") as z:
            walk_kmz(z, prefix=f"{basename}::")

    elif ext == ".kml":
        with open(file, "rb") as f:
            parsed_kml = parse_kml(f, source_prefix=f"{basename}::")
            result.append(parsed_kml)

    else:
        raise ValueError(f"Invalid file format: {ext}")

    # --- Convert to GDF ---
    if not result:
        raise FileNotFoundError("No KML data found in KMZ / nested KMZ.")

    try:
        data_df = pd.concat(result, ignore_index=True)
        data_gdf = gpd.GeoDataFrame(data_df, geometry="geometry", crs="EPSG:4326")

        gt = data_gdf.geometry.geom_type
        points = data_gdf[gt == "Point"]
        lines = data_gdf[gt.isin(["LineString", "MultiLineString"])]
        polygons = data_gdf[gt.isin(["Polygon", "MultiPolygon"])]

        print(f"ℹ️ Total Points data extracted {len(points)}")
        print(f"ℹ️ Total Lines data extracted {len(lines)}")
        print(f"ℹ️ Total Polygon data extracted {len(polygons)}")
        print("✅ Extraction done.")
        return points, lines, polygons

    except Exception as e:
        raise ValueError(f"Error in GeoDataFrame conversion: {e}")


def validate_kmz_design(filepath:str, sep: str = "-"):
    points_kmz, lines_kmz, _ = read_kml(filepath)
    points_kmz = gpd.GeoDataFrame(points_kmz, geometry='geometry', crs='EPSG:4326') 
    lines_kmz = gpd.GeoDataFrame(lines_kmz, geometry='geometry', crs='EPSG:4326')  
    points_kmz = sanitize_header(points_kmz)
    lines_kmz = sanitize_header(lines_kmz)
    points_existing = points_kmz[points_kmz['folder_name'].str.lower().str.contains('site|hub')].copy()
    lines_existing = lines_kmz[lines_kmz['folder_name'].str.lower().str.contains('route')].copy()

    # CLEAN UNUSED DATA
    points_existing = points_existing[~points_existing['folder_name'].str.lower().str.contains("closure|odp|otb|obstacle")]
    points_existing = points_existing[~points_existing['name'].str.lower().str.contains("connection|closure|odp |otb |obstacle")]
    lines_existing = lines_existing[~lines_existing['folder_name'].str.lower().str.contains("bb|backbone|akses|existing")]

    # POINT EXISTING
    points_existing['site_id'] = points_existing['name'].str.strip()
    points_existing['site_name'] = points_existing['Site_Name'] if "Site_Name" in points_existing.columns else points_existing['name']
    points_existing['site_name'] = np.where(points_existing['site_name'].isna(), points_existing['site_id'], points_existing['site_name'])
    points_existing['site_type'] = points_existing['folders'].str.split(";").str[-1]
    points_existing['site_type'] = np.where(points_existing['site_type'].str.lower().str.contains('hub'), "FO Hub", 'Site List')
    points_existing['long'] = round(points_existing.geometry.to_crs(epsg=4326).x, 8)
    points_existing['lat'] = round(points_existing.geometry.to_crs(epsg=4326).y, 8) 
    points_existing['ring_name'] = points_existing['folders'].str.split(";").str[-2]
    points_existing['geometry'] = points_existing.geometry.force_2d()
    points_existing['program'] = points_existing['program'] if "program" in points_existing.columns else points_existing['folders'].str.extract(r';([A-Za-z0-9]{6,});')
    points_existing['region'] = points_existing['region'] if "region" in points_existing.columns else points_existing['folders'].str.extract(r';([A-Z0-9]{3,6});')
    points_existing['program'] = points_existing['program'].fillna("NA")
    points_existing = points_existing.dropna(how='all', axis=1)

    # LINES EXISTING
    sep_re = re.escape(sep)
    lines_existing['segment'] = lines_existing['name'].str.strip().str.replace(
        fr"^(?P<near>.+?)\s*{sep_re}\s*(?P<far>.+?)$",
        fr"\g<near>{sep}\g<far>",
        regex=True
    )
    lines_existing['near_end'] = lines_existing['segment'].str.split(sep).str[0]
    lines_existing['far_end'] = lines_existing['segment'].str.split(sep).str[-1]
    lines_existing['geometry'] = lines_existing.geometry.force_2d()
    lines_existing['length'] = lines_existing.geometry.to_crs(epsg=3857).length
    lines_existing['ring_name'] = lines_existing['folders'].str.split(";").str[-2]
    lines_existing['program'] = lines_existing['program'] if "program" in lines_existing.columns else lines_existing['folders'].str.extract(r';([A-Za-z0-9]{6,});')
    lines_existing['region'] = lines_existing['region'] if "region" in lines_existing.columns else lines_existing['folders'].str.extract(r';([A-Z0-9]{3,6});')
    lines_existing['fo_note'] = 'merged'
    lines_existing['program'] = lines_existing['program'].fillna("NA")
    lines_existing = lines_existing.dropna(how='all', axis=1)
    
    # COMPILE
    existing_col = ['site_id', 'site_name', 'site_type', 'long', 'lat', 'ring_name', 'program', 'region','geometry']
    for col in existing_col:
        if col not in points_existing.columns:
            if col == "region":
                existing_col.pop(existing_col.index('region'))
                continue
            raise ValueError(f"Column {col} not detected in Existing Point Sites data.")
    points_existing = points_existing[existing_col]

    existing_col = ['segment', 'name', 'near_end', 'far_end', 'fo_note', 'ring_name', 'program', 'region','geometry', 'length']
    for col in existing_col:
        if col not in lines_existing.columns:
            if col == "region":
                existing_col.pop(existing_col.index('region'))
                continue
            raise ValueError(f"Column {col} not detected in Existing Lines Sites data.")
    lines_existing = lines_existing[existing_col]

    if points_existing.empty:
        raise ValueError(f"Point data in existing kmz is empty")

    if lines_existing.empty:
        raise ValueError(f"Lines data in existing kmz is empty")

    print(f"ℹ️ Summary Validated Ring:")
    print(f"ℹ️ Total Points      : {len(points_existing):,}")
    print(f"ℹ️ Total LineString  : {len(points_existing):,}")
    return points_existing, lines_existing

def validate_kmz_ipl(filepath:str, sep: str = "-"):
    points_kmz, lines_kmz, _ = read_kml(filepath)
    points_kmz = gpd.GeoDataFrame(points_kmz, geometry='geometry', crs='EPSG:4326') 
    lines_kmz = gpd.GeoDataFrame(lines_kmz, geometry='geometry', crs='EPSG:4326')  
    points_kmz = sanitize_header(points_kmz)
    lines_kmz = sanitize_header(lines_kmz)
    points_data = points_kmz[~points_kmz['name'].str.lower().str.contains('connection')].copy()
    lines_data = lines_kmz.copy()

    # POINT DATA
    points_data['site_id']      = points_data['name'].str.strip().astype(str)
    points_data['site_name']    = points_data['Site_Name'] if "Site_Name" in points_data.columns else points_data['name']
    points_data['site_type']    = points_data['folders'].str.split(";").str[-1]
    points_data['site_type']    = np.where(points_data['site_type'].str.lower().str.contains('hub'), "FO Hub", 'Site List')
    points_data['long']         = round(points_data.geometry.to_crs(epsg=4326).x, 8)
    points_data['lat']          = round(points_data.geometry.to_crs(epsg=4326).y, 8)
    points_data['ring_name']    = points_data['folders'].str.split(";").str[-2]
    points_data['geometry']     = points_data.geometry.force_2d()
    points_data['program']      = points_data['program'] if "program" in points_data.columns else points_data['folders'].str.extract(r';([A-Za-z0-9]{6,});')
    points_data['region']       = points_data['region'] if "region" in points_data.columns else points_data['folders'].str.extract(r';([A-Z0-9]{3,6});')
    points_data['program']      = points_data['program'].fillna("NA")
    points_data = points_data.dropna(how='all', axis=1)

    # LINES DATA
    sep_re = re.escape(sep)
    lines_data['bb_fiber'] = lines_data['name'].str.split("/").str[-1]
    lines_data['name'] = lines_data['name'].str.split("/").str[0]
    lines_extracted = lines_data['name'].str.strip().str.extract(
        fr"^(?P<data_type>BB|Akses|s*)?(?P<near_end>[A-Za-z0-9 -_]+)\s*{sep_re}\s*(?P<far_end>[A-Za-z0-9 -_]+)(\_FO(?P<core>\d{{2}}))?$",
        expand=True
    )
    lines_extracted['near_end'] = lines_extracted['near_end'].str.strip()
    lines_extracted['far_end']  = lines_extracted['far_end'].str.strip()
    if 'core' in lines_extracted.columns:
        lines_extracted['core'] = lines_extracted['core'].fillna(24).astype(int)
    else:
        lines_extracted['core'] = 24

    lines_data['segment'] = lines_extracted['near_end'] + sep + lines_extracted['far_end']
    lines_data['segment'] = lines_data['segment'].str.strip()
    lines_data['near_end']  = lines_data['segment'].str.split(sep).str[0]
    lines_data['far_end']   = lines_data['segment'].str.split(sep).str[-1]
    lines_data['geometry']  = lines_data.geometry.force_2d()
    lines_data['length']    = lines_data.geometry.to_crs(epsg=3857).length
    lines_data['ring_name'] = lines_data['folders'].str.split(";").str[-2]
    lines_data['program']   = lines_data['program'] if "program" in lines_data.columns else lines_data['folders'].str.extract(r';([A-Za-z0-9]{6,});')
    lines_data['region']    = lines_data['region'] if "region" in lines_data.columns else lines_data['folders'].str.extract(r';([A-Z0-9]{3,6});')
    lines_data['fo_note']   = 'merged'
    lines_data['core']      = lines_extracted['core']
    lines_data['program']   = lines_data['program'].fillna("NA")
    lines_data = lines_data.dropna(how='all', axis=1)

    # COMPILE
    existing_col = ['name', 'folder_name', 'site_id', 'site_name', 'site_type', 'long', 'lat', 'ring_name', 'program', 'region','geometry']
    for col in existing_col:
        if col not in points_data.columns:
            if col == "region":
                existing_col.pop(existing_col.index('region'))
                continue
            raise ValueError(f"Column {col} not detected in Existing Point Sites data.")
    points_data = points_data[existing_col]

    existing_col = ['name', 'folder_name', 'segment', 'near_end', 'far_end', 'core', 'fo_note', 'ring_name', 'program', 'region','geometry', 'length']
    for col in existing_col:
        if col not in lines_data.columns:
            if col == "region":
                existing_col.pop(existing_col.index('region'))
                continue
            raise ValueError(f"Column {col} not detected in Existing Lines Sites data.")
    lines_data = lines_data[existing_col]

    if points_data.empty:
        raise ValueError(f"Point data in existing kmz is empty")

    if lines_data.empty:
        raise ValueError(f"Lines data in existing kmz is empty")

    # PARSE FOLDER
    # Lines
    topology = lines_data[lines_data['name'].str.lower().str.strip() == 'connection'].copy()
    route = lines_data[lines_data['folder_name'].str.lower().str.strip() == 'route'].copy()

    backbone = lines_data[lines_data['folder_name'].str.lower().str.contains('backbone')].copy()
    access = lines_data[lines_data['folder_name'].str.lower().str.contains('access|akses')].copy()
    fo_exist = lines_data[lines_data['folder_name'].str.lower().str.strip() == 'fo existing'].copy()
    pole_exist = lines_data[lines_data['folder_name'].str.lower().str.strip() == 'pole existing'].copy()

    # Points
    sites_data = points_data[points_data['folder_name'].str.lower().str.contains('site') | points_data['folder_name'].str.lower().str.contains('hub')].copy()
    fo_hub = sites_data[sites_data['folder_name'].str.lower().str.contains('hub')].copy()
    sitelist = sites_data[sites_data['folder_name'].str.lower().str.contains('site')].copy()

    near_set = set(route['near_end'].astype(str))
    mask_fhub = fo_hub[fo_hub['site_id'].astype(str).isin(near_set)].copy()

    route["hub_ring_id"] = route["ring_name"].astype(str) + "_" + route["near_end"].astype(str)
    mask_fhub = fo_hub.loc[
        fo_hub["site_id"].astype(str).isin(route["near_end"].astype(str)),
        ["site_id", "ring_name"]
    ].copy()

    mask_fhub["hub_ring_id"] = mask_fhub["ring_name"].astype(str) + "_" + mask_fhub["site_id"].astype(str)
    valid_ids = set(mask_fhub["hub_ring_id"])
    route["is_first"] = np.where(route["hub_ring_id"].isin(valid_ids), 1, 0)
    
    sites_data['is_first'] = np.where(
        (sites_data['site_id'].isin(route.loc[route['is_first'] == 1, 'near_end'].dropna()) & sites_data['site_type'].str.lower().str.contains('hub')) |
        (sites_data['site_id'].isin(route.loc[route['is_first'] == 1, 'far_end'].dropna()) & sites_data['site_type'].str.lower().str.contains('site')), 1, 0
    )

    first_points = sites_data[sites_data['is_first'] == 1].copy()
    first_route = route[route['is_first'] == 1].copy()
    first_points = first_points.drop_duplicates(subset=['ring_name', 'site_type'])
    first_route = first_route.drop_duplicates(subset=['ring_name', 'segment'])
    first_points['first_id'] = first_points['ring_name'].astype(str) + "_" + first_points['name'].astype(str)
    first_route['first_id'] = first_route['ring_name'].astype(str) + "_" + first_route['segment'].astype(str)

    first_point_ids = set(first_points['first_id'].astype(str))
    first_route_ids = set(first_route['first_id'].astype(str))

    # DEVICES DATA
    devices = ['odp', 'otb', 'closure']
    devices_mask = points_data['folder_name'].str.lower().str.contains('|'.join(devices))
    devices_data = points_data[devices_mask].copy()
    devices_data['device_name'] = devices_data['name'].str.strip().astype(str)
    devices_data = devices_data.drop(columns=['site_id'])
    devices_data['device_type'] = np.select(
        [
            devices_data['folder_name'].str.lower().str.contains('odp'),
            devices_data['folder_name'].str.lower().str.contains('otb'),
            devices_data['folder_name'].str.lower().str.contains('closure'),
        ],
        [ "ODP", "OTB", "Closure"],
        default="Unknown"
    )
    devices_data['core'] = devices_data['name'].str.extract(r"(?P<device_type>[\w+])\s*(?P<core>\((24|48|72|96|120|144)\))?", expand=True)["core"].fillna(24).astype(int)
    devices_data['identifier'] = np.select(
        [devices_data['device_type'] == "ODP",
        devices_data['device_type'] == "OTB",
        devices_data['device_type'] == "Closure"],
        [
            devices_data['name'].str.extract(r"(?P<device_type>ODP|OTB)(?:[\s\_]+(?P<ext>EXT))?[\s\-]*(?P<core>\((24|48|72|96|120|144)\))?[\s\-]+(?P<site_id>[A-Za-z0-9\ -_]+)$", expand=True)["site_id"].str.strip(),
            devices_data['name'].str.extract(r"(?P<device_type>ODP|OTB)(?:[\s\_]+(?P<ext>EXT))?[\s\-]*(?P<core>\((24|48|72|96|120|144)\))?[\s\-]+(?P<site_id>[A-Za-z0-9\ -_]+)$", expand=True)["site_id"].str.strip(),
            devices_data['name'].str.extract(r"(?P<device_type>\w+)\s*(?P<segment>[A-Za-z0-9\ -;_]+)$", expand=True)["segment"].str.strip(),
        ],
        default=devices_data['name'].str.strip()
    )
    devices_data['first_id'] = np.select(
        [
            devices_data['device_type'] == "ODP",
            devices_data['device_type'] == "OTB",
            devices_data['device_type'] == "Closure",
        ],
        [
            devices_data['ring_name'] + "_" + devices_data['identifier'].astype(str),
            devices_data['ring_name'] + "_" + devices_data['identifier'].astype(str),
            devices_data['ring_name'] + "_" + devices_data['identifier'].astype(str),
        ],
        default=0
    )
    devices_data['is_first'] = np.select(
        [
            devices_data['device_type'] == "ODP",
            devices_data['device_type'] == "OTB",
            devices_data['device_type'] == "Closure",
        ],
        [
            devices_data['first_id'].isin(first_point_ids).astype(int),
            devices_data['first_id'].isin(first_point_ids).astype(int),
            devices_data['first_id'].isin(first_route_ids).astype(int),
        ],
        default=0
    )

    # ASSIGN SEGMENT TO DEVICES
    devices_data['segment'] = None
    grouped_devices = devices_data.groupby('ring_name')
    for ring, group in grouped_devices:
        ring_lines = route[route['ring_name'] == ring]
        first_line = route[route['is_first'] == 1]
        if first_line.empty:
            print(f"🔴 No first line found for ring {ring}")
            continue

        ne_unique = set(ring_lines['near_end'].unique())
        fe_unique = set(ring_lines['far_end'].unique())
        for idx, row in group.iterrows():
            device_name = row['device_name']
            device_type = row['device_type']
            is_first = bool(row['is_first'])
            identifier = row['identifier']

            if device_type in ['ODP', 'OTB']:
                if is_first:
                    segment_line = first_line['segment'].values[0]
                else:
                    fe_line = ring_lines[ring_lines['far_end'] == identifier]
                    if fe_line.empty:
                        print(f"🔴 No far end line found for device {device_name} in ring {ring}")
                        print(f"Identifier {identifier} | First point ids: {first_point_ids}")
                        print(f"Ring Lines \n {ring_lines[['near_end', 'far_end']]}")
                        continue

                    segment_line = fe_line['segment'].values[0]
                devices_data.at[idx, 'segment'] = segment_line
            else:
                segment_line = identifier
                devices_data.at[idx, 'segment'] = segment_line

    odp = devices_data[devices_data['folder_name'].str.lower().str.contains('odp')].copy()
    otb = devices_data[devices_data['folder_name'].str.lower().str.contains('otb')].copy()
    closure = devices_data[devices_data['folder_name'].str.lower().str.contains('closure')].copy()

    # Obstacle
    obstacle = points_data[points_data['folder_name'].str.lower().str.contains('obstacle')].copy()
    obstacle['obstacle_type'] = np.select(
        [
            obstacle['name'].str.lower().str.contains('rail|kai'),
            obstacle['name'].str.lower().str.contains('toll'),
            obstacle['name'].str.lower().str.contains('bridge'),
        ],
        ['Railway','Toll Road','Bridge'],
        default='Not Defined'
    )
    obstacle[['near_end', 'far_end']] = obstacle['name'].str.extract(rf"(?P<obstacle_type>.+?)?\s*{sep}\s*(?P<near>.+?)\s*{sep}\s*(?P<far>.+?)$")[['near', 'far']]
    obstacle['segment'] = obstacle['near_end'] + sep + obstacle['far_end']
    
    # METADATA
    odp['ext_note'] = np.where(odp['name'].str.lower().str.contains('ext_'), 1, 0)
    otb['ext_note'] = np.where(otb['name'].str.lower().str.contains('ext_'), 1, 0)
    closure['ext_note'] = np.where(closure['name'].str.lower().str.contains('ext_'), 1, 0)
    odp_extracted = odp['name'].str.extract(r"^(?P<device>ODP|OTB|Closure)(_?P<core>\d{2})\s*(?P<site_id>\[\w+\s*\])$", expand=True)
    odp['core'] = odp_extracted['core'].fillna(24).astype(int) if 'core' in odp_extracted else 24
    otb_extracted = otb['name'].str.extract(r"^(?P<device>ODP|OTB|Closure)(_?P<core>\d{2})\s*(?P<site_id>\[\w+\s*\])$", expand=True)
    otb['core'] = otb_extracted['core'].fillna(24).astype(int) if 'core' in otb_extracted else 24

    data_compiled = [points_data, lines_data, fo_hub, sitelist, odp, otb, closure,
                     topology, route, backbone, access, fo_exist, pole_exist, obstacle]
    data_compiled_names = ['points_data', 'lines_data', 'fo_hub', 'sitelist', 'odp', 'otb', 'closure',
                           'topology', 'route', 'backbone', 'access', 'fo_exist', 'pole_exist', 'obstacle']

    data_dict = dict(zip(data_compiled_names, data_compiled))

    print(f"ℹ️ Summary Validated IPL:")
    for name, df in data_dict.items():
        print(f"ℹ️ {name:<20} : {len(df):,} records")

    print(f"✅ Extraction done.")
    return data_dict


if __name__ == "__main__":
    kmz_path = r"D:\JACOBS\PROJECT\TASK\2026\JAN\W2\Insert Task\KMZ PLAN FWA SURGE Batch 1 + 3.kmz"

    # KMZ DATA
    points_kmz, lines_kmz, _ = read_kml(kmz_path)
    # if not points_kmz.empty:
    #     points_kmz.to_parquet(r"D:\JACOBS\PROJECT\TASK\2026\JAN\W2\Insert Task\Export\Debug Insert\Point KMZ.parquet")
    # if not lines_kmz.empty:
    #     lines_kmz.to_parquet(r"D:\JACOBS\PROJECT\TASK\2026\JAN\W2\Insert Task\Export\Debug Insert\Lines KMZ.parquet")