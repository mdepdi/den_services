import simplekml
import html
import os
import zipfile
import pandas as pd
import geopandas as gpd
from bs4 import BeautifulSoup
from shapely.geometry import Point, LineString, Polygon

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
    for d in ext.find_all("Data"):
        key = d.get("name")
        val = d.find("value").text.strip() if d.find("value") else None
        attributes[key] = val

    # <SimpleData> format
    for d in ext.find_all("SimpleData"):
        key = d.get("name")
        val = d.text.strip()
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


def parse_folder(folder, parent_name=None):
    results = []
    folder_name = folder.find("name").text if folder.find("name") else "Unnamed Folder"
    full_path = f"{parent_name};{folder_name}" if parent_name else folder_name

    for pm in folder.find_all("Placemark", recursive=False):
        name = pm.find("name").text if pm.find("name") else "Unnamed"
        desc = pm.find("description").text if pm.find("description") else ""
        data = parse_extdata(pm)

        for geom in ["Point", "LineString", "Polygon"]:
            geom_tag = pm.find(geom)
            if geom_tag and geom_tag.find("coordinates"):
                coords = geom_tag.find("coordinates").text.strip()
                geometry = parse_geom(coords, geom_type=geom)
                if geometry:
                    row_data = {
                        "name": name,
                        "folders": full_path,
                        "folder_name": folder_name,
                        "description": desc,
                        **data,
                        "geometry": geometry
                    }
                    results.append(row_data)

    # Recursive folders
    for sub in folder.find_all("Folder", recursive=False):
        results.extend(parse_folder(sub, parent_name=full_path))
    return results


def parse_doc(doc, parent=None):
    result = []
    doc_name = doc.find("name").text if doc.find("name") else None
    full_doc = f"{parent};{doc_name}" if parent else doc_name
    print(f"ℹ️ Parsing {full_doc}")

    all_folders = doc.find_all("Folder", recursive=False)
    for f in all_folders:
        result.extend(parse_folder(f))

    for sub_doc in doc.find_all("Document", recursive=False):
        result.extend(parse_doc(sub_doc, parent=full_doc))
    return result


def parse_kml(kml_file):
    soup = BeautifulSoup(kml_file.read(), "xml")
    doc = soup.find("Document")
    parsed = parse_doc(doc)
    return pd.DataFrame(parsed) if parsed else pd.DataFrame()


def read_kml(file):
    ext = os.path.splitext(file)[1].lower()
    basename = os.path.basename(file)
    print(f"🌏 Extracting KMZ File: {basename}")
    
    result = []
    if ext == ".kmz":
        with zipfile.ZipFile(file, "r") as z:
            kml_files = [f for f in z.namelist() if f.endswith(".kml")]
            print(f"List of KML Files: {kml_files}")
            if not kml_files:
                raise FileNotFoundError("No .kml file found inside the .kmz archive.")

            for kml in kml_files:
                with z.open(kml) as kml_file:
                    print(f"KML File: {kml_file}")
                    parsed_kml = parse_kml(kml_file)
                    result.append(parsed_kml)

    elif ext == ".kml":
        with open(file, "rb") as f:
            parsed_kml = parse_kml(f)
            result.append(parsed_kml)

    else:
        raise ValueError(f"Invalid file format: {ext}")

    # CONVERT TO GDF
    try:
        data_df = pd.concat(result)
        print(data_df.head())
        data_gdf = gpd.GeoDataFrame(data_df, geometry='geometry', crs="EPSG:4326")
        points = data_gdf[data_gdf.geometry.type == "Point"]
        lines = data_gdf[data_gdf.geometry.type.isin(["LineString", "MultiLineString"])]
        polygons = data_gdf[data_gdf.geometry.type.isin(["Polygon", "MultiPolygon"])]

        print(f"ℹ️ Total Points data extracted {len(points)}")
        print(f"ℹ️ Total Lines data extracted {len(lines)}")
        print(f"ℹ️ Total Polygon data extracted {len(polygons)}")
        print(f"✅ Extraction done.")
    except Exception as e:
        raise ValueError(f"Error in GeoDataFrame conversion: {e}")
    
    return points, lines, polygons
