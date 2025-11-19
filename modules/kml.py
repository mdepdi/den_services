import simplekml
import html

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