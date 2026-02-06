'''
Optimized Shade Calculator for OSM Data (Routing Ready)
Copyright (c) 2025
License: MIT
'''

import osmium
import pandas as pd
import numpy as np
import time
import subprocess
import os
import shutil
from shapely.geometry import Polygon, LineString
from shapely.ops import transform, unary_union
from shapely.affinity import translate
from shapely.strtree import STRtree
from pyproj import CRS, Transformer
from pvlib import solarposition

# --- CONFIGURATION ---
INPUT_FILE = r"C:\Users\othma\Desktop\shade_plugin_osmand\backend\nantes.pbf"
OUTPUT_PBF = r"C:\Users\othma\Desktop\shade_plugin_osmand\backend\nantes_with_shade.pbf"
OUTPUT_OBF = r"C:\Users\othma\Desktop\shade_plugin_osmand\backend\nantes_with_shade.obf"
DATE_STR = "2025-06-21"  # Summer Solstice
TARGET_TIMES = ["10:00:00"]  
TIMEZONE = "Europe/Paris"

# OsmAndMapCreator path
OSMAND_MAP_CREATOR_PATH = r"C:\Users\othma\Downloads\OsmAndMapCreator-main"

# Physical Parameters
ROAD_WIDTH = 10.0
BUILDING_SEARCH_RADIUS = 60.0
TREE_SEARCH_RADIUS = 30.0
DEFAULT_BUILDING_HEIGHT = 10.0
DEFAULT_TREE_HEIGHT = 8.0
DEFAULT_TREE_WIDTH = 4.0

class GeometryManager:
    def __init__(self, transformer):
        self.transformer = transformer
        self.buildings = []
        self.trees = []
        self.center_point = None
        hw = DEFAULT_TREE_WIDTH / 2
        self.base_tree_poly = Polygon([(-hw, -hw), (hw, -hw), (hw, hw), (-hw, hw)])

    def load_data(self, input_file):
        print("Loading buildings and trees...")
        class DataHandler(osmium.SimpleHandler):
            def __init__(self, manager):
                super().__init__()
                self.manager = manager
                self.bounds_accum = []
            def node(self, n):
                if "natural" in n.tags and n.tags["natural"] == "tree":
                    x, y = self.manager.transformer.transform(n.lon, n.lat)
                    tree_poly = translate(self.manager.base_tree_poly, xoff=x, yoff=y)
                    self.manager.trees.append(tree_poly)
                if n.id % 1000 == 0:
                    self.bounds_accum.append((n.lon, n.lat))
            def way(self, w):
                if "building" in w.tags:
                    try:
                        coords = [(n.lon, n.lat) for n in w.nodes]
                        if len(coords) >= 3:
                            poly_ll = Polygon(coords)
                            poly_m = transform(self.manager.transformer.transform, poly_ll)
                            self.manager.buildings.append(poly_m)
                    except Exception:
                        pass
        handler = DataHandler(self)
        handler.apply_file(input_file, locations=True)
        if handler.bounds_accum:
            lons = [p[0] for p in handler.bounds_accum]
            lats = [p[1] for p in handler.bounds_accum]
            self.center_point = (sum(lats)/len(lats), sum(lons)/len(lons))
        else:
            self.center_point = (47.218, -1.553)
        print(f"Loaded {len(self.buildings)} buildings and {len(self.trees)} trees.")
        self.building_tree = STRtree(self.buildings) if self.buildings else None
        self.tree_tree = STRtree(self.trees) if self.trees else None

class ShadowCalculator:
    def __init__(self, lat, lon, dates_dict):
        self.configs = {}
        for label, ts in dates_dict.items():
            solpos = solarposition.get_solarposition(ts, lat, lon)
            azimuth = solpos['azimuth'].values[0]
            elevation = solpos['elevation'].values[0]
            if elevation <= 0: continue
            shadow_factor = 1.0 / np.tan(np.radians(elevation))
            theta = np.radians(90 - azimuth)
            self.configs[label] = {
                "dx": shadow_factor * np.cos(theta),
                "dy": shadow_factor * np.sin(theta),
                "elevation": elevation
            }

    def get_shadows_on_road(self, road_poly, geo_manager):
        results = {}
        if not geo_manager.building_tree and not geo_manager.tree_tree:
            return results
            
        search_area = road_poly.buffer(BUILDING_SEARCH_RADIUS)
        candidate_bld = geo_manager.building_tree.query(search_area) if geo_manager.building_tree else []
        candidate_tree = geo_manager.tree_tree.query(road_poly.buffer(TREE_SEARCH_RADIUS)) if geo_manager.tree_tree else []

        for label, config in self.configs.items():
            shadow_polys = []
            dx, dy = config["dx"], config["dy"]
            for idx in candidate_bld:
                bldg = geo_manager.buildings[idx]
                roof = translate(bldg, xoff=dx*DEFAULT_BUILDING_HEIGHT, yoff=dy*DEFAULT_BUILDING_HEIGHT)
                shadow = bldg.convex_hull.union(roof.convex_hull).convex_hull
                if shadow.intersects(road_poly):
                    shadow_polys.append(shadow)
            for idx in candidate_tree:
                tree = geo_manager.trees[idx]
                canopy = translate(tree, xoff=dx*DEFAULT_TREE_HEIGHT, yoff=dy*DEFAULT_TREE_HEIGHT)
                shadow = tree.union(canopy).convex_hull
                if shadow.intersects(road_poly):
                    shadow_polys.append(shadow)
            if shadow_polys:
                merged = unary_union(shadow_polys)
                intersection = merged.intersection(road_poly)
                results[label] = (intersection.area / road_poly.area) * 100 if not intersection.is_empty else 0.0
            else:
                results[label] = 0.0
        return results

class RoadProcessor(osmium.SimpleHandler):
    def __init__(self, output_path, geo_manager, shadow_calc, transformer):
        super().__init__()
        self.writer = osmium.SimpleWriter(output_path)
        self.geo_manager = geo_manager
        self.shadow_calc = shadow_calc
        self.transformer = transformer
        self.modified_count = 0

    def way(self, w):
        if "highway" in w.tags:
            try:
                coords = [(n.lon, n.lat) for n in w.nodes]
                if len(coords) < 2:
                    self.writer.add_way(w)
                    return
                line_m = transform(self.transformer.transform, LineString(coords))
                road_poly = line_m.buffer(ROAD_WIDTH / 2, cap_style=2)
                shades = self.shadow_calc.get_shadows_on_road(road_poly, self.geo_manager)

                if any(v > 0 for v in shades.values()):
                    new_tags = dict(w.tags)
                    total_shade = 0
                    count = 0
                    morning_vals = []
                    
                    for label, percent in shades.items():
                        hour = int(label)
                        val_float = percent / 100.0
                        total_shade += val_float
                        count += 1
                        if 6 <= hour < 12: morning_vals.append(val_float)

                    if count > 0: new_tags["shade_avg"] = f"{total_shade / count:.2f}"
                    if morning_vals: new_tags["shade_morning"] = f"{sum(morning_vals)/len(morning_vals):.2f}"
                    
                    # Add dummy values for afternoon/evening so routing.xml is happy even if we didn't calc them
                    if "shade_afternoon" not in new_tags: new_tags["shade_afternoon"] = "0.0"
                    if "shade_evening" not in new_tags: new_tags["shade_evening"] = "0.0"

                    wk = osmium.osm.mutable.Way(w)
                    wk.tags = new_tags
                    self.writer.add_way(wk)
                    self.modified_count += 1
                else:
                    self.writer.add_way(w)
            except Exception:
                self.writer.add_way(w)
        else:
            self.writer.add_way(w)
    def node(self, n): self.writer.add_node(n)
    def relation(self, r): self.writer.add_relation(r)
    def close(self): self.writer.close()

def convert_pbf_to_obf(pbf_path, obf_path):
    print(f"Converting {pbf_path} to OBF format...")
    
    # 1. Create rendering_types.xml programmatically to ensure it exists
    rt_path = os.path.join(OSMAND_MAP_CREATOR_PATH, "rendering_types.xml")
    rendering_xml_content = """<?xml version="1.0" encoding="utf-8"?>
<rendering_types>
    <category name="routing">
        <type tag="shade_avg" minzoom="1" />
        <type tag="shade_morning" minzoom="1" />
        <type tag="shade_afternoon" minzoom="1" />
        <type tag="shade_evening" minzoom="1" />
    </category>
</rendering_types>
"""
    try:
        with open(rt_path, "w", encoding="utf-8") as f:
            f.write(rendering_xml_content)
        print(f" specific rendering_types.xml created at: {rt_path}")
    except Exception as e:
        print(f" Failed to write rendering_types.xml: {e}")
        return False

    # 2. Setup Java Command
    lib_folder = os.path.join(OSMAND_MAP_CREATOR_PATH, "lib")
    if not os.path.exists(lib_folder):
        print(f" Error: 'lib' folder not found at {lib_folder}")
        return False

    # IMPORTANT: ".;" at start forces Java to look in current dir for rendering_types.xml
    jar_files = [os.path.join(lib_folder, f) for f in os.listdir(lib_folder) if f.endswith('.jar')]
    classpath = ".;OsmAndMapCreator.jar;" + ";".join(jar_files)
    
    cmd = [
        "java",
        "-Djava.util.logging.config.file=logging.properties",
        "-Dfile.encoding=UTF-8",
        "-Xms64M",
        "-Xmx2G",
        "-cp", classpath,
        "net.osmand.MainUtilities",
        "generate-obf",
        # Explicitly pointing to the rendering types file is safer
        f"-Dnet.osmand.rendering_types={rt_path}",
        os.path.abspath(pbf_path)
    ]
    
    try:
        print(f"Running OBF generation...")
        subprocess.run(cmd, cwd=OSMAND_MAP_CREATOR_PATH, check=True, timeout=3600)
        
        # 3. Find and Move Result
        base_name = os.path.basename(pbf_path).replace('.pbf', '.obf').replace('.osm', '.obf')
        capitalized_name = base_name[0].upper() + base_name[1:]
        
        candidates = [
            os.path.join(OSMAND_MAP_CREATOR_PATH, capitalized_name),
            os.path.join(OSMAND_MAP_CREATOR_PATH, base_name),
            os.path.join(OSMAND_MAP_CREATOR_PATH, capitalized_name.replace('.obf', '_2.obf'))
        ]
        
        for cand in candidates:
            if os.path.exists(cand):
                if os.path.exists(obf_path):
                    try: os.remove(obf_path)
                    except: pass
                shutil.move(cand, obf_path)
                return True
        return False

    except subprocess.CalledProcessError as e:
        print(f" Java execution failed with code {e.returncode}")
        return False
    except Exception as e:
        print(f" Error: {e}")
        return False

if __name__ == "__main__":
    start = time.time()
    
    # Setup
    crs_latlon = CRS("EPSG:4326")
    crs_projected = CRS("EPSG:32630")
    transformer = Transformer.from_crs(crs_latlon, crs_projected, always_xy=True)
    
    geo_manager = GeometryManager(transformer)
    geo_manager.load_data(INPUT_FILE)
    
    times_map = {t.split(':')[0]: pd.Timestamp(f"{DATE_STR} {t}").tz_localize(TIMEZONE) for t in TARGET_TIMES}
    lat, lon = geo_manager.center_point
    shadow_calc = ShadowCalculator(lat, lon, times_map)

    # Process
    if os.path.exists(OUTPUT_PBF): os.remove(OUTPUT_PBF)
    processor = RoadProcessor(OUTPUT_PBF, geo_manager, shadow_calc, transformer)
    processor.apply_file(INPUT_FILE, locations=True)
    processor.close()

    print(f"PBF Generated: {OUTPUT_PBF}")
    print(f"Roads modified: {processor.modified_count}")

    # Convert
    print("\n--- Generating OBF ---")
    if convert_pbf_to_obf(OUTPUT_PBF, OUTPUT_OBF):
        print(f" SUCCESS: {OUTPUT_OBF}")
    else:
        print(" FAILURE: OBF not generated.")

    print(f"Time: {time.time() - start:.2f}s")