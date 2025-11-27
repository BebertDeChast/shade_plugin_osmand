'''
Optimized Shade Calculator for OSM Data
Copyright (c) 2025
License: MIT
'''

import osmium
import pandas as pd
import numpy as np
import time
from shapely.geometry import Point, Polygon, LineString, MultiPolygon
from shapely.ops import transform, unary_union
from shapely.affinity import translate
from shapely.strtree import STRtree
from pyproj import CRS, Transformer
from pvlib import solarposition

# --- CONFIGURATION ---
INPUT_FILE = "backend\\nantes.pbf"
OUTPUT_FILE = "backend\\nantes_with_shade.pbf"
DATE_STR = "2025-06-21"  # Solstice d'été
TARGET_TIMES = ["10:00:00"]
TIMEZONE = "Europe/Paris"

# Paramètres physiques
ROAD_WIDTH = 10.0
BUILDING_SEARCH_RADIUS = 60.0  # Distance max d'un bâtiment impactant
TREE_SEARCH_RADIUS = 30.0
DEFAULT_BUILDING_HEIGHT = 10.0 # Moyenne urbaine (R+2/3)
DEFAULT_TREE_HEIGHT = 8.0
DEFAULT_TREE_WIDTH = 4.0

class GeometryManager:
    """Gère le chargement et l'indexation spatiale des obstacles (bâtiments/arbres)."""
    def __init__(self, transformer):
        self.transformer = transformer
        self.buildings = []
        self.trees = []
        self.center_point = None
        
        # Pour les arbres, on crée un modèle de base carré centré sur 0,0
        hw = DEFAULT_TREE_WIDTH / 2
        self.base_tree_poly = Polygon([(-hw, -hw), (hw, -hw), (hw, hw), (-hw, hw)])

    def load_data(self, input_file):
        print("Loading buildings and trees...")
        
        class DataHandler(osmium.SimpleHandler):
            def __init__(self, manager):
                super().__init__()
                self.manager = manager
                self.nodes_coords = {} # Cache temporaire pour construire les ways
                self.bounds_accum = [] # Pour trouver le centre

            def node(self, n):
                # On stocke les coords pour les bâtiments, mais aussi on détecte les arbres
                pt = (n.lon, n.lat)
                if "natural" in n.tags and n.tags["natural"] == "tree":
                    # Projection immédiate
                    x, y = self.manager.transformer.transform(n.lon, n.lat)
                    # On crée directement le polygone de l'arbre positionné
                    tree_poly = translate(self.manager.base_tree_poly, xoff=x, yoff=y)
                    self.manager.trees.append(tree_poly)
                
                # Petit hack pour le centre approx (1 noeud sur 1000)
                if n.id % 1000 == 0:
                    self.bounds_accum.append(pt)

            def way(self, w):
                if "building" in w.tags:
                    try:
                        # Reconstruction rapide de la géométrie
                        coords = []
                        for n in w.nodes:
                            coords.append((n.lon, n.lat))
                        if len(coords) >= 3:
                            # Projection du polygone
                            poly_ll = Polygon(coords)
                            poly_m = transform(self.manager.transformer.transform, poly_ll)
                            self.manager.buildings.append(poly_m)
                    except Exception:
                        pass # Ignorer géométries invalides

        handler = DataHandler(self)
        handler.apply_file(input_file, locations=True)
        
        # Calcul du centre approximatif pour la position solaire
        if handler.bounds_accum:
            lons = [p[0] for p in handler.bounds_accum]
            lats = [p[1] for p in handler.bounds_accum]
            self.center_point = (sum(lats)/len(lats), sum(lons)/len(lons))
        else:
            self.center_point = (47.218, -1.553) # Fallback Nantes

        print(f"Loaded {len(self.buildings)} buildings and {len(self.trees)} trees.")
        
        # Construction des index spatiaux (STRtree est très rapide)
        print("Building spatial indices...")
        self.building_tree = STRtree(self.buildings) if self.buildings else None
        self.tree_tree = STRtree(self.trees) if self.trees else None

class ShadowCalculator:
    """Calcule les paramètres solaires et génère les ombres."""
    def __init__(self, lat, lon, dates_dict):
        self.configs = {}
        
        for label, ts in dates_dict.items():
            solpos = solarposition.get_solarposition(ts, lat, lon)
            azimuth = solpos['azimuth'].values[0]
            elevation = solpos['elevation'].values[0]
            
            # Vecteur d'ombre unitaire (pour une hauteur de 1m)
            # Longueur de l'ombre = h / tan(elevation)
            # Delta X = Longueur * sin(azimuth - 180) (Attention aux conventions geo vs math)
            # PVLib Azimuth: N=0, E=90, S=180, W=270
            # Math radians: E=0, N=pi/2
            
            if elevation <= 0:
                print(f"Warning: Sun below horizon for {label}")
                continue

            # Facteur d'étirement de l'ombre
            shadow_factor = 1.0 / np.tan(np.radians(elevation))
            
            # Conversion Azimuth (Nord=0, Clockwise) vers Math (Est=0, Counter-Clockwise)
            # Math_angle = 90 - Azimuth
            theta = np.radians(90 - azimuth)
            
            dx_per_meter = shadow_factor * np.cos(theta)
            dy_per_meter = shadow_factor * np.sin(theta)
            
            self.configs[label] = {
                "dx": dx_per_meter,
                "dy": dy_per_meter,
                "elevation": elevation
            }
            print(f"Config {label}: Elev={elevation:.1f}°, Factor={shadow_factor:.2f}")

    def get_shadows_on_road(self, road_poly, geo_manager):
        """Retourne un dictionnaire {label: merged_shadow_polygon}."""
        results = {}
        road_bounds = road_poly.bounds
        
        # 1. Récupérer les candidats (Bounding Box Query)
        # On élargit la zone de recherche pour attraper les hauts bâtiments loin
        search_area = road_poly.buffer(BUILDING_SEARCH_RADIUS)
        
        candidate_bld_indices = []
        candidate_tree_indices = []
        
        if geo_manager.building_tree:
            candidate_bld_indices = geo_manager.building_tree.query(search_area)
        
        if geo_manager.tree_tree:
             # Rayon plus court pour les arbres
            candidate_tree_indices = geo_manager.tree_tree.query(road_poly.buffer(TREE_SEARCH_RADIUS))

        # Pour chaque configuration horaire (10h, 14h...)
        for label, config in self.configs.items():
            dx = config["dx"]
            dy = config["dy"]
            shadow_polys = []

            # A. Traitement des bâtiments
            for idx in candidate_bld_indices:
                bldg = geo_manager.buildings[idx]
                # Projection vectorielle simple : on déplace le polygone
                # L'ombre est le "Convex Hull" de (Base U Sommet_Décalé)
                # Mais pour être rapide : on translate le bâtiment et on fait un polygone
                # reliant les points extrêmes.
                
                # Version optimisée : Translate building geometry
                offset_x = dx * DEFAULT_BUILDING_HEIGHT
                offset_y = dy * DEFAULT_BUILDING_HEIGHT
                roof = translate(bldg, xoff=offset_x, yoff=offset_y)
                
                # L'ombre au sol est l'union de la base et du toit projeté
                # (approximation convexe rapide pour bâtiments rectangulaires)
                shadow = bldg.convex_hull.union(roof.convex_hull).convex_hull
                
                if shadow.intersects(road_poly):
                    shadow_polys.append(shadow)

            # B. Traitement des arbres
            for idx in candidate_tree_indices:
                tree = geo_manager.trees[idx]
                offset_x = dx * DEFAULT_TREE_HEIGHT
                offset_y = dy * DEFAULT_TREE_HEIGHT
                
                # Pour un arbre (carré), l'ombre est simplement le carré décalé 
                # + le remplissage entre les deux.
                canopy_proj = translate(tree, xoff=offset_x, yoff=offset_y)
                # Convex hull est très rapide sur des carrés
                shadow = tree.union(canopy_proj).convex_hull
                
                if shadow.intersects(road_poly):
                    shadow_polys.append(shadow)

            # C. Fusion et calcul
            if shadow_polys:
                # unary_union est optimisé dans Shapely 2.0
                merged_shadows = unary_union(shadow_polys)
                # On coupe l'ombre par la route
                intersection = merged_shadows.intersection(road_poly)
                if not intersection.is_empty:
                    results[label] = (intersection.area / road_poly.area) * 100
                else:
                    results[label] = 0.0
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
                # Conversion géométrique Route
                coords = [(n.lon, n.lat) for n in w.nodes]
                if len(coords) < 2:
                    self.writer.add_way(w)
                    return

                line = LineString(coords)
                line_m = transform(self.transformer.transform, line)
                road_poly = line_m.buffer(ROAD_WIDTH / 2, cap_style=2) # Flat cap pour précision

                # Calcul des ombres
                shades = self.shadow_calc.get_shadows_on_road(road_poly, self.geo_manager)

                # Mise à jour des tags
                if any(v > 0 for v in shades.values()):
                    new_tags = dict(w.tags) # Convertir en dict python standard
                    for label, percent in shades.items():
                        new_tags[f"shade:{label}"] = f"{int(round(percent))}%"
                    
                    # Reconstruction objet Osmium (un peu verbeux mais nécessaire)
                    wk = osmium.osm.mutable.Way(w)
                    wk.tags = new_tags
                    self.writer.add_way(wk)
                    self.modified_count += 1
                else:
                    self.writer.add_way(w)

            except Exception as e:
                # En cas d'erreur géométrique, on écrit le way original
                # print(f"Error processing way {w.id}: {e}")
                self.writer.add_way(w)
        else:
            self.writer.add_way(w)

    def node(self, n):
        self.writer.add_node(n)

    def relation(self, r):
        self.writer.add_relation(r)

    def close(self):
        self.writer.close()

if __name__ == "__main__":
    start_total = time.time()
    
    # 1. Setup Coordinates
    crs_latlon = CRS("EPSG:4326")
    crs_projected = CRS("EPSG:32630") # UTM Zone 30N (Adaptez si hors France Ouest/Nantes)
    transformer = Transformer.from_crs(crs_latlon, crs_projected, always_xy=True)

    # 2. Load Geometry (Buildings/Trees)
    geo_manager = GeometryManager(transformer)
    geo_manager.load_data(INPUT_FILE)

    # 3. Setup Solar Config
    print("Configuring solar positions...")
    times_map = {
        t.replace(":00:00", ""): pd.Timestamp(f"{DATE_STR} {t}").tz_localize(TIMEZONE)
        for t in TARGET_TIMES
    }
    
    lat, lon = geo_manager.center_point
    shadow_calc = ShadowCalculator(lat, lon, times_map)

    # 4. Process Roads & Write Output
    print("Processing roads and calculating shade...")
    processor = RoadProcessor(OUTPUT_FILE, geo_manager, shadow_calc, transformer)
    processor.apply_file(INPUT_FILE, locations=True)
    processor.close()

    end_total = time.time()
    print(f"Done! Modified {processor.modified_count} roads.")
    print(f"Total execution time: {end_total - start_total:.2f} seconds")