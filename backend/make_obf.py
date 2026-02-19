import os
import subprocess
import sys
import argparse
import shutil
from dotenv import load_dotenv

# --- GESTION DES ARGUMENTS EN LIGNE DE COMMANDE ---
parser = argparse.ArgumentParser(description="Script d'automatisation pour OsmAndMapCreator (Génération OBF et Inspection)")
parser.add_argument("--inspect", action="store_true", help="Génère un rapport de routing après la création de l'OBF")
args = parser.parse_args()

# --- CHARGEMENT DE LA CONFIGURATION ---
load_dotenv()

OSM_MAP_CREATOR_DIR = os.getenv("OSM_MAP_CREATOR_DIR")
PBF_FILE_PATH = os.getenv("PBF_FILE_PATH")
OUTPUT_DIR = os.getenv("OUTPUT_DIR")

# --- VÉRIFICATIONS ---
if not all([OSM_MAP_CREATOR_DIR, PBF_FILE_PATH, OUTPUT_DIR]):
    print("Erreur : Veuillez définir OSM_MAP_CREATOR_DIR, PBF_FILE_PATH et OUTPUT_DIR dans le fichier .env")
    sys.exit(1)

# Création du dossier de sortie s'il n'existe pas
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- DETECTION DU SYSTEME D'EXPLOITATION ---
is_windows = (os.name == 'nt')
script_ext = ".bat" if is_windows else ".sh"

utilities_script = os.path.join(OSM_MAP_CREATOR_DIR, f"utilities{script_ext}")
inspector_script = os.path.join(OSM_MAP_CREATOR_DIR, f"inspector{script_ext}")

# --- DEFINITION DES CHEMINS ---
nom_pbf = os.path.basename(PBF_FILE_PATH)
nom_base = os.path.splitext(nom_pbf)[0] # Ex: "ma_carte" au lieu de "ma_carte.pbf"
nom_obf = nom_base + ".obf"

# Le chemin final souhaité pour l'OBF
final_obf_path = os.path.join(OUTPUT_DIR, nom_obf)
# Le chemin final pour le rapport CSV (nommé dynamiquement)
csv_report_path = os.path.join(OUTPUT_DIR, f"result_{nom_base}.csv")

def executer_pipeline():
    # 1. GÉNÉRATION DE L'OBF
    print(f"--- Étape 1 : Génération de {nom_obf} ---")
    cmd_generate = [utilities_script, "generate-obf", PBF_FILE_PATH]
    
    try:
        print(f"Exécution : {' '.join(cmd_generate)}")
        # Ajout du paramètre cwd pour exécuter la commande dans le bon dossier
        subprocess.run(cmd_generate, cwd=OSM_MAP_CREATOR_DIR, check=True)
        print("-> Génération OBF terminée avec succès.")
    except subprocess.CalledProcessError as e:
        print(f"Erreur lors de la génération de l'OBF : {e}")
        sys.exit(1)

    # RECHERCHE ET DÉPLACEMENT DE L'OBF GÉNÉRÉ
    # On ajoute la recherche dans le dossier d'OsmAndMapCreator
    obf_genere_osmand = os.path.join(OSM_MAP_CREATOR_DIR, nom_obf)
    obf_genere_courant = os.path.join(os.getcwd(), nom_obf)
    obf_genere_source = os.path.join(os.path.dirname(PBF_FILE_PATH), nom_obf)
    
    obf_trouve = None
    if os.path.exists(obf_genere_osmand):
        obf_trouve = obf_genere_osmand
    elif os.path.exists(obf_genere_courant):
        obf_trouve = obf_genere_courant
    elif os.path.exists(obf_genere_source):
        obf_trouve = obf_genere_source

    if not obf_trouve:
        print(f"Erreur : Le fichier {nom_obf} est introuvable après la génération.")
        sys.exit(1)

    # Déplacement vers le dossier de sortie
    if obf_trouve != final_obf_path:
        if os.path.exists(final_obf_path):
            os.remove(final_obf_path)
        shutil.move(obf_trouve, final_obf_path)
        print(f"-> Fichier OBF déplacé dans : {OUTPUT_DIR}\n")
    else:
        print(f"-> Fichier OBF déjà présent dans : {OUTPUT_DIR}\n")

    # 2. INSPECTION ET RAPPORT ROUTING (Uniquement si --inspect est utilisé)
    if args.inspect:
        print(f"--- Étape 2 : Inspection et création de result_{nom_base}.csv ---")
        cmd_inspect = [inspector_script, "-vrouting", final_obf_path]
        
        try:
            print(f"Exécution : {' '.join(cmd_inspect)} > {csv_report_path}")
            with open(csv_report_path, "w", encoding="utf-8") as fichier_csv:
                # Ajout du paramètre cwd ici aussi
                subprocess.run(cmd_inspect, cwd=OSM_MAP_CREATOR_DIR, stdout=fichier_csv, check=True)
            print(f"-> Rapport généré avec succès à l'emplacement : {csv_report_path}\n")
        except subprocess.CalledProcessError as e:
            print(f"Erreur lors de l'inspection de l'OBF : {e}")
            sys.exit(1)
    else:
        print("--- Étape 2 : Inspection ignorée (utilisez --inspect pour l'activer) ---")
        print("\nProcessus terminé avec succès.")

if __name__ == "__main__":
    if not is_windows:
        os.chmod(utilities_script, 0o755)
        if args.inspect:
            os.chmod(inspector_script, 0o755)
            
    executer_pipeline()