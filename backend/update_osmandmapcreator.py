import os
import zipfile
import tempfile
import shutil
from dotenv import load_dotenv

# --- INITIALISATION ---
# Chargement des variables depuis le fichier .env
load_dotenv()

OSM_MAP_CREATOR_DIR = os.getenv("OSM_MAP_CREATOR_DIR")
CUSTOM_XML_PATH = os.getenv("CUSTOM_XML_PATH")

# Vérification que les variables sont bien définies
if not OSM_MAP_CREATOR_DIR or not CUSTOM_XML_PATH:
    print("Erreur : Les variables OSM_MAP_CREATOR_DIR ou CUSTOM_XML_PATH sont introuvables dans le .env.")
    exit(1)

# Construction des chemins complets
JAR_PATH = os.path.join(OSM_MAP_CREATOR_DIR, "lib", "OsmAnd-java-master-snapshot.jar")
TARGET_INTERNAL_PATH = "net/osmand/osm/rendering_types.xml"

def injecter_xml_dans_jar():
    print("Vérification des fichiers...")
    if not os.path.exists(JAR_PATH):
        print(f"Erreur : Le fichier JAR est introuvable à {JAR_PATH}")
        return
    if not os.path.exists(CUSTOM_XML_PATH):
        print(f"Erreur : Le fichier XML personnalisé est introuvable à {CUSTOM_XML_PATH}")
        return

    print(f"Ouverture de {os.path.basename(JAR_PATH)}...")
    
    # Création d'un chemin pour notre fichier jar temporaire
    fd, temp_jar_path = tempfile.mkstemp(suffix=".jar")
    os.close(fd) # On ferme le descripteur, on utilisera le chemin avec zipfile

    try:
        # On ouvre l'ancien JAR en lecture et le temporaire en écriture
        with zipfile.ZipFile(JAR_PATH, 'r') as jar_in:
            with zipfile.ZipFile(temp_jar_path, 'w', compression=zipfile.ZIP_DEFLATED) as jar_out:
                
                # 1. Copier tous les fichiers SAUF l'ancien rendering_types.xml
                for item in jar_in.infolist():
                    if item.filename != TARGET_INTERNAL_PATH:
                        # Lecture du fichier depuis l'ancien jar et écriture dans le nouveau
                        jar_out.writestr(item, jar_in.read(item.filename))
                
                # 2. Injecter notre fichier customisé avec le nom/chemin attendu dans le JAR
                print(f"Injection de {os.path.basename(CUSTOM_XML_PATH)} en tant que {TARGET_INTERNAL_PATH}...")
                jar_out.write(CUSTOM_XML_PATH, arcname=TARGET_INTERNAL_PATH)

        # 3. Remplacement propre de l'ancien JAR par le nouveau
        print("Remplacement de l'ancien fichier JAR...")
        shutil.move(temp_jar_path, JAR_PATH)
        
        print("\nTerminé avec succès ! Le fichier XML a été remplacé dans le JAR.")

    except Exception as e:
        print(f"\nUne erreur inattendue est survenue : {e}")
        # Nettoyage du fichier temporaire en cas d'échec
        if os.path.exists(temp_jar_path):
            os.remove(temp_jar_path)

if __name__ == "__main__":
    injecter_xml_dans_jar()