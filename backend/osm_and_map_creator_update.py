import os
import shutil

# --- CONFIGURATION ---
DOSSIER_SOURCE = r"C:\\Users\\Humbert\\Downloads\\OsmAnd-java-master-snapshot"
DOSSIER_DESTINATION = r"C:\\Users\\Humbert\\Downloads\\OsmAndMapCreator-2025-06-01\\lib"
NOM_JAR = "OsmAnd-java-master-snapshot.jar"
# ---------------------

def creer_et_deplacer_jar():
    print("Création de l'archive en cours...")
    
    # Nom temporaire pour l'archive zip générée par Python
    nom_base = NOM_JAR.replace(".jar", "")
    
    # 1. Création de l'archive (format zip) à partir du dossier source
    archive_generee = shutil.make_archive(base_name=nom_base, format='zip', root_dir=DOSSIER_SOURCE)
    
    # On renomme le .zip en .jar
    jar_temporaire = nom_base + ".jar"
    if os.path.exists(jar_temporaire):
        os.remove(jar_temporaire)
    os.rename(archive_generee, jar_temporaire)
    
    # Chemin final
    chemin_final = os.path.join(DOSSIER_DESTINATION, NOM_JAR)
    
    print("Déplacement et écrasement...")
    # 2. Suppression de l'ancienne version si elle existe (pour assurer un écrasement propre)
    if os.path.exists(chemin_final):
        os.remove(chemin_final)
        print(" -> Ancienne version supprimée.")
        
    # 3. Déplacement du nouveau .jar vers sa destination
    shutil.move(jar_temporaire, chemin_final)
    
    print(f"Terminé ! Le fichier se trouve ici : {chemin_final}")

if __name__ == "__main__":
    creer_et_deplacer_jar()