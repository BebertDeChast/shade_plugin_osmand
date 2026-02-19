# Backend Processing for Shaded Maps

This document outlines the procedure for enriching OpenStreetMap data with shade calculations and generating a custom OsmAnd map file (`.obf`).

## Prerequisites

* A `.pbf` file of the desired geographical area (e.g., from Geofabrik).
* OsmAndMapCreator installed on your machine.
* Python installed with necessary dependencies (`osmium`, `shapely`, `pvlib`, `pandas`, `python-dotenv`).

## Configuration (.env)

Before running the scripts, configure your environment by creating a `.env` file based on `.env.example`:

* **OSM_MAP_CREATOR_DIR**: Path to the root directory of OsmAndMapCreator.
* **CUSTOM_XML_PATH**: Path to your `rendering_types_custom.xml` file.
* **PBF_FILE_PATH**: Path to your source `.pbf` file.
* **OUTPUT_DIR**: Destination folder for the OBF and reports.

---

## Script Usage

### 1. Shade Calculation
The `calculate_shade_v2.py` script calculates the percentage of shade cast by buildings and trees onto the road at specific times. 
* **Action**: Run `python calculate_shade_v2.py`.
* **Default Settings**: It targets 10:00, 12:00, and 14:00 using a specific date (e.g., Summer Solstice).
* **Result**: Generates an enriched PBF file containing `shade10`, `shade12`, and `shade14` tags.

### 2. Updating OsmAndMapCreator
For the map creation tool to recognize your new shade tags, you must inject the XML configuration into the Java engine.
* **Action**: Run `python update_osmandmapcreator.py`.
* **Result**: Automatically replaces `rendering_types.xml` inside the `OsmAnd-java-master-snapshot.jar` with your custom version.

### 3. OBF File Generation
The `make_obf.py` script automates the final map creation and can verify data integrity.
* **Action**: Run `python make_obf.py`.
* **Inspection**: Use the `--inspect` flag to generate a CSV routing report in the output directory to verify the inclusion of shade data.

---

## Components Summary

| File | Role |
| :--- | :--- |
| `calculate_shade_v2.py` | Calculates shadows using solar position and spatial indexing (STRtree). |
| `update_osmandmapcreator.py` | Modifies the OsmAnd JAR to include custom rendering types. |
| `make_obf.py` | Runs OsmAndMapCreator to compile the PBF into an OBF file. |
| `.env` | Centralizes file paths and configuration variables. |