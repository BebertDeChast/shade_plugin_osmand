# Backend Processing for Shaded Maps

This document outlines the steps to generate a custom OsmAnd map file (`.obf`) with shade data.

## Prerequisites

- An `.pbf` file of the desired map area, downloadable from sources like [Geofabrik](https://www.geofabrik.de/).
- OsmAndMapCreator.

## Steps

1.  **Configure Shade Calculation:**
    -   Adjust the parameters in the `calculate_shade_v2.py` script to suit your needs.

2.  **Run Shade Calculation:**
    -   Execute the `calculate_shade_v2.py` script.
    -   Note: This process can be time-consuming.

3.  **Update Rendering Types:**
    -   Copy the `rendering_types_custom.xml` file.
    -   Place it inside the `OsmAnd-java-master-snapshot.jar` located in the `OSMand-map-creator/lib/` directory, at the path `net/osmand/osm/rendering_types.xml`.

4.  **Generate `.obf` File:**
    -   Run OsmAndMapCreator to generate the final `.obf` map file.
    -   It is recommended to adjust the RAM allocation for OsmAndMapCreator via the command line for better performance.

5.  **Verify Data:**
    -   Use the Inspector tool with the `-vrouting` parameter to verify that the routing and shade data are correctly included in the generated `.obf` file.