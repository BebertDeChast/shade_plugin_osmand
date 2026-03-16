# OsmAnd Shade Plugin

This project integrates a shade calculation algorithm into the OsmAnd application, allowing for rendering shaded areas on the map and using this data in routing decisions. The original algorithm is from [Jihene556/procom_calcul](https://github.com/Jihene556/procom_calcul.git).

## Project Structure

This repository is divided into two main parts:

-   **`/backend`**: Contains the Python scripts and tools necessary to process OpenStreetMap `.pbf` data. It calculates the shade from buildings and adds this information back into a custom `.obf` map file that OsmAnd can use. See the `backend/README.md` for detailed instructions.

-   **`/plugin`**: Contains the source code for the OsmAnd plugin itself. This plugin utilizes the custom `.obf` file to:
    -   Render shaded areas on the map.
    -   Provide a custom routing profile that favors shaded routes.

The **`images`** folder contains images for the osm diary

## Goal

The primary goal is to provide users with routing options that take shade into account, offering more comfortable routes during sunny or hot weather conditions.
