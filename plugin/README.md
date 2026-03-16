# Plugin folder

This folder is composed of two elements :

- **`generate_routing_xml.py`** : a code to generate the shade-routing.xml for a given function. Modify the `priority_function` to change the priority computing. This function takes `p` the percentage (0 to 100) and return the priority factor associated to that percentage.

- **`Plugin_Shade/Plugin_Shade`** : the folder containing the source code of the plugin. There is also the last version of .osf file. If a modification is performed, recreate a .osf file : go in the folder and zip all the elements (not the folder itself!!) then rename .zip in .osf.