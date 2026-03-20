# OsmAnd Shade Plugin Project

## rendering_plugins folder

This folder contains the latest work we did. We were trying to use our custom tags to make a custom rendering. In the releases folder, you will see that we managed to use our custom tags for the routing. However, we did not manage to use the tags for the rendering. 

In this folder, you will find ...

-   **`rendering_types_custom2.xml`** and **`rendering_types_custom3.xml`** : files to try adding custom tag shade08 in the map section of our custom .obf

-   **`march_tests_x/`** : folder with plugin's code to try to use the custom tag shade08. 

- **`march_test5.osf`** : the plugin built from **`march_tests_x/`**. It does not do what we wanted it to do (use the custom tag in it).

We do not know if we are not able to effectively add the tag in the map .obf, or if we manage to do this, and thus we do not call correctly the tag in the .render.xml file... (the problem is either in the definition of tags in the .obf, either in the call of tags in the .render.xml file). 

### Process we followed throughout our tests

The process that is "supposed to be followed" so that it "should" work : 

- 1. Create the map with the tag in it, in the "Map" section (so it can be used but the rendering calculator). For that, refer to the backend/ folder. The rendering_types_custom.xml should be used. However, the tag should be correctly add. We try two way to add them, with **`rendering_types_custom2.xml`** **`rendering_types_custom3.xml`** : 
In rendering_types_custom2.xml : 
```xml
<!-- First Try : Modification to add tag shade08 in the Map section -->
<type tag="shade08" additional="text" minzoom="1" order="70"/>
<entity_convert pattern="tag_transform" from_tag="shade08" to_tag1="shade08" to_value1="tag" map="yes" apply_to="way"/>
<!-- First Try -->
```
In rendering_types_custom3.xml : 
```xml
<!-- Second Try : Modification to add tag shade08 in the Map section -->
<type tag="shade08" additional="text" minzoom="6" order="100"/>
<!-- try to use the example of winter_road, only occured 2 times in the files :
<type tag="winter_road" value="yes" minzoom="9" additional="true"/>
-->
<!-- Second Try -->
```

- 2. Inspect the .obf to be sure you added the tag. For that : in the terminal (see backend/ folder again) : 
```
java -cp "OsmAndMapCreator.jar;lib/*.jar" net.osmand.obf.BinaryInspector -vmap nantes_with_all_shades_test2.obf > test_rendu.txt
```
It will inspect the map "nantes_with_all_shades_test2.obf" (thus, use your map's name) and put the result of the inspection in a file "test_rendu.txt". In this file, you should see the tag you added inside the "1. Map" section. We managed to do that with our 2 try (custom2 and custom3). 

- 3. Create your .render.xml file, call the custom tag (idk how actually...). Add your file in a plugin, add your map in the plugin, and test. To see how to add a map in a plugin, see the latest plugin we made : **`march_tests_x/`**. 


