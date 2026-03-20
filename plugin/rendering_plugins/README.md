# OsmAnd Shade Plugin Project

## rendering_plugins folder

In this folder, you will find ...

Three functional demonstrations on how to change the rendering with a plugin :

-   **`rendering1_everyRoadsRed/`** and **`rendering1_everyRoadsRed.osf`**: the first functional example of how to change the rendering with a plugin (changing roads color and width). **`rendering1_everyRoadsRed/`** is the folder with the code of the plugin, the .osf file **`rendering1_everyRoadsRed.osf`** is the zipped plugin that can be used directly.

-   **`rendering2_routesRed/`** and **`rendering2_routesRed.osf`**: the second functional example of how to change the rendering with a plugin (changing roads color and width).

-   **`rendering3_everyRoadsRedVersion2/`** and **`rendering3_everyRoadsRed_version2.osf`**: the third functional example of how to change the rendering with a plugin (changing roads color and width). It does almost the same as the first example, with a different method (the width changes from the first example).

One folder about unfinished tests due to technical issues :

-   **`WorkInProgess`**: Contains our latest work, not functionnal, with explanation on what we were trying to do.


#### How build a rendering.render.xml file : 

The best way to understand how to change the rendering is to look at the three examples we made. Here are simple explanations : 

- With the rendering.render.xml file, your create a new rendering, either by modifying already created parameters (examples 1 and 2) either by defining new rendering rules (example 3).
A rendering.render.xml always have the same structure : 
```xml
<!-- Define the rendering -->
<renderingStyle name="Shade rendering1" depends="default" defaultColor="#ff0000" version="1">

    <!-- Define the Properties, Attributes and Constants. Note that lots of Properties and Attribute already exist, see the default.render.xml file for more information -->
    <renderingProperty attr="roadStyle" category="ui_hidden" type="string"/>

    <renderingAttribute name="motorwayRoadColor">
        <case attrColorValue="#ff0000"/>
    </renderingAttribute>

    <!-- Define the new rules with order, text, point, polygon abd line. For more information, see the official documentation at https://osmand.net/fr/docs/technical/osmand-file-formats/osmand-rendering-style -->
    <order>
    </order>

    <text>
    </text>

    <point>
	</point>

    <polygon>
	</polygon>

    <line>
        <switch>
            <case tag="highway" value="motorway" color="#ff0000" strokeWidth="5:3"/>
            <case tag="highway" value="trunk" color="#ff0000" strokeWidth="5:3"/>
        </switch>
    </line>

</renderingStyle>
```
Note that in the example 2, we modify the
```xml
<renderingAttribute name="route">
```
This attribute is used for rendering a route, so in order to see the change bring by the new .render.xml, you need to use the navigation tool to calculate a route in OsmAnd. 
Note also that we did not find a way, through plugins, to use several colors in a route (OsmAnd seems to only accept one color for routes when modifying via plugins).

- Then, with the items.json file, you will add your rendering.render.xml file to the app by using this block of code : 
```json
{
    "type": "FILE",
    "pluginId": "shade.plugin1",
    "subtype": "rendering_style",
    "file": "\/rendering\/shade_rendering1.render.xml"
}
```
And you need to link it with your plugin with this block of code inside the definition of the plugin : 
```json
"prefs": {
    "renderer":"shade_rendering1.render.xml"
}
```


#### Remarks about the three functional examples :
Note that the 3 functional examples where created by inspiring from the official OsmAnd-resources GitHub, especially by the files : 
- default.render.xml, 
- Touring-view_(more-contrast-and-details).render.xml and
- snowmobile.render.xml

files that you can find here : https://github.com/osmandapp/OsmAnd-resources/tree/master/rendering_styles

Hints of the inspiration from Touring view and Snowmobile : 

Line 132 from Touring-view_(more-contrast-and-details).render.xml and line 3 from snowmobile.render.xml
```xml
<!-- The "Road atlas style" color options align with some conventional road atlas schemes. This was moved to default render, not any more supported in Touring view -->
<renderingProperty attr="roadStyle" category="ui_hidden" type="string"/>
```

Line 13 from snowmobile.render.xml
```xml
<renderingAttribute name="motorwayRoadColor">
    <case additional="construction=yes" attrColorValue="#ff0000">
        <apply_if nightMode="true" attrColorValue="#48616C"/>
    </case>
    <case nightMode="true" attrColorValue="#384b54">
        <apply_if additional="tunnel=yes" attrColorValue="#304048"/>
        <apply_if additional="covered=yes" attrColorValue="#304048"/>
    </case>
    <case attrColorValue="#ff0000">
        <apply_if additional="tunnel=yes" attrColorValue="#F2EBFF"/>
        <apply_if additional="covered=yes" attrColorValue="#F2EBFF"/>
    </case>
</renderingAttribute>
```

Line 136 from Touring-view_(more-contrast-and-details).render.xml
```xml
<renderingAttribute name="route">
    <case color="#960000FF" strokeWidth="10:6" color_3="#2EFF00" strokeWidth_3="5:3">
        <apply_if nightMode="true" color="#b400a0ff"/>
    </case>
</renderingAttribute>
```

Line 525 from snowmobile.render.xml
```xml
<line>
    <case minzoom="9" tag="osmand_highway" value="snowmobile_road">
        <case additional="snowmobile=yes" color="#55000000">
            <apply_if nightMode="true" color="#55999999"/>
        </case>
        ...
    </case>
    <case tag="highway" value="*">
        <apply color="#FF00FF" strokeWidth="6:6"/>
    </case>
</line>
```
