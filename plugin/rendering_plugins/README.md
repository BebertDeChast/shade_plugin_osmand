# OsmAnd Shade Plugin Project

## rendering_plugins folder

In this folder, you will find ...

-   **`/backend`**: Contains the Python scripts and tools necessary to process OpenStreetMap `.pbf` data. It calculates the shade from buildings and adds this information back into a custom `.obf` map file that OsmAnd can use. See the `backend/README.md` for detailed instructions.

-   **`/plugin`**: Contains the source code for the OsmAnd plugin itself. This plugin utilizes the custom `.obf` file to:
    -   Render shaded areas on the map.
    -   Provide a custom routing profile that favors shaded routes.

Note that these examples where created by inspiring from: 
- default.render.xml, 
- Touring-view_(more-contrast-and-details).render.xml and
- snowmobile.render.xml
files that you can find here : https://github.com/osmandapp/OsmAnd-resources/tree/master/rendering_styles

Hints of the inspiration from Touring view and Snowmobile : 
```
<!-- The "Road atlas style" color options align with some conventional road atlas schemes. This was moved to default render, not any more supported in Touring view -->
<renderingProperty attr="roadStyle" category="ui_hidden" type="string"/>

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

<renderingAttribute name="route">
    <case color="#960000FF" strokeWidth="10:6" color_3="#2EFF00" strokeWidth_3="5:3">
        <apply_if nightMode="true" color="#b400a0ff"/>
    </case>
</renderingAttribute>

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
lalala