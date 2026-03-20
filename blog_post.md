# OsmAnd shade plugin 
*This post presents the results of a student project carried out by the following group: Pierrick Causeret, Humbert de Chastellux, Paul Enjalbert and Othman Ouhaddou.*

## Introduction
This project is the follow up of the project [zorun's Diary Shade-optimized routing paths for pedestrians OpenStreetMap](https://www.openstreetmap.org/user/zorun/diary/406587) which introduced a shade calculator for OSM. Our objective was to make this project available on mobile.

[Our github repository](https://github.com/BebertDeChast/shade_plugin_osmand/tree/main)

## Choosing an App

We first had to choose an app or website to work on. We wanted something easy to publish for us and easy to install for potential users. It also needed to work on both Android and iOS. This excluded all apps that didn’t have plugins or ways to customize them without editing the core code. With all those constraints, we settled on OsmAnd.

## Creating the shade data

The full pipeline integrates the [previous project](https://github.com/Jihene556/procom_calcul/tree/optimisation), which is the calculation of the shade percentage for each way. This information is then added in the .pbf file as a tag, and then converted to .obf for the app.

![Shéma du pipeline](https://github.com/BebertDeChast/shade_plugin_osmand/blob/main/images/pipeline.png?raw=true)

### Shade data computing

Shades are computed by a python script based on previous [previous project](https://github.com/Jihene556/procom_calcul/tree/optimisation). Based on Osm data, it project the shadows of buidings and trees on ways. The output is a percentage of shading. The tags are then added to each ways and storred in a new .pbf.

Format of tags ( shade then the hour is equal to the percentage shade_h_=%) :

~~~
shade10=60
shade12=80
shade14=70
~~~

*No tags are generated for hours without sun*

#### Known limitations

* The previous project was reworked to improve performance but the script is still really slow on large areas. It is a simple python script and this is probably not the best way to do it. It is also memory hungry. The new method uses a projection vector computed once per timeslot.

* The shade computation depends on the position of the latitude and longitude of the map. For optimization purposes we compute the position of the sun at the center of the map. All of this is only setted up and tested for France.

### Integration in the obf

For the shading to be accessible inside the app we have to find a way of giving it to the app.
OsmAnd uses a format called .obf for maps. We used OsmAndMapCreator to convert from .pbf to .obf. 

#### Difficulties

When the conversion between .pbf and .obf is performed, OsmAndMapCreator uses [rendering_types.xml](https://github.com/osmandapp/OsmAnd-resources/blob/master/obf_creation/rendering_types.xml) to convert tags into the new format. As we add some tags in the .pbf (see previous project). They are ignored during the process of conversion.

The first solution we found is to add the tag in the rendering_types.xml in a new section. But the documentation on that subject is limited. After a lot of testing we turned to the community and got an answer (thanks again !).

We figured out that new sections can’t be created in the osmand map creator, because in that case, the osmand app doesn’t recognize the added tag.
So that is how we finally modify the rendering_types.xml : 

~~~XML
<osmand_types>
  [...]
  <category name="routing">
    [...]
    <routing_type tag="shade00" mode="amend" base="true"/>
    <routing_type tag="shade02" mode="amend" base="true"/>
    <routing_type tag="shade04" mode="amend" base="true"/>
    <routing_type tag="shade06" mode="amend" base="true"/>
    <routing_type tag="shade08" mode="amend" base="true"/>
    <routing_type tag="shade10" mode="amend" base="true"/>
    <routing_type tag="shade12" mode="amend" base="true"/>
    <routing_type tag="shade14" mode="amend" base="true"/>
    <routing_type tag="shade16" mode="amend" base="true"/>
    <routing_type tag="shade18" mode="amend" base="true"/>
    <routing_type tag="shade20" mode="amend" base="true"/>
    <routing_type tag="shade22" mode="amend" base="true"/>
  </category>
</osmand_types>
~~~

## Routing

![Routing pipeline](https://github.com/BebertDeChast/shade_plugin_osmand/blob/main/images/pipeline_routing.png?raw=true)

The core of the project is to add a routing profile into the OsmAnd plugin. To do that, the plugin require two main elements :

* The map with the shade tags we added, in a .obf format. This map can be then activated alongside with other maps, or standalone.

* A routing.xml describing the modifications of the routine engine according to the shade tag. This file is edited following the instructions provided [here](https://github.com/osmandapp/OsmAnd-resources/blob/master/routing/routing.xml). 

There is two parts for the routing.xml:

* A parameter for the hour of the day. As we didn’t find a way of getting the current hour during the routing calculation, this information has to be provided by the user by the modification of the parameter. 

~~~XML
<parameter id="shading_hour" name="Shading Hour" description="Hour for shading" type="numeric" values="0,8,10,12,14,16,18,20" valueDescriptions="night,8h,10h,12h,14h,16h,18h,20h" default="14"/>
~~~


* The modification of the priority attribute of a way according to its shade value. Indeed the priority value is multiplied by the speed, so shaded routes are prioritized. The implementation modifies the attribute `priority`, then tests one by one which hour is selected by the user, finally tests one by one the value `v` of the corresponding tag and sets the `value` of the priority.

~~~XML
<way attribute="priority">
      <eq value1=":shading_hour" value2="8">
        <select value="0.5" t="shade08" v="0"/>
        <select value="0.505" t="shade08" v="1"/>
        <select value="0.51" t="shade08" v="2"/>
        [...]
        <select value="0.125" t="shade08" v="100"/>
	</eq>
      [...]
</way>
~~~

For the relation between shade percentage and the priority we use the one presented in the diary of last year's project.
A script is available to modify this relation (see [plugin folder](https://github.com/BebertDeChast/shade_plugin_osmand/tree/main/plugin) of the github).

## Demonstration

Here is a simple demonstration of the core result we obtained :

![Demonstration](https://github.com/BebertDeChast/shade_plugin_osmand/blob/main/images/demonstration.png?raw=true)

As you can see on the picture above, we can use the plugin directly in OsmAnd to change the route. The screenshot on the left is without the custom shade routing, the one on the right is with the custom routing. To obtain this result, we asked the app a route to get from point A to point B, first with the default configuration, and then with our custom configuration. To use the custom configuration, we selected our profile (created with the plugin) and activated our custom routing which takes into account the shadow on the path. 

#### Try it yourself 

In our GitHub, you will find a plugin (.osf file in the [Releases](https://github.com/BebertDeChast/shade_plugin_osmand/tree/main/plugin/Releases) folder) which is the functional version of our project. This plugin can be downloaded into OsmAnd mobile app, and directly used. 
At the moment, there are some limitations for the using of the plugin :
- As we only generate tags for Nantes(France), the plugin only works for this city. If you try outside of this area, there is no error, you can use the plugin as a normal pedestrian routing.
- The map is included in the plugin, thus the plugin is a bit heavy. A possible improvement is to allow the user to download this map remotely

## User Guide
We added in the GitHub some documents to explain in detail plugins and how they work. You will find this information inside the folder [plugin](https://github.com/BebertDeChast/shade_plugin_osmand/tree/main/plugin). There are several READMEs giving some details, and the pdf  [Plugin Documentation](https://github.com/BebertDeChast/shade_plugin_osmand/blob/main/plugin/Plugin%20Documentation.pdf) where you will find general documentation on plugins (how to make a plugin, how to use it, …).
You can refer to this pdf and its section 5 to have a step by step visual guide of the use of a plugin with IOS. Note that despite some minor differences, it is very similar on Android.
Moreover, there may be explanations directly inside our code.

## User Test

**Methodology & Demographics**

To evaluate the usability and integration of our *Itinéraire Ombragé* (Shaded Routing) plugin, we conducted a series of user tests using a think-aloud observation protocol.

* **Number of testers:** 20 users.

* **Demographics:** Users from IMT Atlantique.

* **Prior experience:** None of the testers had previously used the OsmAnd application.

* **Scenario:** Users were asked to install the plugin and calculate a shaded pedestrian route from IMT Atlantique to Parc de Chantrerie, with a departure time set to 14:00.

**Key Results & Observations**

The primary objective was to measure the technical and cognitive friction from discovery to generating a shaded route. Overall, **0 out of 20 testers** were able to complete the scenario successfully without intervention.

* **Installation & OS differences:** There was a noticeable difference in installation experience between operating systems. For iOS users, plugin installation was generally straightforward. Android users struggled more during this phase, indicating a need for clearer, OS-specific installation instructions.

* **Primary blocker (time selection):** The most significant friction point occurred when setting the departure time. All users experienced major difficulties or complete blocks when trying to set the time to 14:00, which prevented them from completing the test independently.

For more information see the folder [user tests](https://github.com/BebertDeChast/shade_plugin_osmand/tree/main/user%20tests)

## Future Improvement

* Improve `.obf` computation (possibly by using PostgreSQL).
* Enable remote map downloads so users can choose areas.
* Find a way to retrieve the real current time for custom routing.
* Add shade rendering.
* Keep only shading data in the created map, to reduce the size of the map.

#### About rendering
We faced throughout our project technical issues about rendering. We wanted to make a specific map rendering, which is supposed to be possible in a plugin, by using a custom rendering.render.xml file. This rendering should have taken into account the custom tags we created in our customized map, so that it renders the shade on the different roads (to get a visual representation of what the map looks like if we only look at our tag shades). However, we did not achieve this goal. We managed to change the map rendering with our custom rendering.render.xml in a plugin : see the functional examples on our GitHub [here](https://github.com/BebertDeChast/shade_plugin_osmand/tree/main/plugin/rendering_plugins). However, we could not use our custom tag. We tried to generate the .obf correctly with the custom tag in the map section, we tried to use the right syntax to call the custom tags, but no matter how far we've gone, it still doesn't work and we still don't know why.

To find more information about this subject, you can refer to the folder [rendering_plugins](https://github.com/BebertDeChast/shade_plugin_osmand/tree/main/plugin/rendering_plugins), which contains some functional examples of the use of custom rendering in a plugin, some explanations on the syntax, in addition of our latest try to use our custom tags (latest try not functional thus - documentation was added to help understand what we were trying to do). 

## Conclusion

The current state of the project makes it possible to use the shade tags introduced by the latest project. However, many avenues for improvement remain. A better understanding of how the OsmAnd plugins work could help improve usability. We thank everyone who will contribute to resolving the issues raised during this project.

If you want to download the latest version of the plugin see [here](https://github.com/BebertDeChast/shade_plugin_osmand/tree/main/plugin/Releases).
