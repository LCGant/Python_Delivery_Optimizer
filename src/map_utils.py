import folium
from folium.plugins import MarkerCluster
import pandas as pd
import random

from function_utils import adjust_order

def generate_unique_color(colors_used):
    predefined_colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink',
                          'darkred', 'lightblue', 'lightgreen', 'magenta', 'beige', 'salmon',
                          'gray', 'black', 'white']
    available_colors = [color for color in predefined_colors if color not in colors_used]
    if not available_colors:
        raise ValueError("There are no more colors available to assign.")
    return random.choice(available_colors)

def create_map(df, df2):
    df2['Lat'] = pd.to_numeric(df2['Lat'], errors='coerce')
    df2['Long'] = pd.to_numeric(df2['Long'], errors='coerce')

    list_localities_names = []
    color_list = []

    for local_list in df['Locations']:
        local_names = [item for item in local_list if item in df2["Locations"].values]
        local_names = adjust_order(local_names, df2)
        list_localities_names.append(local_names)

    colors_used = set()

    for _ in list_localities_names:
        preset_color = generate_unique_color(colors_used)
        colors_used.add(preset_color)
        color_list.append(preset_color)

    name_dictionary = {ix: element for ix, element in enumerate(list_localities_names)}
    color_dictionary = {ix: color_list[ix] for ix in name_dictionary.keys()}

    center_lat = df2['Lat'].mean()
    center_lon = df2['Long'].mean()
    m = folium.Map(location=[center_lat, center_lon], zoom_start=12)

    marker_cluster = MarkerCluster().add_to(m)

    for key, name_list in name_dictionary.items():
        color = color_dictionary[key]
        route_coordinates = []

        for name in name_list:
            if name in df2["Locations"].values:
                line = df2[df2["Locations"] == name]
                for _, row in line.iterrows():
                    lat = row["Lat"]
                    lon = row["Long"]
                    route_coordinates.append([lat, lon])

                    folium.Marker(
                        location=[lat, lon],
                        popup=name,
                        icon=folium.Icon(color=color)
                    ).add_to(marker_cluster)

        if len(route_coordinates) > 1:
            folium.PolyLine(route_coordinates, color=color, weight=2.5, opacity=1).add_to(m)

    map_path = r"results\map.html"
    m.save(map_path)
