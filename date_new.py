import json
import subprocess
import math
from geopy.distance import great_circle
import networkx as nx
import matplotlib.pyplot as plt


def mercator_projection(lon, lat):
    R = 6371  # Earth's radius in kilometers
    x = R * math.radians(lon)
    y = R * math.log(math.tan(math.radians(45 + lat / 2)))
    return x, y


def calculate_distance(lat1, lon1, lat2, lon2):
    R = 6371  # Radius of the Earth in kilometers

    # Convert latitude and longitude to radians
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)

    # Calculate the differences in latitude and longitude
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    # Use the Haversine formula to calculate the distance
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    Distance = R * c
    return Distance


def calculate_bearing(lat1, lon1, lat2, lon2):
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    lon_diff = math.radians(lon2 - lon1)

    x = math.sin(lon_diff) * math.cos(lat2_rad)
    y = math.cos(lat1_rad) * math.sin(lat2_rad) - math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(lon_diff)
    bearing = math.atan2(x, y)
    return math.degrees(bearing)

if __name__ == "__main__":
    with open('jobs_3267204_results_third_try.json', 'r') as file:
        json_data = file.readline()
        data = json.loads(json_data)
    # Access specific data points
    job_name = data["jobName"]
    creation_time = data["creationTime"]
    distance_unit = data["userPreference"]["distanceUnit"]
    time_sets = data["timeSets"]
    network_name = data["network"]["networkName"]
    segment_results = data["network"]["segmentResults"]

    G = nx.Graph()
    # Create a list to store connected streets
    connected_streets = []
    name_counter = {}
    pos = {}  # Define pos dictionary to store node positions
    #g = Geod(ellps="WGS84")  # Create a Geod object for coordinate transformations
    i = 0
    for segment_result in segment_results:
        if i == 7:
            break
        i += 1
        street_name = segment_result["streetName"]
        street_name = street_name.replace(" ", "_")
        segment_id = int(segment_result["segmentId"])
        # if (segment_id < 0):
        # segment_id = segment_id * (-1)
        new_segment_id = segment_result["newSegmentId"]
        speed_limit = segment_result["speedLimit"]
        frc = segment_result["frc"]
        distance = segment_result["distance"]
        shape = segment_result["shape"]
        start_lat = shape[0]["latitude"]
        start_lon = shape[0]["longitude"]
        end_lat = shape[1]["latitude"]
        end_lon = shape[1]["longitude"]
        segment_time_results = segment_result["segmentTimeResults"]
        connected = False

        if street_name not in name_counter:
            name_counter[street_name] = 1
        else:
            name_counter[street_name] += 1

        for connected_street in connected_streets:
            connected_start_lat = connected_street["start_lat"]
            connected_start_lon = connected_street["start_lon"]
            connected_end_lat = connected_street["end_lat"]
            connected_end_lon = connected_street["end_lon"]

            if start_lat == connected_start_lat and start_lon == connected_start_lon:
                connected = True
                print(f"{street_name}'start is connected to {connected_street['street_name']}'start")

            if start_lat == connected_end_lat and start_lon == connected_end_lon:
                connected = True
                print(f"{street_name}'start is connected to {connected_street['street_name']}'end ")
                start_node = f"{street_name}_start"
                end_node = f"{street_name}_end"
                old_distance = great_circle((connected_start_lat, connected_start_lon), (connected_end_lat, connected_end_lon)).kilometers
                new_distance = great_circle((start_lat, start_lon), (end_lat, end_lon)).kilometers
                distance_between_nodes = new_distance + old_distance
                print(old_distance + new_distance)
                G.add_node(end_node, label=f"{street_name}_end", pos="{end_lon},{end_lat}")
                G.add_edge(start_node, end_node, label=f"Distance: {distance_between_nodes:.4f}", weight=distance_between_nodes)
                #pos[start_node] = (connected_start_lon, connected_start_lat)
                pos[end_node] = (end_lon, end_lat)

            if end_lat == connected_start_lat and end_lon == connected_start_lon:
                connected = True
                print(f"{street_name}'end is connected to {connected_street['street_name']}'start ")

            if end_lat == connected_end_lat and end_lon == connected_end_lon:
                connected = True
                print(f"{street_name}'end is connected to {connected_street['street_name']}'end ")

        connected_streets.append({
            "street_name": street_name,
            "start_lat": start_lat,
            "start_lon": start_lon,
            "end_lat": end_lat,
            "end_lon": end_lon
        })
        if connected:
            continue

        # Create unique node names
        start_node = f"{street_name}_start"
        end_node = f"{street_name}_end"
        # print(start_node + ' '+ end_node)
        # Calculate the distance between start and end coordinates
        distance_between_nodes = great_circle((start_lat, start_lon), (end_lat, end_lon)).kilometers
        #print(distance_between_nodes)
        # Write start and end nodes
        G.add_node(start_node, label=f"{street_name}_start", pos=f"{'start_lon'},{'start_lat'}")
        G.add_node(end_node, label=f"{street_name}_end", pos=f"{'end_lon'},{'end_lat'}")
        # Write edge between start and end nodes
        G.add_edge(start_node, end_node, label=f"Distance: {distance_between_nodes:.4f}", weight=distance_between_nodes)
        pos[start_node] = (start_lon, start_lat)
        pos[end_node] = (end_lon, end_lat)

    # Set pygraphviz graph attributes
    G.graph['edge'] = {'splines': 'true'}  # Add other attributes as needed
    G.graph['node'] = {'shape': 'circle'}  # Add other attributes as needed

    # Create the graph with positions
    nx.draw(G, pos, with_labels=True, node_size=200, node_color="lightblue", font_size=5)
    nx.draw_networkx_edges(G, pos, edge_color="gray")

    edge_labels = {(u, v): f"{G[u][v]['weight']:.4f} km" for u, v in G.edges()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color="red", font_size=6)

    # Display the graph
    plt.show()
