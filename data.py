import json
import math
import networkx as nx
import matplotlib.pyplot as plt




# Calculate distance between two points given their longitude and latitude
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

    G = nx.DiGraph()
    # Create a list to store connected streets
    connected_streets = []

    i = 0
    for segment_result in segment_results:
        if (i == 10):
            break
        i += 1
        street_name = segment_result["streetName"]
        street_name = street_name.split()[0]
        segment_id = int(segment_result["segmentId"])
        #new_segment_id = segment_result["newSegmentId"]
        speed_limit = segment_result["speedLimit"]
        frc = segment_result["frc"]
        distance = segment_result["distance"]
        shape = segment_result["shape"]
        start_lat = shape[0]["latitude"]
        start_lon = shape[0]["longitude"]
        end_lat = shape[1]["latitude"]
        end_lon = shape[1]["longitude"]
        segment_time_results = segment_result["segmentTimeResults"]

        # Create unique node names
        start_node = f"{street_name}_start"
        end_node = f"{street_name}_end"

        # Calculate the distance between start and end coordinates
        distance_between_nodes = calculate_distance(start_lat, start_lon, end_lat, end_lon)

        # Write start and end nodes
        G.add_node(start_node, label=f"{street_name}")
        G.add_node(end_node, label=f"{street_name}")

        # Write edge between start and end nodes
        G.add_edge(start_node, end_node, label=f"Distance: {distance_between_nodes:.4f}", weight=distance_between_nodes)

    pos = nx.spring_layout(G, weight='weight')
    nx.draw_networkx(G, pos)
    # Display the graph
    plt.show()
