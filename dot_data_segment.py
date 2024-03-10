import json
import subprocess
import math


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


def mercator_projection(lon, lat):
    R = 6371  # Earth's radius in kilometers
    x = R * math.radians(lon)
    y = R * math.log(math.tan(math.radians(45 + lat / 2)))
    return x, y

"""
def get_relative_direction(start_lon, start_lat, end_lon, end_lat, shape):
    start_coord = shape[0]
    end_coord = shape[-1]

    if start_coord["latitude"] == start_lat and start_coord["longitude"] == start_lon:
        if end_coord["latitude"] == end_lat and end_coord["longitude"] == end_lon:
            return "Same Location"
        else:
            return "North" if end_lat > start_lat else "South"
    elif start_coord["latitude"] == end_lat and start_coord["longitude"] == end_lon:
        if end_coord["latitude"] == start_lat and end_coord["longitude"] == start_lon:
            return "Same Location"
        else:
            return "South" if end_lat > start_lat else "North"
    elif end_coord["latitude"] == start_lat and end_coord["longitude"] == start_lon:
        return "East" if end_lon > start_lon else "West"
    elif end_coord["latitude"] == end_lat and end_coord["longitude"] == end_lon:
        return "West" if end_lon > start_lon else "East"
    else:
        return "Unknown"
"""
def get_relative_direction(start_lon, start_lat, end_lon, end_lat):
    if start_lon < end_lon:
        return "East"
    elif start_lon > end_lon:
        return "West"
    elif start_lat < end_lat:
        return "South"
    elif start_lat > end_lat:
        return "North"
    else:
        return "Same Location"


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

    dot_content = "graph.dot"
    all_streets = []
    connected_streets = []
    name_counter = {}
    # Generate DOT content for each segment
    with open(dot_content, "w") as f:
        f.write("graph G {\n")
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

                # Check if start or end coordinates match
                if start_lat == connected_start_lat and start_lon == connected_start_lon:
                    connected = True
                    print(f"{street_name}'start is connected to {connected_street['street_name']}'start")
                    """
                    start_node = f"{street_name}_{name_counter[street_name]}_start"
                    end_node = f"{connected_street['street_name']}_end"
                    distance_between_nodes = calculate_distance(start_lat, start_lon, end_lat, end_lon)

                    f.write(f'{start_node} [label="{street_name}"];\n')
                    f.write(f'{end_node} [label="{connected_street["street_name"]}"];\n')
                    f.write(
                        f'{start_node} -- {end_node} [headlabel="{distance_between_nodes:.4f} km", weight={distance_between_nodes}];\n')
                    """
                if start_lat == connected_end_lat and start_lon == connected_end_lon:
                    connected = True
                    print(f"{street_name}'start is connected to {connected_street['street_name']}'end ")
                    # check if the connected node is in S N W E
                    relative_dir = get_relative_direction(start_lon, start_lat, connected_end_lon, connected_end_lat)
                    
                    start_node = f"{street_name}_{name_counter[street_name]}_start"
                    end_node = f"{connected_street['street_name']}_end"
                    distance_between_nodes = calculate_distance(start_lat, start_lon, end_lat, end_lon)

                    f.write(f'{start_node} [label="{street_name}_start"];\n')
                    f.write(f'{end_node} [label="{connected_street["street_name"]}_end"];\n')

                    if relative_dir == "North":
                        f.write(f'{{rank=same; {end_node}; {start_node};}}\n')
                    elif relative_dir == "South":
                        f.write(f'{{rank=same; {start_node}; {end_node};}}\n')
                    elif relative_dir == "East":
                        f.write(f'{{rank=same; {start_node}; {end_node};}}\n')
                    elif relative_dir == "West":
                        f.write(f'{{rank=same; {end_node}; {start_node};}}\n')
                    f.write(
                        f'{start_node} -- {end_node} [headlabel="{distance_between_nodes:.4f} km", weight={distance_between_nodes}];\n')
                    

                if end_lat == connected_start_lat and end_lon == connected_start_lon:
                    connected = True
                    print(f"{street_name}'end is connected to {connected_street['street_name']}'start ")
                    """
                    start_node = f"{connected_street['street_name']}_start"
                    end_node = f"{street_name}_{name_counter[street_name]}_end"
                    distance_between_nodes = calculate_distance(start_lat, start_lon, end_lat, end_lon)

                    f.write(f'{start_node} [label="{connected_street["street_name"]}"];\n')
                    f.write(f'{end_node} [label="{street_name}"];\n')
                    f.write(
                        f'{start_node} -- {end_node} [headlabel="{distance_between_nodes:.4f} km", weight={distance_between_nodes}];\n')
                    """
                if end_lat == connected_end_lat and end_lon == connected_end_lon:
                    connected = True
                    print(f"{street_name}'end is connected to {connected_street['street_name']}'end ")
                    """
                    start_node = f"{connected_street['street_name']}_start"
                    end_node = f"{street_name}_{name_counter[street_name]}_end"
                    distance_between_nodes = calculate_distance(start_lat, start_lon, end_lat, end_lon)

                    f.write(f'{start_node} [label="{connected_street["street_name"]}"];\n')
                    f.write(f'{end_node} [label="{street_name}"];\n')
                    f.write(
                        f'{start_node} -- {end_node} [headlabel="{distance_between_nodes:.4f} km", weight={distance_between_nodes}];\n')
                    """

            all_streets.append({
                "street_name": street_name,
                "start_lat": start_lat,
                "start_lon": start_lon,
                "end_lat": end_lat,
                "end_lon": end_lon
            })

            if connected == True:
                continue

            # Create unique node names
            start_node = f"{street_name}_start"
            end_node = f"{street_name}_end"
            # print(start_node + ' '+ end_node)

            # Calculate the distance between start and end coordinates
            distance_between_nodes = calculate_distance(start_lat, start_lon, end_lat, end_lon)

            start_x , start_y = mercator_projection(start_lon, start_lat)
            end_x, end_y = mercator_projection(end_lon, end_lat)
            # Write start and end nodes
            f.write(f' {start_node} [label="{street_name}_start" pos="{start_x},{start_y}"];\n')
            f.write(f' {end_node} [label="{street_name}_end" pos="{end_x},{end_y}"];\n')

            # Write edge between start and end nodes
            # Apply rank=same attribute based on relative direction
            direction = get_relative_direction(start_lon, start_lat, end_lon, end_lat)
            print(direction)
            if direction == "North":
                f.write(f'{{rank=same; {end_node}; {start_node};}}\n')
            elif direction == "South":
                f.write(f'{{rank=same; {start_node}; {end_node};}}\n')
            elif direction == "East":
                f.write(f'{{rank=same; {start_node}; {end_node};}}\n')
            elif direction == "West":
                f.write(f'{{rank=same; {end_node}; {start_node};}}\n')

            f.write(
                f'{start_node} -- {end_node} [headlabel="{distance_between_nodes:.4f} km", weight={distance_between_nodes}];\n')

        f.write("}")

    # print(f"DOT file '{dot_file}' generated successfully.")

    subprocess.call(['../bin/dot', '-Tjpg', '-o', 'graph.jpg', 'graph.dot'])
