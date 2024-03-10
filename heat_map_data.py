import json
import subprocess
import math
from geopy.distance import great_circle
import networkx as nx
import matplotlib.pyplot as plt
import csv
import openpyxl



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
    workbook = openpyxl.Workbook()
    worksheet = workbook.active
    data = [
        ['Value', 'Latitude', 'Longitude']
    ]
    i = 0
    for segment_result in segment_results:
            #if i == 100:
            #    break
        i += 1
        #street_name = segment_result["streetName"]
        #street_name = street_name.replace(" ", "_")
        segment_id = int(segment_result["segmentId"])
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
        #connected = False
        data.append([speed_limit, start_lat, start_lon])

    for row in data:
        worksheet.append(row)

    excel_file_name = 'data.xlsx'
    workbook.save(excel_file_name)

    print(f'{len(data) - 1} rows written to {excel_file_name}')
