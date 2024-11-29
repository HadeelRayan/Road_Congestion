import json
import pandas as pd

# Load the JSON file
file_path = 'jobs_3267204_results_third_try.json'  # Replace with your local file path
with open(file_path, 'r') as file:
    data = json.load(file)

# Extract data for each segment
rows = []
for segment in data['network']['segmentResults']:
    speed_limit = segment.get('speedLimit', None)  # Extract speed limit
    shape = segment.get('shape', [])  # Extract shape (latitude, longitude points)
    time_results = segment.get('segmentTimeResults', [])

    for point in shape:
        latitude = point.get('latitude')
        longitude = point.get('longitude')

        for time_result in time_results:
            average_speed = time_result.get('averageSpeed', None)

            # Calculate ratio (if speed limit and average speed are available)
            if speed_limit and average_speed:
                speed_ratio = average_speed / speed_limit if speed_limit != 0 else 0
            else:
                speed_ratio = 0

            rows.append({
                'latitude': latitude,
                'longitude': longitude,
                'speed_limit': speed_limit,
                'average_speed': average_speed,
                'speed_ratio': speed_ratio
            })

# Create a DataFrame
df = pd.DataFrame(rows)

# Step 1: Filter out rows where average_speed <= 0
df = df[df['average_speed'] > 0]

# Step 2: Add the inverted_ratio column (1 - speed_ratio)
df['inverted_ratio'] = 1 - df['speed_ratio']

# Step 3: Save to CSV
output_path = 'roads_speeds_with_ratios_filtered.csv'  # File will be saved in the same directory as the script
df.to_csv(output_path, index=False)

print(f"Filtered data with inverted_ratio saved to {output_path}")
