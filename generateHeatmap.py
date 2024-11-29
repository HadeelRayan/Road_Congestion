import pandas as pd
import folium
from folium.plugins import HeatMap

# Step 1: Load the CSV file with speed ratios
data = pd.read_csv("roads_speeds_with_ratios_filtered.csv")

# Step 2: Filter out rows with missing or zero speed data
filtered_data = data[data['average_speed'] > 0]#.copy()

#filtered_data['inverted_ratio'] = 1 - filtered_data['speed_ratio']
# Step 3: Create a list of [latitude, longitude, intensity] for the heatmap
heat_data = [[row['latitude'], row['longitude'], row['inverted_ratio']] for _, row in filtered_data.iterrows()]

# Step 4: Initialize a Folium map centered around the average location
m = folium.Map(location=[filtered_data['latitude'].mean(), filtered_data['longitude'].mean()], zoom_start=14)

# Step 5: Add a heatmap layer using the original speed ratio
#HeatMap(heat_data, radius=10, blur=15, max_zoom=1, min_opacity=0.5).add_to(m)
HeatMap(heat_data).add_to(m)
# Step 6: Save the map to an HTML file
m.save("heatmap_filtered.html")

print("Heatmap with filtered data saved as 'heatmap_filtered.html'")

