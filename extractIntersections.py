import pandas as pd
import folium
from sklearn.cluster import KMeans

# Step 1: Load the CSV file
csv_file = 'roads_speeds_with_ratios_filtered.csv'  # Replace with your CSV file path
data = pd.read_csv(csv_file)

# Step 2: Reduce intersections using K-Means clustering
n_clusters = 450  # Adjust this to control the number of intersections
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
data['cluster'] = kmeans.fit_predict(data[['latitude', 'longitude']])

# Step 3: Select one representative point per cluster
reduced_intersections = data.groupby('cluster').apply(lambda x: x.iloc[0])[['latitude', 'longitude']]

# Step 4: Save reduced intersections to a new CSV file
output_csv_file = "reduced_intersections.csv"
reduced_intersections.to_csv(output_csv_file, index=False)
print(f"Reduced intersections saved to '{output_csv_file}'")

# Step 5: Initialize a Folium map centered on the average location
avg_lat = reduced_intersections['latitude'].mean()
avg_lon = reduced_intersections['longitude'].mean()
m = folium.Map(location=[avg_lat, avg_lon], zoom_start=14)

# Step 6: Add green markers for each reduced intersection
for _, row in reduced_intersections.iterrows():
    folium.CircleMarker(
        location=[row['latitude'], row['longitude']],
        radius=5,  # Marker size
        color='green',
        fill=True,
        fill_color='green',
        fill_opacity=0.7
    ).add_to(m)

# Step 7: Save the map to an HTML file
html_file = "reduced_intersections_map.html"
m.save(html_file)
print(f"Map with reduced intersections saved as '{html_file}'")
