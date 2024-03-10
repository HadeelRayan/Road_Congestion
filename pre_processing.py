import numpy as np
import cv2

# Paths to the images
green_dots_path = '/mnt/data/photo-46235789-7B5E-4BF6-B2E4-1E96A5945D54.png'
convex_hull_path = '/mnt/data/photo-70415C81-50E3-42F6-9C57-105F947ED97F.png'
heatmap_path = '/mnt/data/photo-3C306427-622D-4001-9602-3F67AA89AC5B.png'
# Load the images
green_dots_image = cv2.imread(green_dots_path, cv2.IMREAD_GRAYSCALE)
convex_hull_image = cv2.imread(convex_hull_path, cv2.IMREAD_GRAYSCALE)
heatmap_image = cv2.imread(heatmap_path, cv2.IMREAD_GRAYSCALE)
# Check if images are loaded correctly
if any(map(lambda x: x is None, [green_dots_image, convex_hull_image, heatmap_image])):
    raise ValueError("One or more images did not load correctly. Please check the file paths.")
# Normalize the images
green_dots_image = green_dots_image / 255.0
convex_hull_image = convex_hull_image / 255.0
heatmap_image = heatmap_image / 255.0

# Assuming all images should be the same size as the green dots image
target_size = green_dots_image.shape
# Resize images if necessary
convex_hull_image = cv2.resize(convex_hull_image, target_size, interpolation=cv2.INTER_AREA)
heatmap_image = cv2.resize(heatmap_image, target_size, interpolation=cv2.INTER_AREA)
# Stack the images to create a multi-channel input for the CNN
# Assuming we want to stack them along the last dimension
cnn_input = np.stack((green_dots_image, convex_hull_image, heatmap_image), axis=-1)
