import numpy as np
import cv2

# Load the original heatmap image (without the convex hull)
original_heatmap = cv2.imread('HeatMap.jpg')

# Load the image with the convex hull and the green dots
heatmap_with_hull_and_dots = cv2.imread('HeatMapWithTrafficLight_full.jpg') # The image you provided with the convex hull and green dots

# Define the color range for the green dots
lower_green = np.array([0, 150, 0])
upper_green = np.array([100, 255, 100])

# Create a mask where the green dots are on the image with the convex hull
mask_green_dots = cv2.inRange(heatmap_with_hull_and_dots, lower_green, upper_green)

# Mask out everything but the green dots from the image with the hull
green_dots = cv2.bitwise_and(heatmap_with_hull_and_dots, heatmap_with_hull_and_dots, mask=mask_green_dots)

# Now, overlay this onto the original heatmap
# Find the contours of the green dots to create a minimal bounding box to overlay
contours_green, _ = cv2.findContours(mask_green_dots, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for contour in contours_green:
    x, y, w, h = cv2.boundingRect(contour)
    # Take the corresponding area from the green_dots image
    green_dot_area = green_dots[y:y+h, x:x+w]
    # Add this to the original heatmap image
    original_heatmap[y:y+h, x:x+w] = original_heatmap[y:y+h, x:x+w] * (1 - (green_dot_area.any(axis=-1, keepdims=True).astype(original_heatmap.dtype) / 255)) + green_dot_area

# Save the result
cv2.imwrite('heatmap_with_green_dots_only.png', original_heatmap)
